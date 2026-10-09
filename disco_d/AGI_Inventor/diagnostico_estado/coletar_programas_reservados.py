"""Programas novos pré-fixados, sem sondas intermediárias ou reset do ramo."""

import argparse
import json
import time
from collections import deque

import numpy as np

from coleta import gray, make_ale, small
from coletar_estado_preditivo import probe
from coletar_transicoes_mistas import advance_history, compressed_history
from confirmar_futuros_distribuidos import verify
from confirmar_leituras_variaveis import verify_manifest
from piloto_planos_propostos import HERE, sha

LIMIT = 24000


def stem(game):
    return HERE / "resultados" / f"{game}_programas_reservados_103"


def play(ale, actions, plan, history, budget):
    branch = deque([f.copy() for f in history], maxlen=16)
    rewards, valid, ended, elapsed = [], [True], False, 0
    for action in plan:
        if ended:
            gain = 0.0
        else:
            gain, used, ended = advance_history(ale, actions[int(action)], branch, budget)
            elapsed += used
        rewards.append(gain)
        valid.append(not ended)
    return np.asarray(rewards), np.asarray(valid), elapsed, branch[-1].copy(), compressed_history(branch)


def collect(game):
    output = stem(game)
    if any(output.with_suffix(s).exists() for s in (".npz", ".json", ".jsonl")):
        raise RuntimeError("banco reservado já existe; preservar")
    verify()
    verify_manifest()
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    expected_path = HERE / "resultados" / f"{game}_horizontes_103.npz"
    short_path = HERE / "resultados" / f"{game}_efeitos_locais_103.npz"
    expected, short = np.load(expected_path)["histories"], np.load(short_path)
    plans = np.random.default_rng(9811).integers(0, 4, (64, 2, 16))
    plan_path = output.with_name(output.name + "_planos.json")
    if plan_path.exists():
        raise RuntimeError("planos reservados já existem; preservar")
    plan_path.write_text(json.dumps({"status": "fixed_before_execution", "plans": plans.tolist(),
                        "game": game, "world_seed": 103, "plan_seed": 9811,
                        "source_sha256": sha(HERE / "coletar_programas_reservados.py"),
                        "protocol_sha256": sha(HERE / "CONTRASTES_PROGRAMAS.md")}, indent=2), encoding="utf-8")
    ale, rng, budget = make_ale(game, 103), np.random.default_rng(103), [LIMIT]
    actions = list(ale.getMinimalActionSet()[:4])
    histories, signatures, rewards, valid, lengths, program_rows, roots, episodes, events = [], [], [], [], [], [], [], [], []
    certified_roots, certified_probes, certified_replays = 0, 0, 0
    for episode in range(8):
        ale.reset_game()
        frame = gray(ale)
        history = deque([frame.copy() for _ in range(16)], maxlen=16)
        for checkpoint in range(8):
            for _ in range(128):
                if budget[0] <= 0:
                    raise RuntimeError("orçamento esgotado")
                ale.act(actions[int(rng.integers(0, 4))])
                budget[0] -= 1
                if ale.game_over():
                    ale.reset_game()
                    frame = gray(ale)
                    history = deque([frame.copy() for _ in range(16)], maxlen=16)
                else:
                    history.append(gray(ale))
            root = episode * 8 + checkpoint
            if not np.array_equal(np.stack(history), expected[root]):
                raise RuntimeError("raiz reservada diverge")
            certified_roots += 1
            if checkpoint >= 4:
                continue
            state = ale.cloneSystemState()
            pair = np.stack(history)[-2:]
            before = budget[0]
            signature, gain, length, done = probe(ale, actions, state, pair, budget)
            reference = np.stack([small(short["outcomes"][root, a], 21) - small(pair[-1], 21) for a in range(4)])
            if (not np.array_equal(signature, reference) or not np.array_equal(gain, short["rewards"][root])
                    or not np.array_equal(length, short["lengths"][root]) or not np.array_equal(done, short["terminal"][root])):
                raise RuntimeError("sonda reservada diverge")
            certified_probes += 4
            events.append({"event": "root_probes", "root": root, "physical_steps": before - budget[0]})
            rows = []
            for j in range(2):
                ale.restoreSystemState(state)
                before = budget[0]
                row = play(ale, actions, plans[root, j], history, budget)
                rows.append(row)
                events.append({"event": "program_executed", "root": root, "episode": episode, "program_index": j,
                               "plan": plans[root, j].tolist(), "rewards_by_block": row[0].tolist(),
                               "valid": row[1].tolist(), "physical_steps": before - budget[0], "program_steps": row[2]})
                if j == 0 and checkpoint == 0:
                    ale.restoreSystemState(state)
                    replay = play(ale, actions, plans[root, j], history, budget)
                    if any(not np.array_equal(a, b) for a, b in zip(row, replay)):
                        raise RuntimeError("replay reservado divergiu")
                    certified_replays += 1
                    events.append({"event": "replay_certified", "root": root, "physical_steps": replay[2]})
            histories.append(compressed_history(history))
            signatures.append(signature)
            rewards.append(np.stack([r[0] for r in rows]))
            valid.append(np.stack([r[1] for r in rows]))
            lengths.append([r[2] for r in rows])
            program_rows.append(plans[root])
            roots.append(root)
            episodes.append(episode)
            ale.restoreSystemState(state)
        print(json.dumps({"game": game, "episode": episode, "steps": LIMIT - budget[0]}), flush=True)
    np.savez_compressed(output.with_suffix(".npz"), histories=np.stack(histories), signatures=np.stack(signatures),
                        rewards=np.stack(rewards), valid=np.stack(valid), lengths=np.asarray(lengths),
                        actions=np.stack(program_rows), roots=np.asarray(roots), episode=np.asarray(episodes))
    output.with_suffix(".jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    if 8192 + sum(e["physical_steps"] for e in events) != LIMIT - budget[0]:
        raise RuntimeError("contabilidade divergente")
    report = {"status": "reserved_programs_measured_not_agent", "game": game, "world_seed": 103, "plan_seed": 9811,
              "steps_used": LIMIT - budget[0], "certified_roots": certified_roots,
              "certified_short_probes": certified_probes, "certified_replays": certified_replays,
              "source_sha256": sha(HERE / "coletar_programas_reservados.py"),
              "protocol_sha256": sha(HERE / "CONTRASTES_PROGRAMAS.md"), "plans_sha256": sha(plan_path),
              "bank_sha256": sha(output.with_suffix(".npz")), "events_sha256": sha(output.with_suffix(".jsonl")),
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall,
              "human_level_goal_achieved": False}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    print(json.dumps(collect(parser.parse_args().game), indent=2))
