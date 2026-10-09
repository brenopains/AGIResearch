"""História passada real e sondas custeadas em sequências mistas pré-fixadas."""

import argparse
import json
import time
from collections import deque

import numpy as np

from coleta import gray, make_ale, small
from coletar_estado_preditivo import probe
from confirmar_futuros_distribuidos import verify
from confirmar_leituras_variaveis import verify_manifest
from piloto_planos_propostos import HERE, sha

LIMIT = 70000


def stem(game, seed):
    return HERE / "resultados" / f"{game}_transicoes_mistas_{seed}"


def advance_history(ale, action, history, budget):
    total, used = 0.0, 0
    for _ in range(8):
        if budget[0] <= 0:
            raise RuntimeError("orçamento esgotado")
        total += float(ale.act(action))
        budget[0] -= 1
        used += 1
        history.append(gray(ale))
        if ale.game_over():
            break
    return total, used, bool(ale.game_over())


def compressed_history(history):
    return np.stack([small(frame, 21) for frame in history]).reshape(16, 336)


def collect(game, seed):
    output = stem(game, seed)
    if any(output.with_suffix(s).exists() for s in (".npz", ".json", ".jsonl")):
        raise RuntimeError("banco já existe; preservar")
    verify()
    verify_manifest()
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    horizon_path = HERE / "resultados" / f"{game}_horizontes_{seed}.npz"
    short_path = HERE / "resultados" / f"{game}_efeitos_locais_{seed}.npz"
    expected = np.load(horizon_path)["histories"]
    short = np.load(short_path)
    ale, walk_rng, budget = make_ale(game, seed), np.random.default_rng(seed), [LIMIT]
    actions = list(ale.getMinimalActionSet()[:4])
    plans = np.random.default_rng(9791).integers(0, 4, (64, 2, 16))
    # Save the exogenous plans before measuring any of their outcomes.
    plan_path = output.with_name(output.name + "_planos.json")
    if plan_path.exists():
        raise RuntimeError("planos anteriores existem; preservar")
    plan_record = {"status": "plans_fixed_before_collection", "game": game, "seed": seed,
                   "plans": plans.tolist(), "source_sha256": sha(HERE / "coletar_transicoes_mistas.py"),
                   "protocol_sha256": sha(HERE / "MEMORIA_RESIDUAL.md")}
    plan_path.write_text(json.dumps(plan_record, indent=2), encoding="utf-8")
    all_h, all_s, all_v, all_r, all_a, episodes, roots, events = [], [], [], [], [], [], [], []
    certified_roots, certified_short, certified_replays = 0, 0, 0
    for episode in range(8):
        ale.reset_game()
        frame = gray(ale)
        history = deque([frame.copy() for _ in range(16)], maxlen=16)
        for checkpoint in range(8):
            for _ in range(128):
                if budget[0] <= 0:
                    raise RuntimeError("orçamento esgotado na caminhada")
                ale.act(actions[int(walk_rng.integers(0, 4))])
                budget[0] -= 1
                if ale.game_over():
                    ale.reset_game()
                    frame = gray(ale)
                    history = deque([frame.copy() for _ in range(16)], maxlen=16)
                else:
                    history.append(gray(ale))
            root = episode * 8 + checkpoint
            if not np.array_equal(np.stack(history), expected[root]):
                raise RuntimeError("histórico de raiz divergiu")
            certified_roots += 1
            if checkpoint >= 4:
                continue
            state, pair = ale.cloneSystemState(), np.stack(history)[-2:]
            before = budget[0]
            signature, r, length, done = probe(ale, actions, state, pair, budget)
            reference = np.stack([small(short["outcomes"][root, a], 21) - small(pair[-1], 21) for a in range(4)])
            if (not np.array_equal(signature, reference) or not np.array_equal(r, short["rewards"][root])
                    or not np.array_equal(length, short["lengths"][root]) or not np.array_equal(done, short["terminal"][root])):
                raise RuntimeError("sondas iniciais divergem")
            certified_short += 4
            events.append({"event": "root_probes", "root": root, "physical_steps": before - budget[0]})
            paths = []
            for j in range(2):
                ale.restoreSystemState(state)
                branch = deque([f.copy() for f in history], maxlen=16)
                hs, ss, vs, rs = [compressed_history(branch)], [signature], [True], []
                ended, elapsed, probes_used = False, 0, 0
                for t, action in enumerate(plans[root, j]):
                    if ended:
                        gain = 0.0
                    else:
                        gain, used, ended = advance_history(ale, actions[int(action)], branch, budget)
                        elapsed += used
                    rs.append(gain)
                    hs.append(compressed_history(branch))
                    vs.append(not ended)
                    if ended:
                        ss.append(np.zeros((4, 21, 16), np.float32))
                    else:
                        waypoint = ale.cloneSystemState()
                        before_probe = budget[0]
                        signature_next, _, _, _ = probe(ale, actions, waypoint, np.stack(branch)[-2:], budget)
                        probes_used += before_probe - budget[0]
                        ss.append(signature_next)
                paths.append((np.stack(hs), np.stack(ss), np.asarray(vs), np.asarray(rs)))
                events.append({"event": "mixed_trajectory", "root": root, "episode": episode, "path": j,
                               "plan": plans[root, j].tolist(), "trajectory_steps": elapsed,
                               "probe_steps": probes_used, "physical_steps": elapsed + probes_used,
                               "rewards_by_block": rs, "valid": vs, "history_includes_probe_frames": False})
                if j == 0 and checkpoint == 0:
                    final = gray(ale)
                    ale.restoreSystemState(state)
                    replay_h = deque([f.copy() for f in history], maxlen=16)
                    replay_gains, replay_elapsed, replay_ended = [], 0, False
                    for action in plans[root, j]:
                        if replay_ended:
                            replay_gains.append(0.0)
                        else:
                            gain, used, replay_ended = advance_history(ale, actions[int(action)], replay_h, budget)
                            replay_gains.append(gain)
                            replay_elapsed += used
                    if (not np.array_equal(rs, replay_gains) or elapsed != replay_elapsed or ended != replay_ended
                            or not np.array_equal(gray(ale), final) or not np.array_equal(compressed_history(replay_h), hs[-1])):
                        raise RuntimeError("replay da trajetória divergiu")
                    certified_replays += 1
                    events.append({"event": "trajectory_replay", "root": root, "physical_steps": replay_elapsed})
            all_h.append(np.stack([p[0] for p in paths]))
            all_s.append(np.stack([p[1] for p in paths]))
            all_v.append(np.stack([p[2] for p in paths]))
            all_r.append(np.stack([p[3] for p in paths]))
            all_a.append(plans[root])
            episodes.append(episode)
            roots.append(root)
            ale.restoreSystemState(state)
        print(json.dumps({"game": game, "seed": seed, "episode": episode,
                          "mixed_roots": len(roots), "steps": LIMIT - budget[0]}), flush=True)
    np.savez_compressed(output.with_suffix(".npz"), histories=np.stack(all_h), signatures=np.stack(all_s),
                        valid=np.stack(all_v), rewards=np.stack(all_r), actions=np.stack(all_a),
                        roots=np.asarray(roots), episode=np.asarray(episodes))
    with output.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    if 8192 + sum(e["physical_steps"] for e in events) != LIMIT - budget[0]:
        raise RuntimeError("contabilidade divergente")
    verify()
    verify_manifest()
    report = {"status": "mixed_observations_collected_not_agent", "game": game, "seed": seed,
              "steps_used": LIMIT - budget[0], "certified_walk_roots": certified_roots,
              "certified_root_probes": certified_short, "certified_replays": certified_replays,
              "collected_roots": roots, "observed_valid_transitions": int(np.stack(all_v)[..., :-1].sum()),
              "source_sha256": sha(HERE / "coletar_transicoes_mistas.py"),
              "protocol_sha256": sha(HERE / "MEMORIA_RESIDUAL.md"), "plans_sha256": sha(plan_path),
              "bank_sha256": sha(output.with_suffix(".npz")), "events_sha256": sha(output.with_suffix(".jsonl")),
              "existing_bank_hashes": {p.name: sha(p) for p in (horizon_path, short_path)},
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall,
              "human_level_goal_achieved": False}
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("--seed", type=int, choices=(97, 103), required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.game, args.seed), ensure_ascii=False, indent=2))
