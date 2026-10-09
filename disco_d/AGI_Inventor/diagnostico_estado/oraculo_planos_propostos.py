"""Mede estados reais intermediários para localizar a falha dos planos."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from avaliar_estado_preditivo import encode, load
from coleta import gray, make_ale, small
from coletar_estado_preditivo import advance, probe
from confirmar_futuros_distribuidos import verify
from confirmar_leituras_variaveis import verify_manifest
from futuros_distribuidos import make_edges, weights
from piloto_planos_propostos import proposal_path, sha, stem
from valor_composto import fit_credit

HERE = Path(__file__).resolve().parent
POLICIES = ("distributed_search", "reward_search")
MAX_STEPS = 65000


def collect(game):
    out = HERE / "resultados" / f"{game}_oraculo_planos_propostos_103"
    if out.with_suffix(".json").exists() or out.with_suffix(".npz").exists():
        raise RuntimeError("diagnóstico já possui artefatos; preservar")
    verify()
    verify_manifest()
    proposals = json.loads(proposal_path(game).read_text(encoding="utf-8"))
    pilot_path = stem(game).with_suffix(".json")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    pilot_events = [json.loads(s) for s in pilot_path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    prior = {(row["root"], row["policy"]): row for row in pilot_events}
    for p, expected in proposals["source_hashes"].items():
        if sha(HERE / p) != expected:
            raise RuntimeError("fonte da proposta mudou")
    pilot_hash = sha(pilot_path)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    horizon = np.load(HERE / "resultados" / f"{game}_horizontes_103.npz")
    short = np.load(HERE / "resultados" / f"{game}_efeitos_locais_103.npz")
    expected = horizon["histories"]
    ale, rng, budget = make_ale(game, 103), np.random.default_rng(103), [MAX_STEPS]
    actions = list(ale.getMinimalActionSet()[:4])
    all_signature, all_valid, all_reward, roots, groups, events = [], [], [], [], [], []
    certified_roots, certified_short, certified_returns = 0, 0, 0
    for episode in range(8):
        ale.reset_game()
        initial = gray(ale)
        history = deque([initial.copy() for _ in range(16)], maxlen=16)
        for checkpoint in range(8):
            for _ in range(128):
                if budget[0] <= 0:
                    raise RuntimeError("orçamento esgotado")
                ale.act(actions[int(rng.integers(0, 4))])
                budget[0] -= 1
                if ale.game_over():
                    ale.reset_game()
                    initial = gray(ale)
                    history = deque([initial.copy() for _ in range(16)], maxlen=16)
                else:
                    history.append(gray(ale))
            i = episode * 8 + checkpoint
            current = np.stack(history)
            if not np.array_equal(current, expected[i]):
                raise RuntimeError("histórico divergiu")
            certified_roots += 1
            if checkpoint >= 4:
                continue
            root_state, root_pair = ale.cloneSystemState(), current[-2:]
            root_signature, rewards, lengths, terminal = probe(ale, actions, root_state, root_pair, budget)
            reference = np.stack([small(short["outcomes"][i, a], 21) - small(root_pair[-1], 21) for a in range(4)])
            if (not np.array_equal(root_signature, reference)
                    or not np.array_equal(rewards, short["rewards"][i])
                    or not np.array_equal(lengths, short["lengths"][i])
                    or not np.array_equal(terminal, short["terminal"][i])):
                raise RuntimeError("sonda inicial divergiu")
            certified_short += 4
            rows = []
            measured = {}
            for policy in POLICIES:
                plan = tuple(proposals["policies"][policy]["plans"][i])
                if plan in measured:
                    row = measured[plan]
                    events.append({"event": "measurement_reused", "root": i, "policy": policy,
                                   "physical_steps": 0})
                else:
                    before_steps = budget[0]
                    pair, state, ended, elapsed = root_pair.copy(), root_state, False, 0
                    signatures, valid, gains = [root_signature], [True], []
                    for stage, action in enumerate(plan):
                        if ended:
                            gain = 0.0
                        else:
                            ale.restoreSystemState(state)
                            pair, gain, used, ended = advance(ale, actions[action], pair, budget)
                            elapsed += used
                            state = ale.cloneSystemState()
                        gains.append(gain)
                        if ended:
                            signatures.append(np.zeros((4, 21, 16), np.float32))
                        else:
                            signature, _, _, _ = probe(ale, actions, state, pair, budget)
                            signatures.append(signature)
                        valid.append(not ended)
                    row = (np.stack(signatures), np.asarray(valid), np.asarray(gains), elapsed, ended)
                    measured[plan] = row
                    events.append({"event": "trajectory_measured", "root": i, "policy": policy,
                                   "physical_steps": before_steps - budget[0], "plan": list(plan),
                                   "rewards_by_block": gains, "valid": valid,
                                   "scope": "oracle_real_interventions_not_predicted_states"})
                if (float(row[2].sum()) != pilot["returns"][policy]["per_root"][i]
                        or row[3] != prior[(i, policy)]["executed_steps"]
                        or row[4] != prior[(i, policy)]["terminal"]):
                    raise RuntimeError("plano divergiu do piloto")
                certified_returns += 1
                rows.append(row)
            all_signature.append(np.stack([r[0] for r in rows]))
            all_valid.append(np.stack([r[1] for r in rows]))
            all_reward.append(np.stack([r[2] for r in rows]))
            roots.append(i)
            groups.append(episode)
            ale.restoreSystemState(root_state)
        print(json.dumps({"game": game, "episode": episode, "diagnostic_roots": len(roots),
                          "steps": MAX_STEPS - budget[0]}), flush=True)
    np.savez_compressed(out.with_suffix(".npz"), signatures=np.stack(all_signature),
                        valid=np.stack(all_valid), rewards=np.stack(all_reward),
                        roots=np.asarray(roots), episode=np.asarray(groups))
    with out.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    _, _, train_signature, train_valid, _ = load(game, 97)
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    train_z = encode(train_signature, basis["mean"], basis["basis"])
    train_reward = np.load(HERE / "resultados" / f"{game}_horizontes_97.npz")["rewards"]
    credits, ranges, _ = fit_credit(train_z, train_valid, train_reward)
    edges = make_edges(train_z, train_valid, train_reward)
    real_z = encode(np.stack(all_signature).reshape(32, 2, 17, 4, 336), basis["mean"], basis["basis"])
    real_valid, real_reward = np.stack(all_valid), np.stack(all_reward)
    diagnostic = {}
    for j, policy in enumerate(POLICIES):
        observed = real_reward[:, j].sum(1)
        original = np.asarray(proposals["policies"]["distributed_search"]["values"])[roots] if j == 0 else None
        # The reward-shuffled policy must be judged by the normal model too.
        from planejador_distribuido import ExperiencePlanner
        normal = ExperiencePlanner(edges)
        original_values = []
        predicted_nn, predicted_ridge = np.zeros(32), np.zeros(32)
        for k, i in enumerate(roots):
            plan = proposals["policies"][policy]["plans"][i]
            original_values.append(float(normal.plan_values(real_z[k:k + 1, j, 0], [plan])[0, 0]))
            for stage, action in enumerate(plan):
                if not real_valid[k, j, stage]:
                    continue
                query = real_z[k:k + 1, j, stage]
                predicted_nn[k] += float((weights(query, edges[action]["source"]) @ edges[action]["reward"])[0])
                predicted_ridge[k] += float(np.clip(credits[action].predict(query)[0, 0], *ranges[action]))
        metrics = {}
        candidates = {"real_initial_only_normal_model": np.asarray(original_values),
                      "real_states_local_neighbor_credit": predicted_nn,
                      "real_states_local_ridge_credit": predicted_ridge,
                      "true_reward_oracle": real_reward[:, j].sum(1)}
        if original is not None:
            candidates["frozen_root_prediction"] = original
        for name, predicted in candidates.items():
            metrics[name] = {"mse": float(np.mean((predicted - observed) ** 2)),
                             "bias": float((predicted - observed).mean()), "mean_prediction": float(predicted.mean()),
                             "per_root": predicted.tolist()}
        diagnostic[policy] = {"observed_mean_return": float(observed.mean()), "estimators": metrics}
    if sha(pilot_path) != pilot_hash:
        raise RuntimeError("piloto mudou")
    verify()
    verify_manifest()
    report = {"status": "post_pilot_oracle_diagnostic_not_promoted", "game": game,
              "diagnostic": diagnostic, "sample_roots": roots, "certified_roots": certified_roots,
              "certified_short_branches": certified_short, "certified_program_returns": certified_returns,
              "steps_used": MAX_STEPS - budget[0], "bank_sha256": sha(out.with_suffix(".npz")),
              "pilot_sha256_unchanged": pilot_hash, "proposals_sha256": sha(proposal_path(game)),
              "source_sha256": sha(Path(__file__)), "protocol_sha256": sha(HERE / "ORACULO_PLANOS_PROPOSTOS.md"),
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall,
              "scope": "measured_future_states_not_available_to_root_policy"}
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    report = collect(args.game)
    print(json.dumps({"game": args.game, "steps": report["steps_used"],
                      "diagnostic": {p: {n: {k: v for k, v in x.items() if k != "per_root"}
                                         for n, x in r["estimators"].items()}
                                     for p, r in report["diagnostic"].items()}}, indent=2))
