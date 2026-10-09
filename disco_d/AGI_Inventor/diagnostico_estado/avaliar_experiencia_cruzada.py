"""Ajustar somente131; avaliar as duas partições137 sem alterar corpos."""

import argparse
import json
import time
from collections import defaultdict

import numpy as np

from contrastes_programas import ProgramComparison
from experiencia_cruzada import ARMS, PARTITIONS, PROPOSAL, TEST_SEED, TRAIN_SEED, base, fitted_model, gate, manifest_path, predict, result_path, save_json, training_data, verify_proposal
from leituras_por_programa import ConditionalReading
from piloto_planos_propostos import HERE, sha


def fit(game):
    verify_proposal()
    output = manifest_path(game)
    if output.exists() or base(game, TEST_SEED).with_suffix(".npz").exists():
        raise RuntimeError("ajuste já existe ou teste foi coletado; preservar")
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    h, left, right, yl, yr, roots = training_data(game)
    bodies, costs, events = {}, {}, []
    for arm in ARMS:
        before_cpu, before_wall = time.process_time(), time.perf_counter()
        if arm == "raw_rbf":
            model = ProgramComparison().fit(h.reshape(512, -1), left, right, yl, yr, roots, "conditional")
        else:
            model = ConditionalReading().fit(h, left, right, yl, yr, roots, arm)
        path = base(game, TRAIN_SEED).with_name(base(game, TRAIN_SEED).name + f"_{arm}.npz")
        if path.exists() or path.with_suffix(".json").exists():
            raise RuntimeError("corpo existente; preservar")
        model.save(path)
        restored = fitted_model(path, arm)
        error = float(np.max(np.abs(predict(model, arm, h[:32], left[:32], right[:32])
                                    - predict(restored, arm, h[:32], left[:32], right[:32]))))
        if error > 1e-10:
            raise RuntimeError("exportação diverge")
        bodies[arm] = path.relative_to(HERE).as_posix()
        costs[arm] = {"cpu_seconds": time.process_time() - before_cpu, "wall_seconds": time.perf_counter() - before_wall,
                      "system_size": model.system_size, "correlated_prefix_pairs": 512,
                      "scalar_outcomes_used_for_fit": 1024, "export_error": error}
        events.append({"event": "crossed_reading_body_generated", "arm": arm, "parameters": bodies[arm],
                       "reserved_outcomes_used_for_fit": 0, **costs[arm]})
        checkpoint = output.with_name(output.stem + "_checkpoint.json")
        save_json(checkpoint, {"status": "partial_fit_not_eligible_for_evaluation", "completed_arms": list(bodies),
                               "proposal_sha256": sha(PROPOSAL)})
        print(json.dumps({"game": game, "fitted": arm, **costs[arm]}), flush=True)
    event_path = output.with_suffix(".jsonl")
    event_path.write_text("".join(json.dumps(e, allow_nan=False) + "\n" for e in events), encoding="utf-8")
    files = [PROPOSAL, event_path]
    files += [base(game, TRAIN_SEED).with_suffix(s) for s in (".npz", ".json", ".jsonl")]
    files += [HERE / p for b in bodies.values() for p in (b, b.replace(".npz", ".json"))]
    report = {"status": "frozen_before137_collection", "game": game, "train_seed": TRAIN_SEED, "bodies": bodies, "costs": costs,
              "source_hashes": {p.relative_to(HERE).as_posix(): sha(p) for p in files}, "proposal_sha256": sha(PROPOSAL),
              "training_real_trajectories": 128, "training_physical_scalar_returns_at_all_prefixes": 2048,
              "scalar_returns_used_for_fit": 1024, "correlated_prefix_pairs": 512, "reserved_outcomes_used_for_fit": 0,
              "primary": "quadratic", "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall}
    save_json(output, report)
    return report


def observation_collisions(history, real, roots):
    groups = defaultdict(list)
    for i, h in enumerate(history):
        groups[np.ascontiguousarray(h).tobytes()].append(i)
    witnesses = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        first = indices[0]
        for other in indices[1:]:
            if not np.array_equal(real[first], real[other]):
                witnesses.append({"root_a": int(roots[first]), "root_b": int(roots[other]),
                                  "returns_a": real[first].tolist(), "returns_b": real[other].tolist()})
    return {"identical_history_groups": sum(len(v) > 1 for v in groups.values()), "witnesses": witnesses,
            "insufficiency_witness_found": bool(witnesses), "absence_does_not_prove_sufficiency": True}


def evaluate(game):
    verify_proposal()
    output = result_path(game)
    if output.exists():
        raise RuntimeError("avaliação existente; preservar")
    manifest = json.loads(manifest_path(game).read_text(encoding="utf-8"))
    for p, expected in manifest["source_hashes"].items():
        if sha(HERE / p) != expected:
            raise RuntimeError("corpo/banco mudou")
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    with np.load(base(game, TEST_SEED).with_suffix(".npz")) as archive:
        history, programs, rewards, episodes, roots = (archive[k] for k in ("histories", "programs", "rewards", "episode", "roots"))
    observed = rewards.cumsum(-1)
    models = {arm: fitted_model(HERE / p, arm) for arm, p in manifest["bodies"].items()}
    partitions, events = {}, []
    for name, pairs in PARTITIONS.items():
        curves = {}
        for horizon in (32, 64, 128):
            length, t = horizon // 8, horizon // 8 - 1
            left = [programs[p, :length].tolist() for _ in range(32) for p, q in pairs]
            right = [programs[q, :length].tolist() for _ in range(32) for p, q in pairs]
            h = np.repeat(history, 2, axis=0)
            real = np.stack([observed[:, [p, q], t] for p, q in pairs], axis=1)
            rows = {}
            for arm, model in models.items():
                delta = predict(model, arm, h, left, right).reshape(32, 2)
                choices = (delta < 0).astype(int)
                selected = np.take_along_axis(real, choices[:, :, None], axis=2)[:, :, 0]
                regret = real.max(2) - selected
                error = (delta - (real[:, :, 0] - real[:, :, 1])) ** 2
                rows[arm] = {"mean_regret": float(regret.mean()), "per_root_regret": regret.mean(1).tolist(),
                             "per_comparison_regret": regret.tolist(), "contrast_mse": float(error.mean()),
                             "predicted_difference": delta.tolist(), "choices": choices.tolist()}
                for i in range(32):
                    for pair, (p, q) in enumerate(pairs):
                        events.append({"event": "crossed_reserved_decision_measured", "partition": name,
                                       "arm": arm, "horizon": horizon, "root": int(roots[i]), "episode": int(episodes[i]),
                                       "program_indices": [p, q], "pair": pair, "predicted_difference": float(delta[i, pair]),
                                       "observed_returns": real[i, pair].tolist(), "choice": int(choices[i, pair]),
                                       "regret": float(regret[i, pair]), "reserved_outcomes_used_as_input": False})
            curves[str(horizon)] = rows
        real = np.stack([observed[:, [p, q], -1] for p, q in pairs], axis=1)
        verdict = gate(curves["128"], real, episodes)
        verdict.update({"curves": curves, "observation_collisions128": observation_collisions(history, real, roots)})
        partitions[name] = verdict
        events.append({"event": "crossed_partition_gate_decided", "partition": name, "status": verdict["status"],
                       "positive": verdict["positive"], "failed_checks": [k for k, v in verdict["checks"].items() if not v], "promoted": False})
    positive = bool(all(p["positive"] for p in partitions.values()))
    report = {"status": "joint_reserved_test_positive_needs_other_worlds" if positive else "joint_capability_not_demonstrated",
              "joint_positive": positive, "game": game, "partitions": partitions, "train_seed": TRAIN_SEED, "test_seed": TEST_SEED,
              "manifest_sha256": sha(manifest_path(game)), "proposal_sha256": sha(PROPOSAL),
              "test_bank_sha256": sha(base(game, TEST_SEED).with_suffix(".npz")), "scope": "new_seeds_in_two_known_worlds",
              "human_level_goal_achieved": False, "promoted": False, "test_outcomes_used_for_fit": 0, "new_ale_calls_during_evaluation": 0,
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall}
    save_json(output, report)
    output.with_suffix(".jsonl").write_text("".join(json.dumps(e, allow_nan=False) + "\n" for e in events), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ajustar", action="store_true")
    mode.add_argument("--avaliar", action="store_true")
    args = parser.parse_args()
    result = fit(args.game) if args.ajustar else evaluate(args.game)
    summary = {"game": args.game, "status": result["status"], "cpu_seconds": result["cpu_seconds"], "wall_seconds": result["wall_seconds"]}
    if args.avaliar:
        summary["partitions"] = {name: {"status": p["status"], "identification": p["identification"],
                                        "regrets128": {a: row["mean_regret"] for a, row in p["curves"]["128"].items()},
                                        "failed_checks": [k for k, v in p["checks"].items() if not v]}
                                 for name, p in result["partitions"].items()}
    print(json.dumps(summary, indent=2, allow_nan=False))
