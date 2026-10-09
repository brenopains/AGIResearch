"""Ajustar apenas97; comparar palavras novas com corpos congelados."""

import argparse
import json
import time

import numpy as np

from avaliar_estado_preditivo import encode
from avaliar_memoria_residual import load_data
from coletar_programas_reservados import stem as test_stem
from contrastes_programas import ProgramComparison
from confirmar_futuros_distribuidos import verify
from confirmar_leituras_variaveis import verify_manifest
from memoria_residual import MODES, MemoryDynamics
from piloto_planos_propostos import HERE, sha

ARMS = ("conditional", "shuffled_targets", "shuffled_states", "word_only", "absolute_returns")


def stem(game):
    return HERE / "resultados" / f"{game}_contrastes_programas"


def manifest_path(game):
    return stem(game).with_name(stem(game).name + "_97_manifesto.json")


def memory_path(game, mode):
    return HERE / "resultados" / f"{game}_memoria_residual_97_{mode}.npz"


def fit(game):
    output = manifest_path(game)
    if output.exists():
        raise RuntimeError("corpos de comparação já existem; preservar")
    verify()
    verify_manifest()
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    bank, z, ids, _ = load_data(game, 97)
    prior = MemoryDynamics.load(memory_path(game, "residual_history"))
    root = prior.state(z[:, 0, 0], bank["histories"][:, 0, 0], ids[:, 0, 0])
    programs, observed = bank["actions"], bank["rewards"].cumsum(-1)
    x, left, right, yl, yr, roots = [], [], [], [], [], []
    for i in range(32):
        for length in range(1, 17):
            x.append(root[i])
            left.append(programs[i, 0, :length].tolist())
            right.append(programs[i, 1, :length].tolist())
            yl.append(observed[i, 0, length - 1])
            yr.append(observed[i, 1, length - 1])
            roots.append(i)
    x, roots = np.asarray(x), np.asarray(roots)
    bodies, costs, agreement = {}, {}, {}
    for arm in ARMS:
        before_cpu, before_wall = time.process_time(), time.perf_counter()
        model = ProgramComparison().fit(x, left, right, yl, yr, roots, arm)
        path = stem(game).with_name(stem(game).name + f"_97_{arm}.npz")
        if path.exists() or path.with_suffix(".json").exists():
            raise RuntimeError("corpo existente; preservar")
        model.save(path)
        restored = ProgramComparison.load(path)
        value = model.predict(x[:32], left[:32], right[:32])
        exported = restored.predict(x[:32], left[:32], right[:32])
        error = float(np.max(np.abs(value - exported)))
        if error > 1e-12:
            raise RuntimeError("corpo exportado diverge")
        bodies[arm] = path.relative_to(HERE).as_posix()
        costs[arm] = {"cpu_seconds": time.process_time() - before_cpu, "wall_seconds": time.perf_counter() - before_wall,
                      "system_size": model.system_size, "physical_pairs": len(yl), "actual_scalar_observations": len(yl) * 2}
        agreement[arm] = error
        print(json.dumps({"game": game, "arm_fitted": arm, **costs[arm]}), flush=True)
    files = ["avaliar_contrastes_programas.py", "contrastes_programas.py", "CONTRASTES_PROGRAMAS.md",
             "coletar_programas_reservados.py", "coletar_transicoes_mistas.py", "coleta.py",
             "coletar_estado_preditivo.py", "avaliar.py", "memoria_residual.py", "avaliar_memoria_residual.py",
             f"resultados/{game}_base_estado_preditivo.npz", f"resultados/{game}_transicoes_mistas_97.npz"]
    for mode in MODES:
        p = memory_path(game, mode).relative_to(HERE).as_posix()
        files += [p, p.replace(".npz", ".json")]
    files += [p for b in bodies.values() for p in (b, b.replace(".npz", ".json"))]
    # Includes all sources and bodies; new test outcomes do not exist yet.
    report = {"status": "frozen_before_new_program_collection", "game": game, "train_seed": 97,
              "bodies": bodies, "costs": costs, "export_max_errors": agreement,
              "source_hashes": {p: sha(HERE / p) for p in files},
              "constant_program_index": int(observed[:, :, -1].mean(0).argmax()),
              "training_complete_programs": programs.reshape(64, 16).tolist(),
              "training_root_count": 32, "correlated_prefix_pairs": 512,
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    events = [{"event": "comparison_body_generated", "arm": arm, "parameters": bodies[arm],
               "test_outcomes_used_for_fit": 0, **costs[arm]} for arm in ARMS]
    output.with_suffix(".jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return report


def paired_gain(primary, reference, episode):
    values = np.asarray([(reference - primary)[episode == e].mean() for e in range(8)])
    rng = np.random.default_rng(9817)
    samples = values[rng.integers(0, 8, (10000, 8))].mean(1)
    return {"mean": float(values.mean()), "bootstrap_95": np.quantile(samples, (.025, .975)).tolist(),
            "positive_episodes": int(np.sum(values > 1e-12)), "regressed_episodes": int(np.sum(values < -1e-12)),
            "per_episode": values.tolist()}


def evaluate(game):
    output = stem(game).with_name(stem(game).name + "_novas_103.json")
    if output.exists():
        raise RuntimeError("avaliação já existe; preservar")
    manifest = json.loads(manifest_path(game).read_text(encoding="utf-8"))
    for p, expected in manifest["source_hashes"].items():
        if sha(HERE / p) != expected:
            raise RuntimeError(f"fonte/corpo/banco mudou: {p}")
    verify()
    verify_manifest()
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    test_path = test_stem(game).with_suffix(".npz")
    collection = json.loads(test_path.with_suffix(".json").read_text(encoding="utf-8"))
    if sha(test_path) != collection["bank_sha256"]:
        raise RuntimeError("banco reservado mudou")
    with np.load(test_path) as archive:
        bank = {name: archive[name] for name in archive.files}
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    z = encode(bank["signatures"].reshape(32, 4, 336), basis["mean"], basis["basis"])
    prior = MemoryDynamics.load(memory_path(game, "residual_history"))
    ids = np.arange(32) + 103 * 100000
    x = prior.state(z, bank["histories"], ids)
    observed = bank["rewards"].cumsum(-1)
    plans, episode = bank["actions"], bank["episode"]
    models = {arm: ProgramComparison.load(HERE / path) for arm, path in manifest["bodies"].items()}
    memory_predictions = {}
    for mode in MODES:
        model = MemoryDynamics.load(memory_path(game, mode))
        # Preserve the exact sample identifiers used for this root by the old control.
        old_ids = np.arange(32 * 2 * 17).reshape(32, 2, 17)[:, 0, 0] + 103 * 100000
        root = model.state(z, bank["histories"], old_ids)
        duplicated = np.repeat(root, 2, axis=0)
        memory_predictions[f"memory_{mode}"] = model.rollout(duplicated, plans.reshape(64, 16))["returns"].reshape(32, 2, 16)
    curves, events, contrasts, regrets = {}, [], {}, {}
    train_words = {tuple(p) for p in manifest["training_complete_programs"]}
    overlap = sum(tuple(p) in train_words for p in plans.reshape(64, 16))
    for horizon in (32, 64, 128):
        length, t = horizon // 8, horizon // 8 - 1
        left, right = plans[:, 0, :length].tolist(), plans[:, 1, :length].tolist()
        real = observed[:, :, t]
        real_difference = real[:, 0] - real[:, 1]
        predictions = {arm: model.predict(x, left, right) for arm, model in models.items()}
        predictions.update({name: value[:, 0, t] - value[:, 1, t] for name, value in memory_predictions.items()})
        constant = manifest["constant_program_index"]
        predictions["constant_program_index"] = np.full(32, 1. if constant == 0 else -1.)
        row = {}
        for name, difference in predictions.items():
            choice = (difference < 0).astype(int)
            regret = real.max(1) - real[np.arange(32), choice]
            error = (difference - real_difference) ** 2
            row[name] = {"mean_regret": float(regret.mean()), "contrast_mse": float(error.mean()),
                         "contrast_bias": float((difference - real_difference).mean()),
                         "predicted_difference": difference.tolist(), "per_root_regret": regret.tolist(),
                         "choices": choice.tolist(), "per_root_contrast_error": error.tolist()}
            if horizon == 128:
                regrets[name], contrasts[name] = regret, error
            for i in range(32):
                events.append({"event": "new_words_compared", "root": int(bank["roots"][i]), "episode": int(episode[i]),
                               "arm": name, "horizon": horizon, "predicted_difference": float(difference[i]),
                               "observed_returns": real[i].tolist(), "choice": int(choice[i]),
                               "reserved_outcomes_used_as_input": False})
        curves[str(horizon)] = row
    comparisons = {name: paired_gain(regrets["conditional"], value, episode) for name, value in regrets.items() if name != "conditional"}
    checks = {}
    for name, comp in comparisons.items():
        checks[f"mean_gain_vs_{name}"] = comp["mean"] > 1e-12
        checks[f"positive_lower_vs_{name}"] = comp["bootstrap_95"][0] > 1e-12
        checks[f"two_episodes_vs_{name}"] = comp["positive_episodes"] >= 2
    checks["no_regression_vs_constant"] = comparisons["constant_program_index"]["regressed_episodes"] == 0
    checks["contrast_mse_vs_absolute_returns"] = contrasts["conditional"].mean() < contrasts["absolute_returns"].mean() - 1e-12
    checks["contrast_mse_vs_shuffled_targets"] = contrasts["conditional"].mean() < contrasts["shuffled_targets"].mean() - 1e-12
    checks["all_complete_words_unseen_by_comparison_fit"] = overlap == 0
    passed = bool(all(checks.values()))
    status = "positive_oracle_development_needs_confirmation" if passed else "failed_development_gate"
    if not np.any(observed[:, 0, -1] != observed[:, 1, -1]):
        status, passed = "inconclusive_no_real_program_contrast", False
    report = {"status": status, "positive_pilot": passed, "game": game, "train_seed": 97, "test_seed": 103,
              "test_plan_seed": 9811, "curves": curves, "comparisons": comparisons, "checks": checks,
              "overlapping_complete_words": overlap, "manifest_sha256": sha(manifest_path(game)),
              "test_bank_sha256": collection["bank_sha256"], "test_collection_steps": collection["steps_used"],
              "new_ale_steps_during_evaluation": 0, "human_level_goal_achieved": False, "promoted": False,
              "scope": "initial_measured_signature_oracle_new_words_in_known_development_worlds",
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    output.with_suffix(".jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ajustar", action="store_true")
    mode.add_argument("--avaliar", action="store_true")
    args = parser.parse_args()
    result = fit(args.game) if args.ajustar else evaluate(args.game)
    print(json.dumps({"game": args.game, "status": result["status"], "cpu_seconds": result["cpu_seconds"],
                      **({} if args.ajustar else {"horizon128": {name: {"regret": row["mean_regret"], "mse": row["contrast_mse"]}
                          for name, row in result["curves"]["128"].items()}, "word_overlap": result["overlapping_complete_words"]})}, indent=2))
