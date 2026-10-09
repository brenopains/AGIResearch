"""Leituras de pixels condicionadas ao programa: ajuste97, desenvolvimento103."""

import argparse
import json
import time

import numpy as np

from contrastes_programas import ProgramComparison
from confirmar_futuros_distribuidos import verify
from confirmar_leituras_variaveis import verify_manifest
from leituras_por_programa import ConditionalReading, ReadingBody
from piloto_planos_propostos import HERE, sha

ARMS = ("quadratic", "affine", "two_frames", "shuffled_targets", "shuffled_histories", "word_only", "raw_rbf")
HORIZONS = (32, 64, 128)


def stem(game):
    return HERE / "resultados" / f"{game}_leituras_por_programa"


def manifest_path(game):
    return stem(game).with_name(stem(game).name + "_97_manifesto.json")


def reference_path(game):
    return HERE / "resultados" / f"{game}_contrastes_programas_novas_103.json"


def bank_path(game, train=False):
    name = f"{game}_transicoes_mistas_97" if train else f"{game}_programas_reservados_103"
    return HERE / "resultados" / (name + ".npz")


def training_data(game):
    # Deliberately never access signatures, hidden state, or any103 outcomes.
    with np.load(bank_path(game, True)) as archive:
        history = archive["histories"][:, 0, 0].copy()
        plans, rewards = archive["actions"], archive["rewards"].cumsum(-1)
    left, right, yl, yr = [], [], [], []
    for i in range(len(history)):
        for length in range(1, 17):
            left.append(plans[i, 0, :length].tolist())
            right.append(plans[i, 1, :length].tolist())
            yl.append(rewards[i, 0, length - 1])
            yr.append(rewards[i, 1, length - 1])
    return np.repeat(history, 16, axis=0), left, right, np.asarray(yl), np.asarray(yr), np.repeat(np.arange(len(history)), 16), plans


def validate_dependencies(game):
    verify()
    verify_manifest()
    prior_path = HERE / "resultados" / f"{game}_contrastes_programas_97_manifesto.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    for p, expected in prior["source_hashes"].items():
        if sha(HERE / p) != expected:
            raise RuntimeError(f"dependência anterior mudou: {p}")


def fitted_model(path, arm):
    return ProgramComparison.load(path) if arm == "raw_rbf" else ConditionalReading.load(path)


def predict(model, arm, history, left, right):
    value = history.reshape(len(history), -1) if arm == "raw_rbf" else history
    return model.predict(value, left, right)


def fit(game):
    output = manifest_path(game)
    if output.exists():
        raise RuntimeError("ajuste já congelado; preservar")
    validate_dependencies(game)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    h, left, right, yl, yr, roots, plans = training_data(game)
    bodies, readings, costs, agreement, events = {}, {}, {}, {}, []
    for arm in ARMS:
        before_cpu, before_wall = time.process_time(), time.perf_counter()
        if arm == "raw_rbf":
            model = ProgramComparison().fit(h.reshape(len(h), -1), left, right, yl, yr, roots, "conditional")
        else:
            model = ConditionalReading().fit(h, left, right, yl, yr, roots, arm)
        path = stem(game).with_name(stem(game).name + f"_97_{arm}.npz")
        if path.exists() or path.with_suffix(".json").exists():
            raise RuntimeError("corpo existente; preservar")
        model.save(path)
        restored = fitted_model(path, arm)
        value = predict(model, arm, h[:32], left[:32], right[:32])
        exported = predict(restored, arm, h[:32], left[:32], right[:32])
        error = float(np.max(np.abs(value - exported)))
        dual_error = 0. if arm == "raw_rbf" else float(np.max(np.abs(value - model.predict_dual(h[:32], left[:32], right[:32]))))
        if max(error, dual_error) > 1e-10:
            raise RuntimeError("execução do corpo diverge da definição ou da exportação")
        examples = []
        if arm != "raw_rbf":
            for root in range(2):
                for side in range(2):
                    word = plans[root, side].tolist()
                    body_path = stem(game).with_name(stem(game).name + f"_97_{arm}_leitura_{root}_{side}.npz")
                    if body_path.exists():
                        raise RuntimeError("leitura existente; preservar")
                    body = model.construct([word])
                    body.save(body_path)
                    reading_error = float(np.max(np.abs(body.apply(h[:32]) - ReadingBody.load(body_path).apply(h[:32]))))
                    if reading_error > 1e-10:
                        raise RuntimeError("leitura exportada diverge")
                    examples.append({"word": word, "parameters": body_path.relative_to(HERE).as_posix(), "export_error": reading_error})
        bodies[arm], readings[arm] = path.relative_to(HERE).as_posix(), examples
        costs[arm] = {"cpu_seconds": time.process_time() - before_cpu, "wall_seconds": time.perf_counter() - before_wall,
                      "system_size": model.system_size, "correlated_prefix_pairs": len(yl), "actual_scalar_observations": len(yl) * 2}
        agreement[arm] = {"restored": error, "body_vs_dual": dual_error}
        events.append({"event": "reading_generator_fitted", "arm": arm, "parameters": bodies[arm],
                       "test_outcomes_used_for_fit": 0, "probe_signatures_used_for_fit": 0, **costs[arm]})
        for example in examples:
            events.append({"event": "complete_reading_exported", "arm": arm, **example})
        print(json.dumps({"game": game, "fitted": arm, **costs[arm]}), flush=True)
    output.with_suffix(".jsonl").write_text("".join(json.dumps(e, allow_nan=False) + "\n" for e in events), encoding="utf-8")
    files = ["avaliar_leituras_por_programa.py", "leituras_por_programa.py", "LEITURAS_POR_PROGRAMA.md",
             "testes_leituras_por_programa.py", "auditar_leituras_por_programa.py", "contrastes_programas.py", "avaliar.py",
             bank_path(game, True).relative_to(HERE).as_posix(), bank_path(game).relative_to(HERE).as_posix(),
             reference_path(game).relative_to(HERE).as_posix(),
             output.with_suffix(".jsonl").relative_to(HERE).as_posix(),
             f"resultados/{game}_contrastes_programas_97_manifesto.json"]
    files += [p for b in bodies.values() for p in (b, b.replace(".npz", ".json"))]
    files += [p for examples in readings.values() for e in examples for p in (e["parameters"], e["parameters"].replace(".npz", ".json"))]
    report = {"status": "frozen_before_development_evaluation", "game": game, "train_seed": 97,
              "bodies": bodies, "readings": readings, "costs": costs, "export_max_errors": agreement,
              "source_hashes": {p: sha(HERE / p) for p in files}, "training_complete_programs": plans.reshape(-1, 16).tolist(),
              "training_root_count": len(h) // 16, "correlated_prefix_pairs": len(h),
              "reserved_outcomes_used_as_fit_input": False, "probe_signatures_used_as_fit_input": False,
              "test_bank_already_used_in_previous_development": True, "evaluation_is_independent_confirmation": False,
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def paired_gain(primary, reference, episode):
    episode = np.asarray(episode)
    if set(episode.tolist()) != set(range(8)):
        raise ValueError("a bancada exige oito episódios declarados")
    values = np.asarray([(reference - primary)[episode == e].mean() for e in range(8)])
    rng = np.random.default_rng(9827)
    samples = values[rng.integers(0, 8, (10000, 8))].mean(1)
    return {"mean": float(values.mean()), "bootstrap_95": np.quantile(samples, (.025, .975)).tolist(),
            "positive_episodes": int(np.sum(values > 1e-12)), "regressed_episodes": int(np.sum(values < -1e-12)),
            "per_episode": values.tolist()}


def gate(curve, episode, real):
    primary = np.asarray(curve["quadratic"]["per_root_regret"])
    comparisons = {name: paired_gain(primary, np.asarray(row["per_root_regret"]), episode)
                   for name, row in curve.items() if name != "quadratic"}
    checks = {}
    for name, comp in comparisons.items():
        checks[f"mean_gain_vs_{name}"] = bool(comp["mean"] > 1e-12)
        checks[f"positive_lower_vs_{name}"] = bool(comp["bootstrap_95"][0] > 1e-12)
        checks[f"two_episodes_vs_{name}"] = bool(comp["positive_episodes"] >= 2)
    checks["no_regression_vs_old_constant"] = bool(comparisons["old_constant_program_index"]["regressed_episodes"] == 0)
    checks["contrast_mse_vs_affine"] = bool(curve["quadratic"]["contrast_mse"] < curve["affine"]["contrast_mse"] - 1e-12)
    checks["contrast_mse_vs_shuffled_targets"] = bool(curve["quadratic"]["contrast_mse"] < curve["shuffled_targets"]["contrast_mse"] - 1e-12)
    checks["real_contrast_exists"] = bool(np.any(real[:, 0] != real[:, 1]))
    return comparisons, checks, bool(all(checks.values()))


def evaluate(game):
    output = stem(game).with_name(stem(game).name + "_novas_103.json")
    if output.exists():
        raise RuntimeError("avaliação já existe; preservar")
    manifest = json.loads(manifest_path(game).read_text(encoding="utf-8"))
    for p, expected in manifest["source_hashes"].items():
        if sha(HERE / p) != expected:
            raise RuntimeError(f"fonte/corpo/banco mudou: {p}")
    validate_dependencies(game)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    with np.load(bank_path(game)) as archive:
        h, plans, rewards, episode, roots = (archive[k] for k in ("histories", "actions", "rewards", "episode", "roots"))
    observed = rewards.cumsum(-1)
    reference = json.loads(reference_path(game).read_text(encoding="utf-8"))
    if reference["test_bank_sha256"] != sha(bank_path(game)):
        raise RuntimeError("referências não pertencem ao mesmo banco")
    models = {arm: fitted_model(HERE / p, arm) for arm, p in manifest["bodies"].items()}
    train_words = {tuple(p) for p in manifest["training_complete_programs"]}
    overlap = sum(tuple(p) in train_words for p in plans.reshape(-1, 16))
    if overlap:
        raise RuntimeError("palavras completas reservadas coincidem com treino")
    curves, events = {}, []
    for horizon in HORIZONS:
        length, t = horizon // 8, horizon // 8 - 1
        left, right = plans[:, 0, :length].tolist(), plans[:, 1, :length].tolist()
        real, row = observed[:, :, t], {}
        for arm, model in models.items():
            difference = predict(model, arm, h, left, right)
            choice = (difference < 0).astype(int)
            regret = real.max(1) - real[np.arange(len(real)), choice]
            error = (difference - (real[:, 0] - real[:, 1])) ** 2
            row[arm] = {"mean_regret": float(regret.mean()), "contrast_mse": float(error.mean()),
                        "contrast_bias": float((difference - (real[:, 0] - real[:, 1])).mean()),
                        "predicted_difference": difference.tolist(), "per_root_regret": regret.tolist(),
                        "choices": choice.tolist(), "per_root_contrast_error": error.tolist()}
            events.append({"event": "program_readings_executed" if arm != "raw_rbf" else "raw_rbf_comparison_executed",
                           "arm": arm, "horizon": horizon, "reading_count": len(h) * 2,
                           "input": "past_pixels_and_words_only", "reserved_outcomes_used_as_input": False})
            for i in range(len(h)):
                events.append({"event": "reserved_program_decision_measured", "root": int(roots[i]), "episode": int(episode[i]),
                               "arm": arm, "horizon": horizon, "predicted_difference": float(difference[i]),
                               "observed_returns": real[i].tolist(), "choice": int(choice[i]), "regret": float(regret[i]),
                               "reserved_outcomes_used_as_input": False})
        # Previous oracle-containing controls stay labelled as old references.
        for name, old_row in reference["curves"][str(horizon)].items():
            row["old_" + name] = old_row
        curves[str(horizon)] = row
    comparisons, checks, passed = gate(curves["128"], episode, observed[:, :, -1])
    status = "positive_raw_development_needs_confirmation" if passed else "failed_development_gate"
    if not checks["real_contrast_exists"]:
        status = "inconclusive_no_real_program_contrast"
    events.append({"event": "development_gate_decided", "primary": "quadratic", "positive": passed,
                   "promoted": False, "failed_checks": [k for k, v in checks.items() if not v]})
    report = {"status": status, "positive_pilot": passed, "game": game, "train_seed": 97, "test_seed": 103,
              "test_plan_seed": 9811, "curves": curves, "comparisons": comparisons, "checks": checks,
              "overlapping_complete_words": overlap, "manifest_sha256": sha(manifest_path(game)),
              "test_bank_sha256": sha(bank_path(game)), "old_reference_sha256": sha(reference_path(game)),
              "new_ale_steps": 0, "human_level_goal_achieved": False, "promoted": False,
              "independent_confirmation": False, "probe_signatures_used_as_input": False,
              "scope": "known_world_development_raw_past_readings_with_fixed_feature_equivalent",
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
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
    print(json.dumps({"game": args.game, "status": result["status"], "cpu_seconds": result["cpu_seconds"],
                      "wall_seconds": result["wall_seconds"], **({} if args.ajustar else {
                          "horizon128": {name: {"regret": row["mean_regret"], "mse": row["contrast_mse"]}
                                         for name, row in result["curves"]["128"].items()},
                          "failed_checks": [k for k, v in result["checks"].items() if not v]})}, indent=2, allow_nan=False))
