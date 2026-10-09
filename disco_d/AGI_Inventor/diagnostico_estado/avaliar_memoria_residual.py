"""Ajuste separado, corpos congelados e avaliação oracular explícita."""

import argparse
import json
import time

import numpy as np

from avaliar_estado_preditivo import encode
from coletar_transicoes_mistas import stem as bank_stem
from confirmar_futuros_distribuidos import verify
from confirmar_leituras_variaveis import verify_manifest
from memoria_residual import MODES, MemoryDynamics, crossfit_residual
from piloto_planos_propostos import HERE, sha


def stem(game):
    return HERE / "resultados" / f"{game}_memoria_residual"


def manifest_path(game):
    return stem(game).with_name(stem(game).name + "_97_manifesto.json")


def load_data(game, seed):
    path = bank_stem(game, seed).with_suffix(".npz")
    report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if sha(path) != report["bank_sha256"]:
        raise RuntimeError("banco mudou após coleta")
    bank = np.load(path)
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    z = encode(bank["signatures"].reshape(32, 2, 17, 4, 336), basis["mean"], basis["basis"])
    ids = np.arange(32 * 2 * 17).reshape(32, 2, 17) + seed * 100000
    return bank, z, ids, report


def transitions(bank, z, ids):
    mask = bank["valid"][..., :-1].reshape(-1)
    def selected(value):
        return value.reshape(-1, *value.shape[3:])[mask]
    return {"z": selected(z[..., :-1, :]), "next_z": selected(z[..., 1:, :]),
            "history": selected(bank["histories"][..., :-1, :, :]),
            "next_history": selected(bank["histories"][..., 1:, :, :]),
            "reward": bank["rewards"].reshape(-1)[mask],
            "continuation": bank["valid"][..., 1:].reshape(-1)[mask].astype(float),
            "actions": bank["actions"].reshape(-1)[mask],
            "episodes": np.broadcast_to(bank["episode"][:, None, None], (32, 2, 16)).reshape(-1)[mask],
            "ids": ids[..., :-1].reshape(-1)[mask], "next_ids": ids[..., 1:].reshape(-1)[mask]}


def fit(game):
    output = manifest_path(game)
    if output.exists():
        raise RuntimeError("corpos congelados já existem; preservar")
    verify()
    verify_manifest()
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    bank, z, ids, collection = load_data(game, 97)
    data = transitions(bank, z, ids)
    residual = crossfit_residual(data["z"], data["next_z"], data["reward"], data["continuation"],
                                 data["actions"], data["episodes"])
    bodies, agreements, counts, events = {}, {}, {}, []
    for mode in MODES:
        model = MemoryDynamics().fit(**{k: v for k, v in data.items() if k != "episodes"},
                                     residual=residual, mode=mode)
        path = stem(game).with_name(stem(game).name + f"_97_{mode}.npz")
        if path.exists() or path.with_suffix(".json").exists():
            raise RuntimeError("corpo existente; preservar")
        model.save(path)
        restored = MemoryDynamics.load(path)
        query = model.state(data["z"][:16], data["history"][:16], data["ids"][:16])
        exported = restored.state(data["z"][:16], data["history"][:16], data["ids"][:16])
        error = float(np.max(np.abs(query - exported)))
        plans = np.tile([0, 1, 2, 3], (16, 4))
        prediction = model.rollout(query, plans)["returns"]
        reproduced = restored.rollout(exported, plans)["returns"]
        rollout_error = float(np.max(np.abs(prediction - reproduced)))
        if error > 1e-12 or rollout_error > 1e-12:
            raise RuntimeError("corpo exportado diverge")
        bodies[mode] = path.relative_to(HERE).as_posix()
        agreements[mode] = {"state_max_error": error, "return_max_error": rollout_error}
        counts[mode] = model.counts
        events.append({"event": "memory_body_generated", "mode": mode, "rank": 32,
                       "train_seed": 97, "cases_by_action": model.counts, "parameters": bodies[mode],
                       "test_observations_used_for_fit": 0})
        print(json.dumps({"game": game, "body_fitted": mode, "counts": model.counts,
                          "cpu_seconds": time.process_time() - start_cpu}), flush=True)
    if len({tuple(v) for v in counts.values()}) != 1:
        raise RuntimeError("orçamento de observações diferente entre braços")
    files = ["avaliar_memoria_residual.py", "memoria_residual.py", "MEMORIA_RESIDUAL.md", "avaliar.py",
             "coletar_transicoes_mistas.py", "CORRECAO_COLETA_MISTA.md", "avaliar_estado_preditivo.py"]
    files += [f"resultados/{game}_base_estado_preditivo.npz"]
    files += [bank_stem(game, 97).with_suffix(s).relative_to(HERE).as_posix() for s in (".npz", ".json", ".jsonl")]
    files += [value for p in bodies.values() for value in (p, p.replace(".npz", ".json"))]
    report = {"status": "bodies_frozen_before_test_evaluation", "game": game, "train_seed": 97,
              "bodies": bodies, "training_cases": len(data["reward"]), "cases_by_action": counts,
              "source_hashes": {p: sha(HERE / p) for p in files}, "export_agreement": agreements,
              "training_collection_steps": collection["steps_used"], "new_ale_steps_during_fit": 0,
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    output.with_suffix(".jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return report


def paired_gain(proposed_error, reference_error, episode):
    value = np.asarray([(reference_error - proposed_error)[episode == e].mean() for e in range(8)])
    rng = np.random.default_rng(9797)
    distribution = value[rng.integers(0, 8, (10000, 8))].mean(1)
    return {"mean": float(value.mean()), "bootstrap_95": np.quantile(distribution, (.025, .975)).tolist(),
            "positive_episodes": int(np.sum(value > 1e-12)), "regressed_episodes": int(np.sum(value < -1e-12)),
            "per_episode": value.tolist()}


def evaluate(game):
    output = stem(game).with_name(stem(game).name + "_103.json")
    if output.exists():
        raise RuntimeError("avaliação já existe; preservar")
    manifest = json.loads(manifest_path(game).read_text(encoding="utf-8"))
    for p, expected in manifest["source_hashes"].items():
        if sha(HERE / p) != expected:
            raise RuntimeError(f"arquivo congelado mudou: {p}")
    verify()
    verify_manifest()
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    bank, z, ids, collection = load_data(game, 103)
    data = transitions(bank, z, ids)
    observed = bank["rewards"].cumsum(-1)
    models, scores, predictions, local, contrast_errors, regrets = {}, {}, {}, {}, {}, {}
    episode = bank["episode"]
    for mode, body in manifest["bodies"].items():
        model = MemoryDynamics.load(HERE / body)
        models[mode] = model
        query = model.state(data["z"], data["history"], data["ids"])
        predicted_r, predicted_c, predicted_z = np.zeros(len(query)), np.zeros(len(query)), np.zeros_like(data["z"])
        for action in range(4):
            selected = data["actions"] == action
            successor, r, c = model.predict(query[selected], action)
            predicted_r[selected], predicted_c[selected] = r, c
            predicted_z[selected] = successor[:, :32] * np.sqrt(2) * model.scale_z + model.zmean
        survives = data["continuation"] > 0
        local[mode] = {"reward_mse": float(np.mean((predicted_r - data["reward"]) ** 2)),
                       "reward_bias": float((predicted_r - data["reward"]).mean()),
                       "continuation_brier": float(np.mean((predicted_c - data["continuation"]) ** 2)),
                       "surviving_state_mse": float(np.mean((predicted_z[survives] - data["next_z"][survives]) ** 2))}
        # The composition receives only the two real initial observations/histories.
        root = model.state(z[:, :, 0].reshape(64, 32), bank["histories"][:, :, 0].reshape(64, 16, 336),
                           ids[:, :, 0].reshape(64))
        rollout = model.rollout(root, bank["actions"].reshape(64, 16))
        returns = rollout["returns"].reshape(32, 2, 16)
        predictions[mode] = returns
        row = {}
        for horizon in (32, 64, 128):
            t = horizon // 8 - 1
            true, pred = observed[:, :, t], returns[:, :, t]
            difference_error = ((pred[:, 0] - pred[:, 1]) - (true[:, 0] - true[:, 1])) ** 2
            choices = pred.argmax(1)
            regret = true.max(1) - true[np.arange(32), choices]
            row[str(horizon)] = {"return_mse": float(np.mean((pred - true) ** 2)),
                                 "return_bias": float((pred - true).mean()),
                                 "contrast_mse": float(difference_error.mean()),
                                 "mean_regret": float(regret.mean()), "choices": choices.tolist(),
                                 "per_root_contrast_error": difference_error.tolist(), "per_root_regret": regret.tolist(),
                                 "per_root_prediction": pred.tolist()}
            if horizon == 128:
                contrast_errors[mode], regrets[mode] = difference_error, regret
        scores[mode] = row
    primary = "residual_history"
    comparisons = {mode: {"contrast": paired_gain(contrast_errors[primary], contrast_errors[mode], episode),
                           "decision": paired_gain(regrets[primary], regrets[mode], episode)}
                   for mode in MODES if mode != primary}
    checks = {}
    for mode, comp in comparisons.items():
        checks[f"contrast_mean_vs_{mode}"] = comp["contrast"]["mean"] > 1e-12
        checks[f"contrast_lower_vs_{mode}"] = comp["contrast"]["bootstrap_95"][0] > 1e-12
        checks[f"contrast_two_episodes_vs_{mode}"] = comp["contrast"]["positive_episodes"] >= 2
        checks[f"decision_mean_vs_{mode}"] = comp["decision"]["mean"] > 1e-12
    checks["no_regression_vs_state_only"] = comparisons["state_only"]["decision"]["regressed_episodes"] == 0
    for mode in ("state_only", "shuffled_residuals"):
        checks[f"local_reward_vs_{mode}"] = local[primary]["reward_mse"] < local[mode]["reward_mse"] - 1e-12
    passed = bool(all(checks.values()))
    status = "positive_oracle_diagnostic_not_agent" if passed else "failed_diagnostic_gate"
    if not np.any(observed[:, 0, -1] != observed[:, 1, -1]):
        status, passed = "inconclusive_no_sequence_contrast", False
    report = {"status": status, "positive_diagnostic": passed, "game": game,
              "train_seed": 97, "test_seed": 103, "test_valid_transitions": len(data["reward"]),
              "local": local, "curves": scores, "comparisons": comparisons, "checks": checks,
              "human_level_goal_achieved": False, "promoted": False,
              "scope": "measured_signature_and_history_initially_oracle_diagnostic_not_perception_agent",
              "manifest_sha256": sha(manifest_path(game)), "test_bank_sha256": collection["bank_sha256"],
              "test_collection_steps": collection["steps_used"], "new_ale_steps_during_evaluation": 0,
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    events = []
    for mode in MODES:
        for root in range(32):
            events.append({"event": "frozen_model_evaluated", "mode": mode, "root": int(bank["roots"][root]),
                           "episode": int(episode[root]), "predicted_returns": predictions[mode][root].tolist(),
                           "observed_returns": observed[root].tolist(), "future_inputs_during_rollout": False})
    output.with_suffix(".jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--ajustar", action="store_true")
    action.add_argument("--avaliar", action="store_true")
    args = parser.parse_args()
    result = fit(args.game) if args.ajustar else evaluate(args.game)
    summary = {"game": args.game, "status": result["status"], "cpu_seconds": result["cpu_seconds"]}
    if args.avaliar:
        summary.update({"local": result["local"], "horizon128": {mode: {k: v for k, v in row["128"].items()
                         if not k.startswith("per_root") and k != "choices"} for mode, row in result["curves"].items()}})
    print(json.dumps(summary, indent=2))
