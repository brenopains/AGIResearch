"""Diagnóstico somente leitura da lei de execução nos corpos congelados."""

import os
import platform
import sys

if sys.platform == "win32":
    platform.machine = platform._get_machine_win32
    platform.system = lambda: "Windows"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from experiencia_cruzada_v2 import (ARMS, fitted_model, manifest_path,
                                   predict, validate_dependencies, verify_proposal)

HERE = Path(__file__).resolve().parent
STEM = HERE / "resultados" / "semantica_prefixo_comum"
FREEZE = STEM.with_name(STEM.name + "_manifesto.json")
RESULT = STEM.with_suffix(".json")
EVENTS = STEM.with_suffix(".jsonl")
QUERIES = ("full_initial", "suffix_initial", "suffix_actual_midpoint_oracle")
EPS = 1e-12


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_new(path, value):
    # Exclusive creation: old evidence is never overwritten.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def common_prefix(left, right):
    k = 0
    while k < min(len(left), len(right)) and left[k] == right[k]:
        k += 1
    return k


def bank_path(game, seed):
    return HERE / "resultados" / f"{game}_transicoes_mistas_{seed}.npz"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def physical_case(bank, row):
    left, right = bank["actions"][row]
    k = common_prefix(left, right)
    require(0 < k < len(left) == len(right) == 16, "prefixo fora do contrato")
    h, v, r = bank["histories"][row], bank["valid"][row], bank["rewards"][row]
    require(np.array_equal(h[0, :k + 1], h[1, :k + 1]), "histórias de prefixo divergem")
    require(np.array_equal(v[0, :k + 1], v[1, :k + 1]), "validade de prefixo diverge")
    require(np.array_equal(r[0, :k], r[1, :k]), "recompensa de prefixo diverge")
    require(np.isfinite(h).all() and np.isfinite(r).all(), "banco não finito")
    require(not np.any((~v[:, :-1]) & v[:, 1:]), "validade revive depois do término")
    require(not np.any(r[~v[:, :-1]] != 0), "recompensa após término")
    difference = r[0].sum() - r[1].sum()
    suffix = r[0, k:].sum() - r[1, k:].sum()
    require(difference == suffix, "lei física de cancelamento falhou")
    return {"row": int(row), "root": int(bank["roots"][row]), "episode": int(bank["episode"][row]),
            "prefix_length": k, "left": left.tolist(), "right": right.tolist(),
            "actual_delta": float(difference), "actual_suffix_delta": float(suffix),
            "terminal_at_midpoint": (not bool(v[0, k])),
            "terminal_at_end": [not bool(value) for value in v[:, -1]],
            "physical_identity_error": float(abs(difference - suffix))}


def metrics(actual, prediction):
    actual, prediction = np.asarray(actual, float), np.asarray(prediction, float)
    require(actual.shape == prediction.shape and actual.size > 0, "métriques incompatíveis")
    require(np.isfinite(actual).all() and np.isfinite(prediction).all(), "prédictions non finies")
    left_cost, right_cost = np.maximum(-actual, 0), np.maximum(actual, 0)
    tied = np.abs(prediction) <= EPS
    cost = np.where(prediction >= -EPS, left_cost, right_cost)
    fair = np.where(tied, np.abs(actual) / 2, cost)
    real = np.abs(actual) > EPS
    directional = np.where(tied, .5, (np.sign(prediction) == np.sign(actual)).astype(float))
    return {"count": len(actual), "real_nonzero": int(real.sum()), "prediction_ties": int(tied.sum()),
            "mse": float(np.mean((prediction - actual) ** 2)),
            "deterministic_left_tie_regret": float(cost.mean()),
            "fair_tie_regret": float(fair.mean()),
            "directional_accuracy_on_real_nonzero": float(directional[real].mean()) if real.any() else None}


def prepare():
    require(not FREEZE.exists() and not RESULT.exists() and not EVENTS.exists(), "artefatos já existem; preservar")
    verify_proposal()
    sources = ["diagnosticar_semantica_prefixo.py", "auditar_semantica_prefixo.py",
               "testes_semantica_prefixo.py", "SEMANTICA_PREFIXO_COMUM.md",
               "PROXIMA_LEITURA_SEMANTICA.md", "experiencia_cruzada_v2.py",
               "avaliar_leituras_por_programa.py", "leituras_por_programa.py", "contrastes_programas.py", "avaliar.py"]
    selections, models = {}, {}
    for game in ("breakout", "pong"):
        validate_dependencies(game)
        model_manifest = manifest_path(game)
        trained = json.loads(model_manifest.read_text(encoding="utf-8"))
        require(trained["status"] == "frozen_before137_collection", "corpos não congelados")
        for name, expected in trained["source_hashes"].items():
            require(sha(HERE / name) == expected, f"dependência131 mudou: {name}")
        sources.append(model_manifest.relative_to(HERE).as_posix())
        models[game] = trained["bodies"]
        for body in models[game].values():
            sources.extend((body, str(Path(body).with_suffix(".json"))))
        for seed in (97, 103):
            path = bank_path(game, seed)
            sources.extend(path.with_suffix(ext).relative_to(HERE).as_posix() for ext in (".npz", ".json", ".jsonl"))
            certificate = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            require(certificate["bank_sha256"] == sha(path), "certificado físico do banco diverge")
            # Only syntax is read during selection; no rewards or sensory labels.
            with np.load(path, allow_pickle=False) as archive:
                plans = archive["actions"]
            lengths = [common_prefix(*plans[i]) for i in range(len(plans))]
            selections[f"{game}_{seed}"] = {"rows": [i for i, k in enumerate(lengths) if 0 < k < 16],
                                            "all_prefix_lengths": lengths, "all_roots": len(plans)}
    value = {"status": "frozen_before_diagnostic_predictions", "development_only": True,
             "training_seed": 131, "seeds": [97, 103], "selection": "longest_common_prefix_syntax_only",
             "selections": selections, "models": models, "queries": list(QUERIES),
             "primary_horizon_blocks": 16, "secondary_horizons": "k+1..16_correlated",
             "no_fit": True, "no_ale_calls": True, "no_promotion": True,
             "source_hashes": {s: sha(HERE / s) for s in sorted(set(sources))}}
    save_new(FREEZE, value)
    return value


def verify_freeze():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    for name, expected in freeze["source_hashes"].items():
        require(sha(HERE / name) == expected, f"artefato congelado mudou: {name}")
    verify_proposal()
    for game in freeze["models"]:
        validate_dependencies(game)
    return freeze


def run():
    freeze = verify_freeze()
    require(not RESULT.exists() and not EVENTS.exists(), "diagnóstico já registrado; preservar")
    wall, cpu = time.perf_counter(), time.process_time()
    partitions, records = {}, []
    for game in ("breakout", "pong"):
        models = {arm: fitted_model(HERE / freeze["models"][game][arm], arm) for arm in ARMS}
        for seed in freeze["seeds"]:
            key = f"{game}_{seed}"
            # Materialize once; signatures are deliberately never loaded.
            with np.load(bank_path(game, seed), allow_pickle=False) as archive:
                bank = {name: archive[name] for name in ("histories", "actions", "valid", "rewards", "roots", "episode")}
            accepted = freeze["selections"][key]["rows"]
            cases = [physical_case(bank, i) for i in accepted]
            for i in range(len(bank["actions"])):
                if i not in accepted:
                    records.append({"event": "excluded_by_syntax", "partition": key, "row": i,
                                    "prefix_length": common_prefix(*bank["actions"][i])})
            for case in cases:
                records.append({"event": "physical_identity_verified", "partition": key, **case})
            summary = {"eligible_roots": len(cases), "excluded_roots": len(bank["actions"]) - len(cases),
                       "prefix_lengths": sorted(set(c["prefix_length"] for c in cases)),
                       "episodes": sorted(set(c["episode"] for c in cases)),
                       "real_nonzero_at_16": sum(abs(c["actual_delta"]) > EPS for c in cases),
                       "terminal_midpoints": sum(c["terminal_at_midpoint"] for c in cases),
                       "cases": cases, "arms": {}, "no_confirmatory_verdict": True}
            if not cases:
                summary["status"] = "inconclusive_no_syntactic_cases"
                partitions[key] = summary
                continue
            for arm in (*ARMS, "zero"):
                rows = []
                for horizon in range(2, 17):
                    chosen = [c for c in cases if c["prefix_length"] < horizon]
                    if not chosen:
                        continue
                    initial = np.stack([bank["histories"][c["row"], 0, 0] for c in chosen])
                    midpoint = np.stack([bank["histories"][c["row"], 0, c["prefix_length"]] for c in chosen])
                    left = [c["left"][:horizon] for c in chosen]
                    right = [c["right"][:horizon] for c in chosen]
                    sl = [p[c["prefix_length"]:] for p, c in zip(left, chosen)]
                    sr = [p[c["prefix_length"]:] for p, c in zip(right, chosen)]
                    query_inputs = ((initial, left, right), (initial, sl, sr), (midpoint, sl, sr))
                    values = [np.zeros(len(chosen)) if arm == "zero" else predict(models[arm], arm, *inputs)
                              for inputs in query_inputs]
                    for c, *predictions in zip(chosen, *values):
                        r = bank["rewards"][c["row"]]
                        target = float(r[0, :horizon].sum() - r[1, :horizon].sum())
                        suffix_target = float(r[0, c["prefix_length"]:horizon].sum() - r[1, c["prefix_length"]:horizon].sum())
                        require(target == suffix_target, "identité physique de courbe diverge")
                        row = {"event": "frozen_prediction", "partition": key, "arm": arm,
                               "row": c["row"], "root": c["root"], "episode": c["episode"],
                               "prefix_length": c["prefix_length"], "horizon_blocks": horizon,
                               "actual_delta": target, "actual_suffix_delta": suffix_target,
                               "predictions": dict(zip(QUERIES, map(float, predictions)))}
                        rows.append(row)
                        records.append(row)
                final = [r for r in rows if r["horizon_blocks"] == 16]
                def summarize(selected):
                    target = [r["actual_delta"] for r in selected]
                    predictions = {q: np.asarray([r["predictions"][q] for r in selected]) for q in QUERIES}
                    defect = predictions[QUERIES[0]] - predictions[QUERIES[2]]
                    scores = {q: metrics(target, predictions[q]) for q in QUERIES}
                    return {"queries": scores, "mean_absolute_consistency_defect": float(np.abs(defect).mean()),
                            "maximum_absolute_consistency_defect": float(np.abs(defect).max()),
                            "choice_disagreement_full_vs_oracle": int(np.sum((predictions[QUERIES[0]] >= -EPS) != (predictions[QUERIES[2]] >= -EPS))),
                            "oracle_fair_tie_gain_over_full": scores[QUERIES[0]]["fair_tie_regret"] - scores[QUERIES[2]]["fair_tie_regret"],
                            "oracle_fair_tie_gain_over_suffix_initial": scores[QUERIES[1]]["fair_tie_regret"] - scores[QUERIES[2]]["fair_tie_regret"]}
                summary["arms"][arm] = {"final": summarize(final),
                    "per_episode": {str(e): summarize([r for r in final if r["episode"] == e]) for e in summary["episodes"]},
                    "curve": {str(t): summarize([r for r in rows if r["horizon_blocks"] == t]) for t in sorted(set(r["horizon_blocks"] for r in rows))}}
            summary["status"] = "development_diagnostic_completed" if summary["real_nonzero_at_16"] else "inconclusive_zero_real_contrasts"
            partitions[key] = summary
    with EVENTS.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    result = {"status": "development_diagnostic_completed_not_agent", "partitions": partitions,
              "freeze_sha256": sha(FREEZE), "events_sha256": sha(EVENTS), "event_count": len(records),
              "ale_calls": 0, "fit_calls": 0, "human_level_goal_achieved": False,
              "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - wall}
    save_new(RESULT, result)
    return {"status": result["status"], "event_count": len(records), "wall_seconds": result["wall_seconds"],
            "partitions": {key: {"n": value["eligible_roots"], "real": value["real_nonzero_at_16"],
                                  "arms": {arm: body["final"] for arm, body in value["arms"].items()}}
                           for key, value in partitions.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--congelar", action="store_true")
    parser.add_argument("--executar", action="store_true")
    args = parser.parse_args()
    require(args.congelar != args.executar, "escolher congelar ou executar")
    print(json.dumps(prepare() if args.congelar else run(), ensure_ascii=False, indent=2))
