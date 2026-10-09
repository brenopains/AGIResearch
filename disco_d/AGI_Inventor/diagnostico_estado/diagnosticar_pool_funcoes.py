"""Diagnóstico do pool congelado, sem afrouxar seleção nem promover funções."""
import os
import platform
import sys
if sys.platform == "win32":
    platform.machine = platform._get_machine_win32
    platform.system = lambda: "Windows"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import argparse
import json
import time
import numpy as np
from piloto_funcoes_contraexemplos import (HERE, CONSTRUCTION, BODIES, RESULT as PILOT_RESULT,
    GENERATORS, FunctionTransport, GeneratedTransport, PairFunctions, append_slots,
    data_for_episodes, load_bank, require, root_view, save_new, sha, verify_proposal)
from funcoes_contraexemplos import fit_features
from decompor_erros_transporte import trajectory
from auditar_semantica_prefixo import scores, equal_report

STEM = HERE / "resultados" / "pool_funcoes_diagnostico"
MANIFEST = STEM.with_name(STEM.name + "_manifesto.json")
RESULT, EVENTS = STEM.with_suffix(".json"), STEM.with_suffix(".jsonl")


def opportunity(target, baseline_prediction, candidate_predictions):
    """Cobertura empírica: escolher por resposta verdadeira é um oráculo."""
    target, baseline_prediction = np.asarray(target, float), np.asarray(baseline_prediction, float)
    candidate_predictions = np.asarray(candidate_predictions, float)
    require(target.ndim == 1 and baseline_prediction.shape == target.shape and
        candidate_predictions.ndim == 2 and candidate_predictions.shape[1] == len(target) and
        all(np.isfinite(v).all() for v in (target, baseline_prediction, candidate_predictions)), "pool inválido")
    all_predictions = np.vstack((baseline_prediction, candidate_predictions))
    costs = np.where(np.abs(all_predictions) <= 1e-12, np.abs(target)[None] / 2,
        np.where(all_predictions >= -1e-12, np.maximum(-target, 0)[None], np.maximum(target, 0)[None]))
    gains = costs[1:] < costs[0] - 1e-12
    return {
        "baseline_error_positions": np.flatnonzero(costs[0] > 1e-12).tolist(),
        "root_improved_by_any_candidate": np.flatnonzero(gains.any(0)).tolist(),
        "root_not_improved_by_any_candidate": np.flatnonzero((costs[0] > 1e-12) & ~gains.any(0)).tolist(),
        "maximum_roots_improved_by_one_candidate": int(gains.sum(1).max()) if len(gains) else 0,
        "baseline_fair_regret": float(costs[0].mean()),
        "answer_conditioned_pool_oracle_fair_regret": float(costs.min(0).mean()),
        "oracle_is_not_an_executable_decision_rule": True,
    }


def local_error(phi, pn, body, data, base_width, active):
    reward_prediction = np.einsum("ij,ij->i", phi, body.reward[data["actions"]])
    transported = np.einsum("ij,ijk->ik", phi, body.transport[data["actions"]])
    residual = transported - data["continuation"][:, None] * pn
    return {
        "observed_transitions": len(data["reward"]),
        "nonzero_reward_transitions": int(np.count_nonzero(data["reward"])),
        "reward_mse": float(np.mean((reward_prediction - data["reward"]) ** 2)),
        "old_coordinate_transport_mse": float(np.mean(residual[:, :base_width] ** 2)),
        "active_new_coordinate_transport_mse": float(np.mean(residual[:, base_width] ** 2)) if active else None,
    }


def evaluate(body, phi_grid, bank, ids):
    rows = []
    maximum = 0.
    for k, row in enumerate(ids):
        branches = [trajectory(body, phi_grid[k, side], bank["actions"][row, side],
            bank["rewards"][row, side], bank["valid"][row, side]) for side in (0, 1)]
        delta = {field: branches[0][field] - branches[1][field] for field in
            ("actual", "predicted", "reward_error", "transport_error", "boundary_error", "real_readout")}
        error = abs(delta["predicted"] - delta["actual"] - delta["reward_error"] - delta["transport_error"] - delta["boundary_error"])
        require(error < 1e-8, "decomposição do candidato não reconcilia")
        maximum = max(maximum, error, *(b["identity_error"] for b in branches))
        rows.append({"bank_row": int(row), "episode": int(bank["episode"][row]), **delta})
    y = np.asarray([r["actual"] for r in rows])
    r, t, b = [np.asarray([row[field] for row in rows]) for field in
        ("reward_error", "transport_error", "boundary_error")]
    mse = {"reward_squared": float(np.mean(r * r)), "transport_squared": float(np.mean(t * t)),
        "boundary_squared": float(np.mean(b * b)), "reward_transport_cross": float(np.mean(2 * r * t)),
        "reward_boundary_cross": float(np.mean(2 * r * b)), "transport_boundary_cross": float(np.mean(2 * t * b))}
    report = {"composed": scores(y, [v["predicted"] for v in rows]),
        "real_future_history_readout_oracle": scores(y, [v["real_readout"] for v in rows]),
        "remove_reward_error_oracle": scores(y, [v["predicted"] - v["reward_error"] for v in rows]),
        "mse_decomposition": mse, "maximum_identity_error": maximum}
    require(abs(sum(mse.values()) - report["composed"]["mse"]) < 1e-8, "MSE não reconcilia")
    return report, rows


def origins_report(functions, data):
    pairs = [o for o in functions.origins if o["kind"] == "observed_pair"]
    anchor_rows = sorted(set(i for o in pairs for i in o["rows"]))
    counts = {str(int(e)): sum(int(e) in o["episodes"] for o in pairs) for e in sorted(set(data["episodes"]))}
    return {"pair_counts_by_episode": counts, "distinct_anchor_rows": len(anchor_rows),
        "true_nonzero_reward_anchor_rows": sum(abs(float(data["reward"][i])) > 1e-12 for i in anchor_rows),
        "true_nonzero_reward_anchor_episodes": sorted(set(int(data["episodes"][i]) for i in anchor_rows if abs(float(data["reward"][i])) > 1e-12)),
        "synthetic_reward_labels_for_generation": any(o.get("pair_outcomes_recorded") != data["reward"][o["rows"]].tolist() for o in pairs)}


def prepare():
    require(not any(p.exists() for p in (MANIFEST, RESULT, EVENTS)), "diagnóstico já existe; preservar")
    verify_proposal()
    paths = ["diagnosticar_pool_funcoes.py", "testes_pool_funcoes.py", "DIAGNOSTICO_POOL_FUNCOES.md",
        "decompor_erros_transporte.py", "auditar_semantica_prefixo.py",
        "resultados/funcoes_contraexemplos_auditoria.json", CONSTRUCTION.relative_to(HERE).as_posix(),
        BODIES.relative_to(HERE).as_posix(), PILOT_RESULT.relative_to(HERE).as_posix()]
    for manifest in (CONSTRUCTION, BODIES):
        for path, expected in json.loads(manifest.read_text(encoding="utf-8"))["source_hashes"].items():
            require(sha(HERE / path) == expected, "corpo congelado mudou")
            paths.append(path)
    save_new(MANIFEST, {"status": "frozen_before_pool_diagnostic", "development_only": True,
        "episodes": [4, 5], "construction": [0, 1, 2, 3], "seed": 97, "horizon": 16,
        "all_generators_all_candidates": True, "no_gate_change": True, "no_promotion": True,
        "maximum_wall_seconds": 120, "source_hashes": {p: sha(HERE / p) for p in sorted(set(paths))}})
    return {"status": "frozen_before_pool_diagnostic", "manifest_sha256": sha(MANIFEST)}


def run():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path, expected in manifest["source_hashes"].items():
        require(sha(HERE / path) == expected, f"hash mudou: {path}")
    verify_proposal()
    require(not RESULT.exists() and not EVENTS.exists(), "diagnóstico concluído; preservar")
    started = time.perf_counter()
    construction = json.loads(CONSTRUCTION.read_text(encoding="utf-8"))["construction"]
    bodies = json.loads(BODIES.read_text(encoding="utf-8"))["bodies"]
    pilot = json.loads(PILOT_RESULT.read_text(encoding="utf-8"))
    records, games, fits = [], {}, 0
    for game in ("breakout", "pong"):
        bank = load_bank(game, 97)
        data, validation = data_for_episodes(bank, [0, 1, 2, 3]), data_for_episodes(bank, [4, 5])
        view = root_view(bank, [4, 5])
        ids = view["ids"]
        base = FunctionTransport.load(HERE / construction[game]["base"])
        baseline = GeneratedTransport.load(HERE / bodies[game]["no_generated_functions"])
        width = len(base.initial_prior)
        p, pn = base.features(data["source"]), base.features(data["successor"])
        vp, vpn = base.features(validation["source"]), base.features(validation["successor"])
        raw_grid = bank["histories"][ids]
        grid_shape = raw_grid.shape
        flat_grid = raw_grid.reshape(-1, *grid_shape[3:])
        grid_base = base.features(flat_grid)
        baseline_grid = append_slots(grid_base, np.empty((len(grid_base), 0)), []).reshape(*grid_shape[:3], -1)
        baseline_scores, baseline_rows = evaluate(baseline, baseline_grid, bank, ids)
        equal_report(baseline_scores["composed"], pilot["selection"][game]["counterexample_pairs"]["baseline_validation"], "base do pool")
        baseline_prediction = np.asarray([r["predicted"] for r in baseline_rows])
        baseline_local = local_error(append_slots(vp, np.empty((len(vp), 0)), []),
            append_slots(vpn, np.empty((len(vpn), 0)), []), baseline, validation, width, False)
        games[game] = {"validation_actual_deltas": view["actual"].tolist(), "validation_rows": ids.tolist(),
            "baseline": {"queries": baseline_scores, "local_error": baseline_local, "rows": baseline_rows}, "generators": {}}
        for arm in GENERATORS:
            functions = PairFunctions.load(HERE / construction[game]["dictionaries"][arm])
            g, gn = functions.apply(data["source"]), functions.apply(data["successor"])
            vg, vgn, grid_g = functions.apply(validation["source"]), functions.apply(validation["successor"]), functions.apply(flat_grid)
            predictions, candidates = [], []
            for index, expression in enumerate(functions.expressions):
                require(time.perf_counter() - started < manifest["maximum_wall_seconds"], "teto excedido; sem conclusão")
                body = fit_features(base, functions, [index], append_slots(p, g, [index]), append_slots(pn, gn, [index]),
                    data["reward"], data["continuation"], data["actions"])
                fits += 1
                phi_grid = append_slots(grid_base, grid_g, [index]).reshape(*grid_shape[:3], -1)
                result, rows = evaluate(body, phi_grid, bank, ids)
                local = local_error(append_slots(vp, vg, [index]), append_slots(vpn, vgn, [index]), body, validation, width, True)
                prediction = np.asarray([r["predicted"] for r in rows])
                predictions.append(prediction)
                candidate = {"candidate": index, "expression": expression, "queries": result, "local_error": local}
                candidates.append(candidate)
                records.append({"event": "candidate_diagnostic", "game": game, "arm": arm, **candidate, "rows": rows})
            opportunity_report = opportunity(view["actual"], baseline_prediction, np.asarray(predictions))
            games[game]["generators"][arm] = {"origin_diversity": origins_report(functions, data),
                "decision_opportunity": opportunity_report, "candidates": candidates}
        del bank
    with EVENTS.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    result = {"status": "pool_diagnosed_not_promoted", "games": games, "fit_calls": fits,
        "wall_seconds": time.perf_counter() - started, "events": len(records), "events_sha256": sha(EVENTS),
        "manifest_sha256": sha(MANIFEST), "previous_gate_preserved": pilot["status"], "development_only": True,
        "ale_calls": 0, "human_level_goal_achieved": False}
    require(result["wall_seconds"] < manifest["maximum_wall_seconds"], "teto excedido; sem conclusão")
    save_new(RESULT, result)
    return {"status": result["status"], "wall_seconds": result["wall_seconds"], "fit_calls": fits,
        "opportunities": {game: {arm: report["decision_opportunity"] for arm, report in reports["generators"].items()} for game, reports in games.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--congelar", action="store_true")
    parser.add_argument("--executar", action="store_true")
    args = parser.parse_args()
    require(args.congelar != args.executar, "escolher congelar ou executar")
    print(json.dumps(prepare() if args.congelar else run(), ensure_ascii=False, indent=2))
