"""Gerar funções no treino, escolher por decisão em outros episódios, congelar."""

import os
import platform
import sys
if sys.platform == "win32":
    platform.machine = platform._get_machine_win32
    platform.system = lambda: "Windows"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import argparse
import copy
import json
import time
import numpy as np
from piloto_transporte_funcoes import HERE, load_bank, require, save_new, sha, verify
from diagnosticar_semantica_prefixo import metrics
from transporte_funcoes import FunctionTransport
from funcoes_contraexemplos import (PairFunctions, GeneratedTransport, SLOTS,
    append_slots, fit_features, propose)

STEM = HERE / "resultados" / "funcoes_contraexemplos"
PROPOSAL = STEM.with_name(STEM.name + "_proposta.json")
CONSTRUCTION = STEM.with_name(STEM.name + "_construcao_manifesto.json")
BODIES = STEM.with_name(STEM.name + "_corpos_manifesto.json")
RESULT, EVENTS = STEM.with_suffix(".json"), STEM.with_suffix(".jsonl")
GENERATORS = ("counterexample_pairs", "shuffled_outcomes", "random_pairs", "random_directions")
ARMS = (*GENERATORS, "no_generated_functions", "remove_generated", "remove_context", "zero")


def prepare():
    require(not any(p.exists() for p in (PROPOSAL, CONSTRUCTION, BODIES, RESULT, EVENTS)), "piloto existe; preservar")
    verify()
    sources = ["funcoes_contraexemplos.py", "piloto_funcoes_contraexemplos.py", "auditar_funcoes_contraexemplos.py",
               "testes_funcoes_contraexemplos.py", "GERADOR_FUNCOES_CONTRAEXEMPLOS.md", "transporte_funcoes.py",
               "piloto_transporte_funcoes.py", "diagnosticar_semantica_prefixo.py",
               "resultados/credito_sequencias_auditoria.json", "resultados/decomposicao_transporte_auditoria.json"]
    for game in ("breakout", "pong"):
        for seed in (97, 103):
            for ext in (".npz", ".json", ".jsonl"):
                sources.append(f"resultados/{game}_transicoes_mistas_{seed}{ext}")
    report = {"status": "frozen_before_function_generation", "generators": list(GENERATORS), "arms": list(ARMS),
              "construction_episodes": [0, 1, 2, 3], "validation_episodes": [4, 5], "held_episodes97": [6, 7],
              "atoms": 24, "products": 8, "slots": SLOTS, "alpha": .05, "novelty_minimum": 1e-4,
              "maximum_main_wall_seconds": 120, "primary_bank": 103, "primary_horizon": 16,
              "development_only": True, "no_ale": True,
              "source_hashes": {p: sha(HERE / p) for p in sources}}
    save_new(PROPOSAL, report)
    return {"status": report["status"], "proposal_sha256": sha(PROPOSAL)}


def verify_proposal():
    report = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    for p, expected in report["source_hashes"].items():
        require(sha(HERE / p) == expected, f"artefato mudou: {p}")
    verify()
    return report


def data_for_episodes(bank, episodes):
    mask = bank["valid"][:, :, :-1] & np.isin(bank["episode"], episodes)[:, None, None]
    ids = np.argwhere(mask)
    return {"source": bank["histories"][:, :, :-1][mask], "successor": bank["histories"][:, :, 1:][mask],
            "actions": bank["actions"][mask], "reward": bank["rewards"][mask],
            "continuation": bank["valid"][:, :, 1:][mask].astype(float),
            "episodes": bank["episode"][ids[:, 0]], "bank_indices": ids}


def root_view(bank, episodes, horizon=16):
    ids = np.flatnonzero(np.isin(bank["episode"], episodes))
    require(np.array_equal(bank["histories"][ids, 0, 0], bank["histories"][ids, 1, 0]), "histórias iniciais divergem")
    return {"ids": ids, "history": bank["histories"][ids, 0, 0],
            "left": bank["actions"][ids, 0, :horizon].tolist(), "right": bank["actions"][ids, 1, :horizon].tolist(),
            "actual": bank["rewards"][ids, 0, :horizon].sum(1) - bank["rewards"][ids, 1, :horizon].sum(1)}


def forecast(body, phi, left, right):
    if body.mode == "no_context":
        phi = np.broadcast_to(body.initial_prior, phi.shape)
    lv, rv = np.asarray([body.word_value(p) for p in left]), np.asarray([body.word_value(q) for q in right])
    return np.einsum("ij,ij->i", phi, lv - rv)


def regrets(target, predicted):
    return np.where(np.abs(predicted) <= 1e-12, np.abs(target) / 2,
                    np.where(predicted >= -1e-12, np.maximum(-target, 0), np.maximum(target, 0)))


def select(base, functions, data, validation, game, arm, records, check_time):
    p = base.features(data["source"])
    pn = base.features(data["successor"])
    pv = base.features(validation["history"])
    g = functions.apply(data["source"])
    gn = functions.apply(data["successor"])
    gv = functions.apply(validation["history"])
    selected, fit_calls = [], 1
    current = fit_features(base, functions, [], append_slots(p, g, []), append_slots(pn, gn, []),
                           data["reward"], data["continuation"], data["actions"])
    current_prediction = forecast(current, append_slots(pv, gv, []), validation["left"], validation["right"])
    original_score = metrics(validation["actual"], current_prediction)
    if original_score["fair_tie_regret"] <= 1e-12:
        return current, {"status": "inconclusive_no_validation_decision_error", "selected": [], "fit_calls": fit_calls,
                         "baseline_validation": original_score, "final_validation": original_score, "accepted_evidence": []}
    accepted = []
    for iteration in range(SLOTS):
        check_time()
        remaining = [i for i in range(len(functions.expressions)) if i not in selected]
        if not remaining:
            break
        current_phi = append_slots(p, g, selected)
        coeff = np.linalg.lstsq(current_phi, g[:, remaining], rcond=1e-10)[0]
        residual = np.mean((current_phi @ coeff - g[:, remaining]) ** 2, axis=0)
        variance = np.var(g[:, remaining], axis=0)
        novelty = residual / np.maximum(variance, 1e-12)
        previous_score = metrics(validation["actual"], current_prediction)
        previous_regret = regrets(validation["actual"], current_prediction)
        best = None
        for index, ratio, var in zip(remaining, novelty, variance):
            check_time()
            record = {"event": "candidate_evaluated", "game": game, "arm": arm, "iteration": iteration,
                      "candidate": index, "selected_before": selected.copy(), "novelty_ratio_training": float(ratio),
                      "training_variance": float(var), "expression": functions.expressions[index]}
            if var < 1e-12 or ratio <= 1e-4:
                record.update({"eligible": False, "reason": "no_numerical_expansion", "validation": None})
                records.append(record)
                continue
            trial_selected = selected + [index]
            body = fit_features(base, functions, trial_selected, append_slots(p, g, trial_selected), append_slots(pn, gn, trial_selected),
                                data["reward"], data["continuation"], data["actions"])
            fit_calls += 1
            prediction = forecast(body, append_slots(pv, gv, trial_selected), validation["left"], validation["right"])
            if not np.isfinite(prediction).all():
                record.update({"eligible": False, "reason": "nonfinite_prediction", "validation": None})
                records.append(record)
                continue
            score = metrics(validation["actual"], prediction)
            improved = int(np.sum(regrets(validation["actual"], prediction) < previous_regret - 1e-12))
            eligible = bool(score["fair_tie_regret"] < previous_score["fair_tie_regret"] - 1e-12
                            and score["mse"] < previous_score["mse"] - 1e-12 and improved >= 2)
            record.update({"eligible": eligible, "reason": "validation_gain" if eligible else "no_joint_validation_gain",
                           "validation": score, "positive_validation_roots": improved})
            records.append(record)
            ranking = (score["fair_tie_regret"], score["mse"], len(functions.expressions[index]["units"]), index)
            if eligible and (best is None or ranking < best[0]):
                best = (ranking, index, body, prediction, record)
        if best is None:
            break
        _, index, current, current_prediction, record = best
        selected.append(index)
        accepted.append(record)
        records.append({"event": "function_selected_in_development", "game": game, "arm": arm,
                        "iteration": iteration, "candidate": index, "selected_after": selected.copy(),
                        "is_general_library_promotion": False})
    return current, {"status": "functions_selected_in_development" if selected else "no_candidate_passed_validation",
                     "selected": selected, "fit_calls": fit_calls, "baseline_validation": original_score,
                     "final_validation": metrics(validation["actual"], current_prediction), "accepted_evidence": accepted}


def run():
    proposal = verify_proposal()
    require(not any(p.exists() for p in (CONSTRUCTION, BODIES, RESULT, EVENTS)), "piloto já iniciado; preservar")
    wall, cpu = time.perf_counter(), time.process_time()
    def check_time():
        require(time.perf_counter() - wall < proposal["maximum_main_wall_seconds"], "teto atingido; resultados inconclusivos, sem aprovação")
    banks, construction, objects, records, frozen_files = {}, {}, {}, [], [PROPOSAL.relative_to(HERE).as_posix()]
    for game in ("breakout", "pong"):
        check_time()
        bank = load_bank(game, 97)
        banks[game] = bank
        data = data_for_episodes(bank, proposal["construction_episodes"])
        base = FunctionTransport().fit(data["source"], data["successor"], data["actions"], data["reward"], data["continuation"])
        base_path = STEM.with_name(STEM.name + f"_{game}_base_construcao.npz")
        base.save(base_path)
        frozen_files.extend((base_path.relative_to(HERE).as_posix(), base_path.with_suffix(".json").relative_to(HERE).as_posix()))
        construction[game] = {"base": base_path.relative_to(HERE).as_posix(), "observed_transitions": len(data["reward"]),
                              "source_bank_indices": data["bank_indices"].tolist(), "dictionaries": {}, "generation": {}}
        objects[game] = (base, data, {})
        for arm in GENERATORS:
            functions, stats = propose(data["source"], data["actions"], data["reward"], data["episodes"], base.features(data["source"]), arm)
            path = STEM.with_name(STEM.name + f"_{game}_{arm}_dicionario.npz")
            functions.save(path)
            objects[game][2][arm] = functions
            construction[game]["dictionaries"][arm] = path.relative_to(HERE).as_posix()
            construction[game]["generation"][arm] = stats
            frozen_files.extend((path.relative_to(HERE).as_posix(), path.with_suffix(".json").relative_to(HERE).as_posix()))
            records.append({"event": "dictionary_generated", "game": game, "arm": arm, **stats})
    save_new(CONSTRUCTION, {"status": "dictionaries_frozen_before_selector_calls", "construction": construction,
                             "source_hashes": {p: sha(HERE / p) for p in frozen_files}})
    final, reports, paths = {}, {}, {}
    for game in ("breakout", "pong"):
        base, data, dictionaries = objects[game]
        validation = root_view(banks[game], proposal["validation_episodes"])
        final[game], reports[game], paths[game] = {}, {}, {}
        for arm in GENERATORS:
            body, report = select(base, dictionaries[arm], data, validation, game, arm, records, check_time)
            final[game][arm], reports[game][arm] = body, report
        empty = PairFunctions(np.empty((0, base.support.shape[1])), np.empty(0), [], [])
        p, pn = base.features(data["source"]), base.features(data["successor"])
        final[game]["no_generated_functions"] = fit_features(base, empty, [], append_slots(p, np.empty((len(p), 0)), []),
            append_slots(pn, np.empty((len(pn), 0)), []), data["reward"], data["continuation"], data["actions"])
        removed = copy.deepcopy(final[game]["counterexample_pairs"])
        width = len(base.initial_prior)
        removed.selected = [None] * SLOTS
        removed.reward[:, width:] = 0
        removed.transport[:, width:, :] = 0
        removed.transport[:, :, width:] = 0
        removed.initial_prior[width:] = 0
        final[game]["remove_generated"] = removed
        context = copy.deepcopy(final[game]["counterexample_pairs"])
        context.mode = "no_context"
        final[game]["remove_context"] = context
        for arm, body in final[game].items():
            path = STEM.with_name(STEM.name + f"_{game}_{arm}_corpo.npz")
            body.save(path)
            restored = GeneratedTransport.load(path)
            sample = data["source"][:8]
            require(np.array_equal(body.comparison(sample, [[0, 1]] * 8, [[1, 0]] * 8),
                                   restored.comparison(sample, [[0, 1]] * 8, [[1, 0]] * 8)), "corpo exportado diverge")
            paths[game][arm] = path.relative_to(HERE).as_posix()
    files = [CONSTRUCTION.relative_to(HERE).as_posix()]
    for game in paths:
        for path in paths[game].values():
            files.extend((path, path.replace(".npz", ".json")))
    save_new(BODIES, {"status": "selected_bodies_frozen_before_final_evaluation", "bodies": paths, "selection": reports,
                     "source_hashes": {p: sha(HERE / p) for p in files}})
    games = {}
    for game in ("breakout", "pong"):
        check_time()
        games[game] = {}
        for seed, episodes in ((97, [6, 7]), (103, list(range(8)))):
            bank = banks[game] if seed == 97 else load_bank(game, seed)
            horizons = {}
            for horizon in (4, 8, 16):
                view = root_view(bank, episodes, horizon)
                scores = {}
                for arm in ARMS:
                    prediction = np.zeros(len(view["ids"])) if arm == "zero" else final[game][arm].comparison(view["history"], view["left"], view["right"])
                    require(np.isfinite(prediction).all(), "resultado final não finito; sem aprovação")
                    scores[arm] = metrics(view["actual"], prediction)
                    for j, i in enumerate(view["ids"]):
                        records.append({"event": "held_comparison", "game": game, "seed": seed, "arm": arm,
                                        "horizon_blocks": horizon, "row": int(i), "episode": int(bank["episode"][i]),
                                        "actual_delta": float(view["actual"][j]), "predicted_delta": float(prediction[j])})
                horizons[str(horizon)] = scores
            games[game][str(seed)] = horizons
        primary = games[game]["103"]["16"]["counterexample_pairs"]
        checks = {"generated_function_selected": bool(reports[game]["counterexample_pairs"]["selected"])}
        for arm in ARMS:
            if arm == "counterexample_pairs":
                continue
            for metric in ("mse", "fair_tie_regret"):
                checks[f"{metric}_vs_{arm}"] = bool(primary[metric] < games[game]["103"]["16"][arm][metric] - 1e-12)
        games[game]["gate_checks"] = checks
        games[game]["development_gate_passed"] = all(checks.values())
    with EVENTS.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    result = {"status": "development_gate_passed_not_agent" if all(g["development_gate_passed"] for g in games.values()) else "failed_development_gate",
              "games": games, "selection": reports, "construction": construction, "event_count": len(records),
              "proposal_sha256": sha(PROPOSAL), "construction_manifest_sha256": sha(CONSTRUCTION), "body_manifest_sha256": sha(BODIES),
              "events_sha256": sha(EVENTS), "wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
              "trial_model_fits": sum(r["fit_calls"] for gs in reports.values() for r in gs.values()),
              "base_model_fits": 2, "extra_no_function_model_fits": 2,
              "crossfit_reward_head_solves": 2 * len(GENERATORS) * 4 * 4,
              "ale_calls": 0, "human_level_goal_achieved": False, "development_only": True}
    check_time()
    save_new(RESULT, result)
    return {"status": result["status"], "wall_seconds": result["wall_seconds"], "trial_model_fits": result["trial_model_fits"],
            "selected": {g: {a: r["selected"] for a, r in arms.items()} for g, arms in reports.items()},
            "primary103": {g: report["103"]["16"] for g, report in games.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--congelar", action="store_true")
    parser.add_argument("--executar", action="store_true")
    args = parser.parse_args()
    require(args.congelar != args.executar, "escolher congelar ou executar")
    print(json.dumps(prepare() if args.congelar else run(), ensure_ascii=False, indent=2))
