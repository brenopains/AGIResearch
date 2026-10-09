"""Separar reconhecer de prever e testar o campo executável com corpos congelados."""
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
import numpy as np
from piloto_prospectivo import (HERE, BODIES as FEATURE_BODIES, PROPOSAL as PRIOR_PROPOSAL, GeneratedTransport,
    bank_path as prior_bank, load_bank, transitions, view, coverage, require, save_new, sha, verify_bodies)
from leitor_eventos import EventReader, PotentialPlanner, potential_constraints
from diagnosticar_semantica_prefixo import metrics

STEM = HERE / "resultados" / "eventos"
PROPOSAL = STEM.with_name(STEM.name + "_proposta.json")
BODIES = STEM.with_name(STEM.name + "_corpos_manifesto.json")
DIAGNOSTIC = STEM.with_name(STEM.name + "_reconhecimento.json")
RESULT, EVENTS = STEM.with_suffix(".json"), STEM.with_suffix(".jsonl")
BASES = ("no_generated_functions", "future_pairs")
READERS = ("before", "after", "delta", "transition", "potential", "shuffled_successor", "shuffled_rewards", "action_prior", "zero")
ARMS = ("potential", "old_composed", "shuffled_field", "no_terminal_transport", "no_context", "zero")
TEST_SEED = 163


def fresh_bank(game):
    return STEM.with_name(STEM.name + f"_{game}_{TEST_SEED}")


def prepare():
    verify_bodies()
    require(not any(p.exists() for p in (PROPOSAL, BODIES, DIAGNOSTIC, RESULT, EVENTS)), "rodada existe; preservar")
    require(not any(list((HERE / "resultados").glob(fresh_bank(g).name + "*")) for g in ("breakout", "pong")), "semente já usada nesta rodada")
    sources = ("piloto_leitor_eventos.py", "leitor_eventos.py", "testes_leitor_eventos.py", "coletar_eventos.py",
        "auditar_leitor_eventos.py", "EVENTOS.md", "coletar_prospectiva.py", "piloto_prospectivo.py", "auditar_semantica_prefixo.py",
        "resultados/prospectiva_auditoria.json", FEATURE_BODIES.relative_to(HERE).as_posix())
    parent = json.loads(PRIOR_PROPOSAL.read_text(encoding="utf-8"))
    save_new(PROPOSAL, {"status": "frozen_before_event_head_fit", "bases": list(BASES), "readers": list(READERS),
        "planning_arms": list(ARMS), "primary_basis": "no_generated_functions", "training_seed": 149,
        "diagnostic_seeds": [151, 157], "fresh_evaluation_seed": TEST_SEED, "programs": parent["programs"],
        "episodes": 8, "checkpoints": [0, 2, 4, 6], "calls_per_block": 8, "horizons": [4, 8, 16],
        "gate": "strictly_lower_mse_and_fair_regret_than_all_controls_at_16_in_all_primary_cells_with_coverage",
        "alpha": .05, "maximum_fit_wall_seconds": 120, "maximum_calls_per_bank": 50000,
        "maximum_collection_wall_seconds": 120, "maximum_worker_rss_bytes": 6 * 1024 ** 3,
        "known_games_new_seed_not_unseen_worlds": True, "diagnostic_corpus_previously_seen": True,
        "human_level_goal_achieved": False, "source_hashes": {p: sha(HERE / p) for p in sources}})
    return {"status": "event_protocol_frozen", "proposal_sha256": sha(PROPOSAL)}


def verify():
    value = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    for path, expected in value["source_hashes"].items():
        require(sha(HERE / path) == expected, f"fonte mudou: {path}")
    verify_bodies()
    return value


def verify_frozen():
    verify()
    value = json.loads(BODIES.read_text(encoding="utf-8"))
    for path, expected in value["source_hashes"].items():
        require(sha(HERE / path) == expected, f"corpo mudou: {path}")
    return value


def shuffled(values, actions):
    values = np.asarray(values).copy()
    rng = np.random.default_rng(10311)
    for action in range(4):
        ids = np.flatnonzero(actions == action)
        values[ids] = values[rng.permutation(ids)]
    return values


def graph(data):
    canonical, keys = {}, []
    for array in (data["source"], data["successor"]):
        group = []
        for row in array:
            key = hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
            if key in canonical:
                require(np.array_equal(canonical[key], row), "digest igual com observações diferentes")
            else:
                canonical[key] = row
            group.append(key)
        keys.append(group)
    return potential_constraints(keys[0], keys[1], data["reward"])


def fit():
    proposal = verify()
    require(not BODIES.exists(), "ajuste existe; preservar")
    require(not any(fresh_bank(g).with_suffix(".npz").exists() for g in ("breakout", "pong")), "avaliação coletada antes do ajuste")
    start = time.perf_counter()
    parent = json.loads(FEATURE_BODIES.read_text(encoding="utf-8"))
    bodies, reports, paths = {}, {}, [PROPOSAL, FEATURE_BODIES]
    for game in ("breakout", "pong"):
        data = transitions(load_bank(game, 149))
        reports[game] = {"observed_integer_graph": graph(data), "bases": {}}
        bodies[game] = {}
        paths.extend(prior_bank(game, 149).with_suffix(s) for s in (".npz", ".json"))
        for basis in BASES:
            reference = HERE / parent["bodies"][game][basis]
            body = GeneratedTransport.load(reference)
            p, pn = body.features(data["source"]), body.features(data["successor"])
            bodies[game][basis] = {"feature": reference.relative_to(HERE).as_posix(), "readers": {}}
            fields = {}
            for arm in READERS:
                require(time.perf_counter() - start < 120, "teto atingido; inconclusivo")
                mode = "transition" if arm in ("shuffled_successor", "shuffled_rewards") else arm
                next_phi = shuffled(pn, data["actions"]) if arm == "shuffled_successor" else pn
                target = shuffled(data["reward"], data["actions"]) if arm == "shuffled_rewards" else data["reward"]
                reader = EventReader(mode).fit(p, next_phi, data["actions"], target)
                path = STEM.with_name(STEM.name + f"_{game}_{basis}_{arm}.npz")
                reader.save(path, {"path": reference.relative_to(HERE).as_posix(), "sha256": sha(reference)},
                    {"training_seed": 149, "arm": arm, "training_transitions": len(data["reward"]), "diagnostic_not_agi": True})
                restored = EventReader.load(path)
                require(np.array_equal(reader.predict(p[:8], pn[:8], data["actions"][:8]),
                    restored.predict(p[:8], pn[:8], data["actions"][:8])), "corpo exportado diverge")
                if arm == "before":
                    require(np.allclose(reader.coefficient[:, :len(body.initial_prior)], body.reward, atol=1e-8), "baseline antes mudou")
                if arm == "potential":
                    fields[arm] = reader.field
                bodies[game][basis]["readers"][arm] = path.relative_to(HERE).as_posix()
                paths.extend((path, path.with_suffix(".json")))
            shuffled_field = EventReader("potential").fit(p, pn, data["actions"], shuffled(data["reward"], data["actions"]))
            planner = PotentialPlanner.fit(body, fields["potential"], p, pn, data["actions"], data["continuation"])
            path = STEM.with_name(STEM.name + f"_{game}_{basis}_planner.npz")
            require(not path.exists(), "planejador já existe")
            np.savez_compressed(path, field=fields["potential"], shuffled_field=shuffled_field.field,
                terminal_transport=planner.terminal_transport)
            save_new(path.with_suffix(".json"), {"feature": reference.relative_to(HERE).as_posix(), "parameters": path.name,
                "operator_body": "alive_next=alive@M[action];dead_next=dead+alive@terminal_M[action]",
                "value_body": "(alive_final+dead_final-alive_initial)@field", "dead_is_absorbing_protocol_semantics": True,
                "no_future_observations_at_decision": True})
            bodies[game][basis]["planner"] = path.relative_to(HERE).as_posix()
            paths.extend((path, path.with_suffix(".json")))
            delta = pn - p
            exact_fit, _, rank, _ = np.linalg.lstsq(delta, data["reward"], rcond=1e-10)
            residual = delta @ exact_fit - data["reward"]
            reports[game]["bases"][basis] = {"feature_width": p.shape[1], "least_squares_difference_rank": int(rank),
                "minimum_training_mse_with_numeric_rank_cutoff": float(np.mean(residual ** 2)),
                "maximum_training_absolute_residual": float(np.max(np.abs(residual))),
                "scope": "fixed_feature_family_not_proof_of_global_impossibility"}
    require(time.perf_counter() - start < 120, "teto atingido; inconclusivo")
    save_new(BODIES, {"status": "event_bodies_frozen_before_diagnostic_scoring_and_fresh_collection", "bodies": bodies,
        "construction": reports, "wall_seconds": time.perf_counter() - start,
        "source_hashes": {p.relative_to(HERE).as_posix(): sha(p) for p in paths}})
    return {"status": "event_bodies_frozen", "wall_seconds": time.perf_counter() - start, "construction": reports}


def local_metrics(actual, prediction):
    return {"count": len(actual), "mse": float(np.mean((actual - prediction) ** 2)),
        "mae": float(np.mean(np.abs(actual - prediction))), "observed_nonzero": int(np.count_nonzero(actual)),
        "nonzero_mse": float(np.mean((actual[actual != 0] - prediction[actual != 0]) ** 2)) if np.any(actual != 0) else None,
        "rounded_match_fraction_not_exact_proof": float(np.mean(np.rint(prediction) == actual))}


def diagnose():
    frozen = verify_frozen()
    require(not DIAGNOSTIC.exists(), "diagnóstico já existe")
    reports = {}
    for game in ("breakout", "pong"):
        reports[game] = {}
        for seed in (151, 157):
            bank = load_bank(game, seed)
            data = transitions(bank)
            reports[game][str(seed)] = {}
            for basis in BASES:
                files = frozen["bodies"][game][basis]
                body = GeneratedTransport.load(HERE / files["feature"])
                p, pn = body.features(data["source"]), body.features(data["successor"])
                results = {}
                for arm in READERS:
                    reader = EventReader.load(HERE / files["readers"][arm])
                    prediction = reader.predict(p, pn, data["actions"])
                    per_episode = {str(e): local_metrics(data["reward"][data["episodes"] == e], prediction[data["episodes"] == e]) for e in range(8)}
                    predicted = np.zeros_like(bank["rewards"], dtype=float)
                    for i, value in zip(data["bank_indices"], prediction):
                        predicted[tuple(i)] = value
                    paired = {}
                    for partition, offset in (("known", 0), ("new", 4)) if seed == 157 else (("known", 0),):
                        cases = view(bank, offset)
                        delta = np.asarray([predicted[i, a].sum() - predicted[i, b].sum() for i, a, b in cases["bank_cases"]])
                        paired[partition] = metrics(cases["actual"], delta)
                    results[arm] = {"local": local_metrics(data["reward"], prediction), "per_episode": per_episode,
                        "whole_program_with_real_observations_diagnostic": paired,
                        "available_before_action": arm in ("before", "action_prior", "zero")}
                reports[game][str(seed)][basis] = results
    save_new(DIAGNOSTIC, {"status": "recognition_vs_prediction_development_diagnostic_complete", "games": reports,
        "corpus_previously_seen": True, "no_new_head_selection_or_tuning": True,
        "human_level_goal_achieved": False, "frozen_bodies_sha256": sha(BODIES)})
    return {"status": "event_diagnostic_complete", "primary157": {g: {a: r["local"] for a, r in v["157"]["no_generated_functions"].items()} for g, v in reports.items()}}


def load_fresh(game):
    proposal = verify()
    path = fresh_bank(game)
    certificate = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    require(certificate["status"] == "event_evaluation_bank_complete" and certificate["bank_sha256"] == sha(path.with_suffix(".npz"))
        and certificate["events_sha256"] == sha(path.with_suffix(".jsonl")) and certificate["bodies_sha256_before_collection"] == sha(BODIES)
        and certificate["proposal_sha256"] == sha(PROPOSAL) and certificate["collector_sha256"] == sha(HERE / "coletar_eventos.py"), "avaliação não certificada")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
        bank = {k: archive[k].copy() for k in archive.files}
    require(bank["histories"].shape == (32, 8, 17, 16, 336) and bank["rewards"].shape == (32, 8, 16)
        and bank["valid"].shape == (32, 8, 17) and bank["valid"].dtype == bool
        and np.array_equal(bank["programs"], proposal["programs"]) and np.isfinite(bank["histories"]).all()
        and np.isfinite(bank["rewards"]).all(), "banco/planos divergem")
    require(np.array_equal(bank["episode"], np.repeat(np.arange(8), 4))
        and np.array_equal(bank["roots"], np.asarray([8 * e + p for e in range(8) for p in (0, 2, 4, 6)])), "raízes/episódios divergem")
    require(all(np.array_equal(bank["histories"][:, 0, 0], bank["histories"][:, j, 0]) for j in range(8)), "raiz contrafactual difere")
    require(not np.any((~bank["valid"][..., :-1]) & bank["valid"][..., 1:]) and not np.any(bank["rewards"][~bank["valid"][..., :-1]]), "padding diverge")
    return bank


def evaluate():
    frozen = verify_frozen()
    require(not RESULT.exists() and not EVENTS.exists(), "avaliação já existe")
    reports, records, statuses = {}, [], []
    for game in ("breakout", "pong"):
        bank = load_fresh(game)
        reports[game] = {}
        for basis in BASES:
            files = frozen["bodies"][game][basis]
            body = GeneratedTransport.load(HERE / files["feature"])
            with np.load(HERE / files["planner"], allow_pickle=False) as params:
                planner = PotentialPlanner(body, params["field"], params["terminal_transport"])
                shuffled_planner = PotentialPlanner(body, params["shuffled_field"], params["terminal_transport"])
                no_terminal = PotentialPlanner(body, params["field"], np.zeros_like(params["terminal_transport"]))
            reports[game][basis] = {}
            for partition, offset in (("known", 0), ("new", 4)):
                curves = {}
                for horizon in (4, 8, 16):
                    cases = view(bank, offset, horizon)
                    phi = body.features(cases["history"])
                    rows = {}
                    for arm in ARMS:
                        if arm == "zero":
                            predicted = np.zeros(64)
                        elif arm == "old_composed":
                            predicted = body.comparison(cases["history"], cases["left"], cases["right"])
                        else:
                            model = shuffled_planner if arm == "shuffled_field" else no_terminal if arm == "no_terminal_transport" else planner
                            states = np.broadcast_to(body.initial_prior, phi.shape) if arm == "no_context" else phi
                            predicted = np.asarray([model.value(z, a) - model.value(z, b) for z, a, b in zip(states, cases["left"], cases["right"])])
                        require(np.isfinite(predicted).all(), "previsão não finita")
                        rows[arm] = metrics(cases["actual"], predicted)
                        for index, (i, a, b) in enumerate(cases["bank_cases"]):
                            records.append({"event": "event_field_planned_comparison", "game": game, "basis": basis, "partition": partition,
                                "horizon": horizon, "arm": arm, "bank_row": i, "left": a, "right": b,
                                "actual_delta": float(cases["actual"][index]), "predicted_delta": float(predicted[index])})
                    curves[str(horizon)] = rows
                cases = view(bank, offset)
                cov = coverage(cases["actual"], cases["episodes"])
                primary = curves["16"]["potential"]
                checks = {"contextual_coverage": cov["sufficient"]}
                for arm in ARMS:
                    if arm != "potential":
                        for metric in ("mse", "fair_tie_regret"):
                            checks[f"{metric}_vs_{arm}"] = bool(primary[metric] < curves["16"][arm][metric] - 1e-12)
                status = "inconclusive_coverage" if not cov["sufficient"] else "passed" if all(checks.values()) else "failed"
                reports[game][basis][partition] = {"coverage": cov, "curves": curves, "gate_checks": checks, "status": status}
                if basis == "no_generated_functions":
                    statuses.append(status)
    with EVENTS.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    status = "event_field_gate_passed_not_agi" if all(s == "passed" for s in statuses) else "event_field_inconclusive_coverage" if "inconclusive_coverage" in statuses else "event_field_gate_failed"
    save_new(RESULT, {"status": status, "games": reports, "event_count": len(records), "events_sha256": sha(EVENTS),
        "bodies_sha256": sha(BODIES), "proposal_sha256": sha(PROPOSAL), "human_level_goal_achieved": False, "no_future_observations_used_at_decision": True,
        "fit_calls_during_evaluation": 0,
        "primary_basis": "no_generated_functions", "secondary_basis_not_promoted_to_primary": True})
    return {"status": status, "primary_cells": {g: {p: r["status"] for p, r in v["no_generated_functions"].items()} for g, v in reports.items()},
        "primary16": {g: {p: r["curves"]["16"] for p, r in v["no_generated_functions"].items()} for g, v in reports.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "fit", "diagnose", "evaluate"))
    mode = parser.parse_args().mode
    print(json.dumps({"prepare": prepare, "fit": fit, "diagnose": diagnose, "evaluate": evaluate}[mode](), ensure_ascii=False, indent=2))
