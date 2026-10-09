"""Bancada prospectiva: fontes congeladas, corpos antes da coleta de avaliação."""
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
from piloto_funcoes_contraexemplos import HERE, require, save_new, sha, select, verify_proposal as verify_previous
from transporte_funcoes import FunctionTransport
from funcoes_contraexemplos import PairFunctions, GeneratedTransport, append_slots, fit_features, propose
from funcoes_prospectivas import propose_future
from diagnosticar_semantica_prefixo import metrics

STEM = HERE / "resultados" / "prospectiva"
PROPOSAL = STEM.with_name(STEM.name + "_proposta.json")
DICTIONARIES = STEM.with_name(STEM.name + "_dicionarios_manifesto.json")
BODIES = STEM.with_name(STEM.name + "_corpos_manifesto.json")
FIT_EVENTS = STEM.with_name(STEM.name + "_ajuste.jsonl")
RESULT, EVENTS = STEM.with_suffix(".json"), STEM.with_suffix(".jsonl")
GENERATORS = ("future_pairs", "shuffled_future", "reward_pairs", "random_directions")
ARMS = (*GENERATORS, "no_generated_functions", "remove_generated", "remove_context", "zero")
SEEDS = (149, 151, 157)


def bank_path(game, seed):
    return STEM.with_name(STEM.name + f"_{game}_{seed}")


def prepare():
    require(not any(p.exists() for p in (PROPOSAL, DICTIONARIES, BODIES, FIT_EVENTS, RESULT, EVENTS)), "rodada existe; preservar")
    verify_previous()
    for game in ("breakout", "pong"):
        for seed in SEEDS:
            require(not list((HERE / "resultados").glob(bank_path(game, seed).name + "*")), "semente prospectiva já usada")
    words, seen = [], set()
    for seed in (10171, 10173):
        rng = np.random.default_rng(seed)
        count = 0
        while count < 4:
            word = tuple(map(int, rng.integers(0, 4, 16)))
            if word not in seen:
                seen.add(word)
                words.append(list(word))
                count += 1
    sources = ("piloto_prospectivo.py", "coletar_prospectiva.py", "auditar_prospectiva.py", "testes_bancada_prospectiva.py",
        "funcoes_prospectivas.py", "testes_funcoes_prospectivas.py", "PROSPECTIVA.md", "coleta.py",
        "coletar_transicoes_mistas.py", "piloto_funcoes_contraexemplos.py", "funcoes_contraexemplos.py",
        "transporte_funcoes.py", "diagnosticar_semantica_prefixo.py", "auditar_funcoes_contraexemplos.py",
        "auditar_semantica_prefixo.py")
    value = {"status": "frozen_before_new_collection", "seeds": list(SEEDS), "programs": words,
        "training_programs": 4, "evaluation_programs": 8, "episodes": 8, "checkpoints": [0, 2, 4, 6],
        "blocks": 16, "calls_per_block": 8, "maximum_calls_per_bank": 50000,
        "maximum_collection_wall_seconds": 120, "maximum_fit_wall_seconds": 120,
        "maximum_worker_rss_bytes": 6 * 1024 ** 3, "proposal_witness_limit": 512,
        "proposal_witnesses_per_episode_action": 16, "alpha": .05,
        "generators": list(GENERATORS), "arms": list(ARMS),
        "all_cells_required": ["breakout_known", "breakout_new", "pong_known", "pong_new"],
        "human_level_goal_achieved": False, "source_hashes": {p: sha(HERE / p) for p in sources}}
    save_new(PROPOSAL, value)
    return {"status": value["status"], "proposal_sha256": sha(PROPOSAL)}


def verify():
    value = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    for path, expected in value["source_hashes"].items():
        require(sha(HERE / path) == expected, f"fonte mudou: {path}")
    verify_previous()
    return value


def verify_bodies():
    verify()
    value = json.loads(BODIES.read_text(encoding="utf-8"))
    for path, expected in value["source_hashes"].items():
        require(sha(HERE / path) == expected, f"corpo mudou: {path}")
    return value


def load_bank(game, seed):
    proposal = verify()
    stem = bank_path(game, seed)
    certificate = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    require(certificate["status"] == "prospective_bank_complete" and certificate["bank_sha256"] == sha(stem.with_suffix(".npz"))
        and certificate["events_sha256"] == sha(stem.with_suffix(".jsonl")) and certificate["proposal_sha256"] == sha(PROPOSAL), "banco não certificado")
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
        bank = {key: archive[key].copy() for key in archive.files}
    p = 8 if seed == 157 else 4
    require(bank["histories"].shape == (32, p, 17, 16, 336) and bank["rewards"].shape == (32, p, 16)
        and bank["valid"].shape == (32, p, 17), "forma de banco incorreta")
    require(np.array_equal(bank["programs"], proposal["programs"][:p]) and np.isfinite(bank["histories"]).all(), "observações/programas divergem")
    require(not np.any((~bank["valid"][..., :-1]) & bank["valid"][..., 1:])
        and not np.any(bank["rewards"][~bank["valid"][..., :-1]]), "padding terminal incorreto")
    require(all(np.array_equal(bank["histories"][:, 0, 0], bank["histories"][:, j, 0]) for j in range(p)), "raiz contrafactual difere")
    return bank


def transitions(bank):
    mask = bank["valid"][..., :-1]
    ids = np.argwhere(mask)
    actions = np.broadcast_to(bank["programs"], mask.shape)
    return {"source": bank["histories"][..., :-1, :, :][mask], "successor": bank["histories"][..., 1:, :, :][mask],
        "actions": actions[mask], "reward": bank["rewards"][mask], "continuation": bank["valid"][..., 1:][mask].astype(float),
        "episodes": bank["episode"][ids[:, 0]], "bank_indices": ids}


def witness_indices(actions, episodes, per_group=16):
    rows = []
    for episode in sorted(set(episodes)):
        for action in sorted(set(actions)):
            ids = np.flatnonzero((episodes == episode) & (actions == action))
            if len(ids):
                rows.extend(ids[np.linspace(0, len(ids) - 1, min(per_group, len(ids)), dtype=int)].tolist())
    return np.asarray(sorted(rows), int)


def view(bank, pair_offset=0, horizon=16):
    cases = [(i, p, p + 1) for i in range(32) for p in (pair_offset, pair_offset + 2)]
    return {"ids": np.arange(len(cases)), "bank_cases": cases,
        "history": np.asarray([bank["histories"][i, p, 0] for i, p, q in cases]),
        "left": [bank["programs"][p, :horizon].tolist() for i, p, q in cases],
        "right": [bank["programs"][q, :horizon].tolist() for i, p, q in cases],
        "actual": np.asarray([bank["rewards"][i, p, :horizon].sum() - bank["rewards"][i, q, :horizon].sum() for i, p, q in cases]),
        "episodes": np.asarray([bank["episode"][i] for i, p, q in cases])}


def coverage(actual, episodes):
    values = np.asarray(actual).reshape(32, 2)
    opposite = [bool(np.any(values[:, p] < -1e-12) and np.any(values[:, p] > 1e-12)) for p in range(2)]
    active = sorted(map(int, set(np.asarray(episodes)[np.abs(actual) > 1e-12])))
    return {"opposite_preferences_per_pair": opposite, "episodes_with_contrast": active,
        "non_tied_comparisons": int(np.count_nonzero(np.abs(actual) > 1e-12)),
        "sufficient": bool(all(opposite) and len(active) >= 3)}


def fit():
    proposal = verify()
    require(not any(p.exists() for p in (DICTIONARIES, BODIES, FIT_EVENTS)), "ajuste existe; preservar")
    require(not any(bank_path(g, 157).with_suffix(".npz").exists() for g in ("breakout", "pong")), "avaliação coletada antes do congelamento")
    started = time.perf_counter()
    def check_time():
        require(time.perf_counter() - started < proposal["maximum_fit_wall_seconds"], "orçamento esgotado; inconclusivo")
    construction, objects, records, paths = {}, {}, [], []
    for game in ("breakout", "pong"):
        data = transitions(load_bank(game, 149))
        base = FunctionTransport().fit(data["source"], data["successor"], data["actions"], data["reward"], data["continuation"])
        base_path = STEM.with_name(STEM.name + f"_{game}_base.npz")
        base.save(base_path)
        ids = witness_indices(data["actions"], data["episodes"])
        require(len(ids) <= proposal["proposal_witness_limit"], "testemunhas excedem orçamento")
        sample = {key: value[ids] for key, value in data.items()}
        phi, pn = base.features(sample["source"]), base.features(sample["successor"])
        functions = {}
        construction[game] = {"base": base_path.relative_to(HERE).as_posix(), "training_transitions": len(data["reward"]),
            "witness_indices": ids.tolist(), "dictionaries": {}, "generation": {}}
        paths.extend((base_path, base_path.with_suffix(".json"), bank_path(game, 149).with_suffix(".npz"), bank_path(game, 149).with_suffix(".json")))
        for arm in GENERATORS:
            check_time()
            if arm in ("future_pairs", "shuffled_future"):
                functions[arm], stats = propose_future(sample["source"], sample["actions"], pn, sample["continuation"], sample["episodes"], phi, arm == "shuffled_future")
            else:
                functions[arm], stats = propose(sample["source"], sample["actions"], sample["reward"], sample["episodes"], phi,
                    "counterexample_pairs" if arm == "reward_pairs" else "random_directions")
            path = STEM.with_name(STEM.name + f"_{game}_{arm}_dicionario.npz")
            functions[arm].save(path)
            paths.extend((path, path.with_suffix(".json")))
            construction[game]["dictionaries"][arm] = path.relative_to(HERE).as_posix()
            construction[game]["generation"][arm] = stats
            records.append({"event": "prospective_dictionary_generated", "game": game, "arm": arm, **stats})
        objects[game] = (base, data, functions)
    save_new(DICTIONARIES, {"status": "dictionaries_frozen_before_selector_calls", "construction": construction,
        "source_hashes": {p.relative_to(HERE).as_posix(): sha(p) for p in paths}})
    reports, bodies, frozen_paths = {}, {}, [DICTIONARIES, PROPOSAL]
    for game in ("breakout", "pong"):
        base, data, dictionaries = objects[game]
        validation_bank = load_bank(game, 151)
        validation = view(validation_bank)
        cov = coverage(validation["actual"], validation["episodes"])
        reports[game], bodies[game] = {"coverage": cov, "selection": {}}, {}
        empty = PairFunctions(np.empty((0, base.support.shape[1])), np.empty(0), [], [])
        phi, pn = base.features(data["source"]), base.features(data["successor"])
        baseline = fit_features(base, empty, [], append_slots(phi, np.empty((len(phi), 0)), []),
            append_slots(pn, np.empty((len(pn), 0)), []), data["reward"], data["continuation"], data["actions"])
        models = {"no_generated_functions": baseline}
        for arm in GENERATORS:
            check_time()
            if cov["sufficient"]:
                models[arm], report = select(base, dictionaries[arm], data, validation, game, arm, records, check_time)
            else:
                models[arm] = copy.deepcopy(baseline)
                score = metrics(validation["actual"], baseline.comparison(validation["history"], validation["left"], validation["right"]))
                report = {"status": "inconclusive_validation_coverage", "selected": [], "fit_calls": 0,
                    "baseline_validation": score, "final_validation": score, "accepted_evidence": []}
            reports[game]["selection"][arm] = report
        removed = copy.deepcopy(models["future_pairs"])
        width = len(base.initial_prior)
        removed.selected = [None] * 4
        removed.reward[:, width:] = 0
        removed.transport[:, width:, :] = 0
        removed.transport[:, :, width:] = 0
        removed.initial_prior[width:] = 0
        models["remove_generated"] = removed
        context = copy.deepcopy(models["future_pairs"])
        context.mode = "no_context"
        models["remove_context"] = context
        for arm, model in models.items():
            path = STEM.with_name(STEM.name + f"_{game}_{arm}_corpo.npz")
            model.save(path)
            restored = GeneratedTransport.load(path)
            require(np.array_equal(model.comparison(validation["history"], validation["left"], validation["right"]),
                restored.comparison(validation["history"], validation["left"], validation["right"])), "exportação divergiu")
            bodies[game][arm] = path.relative_to(HERE).as_posix()
            frozen_paths.extend((path, path.with_suffix(".json")))
        frozen_paths.extend((bank_path(game, 151).with_suffix(".npz"), bank_path(game, 151).with_suffix(".json")))
    with FIT_EVENTS.open("x", encoding="utf-8") as stream:
        for event in records:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    frozen_paths.append(FIT_EVENTS)
    check_time()
    save_new(BODIES, {"status": "bodies_frozen_before_new_evaluation_collection", "bodies": bodies, "validation": reports,
        "wall_seconds": time.perf_counter() - started, "fit_events": len(records),
        "source_hashes": {p.relative_to(HERE).as_posix(): sha(p) for p in frozen_paths}})
    return {"status": "bodies_frozen_before_new_evaluation_collection", "wall_seconds": time.perf_counter() - started,
        "selection": {g: {a: v["selected"] for a, v in r["selection"].items()} for g, r in reports.items()},
        "coverage": {g: r["coverage"] for g, r in reports.items()}}


def evaluate():
    frozen = verify_bodies()
    require(not RESULT.exists() and not EVENTS.exists(), "avaliação já existe; preservar")
    games, events, cells = {}, [], []
    for game in ("breakout", "pong"):
        bank = load_bank(game, 157)
        certificate = json.loads(bank_path(game, 157).with_suffix(".json").read_text(encoding="utf-8"))
        require(certificate["frozen_bodies_sha256_before_collection"] == sha(BODIES), "corpos não precederam avaliação")
        models = {a: GeneratedTransport.load(HERE / p) for a, p in frozen["bodies"][game].items()}
        games[game] = {}
        for name, offset in (("known", 0), ("new", 4)):
            curves = {}
            for horizon in (4, 8, 16):
                cases = view(bank, offset, horizon)
                row = {}
                for arm in ARMS:
                    prediction = np.zeros(len(cases["actual"])) if arm == "zero" else models[arm].comparison(cases["history"], cases["left"], cases["right"])
                    require(np.isfinite(prediction).all(), "previsão não finita")
                    row[arm] = metrics(cases["actual"], prediction)
                    for index, (i, p, q) in enumerate(cases["bank_cases"]):
                        events.append({"event": "prospective_held_comparison", "game": game, "partition": name, "horizon": horizon,
                            "arm": arm, "bank_row": i, "left": p, "right": q, "episode": int(bank["episode"][i]),
                            "actual_delta": float(cases["actual"][index]), "predicted_delta": float(prediction[index])})
                curves[str(horizon)] = row
            cov = coverage(view(bank, offset)["actual"], view(bank, offset)["episodes"])
            primary = curves["16"]["future_pairs"]
            checks = {"selected_function": bool(frozen["validation"][game]["selection"]["future_pairs"]["selected"]),
                "validation_coverage": frozen["validation"][game]["coverage"]["sufficient"], "held_coverage": cov["sufficient"]}
            for arm in ARMS:
                if arm != "future_pairs":
                    for metric in ("mse", "fair_tie_regret"):
                        checks[f"{metric}_vs_{arm}"] = bool(primary[metric] < curves["16"][arm][metric] - 1e-12)
            status = "inconclusive_coverage" if not checks["validation_coverage"] or not checks["held_coverage"] else "passed" if all(checks.values()) else "failed"
            games[game][name] = {"coverage": cov, "curves": curves, "gate_checks": checks, "status": status}
            cells.append(status)
    with EVENTS.open("x", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    status = "capability_gate_passed_not_agi" if all(s == "passed" for s in cells) else "gate_not_passed_with_inconclusive_cells" if "inconclusive_coverage" in cells else "capability_gate_failed"
    result = {"status": status, "games": games, "event_count": len(events), "events_sha256": sha(EVENTS),
        "proposal_sha256": sha(PROPOSAL), "bodies_sha256": sha(BODIES), "evaluation_source": "new_collection_after_body_freeze",
        "human_level_goal_achieved": False, "trained_function_cross_domain_transfer_demonstrated": False}
    save_new(RESULT, result)
    return {"status": status, "cells": {g: {p: r["status"] for p, r in partitions.items()} for g, partitions in games.items()},
        "primary": {g: {p: r["curves"]["16"]["future_pairs"] for p, r in partitions.items()} for g, partitions in games.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "fit", "evaluate"))
    mode = parser.parse_args().mode
    print(json.dumps({"prepare": prepare, "fit": fit, "evaluate": evaluate}[mode](), ensure_ascii=False, indent=2))
