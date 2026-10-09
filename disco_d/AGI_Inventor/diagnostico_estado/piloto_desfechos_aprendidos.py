"""Propor antes de observar; avaliar desfechos conjuntos com controles reais."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from avaliar_composicao import features, reduce_frames
from avaliar_estado_preditivo import encode, load
from coleta import gray, make_ale
from confirmar_futuros_distribuidos import verify
from confirmar_leituras_variaveis import verify_manifest
from desfechos_aprendidos import OutcomeModel, OutcomePlanner
from extrator_experiencias import ExperienceExtractor
from futuros_distribuidos import make_edges
from piloto_planos_propostos import execute_plan, sha

HERE = Path(__file__).resolve().parent
POLICIES = ("conditional_outcomes", "shuffled_outcomes", "prior_outcomes")
LIMIT = 65000


def stem(game):
    return HERE / "resultados" / f"{game}_desfechos_aprendidos_103"


def proposal_path(game):
    return stem(game).with_name(stem(game).name + "_propostas.json")


def reference_paths(game):
    return {"plans": HERE / "resultados" / f"{game}_planos_propostos_103.json",
            "coordinates": HERE / "resultados" / f"{game}_coordenadas_consequencias_103.json"}


def propose(game):
    output = proposal_path(game)
    if output.exists():
        raise RuntimeError("propostas já existem; preservar")
    verify()
    verify_manifest()
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    _, _, signature, valid, _ = load(game, 97)
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    z = encode(signature, basis["mean"], basis["basis"])
    reward = np.load(HERE / "resultados" / f"{game}_horizontes_97.npz")["rewards"]
    edges = make_edges(z, valid, reward)
    initial = reduce_frames(np.load(HERE / "resultados" / f"{game}_horizontes_103.npz")["histories"][:, -2:])
    root, _, _, counts = ExperienceExtractor(game).predict(features(initial), set(range(8)))
    reports, categories, body_paths, agreement = {}, {}, [], {}
    for policy in POLICIES:
        model = OutcomeModel().fit(edges, per_group=16, shuffled=policy == "shuffled_outcomes",
                                   prior_only=policy == "prior_outcomes")
        path = stem(game).with_name(stem(game).name + f"_{policy}.npz")
        if path.exists() or path.with_suffix(".json").exists():
            raise RuntimeError("corpo já existe; preservar")
        model.save(path)
        restored = OutcomeModel.load(path)
        export_error = max(float(np.max(np.abs(model.posterior(root, a) - restored.posterior(root, a))))
                           for a in range(len(edges)))
        if export_error > 1e-12:
            raise RuntimeError("corpo salvo não reproduz probabilidades")
        planner = OutcomePlanner(restored)
        report = planner.search(root, horizon=16, width=8)
        independent_values = np.diag(planner.plan_values(root, report["plans"]))
        composition_error = float(np.max(np.abs(independent_values - report["values"])))
        if composition_error > 1e-10:
            raise RuntimeError("busca divergiu da composição independente")
        reports[policy] = report
        categories[policy] = [len(o["reward"]) for o in restored.outcomes]
        agreement[policy] = {"export_max_error": export_error, "composition_max_error": composition_error}
        body_paths += [path.relative_to(HERE).as_posix(), path.with_suffix(".json").relative_to(HERE).as_posix()]
        print(json.dumps({"game": game, "policy_generated": policy, "categories": categories[policy],
                          "cpu_seconds": time.process_time() - start_cpu}), flush=True)
    if len({tuple(v) for v in categories.values()}) != 1:
        raise RuntimeError("controles têm categorias diferentes")
    if len({tuple(r["expansions_per_root"]) for r in reports.values()}) != 1:
        raise RuntimeError("controles têm orçamento de busca diferente")
    files = ["piloto_desfechos_aprendidos.py", "desfechos_aprendidos.py", "PILOTO_DESFECHOS_APRENDIDOS.md",
             "planejador_distribuido.py", "piloto_planos_propostos.py", "coleta.py", "avaliar.py",
             "avaliar_composicao.py", "avaliar_efeitos_locais.py", "variaveis_movimento.py",
             "avaliar_estado_preditivo.py", "extrator_experiencias.py", "futuros_distribuidos.py",
             "valor_composto.py", "diagnosticar_credito.py", "selecionar_sondas_decisao.py", "avaliar_sondas.py"]
    files += [f"resultados/{game}_{suffix}" for suffix in (
        "estado_preditivo_97.npz", "base_estado_preditivo.npz", "horizontes_97.npz", "horizontes_103.npz",
        "efeitos_locais_103.npz", "sondas_decisao_definicoes.json", "sondas_decisao_97.npz")]
    files += [p.relative_to(HERE).as_posix() for p in reference_paths(game).values()] + body_paths
    report = {"status": "frozen_before_plan_execution", "game": game, "train_seed": 97, "test_seed": 103,
              "policies": reports, "categories": categories, "agreement": agreement,
              "training_counts": counts, "source_hashes": {p: sha(HERE / p) for p in files},
              "body_parameters": body_paths, "new_ale_steps": 0,
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with output.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for name, data in reports.items():
            for i, trace in enumerate(data["traces"]):
                handle.write(json.dumps({"event": "hypotheses_proposed", "root": i, "policy": name,
                                         "alternatives": trace, "observed_return": None}) + "\n")
    return report


def comparison(proposed, reference, episode):
    paired = np.asarray([(proposed - reference)[episode == e].mean() for e in range(8)])
    rng = np.random.default_rng(9787)
    samples = paired[rng.integers(0, 8, (10000, 8))].mean(1)
    return {"mean_gain": float(paired.mean()), "bootstrap_95": np.quantile(samples, (.025, .975)).tolist(),
            "positive_episodes": int(np.sum(paired > 1e-12)), "regressed_episodes": int(np.sum(paired < -1e-12)),
            "per_episode": paired.tolist()}


def load_references(game):
    references, prior_costs = {}, {}
    for group, path in reference_paths(game).items():
        report = json.loads(path.read_text(encoding="utf-8"))
        # The coordinate report contains old references too; include each actual arm only once.
        for name, value in report["returns"].items():
            if group == "coordinates" and name.startswith("previous_"):
                continue
            references[f"{group}_{name}"] = np.asarray(value["per_root"])
        prior_costs[group] = report["steps_used"]
    return references, prior_costs


def execute(game):
    output = stem(game).with_suffix(".json")
    if output.exists():
        raise RuntimeError("piloto concluído; preservar")
    proposal = json.loads(proposal_path(game).read_text(encoding="utf-8"))
    for p, expected_hash in proposal["source_hashes"].items():
        if sha(HERE / p) != expected_hash:
            raise RuntimeError(f"corpo, parâmetro, referência ou banco mudou: {p}")
    verify()
    verify_manifest()
    references, prior_costs = load_references(game)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    expected = np.load(HERE / "resultados" / f"{game}_horizontes_103.npz")["histories"]
    short = np.load(HERE / "resultados" / f"{game}_efeitos_locais_103.npz")
    ale, rng, budget = make_ale(game, 103), np.random.default_rng(103), [LIMIT]
    actions = list(ale.getMinimalActionSet()[:4])
    returns = {name: np.zeros(64) for name in POLICIES}
    events, roots_checked, short_checked, replay_checked = [], 0, 0, 0
    for episode in range(8):
        ale.reset_game()
        frame = gray(ale)
        history = deque([frame.copy() for _ in range(16)], maxlen=16)
        for checkpoint in range(8):
            for _ in range(128):
                if budget[0] <= 0:
                    raise RuntimeError("orçamento ALE esgotado")
                ale.act(actions[int(rng.integers(0, 4))])
                budget[0] -= 1
                if ale.game_over():
                    ale.reset_game()
                    frame = gray(ale)
                    history = deque([frame.copy() for _ in range(16)], maxlen=16)
                else:
                    history.append(gray(ale))
            i = episode * 8 + checkpoint
            if not np.array_equal(np.stack(history), expected[i]):
                raise RuntimeError("raiz não corresponde ao banco anterior")
            roots_checked += 1
            state = ale.cloneSystemState()
            for action in range(4):
                ale.restoreSystemState(state)
                gain, length, done, frame = execute_plan(ale, actions, [action], budget)
                if (gain != short["rewards"][i, action] or length != short["lengths"][i, action]
                        or done != short["terminal"][i, action] or not np.array_equal(frame, short["outcomes"][i, action])):
                    raise RuntimeError("ramo curto não reproduziu dado")
                short_checked += 1
            measured = {}
            for name in POLICIES:
                plan = proposal["policies"][name]["plans"][i]
                key = tuple(plan)
                before = budget[0]
                reused = key in measured
                if not reused:
                    ale.restoreSystemState(state)
                    measured[key] = execute_plan(ale, actions, plan, budget)
                gain, length, done, final = measured[key]
                physical_steps = before - budget[0]
                returns[name][i] = gain
                events.append({"event": "measurement_reused" if reused else "plan_executed", "root": i,
                               "episode": episode, "policy": name, "plan": plan, "observed_return": gain,
                               "predicted_return": proposal["policies"][name]["values"][i],
                               "physical_steps": physical_steps, "plan_steps": length, "terminal": done})
                if name == "conditional_outcomes" and checkpoint == 0:
                    ale.restoreSystemState(state)
                    replay = execute_plan(ale, actions, plan, budget)
                    if replay[:3] != (gain, length, done) or not np.array_equal(replay[3], final):
                        raise RuntimeError("replay divergiu")
                    replay_checked += 1
            ale.restoreSystemState(state)
        print(json.dumps({"game": game, "episode_evaluated": episode, "steps": LIMIT - budget[0]}), flush=True)
    episode_ids = np.repeat(np.arange(8), 8)
    all_returns = {**returns, **references}
    comparisons = {name: comparison(returns["conditional_outcomes"], data, episode_ids)
                   for name, data in all_returns.items() if name != "conditional_outcomes"}
    checks = {}
    for name, data in comparisons.items():
        checks[f"positive_mean_vs_{name}"] = data["mean_gain"] > 1e-12
        checks[f"positive_lower_vs_{name}"] = data["bootstrap_95"][0] > 1e-12
        checks[f"two_episodes_vs_{name}"] = data["positive_episodes"] >= 2
    checks["no_regression_vs_constant"] = comparisons["plans_constant"]["regressed_episodes"] == 0
    passed = bool(all(checks.values()))
    status = "positive_development_needs_confirmation" if passed else "failed_development_gate"
    if not np.any(np.column_stack(list(all_returns.values())).max(1) > references["plans_constant"]):
        status, passed = "inconclusive_no_margin", False
    report = {"status": status, "positive_pilot": passed, "game": game, "train_seed": 97, "test_seed": 103,
              "human_level_goal_achieved": False, "comparisons": comparisons, "checks": checks,
              "returns": {name: {"mean": float(data.mean()), "per_root": data.tolist(),
                                 "per_episode": [float(data[episode_ids == e].mean()) for e in range(8)]}
                          for name, data in all_returns.items()},
              "prediction_mse": {name: float(np.mean((np.asarray(proposal["policies"][name]["values"]) - returns[name]) ** 2))
                                 for name in POLICIES},
              "nonconstant_plans": {name: sum(len(set(p)) > 1 for p in proposal["policies"][name]["plans"]) for name in POLICIES},
              "steps_used": LIMIT - budget[0], "cached_reference_costs_separate": prior_costs,
              "certified_roots": roots_checked, "certified_short_branches": short_checked,
              "replays_certified": replay_checked, "proposals_sha256": sha(proposal_path(game)),
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with output.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--propor", action="store_true")
    modes.add_argument("--executar", action="store_true")
    args = parser.parse_args()
    r = propose(args.game) if args.propor else execute(args.game)
    print(json.dumps({"game": args.game, "status": r["status"], "cpu_seconds": r["cpu_seconds"],
                      **({"categories": r["categories"], "agreement": r["agreement"]} if args.propor else {
                          "returns": {n: v["mean"] for n, v in r["returns"].items()},
                          "mse": r["prediction_mse"], "nonconstant": r["nonconstant_plans"], "steps": r["steps_used"]})}, indent=2))
