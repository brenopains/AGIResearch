"""A geometria é informada por funções aprendidas; controle de dimensões e sinal."""

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
from coordenadas_consequencias import ConsequenceGeometry, ConsequenceReadout, prepare_edges
from extrator_experiencias import ExperienceExtractor
from futuros_distribuidos import make_edges
from piloto_planos_propostos import execute_plan, sha, stem as previous_stem
from planejador_distribuido import ExperiencePlanner

HERE = Path(__file__).resolve().parent
POLICIES = ("learned_credit", "consequence_geometry", "shuffled_geometry", "random_geometry")
LIMIT = 65000


def stem(game):
    return HERE / "resultados" / f"{game}_coordenadas_consequencias_103"


def proposal_path(game):
    return stem(game).with_name(stem(game).name + "_propostas.json")


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
    credit = ConsequenceReadout().fit(edges)
    shuffled = ConsequenceReadout().fit(edges, shuffled=True)
    body_paths = []
    for name, reader in (("credito", credit), ("embaralhado", shuffled)):
        path = stem(game).with_name(stem(game).name + f"_{name}.npz")
        reader.save(path)
        restored = ConsequenceReadout.load(path)
        if not np.allclose(reader.coordinates(root), restored.coordinates(root), atol=1e-12):
            raise RuntimeError("corpo salvo não reproduz leitura")
        body_paths += [path.relative_to(HERE).as_posix(), path.with_suffix(".json").relative_to(HERE).as_posix()]
    support = np.concatenate([e["source"] for e in edges])
    real_geometry = ConsequenceGeometry(credit, support)
    shuffled_geometry = ConsequenceGeometry(shuffled, support)
    random_geometry = ConsequenceGeometry(credit, support, random=True)
    reports, scales = {}, {}
    for policy, geometry in (("learned_credit", None), ("consequence_geometry", real_geometry),
                             ("shuffled_geometry", shuffled_geometry), ("random_geometry", random_geometry)):
        bank = prepare_edges(edges, credit, geometry)
        query = geometry.transform(root) if geometry is not None else root
        reports[policy] = ExperiencePlanner(bank).search(query, 16, 8)
        scales[policy] = None if geometry is None else {"state": geometry.scale_z, "readout": geometry.scale_q}
        print(json.dumps({"game": game, "policy_generated": policy,
                          "cpu_seconds": time.process_time() - start_cpu}), flush=True)
    files = ["piloto_coordenadas_consequencias.py", "coordenadas_consequencias.py",
             "PILOTO_COORDENADAS_CONSEQUENCIAS.md", "planejador_distribuido.py", "piloto_planos_propostos.py",
             "coleta.py", "avaliar.py", "avaliar_composicao.py", "avaliar_efeitos_locais.py",
             "variaveis_movimento.py", "avaliar_estado_preditivo.py", "extrator_experiencias.py",
             "futuros_distribuidos.py", "valor_composto.py", "diagnosticar_credito.py",
             "selecionar_sondas_decisao.py", "avaliar_sondas.py"]
    files += [f"resultados/{game}_{suffix}" for suffix in (
        "estado_preditivo_97.npz", "base_estado_preditivo.npz", "horizontes_97.npz", "horizontes_103.npz",
        "efeitos_locais_103.npz", "sondas_decisao_definicoes.json", "sondas_decisao_97.npz",
        "planos_propostos_103.json")]
    files += body_paths
    report = {"status": "frozen_before_plan_execution", "game": game, "train_seed": 97, "test_seed": 103,
              "policies": reports, "scales": scales, "training_counts": counts,
              "source_hashes": {p: sha(HERE / p) for p in files},
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
    rng = np.random.default_rng(9773)
    samples = paired[rng.integers(0, 8, (10000, 8))].mean(1)
    return {"mean_gain": float(paired.mean()), "bootstrap_95": np.quantile(samples, (.025, .975)).tolist(),
            "positive_episodes": int(np.sum(paired > 1e-12)),
            "regressed_episodes": int(np.sum(paired < -1e-12)), "per_episode": paired.tolist()}


def execute(game):
    output = stem(game).with_suffix(".json")
    if output.exists():
        raise RuntimeError("piloto concluído; preservar")
    proposal = json.loads(proposal_path(game).read_text(encoding="utf-8"))
    for p, expected in proposal["source_hashes"].items():
        if sha(HERE / p) != expected:
            raise RuntimeError("corpo, parâmetro ou banco mudou")
    verify()
    verify_manifest()
    previous = json.loads(previous_stem(game).with_suffix(".json").read_text(encoding="utf-8"))
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
                if name == "consequence_geometry" and checkpoint == 0:
                    ale.restoreSystemState(state)
                    replay = execute_plan(ale, actions, plan, budget)
                    if replay[:3] != (gain, length, done) or not np.array_equal(replay[3], final):
                        raise RuntimeError("replay divergiu")
                    replay_checked += 1
            ale.restoreSystemState(state)
        print(json.dumps({"game": game, "episode_evaluated": episode, "steps": LIMIT - budget[0]}), flush=True)
    episode_ids = np.repeat(np.arange(8), 8)
    references = {f"previous_{name}": np.asarray(data["per_root"]) for name, data in previous["returns"].items()}
    all_returns = {**returns, **references}
    comparisons = {name: comparison(returns["consequence_geometry"], data, episode_ids)
                   for name, data in all_returns.items() if name != "consequence_geometry"}
    checks = {}
    for name, data in comparisons.items():
        checks[f"positive_mean_vs_{name}"] = data["mean_gain"] > 1e-12
        checks[f"positive_lower_vs_{name}"] = data["bootstrap_95"][0] > 1e-12
        checks[f"two_episodes_vs_{name}"] = data["positive_episodes"] >= 2
    checks["no_regression_vs_constant"] = comparisons["previous_constant"]["regressed_episodes"] == 0
    passed = bool(all(checks.values()))
    status = "positive_development_needs_confirmation" if passed else "failed_development_gate"
    if not np.any(np.column_stack(list(all_returns.values())).max(1) > references["previous_constant"]):
        status, passed = "inconclusive_no_margin", False
    report = {"status": status, "positive_pilot": passed, "game": game, "train_seed": 97, "test_seed": 103,
              "human_level_goal_achieved": False, "comparisons": comparisons, "checks": checks,
              "returns": {name: {"mean": float(data.mean()), "per_root": data.tolist(),
                                 "per_episode": [float(data[episode_ids == e].mean()) for e in range(8)]}
                          for name, data in all_returns.items()},
              "prediction_mse": {name: float(np.mean((np.asarray(proposal["policies"][name]["values"]) - returns[name]) ** 2))
                                 for name in POLICIES},
              "nonconstant_plans": {name: sum(len(set(p)) > 1 for p in proposal["policies"][name]["plans"]) for name in POLICIES},
              "steps_used": LIMIT - budget[0], "certified_roots": roots_checked,
              "certified_short_branches": short_checked, "replays_certified": replay_checked,
              "proposals_sha256": sha(proposal_path(game)), "new_policy_count": len(POLICIES),
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
                      **({"scales": r["scales"]} if args.propor else {
                          "returns": {n: v["mean"] for n, v in r["returns"].items()},
                          "mse": r["prediction_mse"], "nonconstant": r["nonconstant_plans"], "steps": r["steps_used"]})}, indent=2))
