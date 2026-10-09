"""Testa composições mesmo quando nenhuma sonda isolada melhora decisão.

Propostas incluem grupos naturais por programa/testemunha e 96 conjuntos
diversos de 2/4/8 leituras, fixados antes de medir. Controles veem os mesmos
conjuntos e o mesmo seletor. Não promove por ajuste em validação.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from selecionar_sondas_decisao import input_matrix, measure, preferences, regrets
from avaliar_sondas import ScaledRidge
from avaliar_estado_preditivo import load as load_states

HERE = Path(__file__).resolve().parent


def proposals(probes):
    n = len(probes)
    sets = {(i,) for i in range(n)}
    if n:
        sets.add(tuple(range(n)))
    for key in ("program_index", "source_pair"):
        groups = {}
        for index, probe in enumerate(probes):
            value = tuple(probe[key]) if isinstance(probe[key], list) else probe[key]
            groups.setdefault(value, []).append(index)
        sets.update(tuple(group) for group in groups.values())
    rng = np.random.default_rng(9707)
    for size in (2, 4, 8):
        if n < size:
            continue
        for _ in range(32):
            sets.add(tuple(sorted(rng.choice(n, size=size, replace=False).tolist())))
    return sorted(sets, key=lambda s: (len(s), s))


def select_bundle(base_fit, base_val, qfit, qval, target, actual, groups, candidate_sets):
    baseline = regrets(actual, ScaledRidge().fit(base_fit, target).predict(base_val))
    rows, eligible = [], []
    for chosen in candidate_sets:
        prediction = ScaledRidge().fit(input_matrix(base_fit, qfit, list(chosen)), target).predict(
            input_matrix(base_val, qval, list(chosen)))
        regret = regrets(actual, prediction)
        gains = [float((baseline - regret)[groups == e].mean()) for e in (4, 5)]
        passed = all(g > 1e-12 for g in gains)
        rows.append({"indexes": list(chosen), "regret": float(regret.mean()),
                     "paired_gain_each_episode": gains, "eligible": passed})
        if passed:
            eligible.append((float(regret.mean()), len(chosen), tuple(chosen)))
    chosen = list(min(eligible)[2]) if eligible else []
    return chosen, rows


def evaluate(game):
    definition_path = HERE / "resultados" / f"{game}_sondas_decisao_definicoes.json"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    candidate_sets = proposals(definition["probes"])
    prediction_path = HERE / "resultados" / f"{game}_predicoes_sondas_experiencias.npz"
    p = np.load(prediction_path)
    _, _, _, _, groups = load_states(game, 97)
    _, _, _, _, test_groups = load_states(game, 103)
    reward = np.load(HERE / "resultados" / f"{game}_horizontes_97.npz")["rewards"][:, :, -1]
    test_reward = np.load(HERE / "resultados" / f"{game}_horizontes_103.npz")["rewards"][:, :, -1]
    target = preferences(reward)
    fit, val, test = groups < 4, np.isin(groups, (4, 5)), groups >= 6
    constant = int(np.argmax(reward[fit].mean(0)))
    full_constant = int(np.argmax(reward.mean(0)))
    baseline = {
        "selection": measure(p["zcv"][fit], p["zval"], target[fit], reward[val], groups[val], constant),
        "evaluation": measure(p["zcv"][fit], p["zeval"], target[fit], reward[test], groups[test], constant),
        "seed103": measure(p["full_zcv"], p["full_zpred"], target, test_reward, test_groups, full_constant),
    }
    arms = {}
    rng = np.random.default_rng(9708)
    permutation_fit, permutation_full = rng.permutation(int(fit.sum())), rng.permutation(64)
    for name, qfit, qval, qtest, qfull, qpred in (
        ("witness_readout", p["qcv"][fit], p["qval"], p["qeval"], p["full_qcv"], p["full_qpred"]),
        ("random_readout", p["rcv"][fit], p["rval"], p["reval"], p["full_rcv"], p["full_rpred"]),
        ("shuffled_feature_rows", p["qcv"][fit][permutation_fit], p["qval"], p["qeval"],
         p["full_qcv"][permutation_full], p["full_qpred"]),
    ):
        chosen, trace = select_bundle(p["zcv"][fit], p["zval"], qfit, qval,
                                      target[fit], reward[val], groups[val], candidate_sets)
        arms[name] = {"selected": chosen, "selection_trace": trace,
                      "selection": measure(input_matrix(p["zcv"][fit], qfit, chosen),
                                           input_matrix(p["zval"], qval, chosen), target[fit], reward[val], groups[val], constant),
                      "evaluation": measure(input_matrix(p["zcv"][fit], qfit, chosen),
                                            input_matrix(p["zeval"], qtest, chosen), target[fit], reward[test], groups[test], constant),
                      "seed103": measure(input_matrix(p["full_zcv"], qfull, chosen),
                                         input_matrix(p["full_zpred"], qpred, chosen), target, test_reward, test_groups, full_constant)}
    report = {"status": "exploratory_no_independent_confirmation", "game": game,
              "candidate_sets": len(candidate_sets), "baseline": baseline, "arms": arms,
              "strict_gain_gate_zero_margin_episodes": [int(e) for e, value in baseline["selection"]["per_episode"].items() if value <= 1e-12],
              "selected_executable_probes": [definition["probes"][i] for i in arms["witness_readout"]["selected"]],
              "prediction_bank_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if report["strict_gain_gate_zero_margin_episodes"]:
        report["status"] = "inconclusive_strict_gain_gate_has_zero_margin"
    output = HERE / "resultados" / f"{game}_conjuntos_sondas.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    result = evaluate(args.game)
    print(json.dumps({"game": args.game, "sets": result["candidate_sets"],
                      "baseline": {k: v["mean_regret"] for k, v in result["baseline"].items()},
                      "arms": {name: {"selected": arm["selected"],
                                      "selection": arm["selection"]["mean_regret"],
                                      "evaluation": arm["evaluation"]["mean_regret"],
                                      "seed103": arm["seed103"]["mean_regret"]}
                               for name, arm in result["arms"].items()}}, ensure_ascii=False, indent=2))
