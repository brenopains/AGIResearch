"""Seleção de desenvolvimento sem exigir melhora em episódios já perfeitos.

O critério desta nova rodada é fixado antes de executar as leituras nos jogos.
103 é desenvolvimento: nenhuma seleção nela é confirmação. Uma semente nova
só será aberta após vantagem sobre base, melhor ação fixa e ambos os controles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from avaliar_sondas import ScaledRidge
from conjuntos_sondas import proposals
from selecionar_sondas_decisao import input_matrix, measure, preferences, regrets

HERE = Path(__file__).resolve().parent


def eligible(gains):
    gains = np.asarray(gains)
    return bool(np.all(gains >= -1e-12) and np.sum(gains > 1e-12) >= 2)


def choose(base_fit, base_val, qfit, qval, target, actual, groups, parent_sets, channels=4):
    baseline = regrets(actual, ScaledRidge().fit(base_fit, target).predict(base_val))
    records, candidates = [], []
    for parents in parent_sets:
        columns = [channels * index + c for index in parents for c in range(channels)]
        prediction = ScaledRidge().fit(input_matrix(base_fit, qfit, columns), target).predict(
            input_matrix(base_val, qval, columns))
        regret = regrets(actual, prediction)
        gains = [float((baseline - regret)[groups == e].mean()) for e in sorted(set(groups))]
        permitted = eligible(gains)
        records.append({"parents": list(parents), "mean_regret": float(regret.mean()),
                        "gain_each_episode": gains, "eligible": permitted})
        if permitted:
            candidates.append((float(regret.mean()), len(parents), tuple(parents)))
    chosen = list(min(candidates)[2]) if candidates else []
    columns = [channels * index + c for index in chosen for c in range(channels)]
    return chosen, columns, records


def evaluate(game):
    started = time.process_time()
    definition_path = HERE / "resultados" / f"{game}_leituras_variaveis_definicoes.json"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    parent_sets = proposals(definition["probes"])
    bank_path = HERE / "resultados" / f"{game}_leituras_variaveis.npz"
    data = np.load(bank_path)
    prior = np.load(HERE / "resultados" / f"{game}_predicoes_sondas_experiencias.npz")
    base_fit, base_val = prior["full_zcv"], prior["full_zpred"]
    reward = np.load(HERE / "resultados" / f"{game}_horizontes_97.npz")["rewards"][:, :, -1]
    actual = np.load(HERE / "resultados" / f"{game}_horizontes_103.npz")["rewards"][:, :, -1]
    groups = data["test_episode"]
    target = preferences(reward)
    constant = int(np.argmax(reward.mean(0)))
    baseline = measure(base_fit, base_val, target, actual, groups, constant)
    rng = np.random.default_rng(9741)
    shuffled = data["cv"][rng.permutation(64)]
    arms = {}
    for name, fit, test, channels in (
        ("variable_readout", data["cv"], data["predicted"], 4),
        ("random_kernels", data["random_cv"], data["random_predicted"], 4),
        ("shuffled_association", shuffled, data["predicted"], 4),
        ("real_responses_oracle", data["real_train"], data["real_test"], 4),
        ("fixed_linear_readout", prior["full_qcv"], prior["full_qpred"], 1),
    ):
        chosen, columns, trace = choose(base_fit, base_val, fit, test, target,
                                         actual, groups, parent_sets, channels=channels)
        result = measure(input_matrix(base_fit, fit, columns), input_matrix(base_val, test, columns),
                         target, actual, groups, constant)
        arms[name] = {"selected_parents": chosen, "columns": columns,
                      "decision": result, "selection_trace": trace}
    proposed = arms["variable_readout"]["decision"]["mean_regret"]
    checks = {
        "selected_nonempty": bool(arms["variable_readout"]["selected_parents"]),
        "better_than_base": proposed < baseline["mean_regret"] - 1e-12,
        "better_than_constant": proposed < baseline["constant_regret"] - 1e-12,
        "better_than_random": proposed < arms["random_kernels"]["decision"]["mean_regret"] - 1e-12,
        "better_than_shuffled": proposed < arms["shuffled_association"]["decision"]["mean_regret"] - 1e-12,
    }
    report = {"status": "development_only_no_independent_confirmation", "game": game,
              "training_seed": 97, "selection_seed": 103, "parent_sets": len(parent_sets),
              "baseline": baseline, "arms": arms, "pilot_checks": checks,
              "positive_pilot": bool(all(checks.values())), "promoted": False,
              "baseline_zero_regret_episodes": [e for e, r in baseline["per_episode"].items() if r == 0],
              "selected_executable_probes": [definition["probes"][i] for i in arms["variable_readout"]["selected_parents"]],
              "bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
              "definition_sha256": hashlib.sha256(definition_path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cpu_seconds": time.process_time() - started}
    output = HERE / "resultados" / f"{game}_avaliacao_leituras_variaveis.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    r = evaluate(args.game)
    print(json.dumps({"game": args.game, "baseline": r["baseline"]["mean_regret"],
                      "constant": r["baseline"]["constant_regret"],
                      "arms": {name: {"selected": arm["selected_parents"],
                                      "regret": arm["decision"]["mean_regret"]}
                               for name, arm in r["arms"].items()},
                      "positive_pilot": r["positive_pilot"], "checks": r["pilot_checks"]},
                     ensure_ascii=False, indent=2))
