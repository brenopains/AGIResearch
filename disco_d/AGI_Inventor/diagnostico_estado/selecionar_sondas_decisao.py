"""Seleciona por decisão com episódios separados, sem trocar critério depois.

0--3: pares, extração e treino; 4--5: seleção; 6--7: teste de desenvolvimento.
103 é diagnóstico posterior já visto; não é confirmação independente.
Os controles recebem igual número de propostas e o mesmo seletor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from avaliar_composicao import features
from avaliar_estado_preditivo import encode, load as load_states
from avaliar_sondas import ScaledRidge, cross_predict

HERE = Path(__file__).resolve().parent
MAX_SELECTED = 4


def preferences(reward):
    span = np.ptp(reward, axis=1, keepdims=True)
    return np.divide(reward - reward.mean(1, keepdims=True), span,
                     out=np.zeros_like(reward), where=span > 0)


def regrets(reward, predicted):
    choices = np.argmax(predicted, axis=1)
    return reward.max(1) - reward[np.arange(len(reward)), choices]


def cv_first_four(x, y, groups):
    result = np.empty_like(y, dtype=np.float64)
    for group in range(4):
        train, test = groups != group, groups == group
        result[test] = KernelRidge().fit(x[train], y[train]).predict(x[test])
    return result


def input_matrix(base, extra, indexes):
    return np.column_stack((base, extra[:, indexes])) if indexes else base


def select(base_fit, base_val, extra_fit, extra_val, target, actual, val_groups):
    selected, rows = [], []
    current_prediction = ScaledRidge().fit(base_fit, target).predict(base_val)
    current = regrets(actual, current_prediction)
    for round_index in range(MAX_SELECTED):
        candidates = []
        for index in range(extra_fit.shape[1]):
            if index in selected:
                continue
            chosen = selected + [index]
            predicted = ScaledRidge().fit(input_matrix(base_fit, extra_fit, chosen), target).predict(
                input_matrix(base_val, extra_val, chosen))
            regret = regrets(actual, predicted)
            paired_gain = [float((current - regret)[val_groups == e].mean()) for e in (4, 5)]
            eligible = all(gain > 1e-12 for gain in paired_gain)
            rows.append({"round": round_index, "candidate": index,
                         "mean_regret": float(regret.mean()),
                         "gain_each_selection_episode": paired_gain, "eligible": eligible})
            if eligible:
                candidates.append((float(regret.mean()), index, regret))
        if not candidates:
            break
        candidates.sort(key=lambda row: (row[0], row[1]))
        _, index, current = candidates[0]
        selected.append(index)
    return selected, rows


def measure(fit_x, test_x, target, actual, groups, constant):
    prediction = ScaledRidge().fit(fit_x, target).predict(test_x)
    regret = regrets(actual, prediction)
    const = actual.max(1) - actual[:, constant]
    return {"mean_regret": float(regret.mean()), "constant_regret": float(const.mean()),
            "roots_with_contrast": int(np.sum(np.ptp(actual, axis=1) > 0)),
            "per_episode": {str(e): float(regret[groups == e].mean()) for e in sorted(set(groups))},
            "constant_per_episode": {str(e): float(const[groups == e].mean()) for e in sorted(set(groups))},
            "choices": np.argmax(prediction, axis=1).tolist()}


def evaluate(game):
    _, pairs, signature, _, groups = load_states(game, 97)
    _, test_pairs, test_signature, _, test_groups = load_states(game, 103)
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    x, test_x = features(pairs[:, 0, 0]), features(test_pairs[:, 0, 0])
    z = encode(signature[:, 0, 0], basis["mean"], basis["basis"])
    test_z = encode(test_signature[:, 0, 0], basis["mean"], basis["basis"])
    train_path = HERE / "resultados" / f"{game}_sondas_decisao_97.npz"
    test_path = HERE / "resultados" / f"{game}_sondas_decisao_103.npz"
    answers, test_answers = np.load(train_path), np.load(test_path)
    response, random_response = answers["response"], answers["random_response"]
    reward = np.load(HERE / "resultados" / f"{game}_horizontes_97.npz")["rewards"][:, :, -1]
    test_reward = np.load(HERE / "resultados" / f"{game}_horizontes_103.npz")["rewards"][:, :, -1]
    target = preferences(reward)
    construction, selection, evaluation = groups < 4, np.isin(groups, (4, 5)), groups >= 6
    xfit = x[construction]
    zfit = cv_first_four(xfit, z[construction], groups[construction])
    encoder = KernelRidge().fit(xfit, z[construction])
    zval, zeval = encoder.predict(x[selection]), encoder.predict(x[evaluation])
    constant = int(np.argmax(reward[construction].mean(0)))
    baseline = {
        "selection": measure(zfit, zval, target[construction], reward[selection], groups[selection], constant),
        "evaluation": measure(zfit, zeval, target[construction], reward[evaluation], groups[evaluation], constant),
    }
    full_zcv = cross_predict(x, z, groups)
    full_zpred = KernelRidge().fit(x, z).predict(test_x)
    full_constant = int(np.argmax(reward.mean(0)))
    baseline["seed103"] = measure(full_zcv, full_zpred, target, test_reward, test_groups, full_constant)
    rng = np.random.default_rng(9705)
    shuffled = response.copy()
    shuffled[construction] = response[construction][rng.permutation(int(construction.sum()))]
    arms = {}
    for name, raw, raw_test, oracle in (
        ("witness_readout", response, test_answers["response"], False),
        ("random_readout", random_response, test_answers["random_response"], False),
        ("shuffled_answers", shuffled, test_answers["response"], False),
        ("real_answers_oracle", response, test_answers["response"], True),
    ):
        if oracle:
            qfit, qval, qeval = raw[construction], raw[selection], raw[evaluation]
            fit_base, val_base, eval_base = z[construction], z[selection], z[evaluation]
        else:
            qfit = cv_first_four(xfit, raw[construction], groups[construction])
            model = KernelRidge().fit(xfit, raw[construction])
            qval, qeval = model.predict(x[selection]), model.predict(x[evaluation])
            fit_base, val_base, eval_base = zfit, zval, zeval
        selected, trace = select(fit_base, val_base, qfit, qval, target[construction],
                                  reward[selection], groups[selection])
        result = {"selected": selected, "selection_trace": trace,
                  "status": "candidate_selected" if selected else "no_validation_decision_gain",
                  "selection": measure(input_matrix(fit_base, qfit, selected),
                                       input_matrix(val_base, qval, selected), target[construction],
                                       reward[selection], groups[selection], constant),
                  "evaluation": measure(input_matrix(fit_base, qfit, selected),
                                        input_matrix(eval_base, qeval, selected), target[construction],
                                        reward[evaluation], groups[evaluation], constant)}
        if oracle:
            full_fit_base, full_test_base, full_qcv, full_qpred = z, test_z, raw, raw_test
        else:
            full_fit_base, full_test_base = full_zcv, full_zpred
            full_raw = (response[np.random.default_rng(9705).permutation(64)]
                        if name == "shuffled_answers" else raw)
            full_qcv = cross_predict(x, full_raw, groups)
            full_qpred = KernelRidge().fit(x, full_raw).predict(test_x)
        result["seed103"] = measure(input_matrix(full_fit_base, full_qcv, selected),
                                    input_matrix(full_test_base, full_qpred, selected),
                                    target, test_reward, test_groups, full_constant)
        arms[name] = result
    definition_path = HERE / "resultados" / f"{game}_sondas_decisao_definicoes.json"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    report = {"status": "development_not_confirmation", "game": game,
              "split": {"construction": [0, 1, 2, 3], "selection": [4, 5], "evaluation": [6, 7]},
              "train_seed": 97, "retrospective_test_seed": 103,
              "candidate_count": response.shape[1], "baseline": baseline, "arms": arms,
              "selected_executable_probes": [definition["probes"][i] for i in arms["witness_readout"]["selected"]],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "definition_sha256": hashlib.sha256(definition_path.read_bytes()).hexdigest(),
              "train_bank_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
              "test_bank_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest()}
    output = HERE / "resultados" / f"{game}_selecionar_sondas_decisao.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    result = evaluate(args.game)
    print(json.dumps({"game": args.game, "candidates": result["candidate_count"],
                      "baseline": {k: v["mean_regret"] for k, v in result["baseline"].items()},
                      "arms": {name: {"selected": arm["selected"],
                                      "selection": arm["selection"]["mean_regret"],
                                      "evaluation": arm["evaluation"]["mean_regret"],
                                      "seed103": arm["seed103"]["mean_regret"]}
                               for name, arm in result["arms"].items()}},
                     ensure_ascii=False, indent=2))
