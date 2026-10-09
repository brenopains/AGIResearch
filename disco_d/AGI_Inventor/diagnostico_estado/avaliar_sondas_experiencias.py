"""Repete o seletor sem mudar o portão, com o extrator de memória acumulada."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from avaliar_composicao import features
from avaliar_estado_preditivo import load as load_states
from extrator_experiencias import ExperienceExtractor
from selecionar_sondas_decisao import input_matrix, measure, preferences, select

HERE = Path(__file__).resolve().parent


def evaluate(game):
    extractor = ExperienceExtractor(game)
    groups = extractor.episode
    reward = np.load(HERE / "resultados" / f"{game}_horizontes_97.npz")["rewards"][:, :, -1]
    test_reward = np.load(HERE / "resultados" / f"{game}_horizontes_103.npz")["rewards"][:, :, -1]
    _, test_pairs, _, _, test_groups = load_states(game, 103)
    test_x = features(test_pairs[:, 0, 0])
    construction, selection, evaluation = groups < 4, np.isin(groups, (4, 5)), groups >= 6
    zcv, qcv, rcv = extractor.cross_predict(set(range(4)))
    zval, qval, rval, selection_counts = extractor.predict(extractor.root_x[selection], set(range(4)))
    zeval, qeval, reval, _ = extractor.predict(extractor.root_x[evaluation], set(range(4)))
    full_zcv, full_qcv, full_rcv = extractor.cross_predict(set(range(8)))
    full_zpred, full_qpred, full_rpred, full_counts = extractor.predict(test_x, set(range(8)))
    target = preferences(reward)
    constant = int(np.argmax(reward[construction].mean(0)))
    full_constant = int(np.argmax(reward.mean(0)))
    baseline = {
        "selection": measure(zcv[construction], zval, target[construction], reward[selection], groups[selection], constant),
        "evaluation": measure(zcv[construction], zeval, target[construction], reward[evaluation], groups[evaluation], constant),
        "seed103": measure(full_zcv, full_zpred, target, test_reward, test_groups, full_constant),
    }
    arms = {}
    for name, cv, val, ev, full_cv, full_pred in (
        ("witness_readout", qcv, qval, qeval, full_qcv, full_qpred),
        ("random_readout", rcv, rval, reval, full_rcv, full_rpred),
    ):
        chosen, trace = select(zcv[construction], zval, cv[construction], val,
                               target[construction], reward[selection], groups[selection])
        arms[name] = {
            "selected": chosen, "selection_trace": trace,
            "status": "candidate_selected" if chosen else "no_validation_decision_gain",
            "selection": measure(input_matrix(zcv[construction], cv[construction], chosen),
                                 input_matrix(zval, val, chosen), target[construction],
                                 reward[selection], groups[selection], constant),
            "evaluation": measure(input_matrix(zcv[construction], cv[construction], chosen),
                                  input_matrix(zeval, ev, chosen), target[construction],
                                  reward[evaluation], groups[evaluation], constant),
            "seed103": measure(input_matrix(full_zcv, full_cv, chosen),
                               input_matrix(full_zpred, full_pred, chosen), target,
                               test_reward, test_groups, full_constant),
        }
    actual_q = np.load(HERE / "resultados" / f"{game}_sondas_decisao_103.npz")["response"]
    train_q = np.load(HERE / "resultados" / f"{game}_sondas_decisao_97.npz")["response"]
    denominator = np.mean((actual_q - train_q.mean(0)) ** 2, axis=0)
    error = np.mean((actual_q - full_qpred) ** 2, axis=0)
    ratios = np.divide(error, denominator, out=np.full_like(error, np.nan), where=denominator > 1e-12)
    root_only = KernelRidge().fit(extractor.root_x, train_q).predict(test_x)
    root_error = np.mean((actual_q - root_only) ** 2, axis=0)
    root_ratios = np.divide(root_error, denominator, out=np.full_like(error, np.nan), where=denominator > 1e-12)
    prediction_bank = HERE / "resultados" / f"{game}_predicoes_sondas_experiencias.npz"
    np.savez_compressed(prediction_bank, zcv=zcv, qcv=qcv, rcv=rcv,
                        zval=zval, qval=qval, rval=rval,
                        zeval=zeval, qeval=qeval, reval=reval,
                        full_zcv=full_zcv, full_qcv=full_qcv, full_rcv=full_rcv,
                        full_zpred=full_zpred, full_qpred=full_qpred, full_rpred=full_rpred)
    report = {"status": "exploratory_accumulated_experience", "game": game,
              "certified_composition_max_absolute_error": extractor.certified_root_max_error,
              "selection_training_counts": selection_counts, "full_training_counts": full_counts,
              "probe_error_to_constant": ratios.tolist(), "baseline": baseline, "arms": arms,
              "same_probes_root_only_error": root_ratios.tolist(),
              "prediction_bank_sha256": hashlib.sha256(prediction_bank.read_bytes()).hexdigest(),
              "selected_executable_probes": [extractor.probes[i] for i in arms["witness_readout"]["selected"]],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "extractor_sha256": hashlib.sha256((HERE / "extrator_experiencias.py").read_bytes()).hexdigest()}
    output = HERE / "resultados" / f"{game}_sondas_experiencias.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    result = evaluate(args.game)
    print(json.dumps({"game": args.game,
                      "composition_error": result["certified_composition_max_absolute_error"],
                      "training_counts": result["full_training_counts"],
                      "mean_probe_error": float(np.nanmean(result["probe_error_to_constant"])),
                      "baseline": {k: v["mean_regret"] for k, v in result["baseline"].items()},
                      "arms": {name: {"selected": arm["selected"],
                                      "selection": arm["selection"]["mean_regret"],
                                      "evaluation": arm["evaluation"]["mean_regret"],
                                      "seed103": arm["seed103"]["mean_regret"]}
                               for name, arm in result["arms"].items()}}, ensure_ascii=False, indent=2))
