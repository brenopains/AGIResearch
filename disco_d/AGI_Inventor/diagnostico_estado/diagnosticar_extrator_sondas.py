"""Separa resolução/histórico da seleção da pergunta, no mesmo banco."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge, views
from avaliar_composicao import features, load as load_horizons, score
from avaliar_estado_preditivo import encode, load as load_states
from avaliar_sondas import ScaledRidge, cross_predict, reward_metrics

HERE = Path(__file__).resolve().parent


def evaluate(game):
    train_raw = np.load(HERE / "resultados" / f"{game}_efeitos_locais_97.npz")
    test_raw = np.load(HERE / "resultados" / f"{game}_efeitos_locais_103.npz")
    train_views = views(train_raw["histories"])
    test_views = views(test_raw["histories"])
    _, train_pairs, train_sig, _, episode = load_states(game, 97)
    _, test_pairs, _, _, test_episode = load_states(game, 103)
    train_path, _, train_future, _, _ = load_horizons(game, 97)
    test_path, _, test_future, _, _ = load_horizons(game, 103)
    answers_path = HERE / "resultados" / f"{game}_sondas_97.npz"
    answers = np.load(answers_path)["response"]
    true_test = np.load(HERE / "resultados" / f"{game}_sondas_103.npz")["response"]
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    xtrain, xtest = features(train_pairs[:, 0, 0]), features(test_pairs[:, 0, 0])
    root_signature = train_sig[:, 0, 0].reshape(64, -1)
    base_cv = encode(cross_predict(xtrain, root_signature, episode).reshape(64, 4, 336),
                     basis["mean"], basis["basis"])
    base_pred = encode(KernelRidge().fit(xtrain, root_signature).predict(xtest).reshape(64, 4, 336),
                       basis["mean"], basis["basis"])
    target = train_future[:, :, -1, 1].reshape(64, -1)
    actual = test_future[:, :, -1, 1]
    train_reward = np.load(train_path)["rewards"][:, :, -1]
    test_reward = np.load(test_path)["rewards"][:, :, -1]
    denominator = np.mean((true_test - answers.mean(0)) ** 2, axis=0)
    results = {}
    for name in ("r0_21x16_2", "hist_21x16_16", "res_42x32_2", "hist_42x32_16", "native_160x210_2"):
        x, test_x = train_views[name], test_views[name]
        qcv = cross_predict(x, answers, episode)
        qpred = KernelRidge().fit(x, answers).predict(test_x)
        error = np.mean((true_test - qpred) ** 2, axis=0)
        ratio = np.divide(error, denominator, out=np.full_like(error, np.nan),
                          where=denominator > 1e-12)
        fit_input = np.column_stack((base_cv, qcv))
        test_input = np.column_stack((base_pred, qpred))
        forecast = ScaledRidge().fit(fit_input, target).predict(test_input).reshape(64, 4, 336)
        forecast_reward = ScaledRidge().fit(fit_input, train_reward).predict(test_input)
        results[name] = {"sensor_features": x.shape[1],
                         "probe_error_to_constant": ratio.tolist(),
                         "mean_probe_error": float(np.nanmean(ratio)),
                         "visual_error": score(actual, forecast, test_episode),
                         "reward": reward_metrics(test_reward, forecast_reward,
                                                  train_reward, test_episode)}
    report = {"status": "exploratory_same_cohort_same_probes", "game": game,
              "train_seed": 97, "test_seed": 103,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "probe_bank_sha256": hashlib.sha256(answers_path.read_bytes()).hexdigest(),
              "views": results}
    output = HERE / "resultados" / f"{game}_diagnostico_extrator_sondas.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    result = evaluate(args.game)
    print(json.dumps({"game": args.game,
                      "views": {name: {"probe_error": entry["mean_probe_error"],
                                       "visual_error": entry["visual_error"]["mean"],
                                       "regret": entry["reward"]["mean_regret"]}
                                for name, entry in result["views"].items()}},
                     ensure_ascii=False, indent=2))
