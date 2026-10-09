"""A assinatura de efeitos curtos prevê efeitos distantes sem simular pixels?"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from avaliar_composicao import TARGET_STEPS, features, load, score

HERE = Path(__file__).resolve().parent


def crossfit_short(x, signature, episode):
    output = np.empty_like(signature)
    for held in range(8):
        fit = np.flatnonzero(episode != held)
        test = np.flatnonzero(episode == held)
        model = KernelRidge().fit(x[fit], signature[fit].reshape(len(fit), -1))
        output[test] = model.predict(x[test]).reshape(len(test), 4, 336)
    return output


def evaluate(game):
    train_path, train_initial, train_future, train_episode, _ = load(game, 97)
    test_path, test_initial, test_future, test_episode, _ = load(game, 103)
    xtrain = features(train_initial)
    xtest = features(test_initial)
    train_s8 = train_future[:, :, 0, 1] - train_initial[:, None, 1]
    test_s8 = test_future[:, :, 0, 1] - test_initial[:, None, 1]
    train_s8_crossfit = crossfit_short(xtrain, train_s8, train_episode)
    local = KernelRidge().fit(xtrain, train_s8.reshape(64, -1))
    test_s8_pred = local.predict(xtest).reshape(64, 4, 336)
    curves = {}
    for horizon in TARGET_STEPS[1:]:
        index = horizon // 8 - 1
        train_target = train_future[:, :, index, 1] - train_initial[:, None, 1]
        actual = test_future[:, :, index, 1]
        direct = KernelRidge().fit(xtrain, train_target.reshape(64, -1))
        from_pixels = direct.predict(xtest).reshape(64, 4, 336)
        oracle = KernelRidge().fit(train_s8.reshape(64, -1),
                                   train_target.reshape(64, -1))
        from_true_short = oracle.predict(test_s8.reshape(64, -1)).reshape(64, 4, 336)
        bridge = KernelRidge().fit(train_s8_crossfit.reshape(64, -1),
                                   train_target.reshape(64, -1))
        from_predicted_short = bridge.predict(test_s8_pred.reshape(64, -1)).reshape(64, 4, 336)
        template = np.broadcast_to(train_target.mean(axis=0), train_target.shape)
        current = test_initial[:, None, 1]
        curves[str(horizon)] = {
            "pixels_diretos": score(actual, np.clip(current + from_pixels, 0, 1), test_episode),
            "assinatura_curta_real_oracular": score(
                actual, np.clip(current + from_true_short, 0, 1), test_episode),
            "assinatura_curta_prevista": score(
                actual, np.clip(current + from_predicted_short, 0, 1), test_episode),
            "molde_por_acao": score(
                actual, np.clip(current + train_target.mean(axis=0), 0, 1), test_episode),
        }
    report = {"status": "exploratorio_sobre_semente_103_ja_inspecionada",
              "game": game, "train_seed": 97, "test_seed": 103,
              "train_bank_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
              "test_bank_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "crossfit_short_by_episode": True,
              "curves": curves}
    output = HERE / "resultados" / f"{game}_ponte_assinatura.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    report = evaluate(args.game)
    print(json.dumps({"game": args.game,
                      "means": {h: {name: row["mean"] for name, row in part.items()}
                                for h, part in report["curves"].items()}},
                     ensure_ascii=False, indent=2))
