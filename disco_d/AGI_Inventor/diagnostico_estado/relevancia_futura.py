"""Extrai coordenadas do efeito curto que predizem contrastes futuros."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from avaliar_composicao import features, load, score

HERE = Path(__file__).resolve().parent
RANK = 8
HORIZONS = (16, 32, 64, 128)
PAIRS = list(itertools.combinations(range(4), 2))


def prospective_basis(short_signature, future):
    pieces = []
    for horizon in HORIZONS:
        end = future[:, :, horizon // 8 - 1, 1]
        contrast = np.stack([end[:, a] - end[:, b] for a, b in PAIRS], axis=1)
        flat = contrast.reshape(len(end), -1)
        scale = max(float(np.sqrt(np.mean(flat ** 2))), 1e-8)
        pieces.append(flat / scale)
    target = np.concatenate(pieces, axis=1)
    xmean = short_signature.mean(0)
    x = short_signature - xmean
    target -= target.mean(0)
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    matrix = singular[:, None] * (u.T @ target)
    gram = matrix @ matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:RANK]
    directions = vt.T @ eigenvectors[:, order]
    return xmean, directions, eigenvalues[order]


def crossfit(x, target, episode):
    result = np.empty_like(target)
    for held in range(8):
        fit = np.flatnonzero(episode != held)
        test = np.flatnonzero(episode == held)
        result[test] = KernelRidge().fit(x[fit], target[fit]).predict(x[test])
    return result


def evaluate(game):
    train_path, train_initial, train_future, train_episode, _ = load(game, 97)
    test_path, test_initial, test_future, test_episode, _ = load(game, 103)
    train_pixels = features(train_initial)
    test_pixels = features(test_initial)
    short_train = (train_future[:, :, 0, 1] - train_initial[:, None, 1]).reshape(64, -1)
    short_test = (test_future[:, :, 0, 1] - test_initial[:, None, 1]).reshape(64, -1)
    mean, directions, eigenvalues = prospective_basis(short_train, train_future)
    ztrain = (short_train - mean) @ directions
    ztest_oracle = (short_test - mean) @ directions
    ztrain_oof = crossfit(train_pixels, ztrain, train_episode)
    ztest_pred = KernelRidge().fit(train_pixels, ztrain).predict(test_pixels)
    curves = {}
    for horizon in HORIZONS:
        index = horizon // 8 - 1
        target = train_future[:, :, index, 1] - train_initial[:, None, 1]
        actual = test_future[:, :, index, 1]
        current = test_initial[:, None, 1]
        pixels = KernelRidge().fit(train_pixels, target.reshape(64, -1))
        oracle = KernelRidge().fit(ztrain, target.reshape(64, -1))
        predicted = KernelRidge().fit(ztrain_oof, target.reshape(64, -1))
        curves[str(horizon)] = {
            "pixels_diretos": score(actual, np.clip(current + pixels.predict(test_pixels).reshape(64, 4, 336), 0, 1), test_episode),
            "estado_futuro_oracular": score(actual, np.clip(current + oracle.predict(ztest_oracle).reshape(64, 4, 336), 0, 1), test_episode),
            "estado_futuro_previsto": score(actual, np.clip(current + predicted.predict(ztest_pred).reshape(64, 4, 336), 0, 1), test_episode),
        }
    basis_path = HERE / "resultados" / f"{game}_base_relevancia_futura.npz"
    np.savez_compressed(basis_path, mean=mean, directions=directions,
                        eigenvalues=eigenvalues)
    report = {"status": "exploratorio_semente_103_ja_inspecionada",
              "game": game, "train_seed": 97, "test_seed": 103,
              "rank": RANK, "train_bank_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
              "test_bank_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
              "basis_sha256": hashlib.sha256(basis_path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "curves": curves}
    output = HERE / "resultados" / f"{game}_relevancia_futura.json"
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
