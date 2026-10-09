"""Confirma efeito visual de oito passos em semente ALE reservada."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge, views
from avaliar_efeitos_locais import contrast_loss, downsample

HERE = Path(__file__).resolve().parent
TRAIN_SEED = 97
TEST_SEED = 103
SIDE = 21
NULLS = 32


def read_bank(game, seed):
    path = HERE / "resultados" / f"{game}_efeitos_locais_{seed}.npz"
    data = np.load(path)
    history = data["histories"]
    now = downsample(history[:, -1, None], SIDE)[:, 0]
    target = downsample(data["outcomes"], SIDE) - now[:, None]
    return path, views(history)["r0_21x16_2"], target.reshape(len(history), 4, -1), data["episode"]


def ratios(actual, predicted, episodes):
    results, energies = [], []
    for episode in range(8):
        ix = np.flatnonzero(episodes == episode)
        error, baseline = contrast_loss(actual[ix], predicted[ix])
        energy = float(baseline.sum())
        energies.append(energy)
        results.append(float(error.sum() / energy) if energy > 1e-12 else None)
    usable = [r for r in results if r is not None]
    return {"per_episode": results,
            "mean": float(np.mean(usable)) if usable else None,
            "contrast_energy": energies}


def confirm(game):
    train_path, xtrain, ytrain, _ = read_bank(game, TRAIN_SEED)
    test_path, xtest, ytest, episodes = read_bank(game, TEST_SEED)
    model = KernelRidge().fit(xtrain, ytrain.reshape(len(ytrain), -1))
    predicted = model.predict(xtest).reshape(len(xtest), 4, -1)
    template = np.broadcast_to(ytrain.mean(axis=0), ytest.shape)
    primary = ratios(ytest, predicted, episodes)
    fixed = ratios(ytest, template, episodes)
    rng = np.random.default_rng(10309)
    shuffled_actions = []
    for _ in range(NULLS):
        labels = ytrain.copy()
        for index in range(len(labels)):
            labels[index] = ytrain[index, rng.permutation(4)]
        alternative = KernelRidge().fit(xtrain, labels.reshape(len(labels), -1))
        forecast = alternative.predict(xtest).reshape(len(xtest), 4, -1)
        shuffled_actions.append(ratios(ytest, forecast, episodes)["mean"])
    shuffled_history = []
    for _ in range(NULLS):
        permuted = xtest.copy()
        for episode in range(8):
            ix = np.flatnonzero(episodes == episode)
            permuted[ix] = xtest[rng.permutation(ix)]
        forecast = model.predict(permuted).reshape(len(xtest), 4, -1)
        shuffled_history.append(ratios(ytest, forecast, episodes)["mean"])
    action_null_equal = sum(value <= primary["mean"] for value in shuffled_actions)
    history_null_equal = sum(value <= primary["mean"] for value in shuffled_history)
    episodes_better_than_fixed = sum(
        primary["per_episode"][i] is not None
        and primary["per_episode"][i] < fixed["per_episode"][i]
        for i in range(8))
    gate_game = bool(all(energy > 1e-12 for energy in primary["contrast_energy"])
                     and all(r is not None and r < 1 for r in primary["per_episode"])
                     and fixed["mean"] - primary["mean"] >= 0.10
                     and episodes_better_than_fixed >= 6
                     and action_null_equal <= 1 and history_null_equal <= 1)
    report = {"game": game, "train_seed": TRAIN_SEED, "test_seed": TEST_SEED,
              "train_bank_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
              "test_bank_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
              "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                for name in ("confirmar_efeitos_locais.py", "avaliar_efeitos_locais.py",
                                             "avaliar.py", "coletar_efeitos_locais.py")},
              "primary": primary, "fixed_action_template": fixed,
              "episodes_better_than_fixed": episodes_better_than_fixed,
              "shuffled_action_ratios": shuffled_actions,
              "shuffled_history_ratios": shuffled_history,
              "shuffled_action_equal_or_better": action_null_equal,
              "shuffled_history_equal_or_better": history_null_equal,
              "gate_game": gate_game}
    output = HERE / "resultados" / f"{game}_confirmacao_efeitos.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    report = confirm(args.game)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("shuffled_action_ratios", "shuffled_history_ratios")},
                     ensure_ascii=False, indent=2))
