"""Curva de previsão direta versus composição de transições de oito passos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from avaliar_efeitos_locais import contrast_loss

HERE = Path(__file__).resolve().parent
TARGET_STEPS = (8, 16, 32, 64, 128)


def reduce_frames(frames):
    shape = frames.shape
    if shape[-2:] != (210, 160):
        raise ValueError("quadros de tamanho inesperado")
    return frames.reshape(*shape[:-2], 21, 10, 16, 10).mean(
        axis=(-3, -1), dtype=np.float32).reshape(*shape[:-2], 336) / 255


def load(game, seed):
    path = HERE / "resultados" / f"{game}_horizontes_{seed}.npz"
    bank = np.load(path)
    initial = reduce_frames(bank["histories"][:, -2:])
    future = reduce_frames(bank["pairs"])
    episode = bank["episode"].copy()
    terminal = bank["terminal"].copy()
    if future.shape != (64, 4, 16, 2, 336):
        raise RuntimeError("banco de horizontes incompleto")
    return path, initial, future, episode, terminal


def features(pairs):
    current, previous = pairs[..., 1, :], pairs[..., 0, :]
    return np.concatenate((current, current - previous), axis=-1)


def score(actual, predicted, episode):
    results, energies = [], []
    for group in range(8):
        ix = np.flatnonzero(episode == group)
        err, base = contrast_loss(actual[ix], predicted[ix])
        energy = float(base.sum())
        results.append(float(err.sum() / energy) if energy > 1e-12 else None)
        energies.append(energy)
    valid = [r for r in results if r is not None]
    return {"per_episode": results,
            "mean": float(np.mean(valid)) if valid else None,
            "energy": energies}


def chosen_effects(model, states):
    n = len(states)
    flat = states.reshape(n * 4, 2, 336)
    all_actions = model.predict(features(flat)).reshape(n * 4, 4, 2, 336)
    selected = all_actions[np.arange(n * 4), np.tile(np.arange(4), n)]
    return selected.reshape(n, 4, 2, 336)


def evaluate(game):
    train_path, train_initial, train_future, _, _ = load(game, 97)
    test_path, test_initial, test_future, groups, terminal = load(game, 103)
    xtrain = features(train_initial)
    root = np.repeat(test_initial[:, None], 4, axis=1)
    local_change = train_future[:, :, 0] - train_initial[:, None]
    transition = KernelRidge().fit(xtrain, local_change.reshape(64, -1))
    roll = root.copy()
    rollouts = {}
    for step in range(1, 17):
        roll = np.clip(roll + chosen_effects(transition, roll), 0, 1)
        if step * 8 in TARGET_STEPS:
            rollouts[step * 8] = roll[..., 1, :].copy()
    curves = {}
    for steps in TARGET_STEPS:
        index = steps // 8 - 1
        actual = test_future[:, :, index, 1]
        delta = train_future[:, :, index] - train_initial[:, None]
        direct = KernelRidge().fit(xtrain, delta.reshape(64, -1))
        direct_pair = test_initial[:, None] + direct.predict(
            features(test_initial)).reshape(64, 4, 2, 336)
        direct_last = np.clip(direct_pair[..., 1, :], 0, 1)
        template = np.clip(test_initial[:, None, 1, :] + delta[..., 1, :].mean(0), 0, 1)
        curves[str(steps)] = {
            "direct": score(actual, direct_last, groups),
            "composed": score(actual, rollouts[steps], groups),
            "fixed_action_template": score(actual, template, groups),
            "terminal_fraction": float(np.mean(terminal[:, :, index])),
        }
    # Atualizar a cada oito passos usando o estado real é um oráculo de
    # diagnóstico, não uma execução autônoma do modelo.
    teacher = []
    for step in range(16):
        previous = root if step == 0 else test_future[:, :, step - 1]
        following = test_future[:, :, step]
        predicted = np.clip(previous + chosen_effects(transition, previous), 0, 1)
        change = following[..., 1, :] - previous[..., 1, :]
        weight = 0.05 + np.abs(change)
        err = np.mean((following[..., 1, :] - predicted[..., 1, :]) ** 2 * weight)
        base = np.mean(change ** 2 * weight)
        teacher.append(float(err / base) if base > 1e-12 else None)
    report = {"status": "exploratorio_semente_103_previamente_inspecionada",
              "game": game, "train_seed": 97, "test_seed": 103,
              "train_bank_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
              "test_bank_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "curves": curves, "teacher_forced_local_error_ratio_by_step": teacher}
    output = HERE / "resultados" / f"{game}_avaliacao_composicao.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    report = evaluate(args.game)
    print(json.dumps({"game": args.game,
                      "curves": {k: {name: value["mean"] for name, value in row.items()
                                     if isinstance(value, dict)}
                                 for k, row in report["curves"].items()},
                      "teacher_forced_local_error_ratio_by_step":
                          report["teacher_forced_local_error_ratio_by_step"]},
                     ensure_ascii=False, indent=2))
