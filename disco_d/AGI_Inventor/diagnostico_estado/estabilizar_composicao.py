"""Diagnóstico fatorial: cobertura de transições e memória de estados alcançáveis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from avaliar_composicao import (TARGET_STEPS, chosen_effects, features,
                                load, score)

HERE = Path(__file__).resolve().parent
ROOTS = 64
EXTRA_PER_ACTION = 128
LATENT = 32


def train_trajectory(initial, future, terminal):
    rng = np.random.default_rng(9711)
    models, libraries, counts = [], [], []
    for action in range(4):
        start_source = initial
        start_target = future[:, action, 0]
        other_source = np.stack([future[:, action, step - 1]
                                 for step in range(1, 16)], axis=1).reshape(-1, 2, 336)
        other_target = future[:, action, 1:].reshape(-1, 2, 336)
        eligible = np.flatnonzero((~terminal[:, action, :-1]).reshape(-1))
        selected = rng.choice(eligible, min(EXTRA_PER_ACTION, len(eligible)), replace=False)
        source = np.concatenate((start_source, other_source[selected]))
        target = np.concatenate((start_target, other_target[selected]))
        model = KernelRidge().fit(features(source), (target - source).reshape(len(source), -1))
        models.append(model)
        libraries.append(target)
        counts.append({"root": ROOTS, "intermediate": len(selected),
                       "eligible_intermediate": len(eligible)})
    return models, libraries, counts


def principal_space(libraries):
    all_states = np.concatenate(libraries).reshape(-1, 672)
    center = all_states.mean(0)
    matrix = all_states - center
    rng = np.random.default_rng(9732)
    probe = rng.normal(size=(672, 64))
    q, _ = np.linalg.qr(matrix @ probe)
    for _ in range(2):
        q, _ = np.linalg.qr(matrix @ (matrix.T @ q))
    _, _, vt = np.linalg.svd(q.T @ matrix, full_matrices=False)
    basis = vt[:LATENT].T
    embedded = [(library.reshape(-1, 672) - center) @ basis
                for library in libraries]
    return center, basis, embedded


def project(states, libraries, space):
    center, basis, embedded = space
    output = np.empty_like(states)
    for action in range(4):
        point = (states[:, action].reshape(-1, 672) - center) @ basis
        ref = embedded[action]
        distance = ((point * point).sum(1)[:, None]
                    + (ref * ref).sum(1)[None, :] - 2 * point @ ref.T)
        nearest = np.argmin(distance, axis=1)
        output[:, action] = libraries[action][nearest]
    return output


def project_continuous(states, space):
    center, basis, _ = space
    flat = states.reshape(-1, 672)
    reconstructed = center + ((flat - center) @ basis) @ basis.T
    return np.clip((0.5 * flat + 0.5 * reconstructed).reshape(states.shape), 0, 1)


def trajectory_effect(models, states):
    result = np.empty_like(states)
    for action in range(4):
        result[:, action] = models[action].predict(features(states[:, action])).reshape(-1, 2, 336)
    return result


def local_error(models, root_model, initial, future):
    root = np.repeat(initial[:, None], 4, axis=1)
    by_step = {"root_only": [], "trajectory": []}
    for step in range(16):
        previous = root if step == 0 else future[:, :, step - 1]
        following = future[:, :, step]
        alternatives = {"root_only": chosen_effects(root_model, previous),
                        "trajectory": trajectory_effect(models, previous)}
        change = following[..., 1, :] - previous[..., 1, :]
        weight = 0.05 + np.abs(change)
        baseline = np.mean(change ** 2 * weight)
        for name, delta in alternatives.items():
            prediction = np.clip(previous + delta, 0, 1)[..., 1, :]
            error = np.mean((following[..., 1, :] - prediction) ** 2 * weight)
            by_step[name].append(float(error / baseline) if baseline > 1e-12 else None)
    return by_step


def evaluate(game):
    train_path, train_initial, train_future, _, train_terminal = load(game, 97)
    test_path, test_initial, test_future, groups, test_terminal = load(game, 103)
    root_target = train_future[:, :, 0] - train_initial[:, None]
    root_model = KernelRidge().fit(features(train_initial), root_target.reshape(ROOTS, -1))
    models, libraries, counts = train_trajectory(train_initial, train_future, train_terminal)
    space = principal_space(libraries)
    root = np.repeat(test_initial[:, None], 4, axis=1)
    states = {"raiz_livre": root.copy(), "raiz_memoria": root.copy(),
              "raiz_subespaco": root.copy(),
              "trajetoria_livre": root.copy(), "trajetoria_memoria": root.copy(),
              "trajetoria_subespaco": root.copy()}
    by_horizon = {}
    for step in range(1, 17):
        for name, previous in states.items():
            delta = (chosen_effects(root_model, previous)
                     if name.startswith("raiz_") else trajectory_effect(models, previous))
            next_state = np.clip(previous + delta, 0, 1)
            if name.endswith("memoria"):
                next_state = project(next_state, libraries, space)
            elif name.endswith("subespaco"):
                next_state = project_continuous(next_state, space)
            states[name] = next_state
        horizon = step * 8
        if horizon not in TARGET_STEPS:
            continue
        index = step - 1
        actual = test_future[:, :, index, 1]
        nonterminal = ~test_terminal[:, :, index].any(axis=1)
        by_horizon[str(horizon)] = {
            "all": {name: score(actual, state[..., 1, :], groups)
                    for name, state in states.items()},
            "no_terminal": {name: score(actual[nonterminal], state[nonterminal, ..., 1, :],
                                        groups[nonterminal])
                            for name, state in states.items()},
            "no_terminal_roots": int(nonterminal.sum())}
    report = {"status": "exploratorio_apos_ver_semente_103", "game": game,
              "train_seed": 97, "test_seed": 103,
              "train_bank_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
              "test_bank_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "training_counts": counts, "projection_dimensions": LATENT,
              "horizons": by_horizon,
              "teacher_forced_local_error_ratio":
                  local_error(models, root_model, test_initial, test_future)}
    output = HERE / "resultados" / f"{game}_estabilizar_composicao.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    result = evaluate(args.game)
    print(json.dumps({"game": args.game,
                      "all": {h: {name: value["mean"] for name, value in row["all"].items()}
                              for h, row in result["horizons"].items()},
                      "no_terminal": {h: {"roots": row["no_terminal_roots"],
                                             "scores": {name: value["mean"]
                                                        for name, value in row["no_terminal"].items()}}
                                      for h, row in result["horizons"].items()},
                      "teacher_forced": result["teacher_forced_local_error_ratio"]},
                     ensure_ascii=False, indent=2))
