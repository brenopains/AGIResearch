"""Prevê contrastes visuais entre quatro intervenções curtas."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge, views
from variaveis_movimento import batch

HERE = Path(__file__).resolve().parent
PAIRS = list(itertools.combinations(range(4), 2))


def downsample(frames, side):
    n, actions, height, width = frames.shape
    if (height, width) != (210, 160):
        raise ValueError("dimensão ALE inesperada")
    scale = 210 // side
    return frames.astype(np.float32).reshape(n, actions, side, scale,
                                              width // scale, scale).mean((3, 5)) / 255


def contrast_loss(actual, predicted):
    aa = np.stack([actual[:, a] - actual[:, b] for a, b in PAIRS], axis=1)
    pp = np.stack([predicted[:, a] - predicted[:, b] for a, b in PAIRS], axis=1)
    # Máscara extraída apenas dos desfechos para a métrica, jamais do aprendiz.
    weight = 0.05 + np.max(np.abs(aa), axis=1, keepdims=True)
    error = np.mean((aa - pp) ** 2 * weight, axis=(1, 2))
    baseline = np.mean(aa ** 2 * weight, axis=(1, 2))
    return error, baseline


def evaluate(game, side=21):
    path = HERE / "resultados" / f"{game}_efeitos_locais_97.npz"
    data = np.load(path)
    history, ends, group = data["histories"], data["outcomes"], data["episode"]
    now = downsample(history[:, -1, None], side)[:, 0]
    target = downsample(ends, side) - now[:, None]
    target = target.reshape(len(history), 4, -1)
    features = {"pixels": views(history)["r0_21x16_2"],
                "trilhas": batch(history)}
    fold_models = {name: [] for name in features}
    per_episode = {name: [] for name in (*features, "molde_por_acao")}
    baseline_by_episode = []
    for episode in range(8):
        fit = np.flatnonzero(group != episode)
        test = np.flatnonzero(group == episode)
        template = np.broadcast_to(target[fit].mean(axis=0), target[test].shape)
        template_error, template_base = contrast_loss(target[test], template)
        per_episode["molde_por_acao"].append(
            float(template_error.sum() / template_base.sum())
            if template_base.sum() > 1e-12 else None)
        for name, x in features.items():
            model = KernelRidge().fit(x[fit], target[fit].reshape(len(fit), -1))
            predicted = model.predict(x[test]).reshape(len(test), 4, -1)
            error, base = contrast_loss(target[test], predicted)
            ratio = float(error.sum() / base.sum()) if base.sum() > 1e-12 else None
            per_episode[name].append(ratio)
            fold_models[name].append((fit, test, model))
        _, base = contrast_loss(target[test], np.zeros_like(target[test]))
        baseline_by_episode.append(float(base.sum()))
    scores = {}
    for name, rows in per_episode.items():
        valid = [v for v in rows if v is not None]
        scores[name] = {"ratio_to_action_blind": float(np.mean(valid)) if valid else None,
                        "per_episode": rows,
                        "episodes_better_than_blind": int(sum(v is not None and v < 1 for v in rows))}
    rng = np.random.default_rng(9709)
    nulls = []
    x = features["trilhas"]
    for _ in range(32):
        null_ratios = []
        for fit, test, _ in fold_models["trilhas"]:
            randomized = target[fit].copy()
            for i in range(len(fit)):
                randomized[i] = randomized[i, rng.permutation(4)]
            model = KernelRidge().fit(x[fit], randomized.reshape(len(fit), -1))
            predicted = model.predict(x[test]).reshape(len(test), 4, -1)
            error, base = contrast_loss(target[test], predicted)
            if base.sum() > 1e-12:
                null_ratios.append(float(error.sum() / base.sum()))
        nulls.append(float(np.mean(null_ratios)))
    observed = scores["trilhas"]["ratio_to_action_blind"]
    shuffled_history = {}
    for name, entries in fold_models.items():
        shuffled_history[name] = []
        for _ in range(32):
            ratios = []
            for _, test, model in entries:
                permuted = rng.permutation(test)
                predicted = model.predict(features[name][permuted]).reshape(len(test), 4, -1)
                error, base = contrast_loss(target[test], predicted)
                if base.sum() > 1e-12:
                    ratios.append(float(error.sum() / base.sum()))
            shuffled_history[name].append(float(np.mean(ratios)))
    report = {"status": "desenvolvimento_semente_97",
              "game": game, "side": side, "horizon": 8,
              "bank_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                for name in ("avaliar_efeitos_locais.py", "variaveis_movimento.py")},
              "baseline_contrast_energy_per_episode": baseline_by_episode,
              "scores": scores, "shuffled_action_ratios": nulls,
              "shuffled_action_equal_or_better": int(sum(v <= observed for v in nulls)),
              "shuffled_history_ratios": shuffled_history,
              "shuffled_history_equal_or_better": {
                  name: int(sum(v <= scores[name]["ratio_to_action_blind"] for v in rows))
                  for name, rows in shuffled_history.items()}}
    output = HERE / "resultados" / f"{game}_avaliacao_efeitos_{side}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("--side", type=int, choices=(21, 42), default=21)
    args = parser.parse_args()
    result = evaluate(args.game, args.side)
    print(json.dumps({"game": args.game, "side": args.side,
                      "scores": result["scores"],
                      "contrast_energy": result["baseline_contrast_energy_per_episode"],
                      "shuffled_action_equal_or_better": result["shuffled_action_equal_or_better"],
                      "shuffled_history_equal_or_better": result["shuffled_history_equal_or_better"]},
                     ensure_ascii=False, indent=2))
