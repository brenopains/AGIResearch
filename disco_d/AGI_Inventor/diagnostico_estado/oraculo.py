"""Diagnóstico privilegiado e separado da entrada do aprendiz."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from avaliar import KernelRidge, group_split
from geral import _sensor_values
from prefixo import decision, predict

HERE = Path(__file__).resolve().parent
TIMES = (20, 28, 36, 44, 52, 60, 68, 76)


def reproduce_oracle(bank: Path, seed=53):
    sys.path.insert(0, str(HERE.parent / "representacao"))
    import mundo_orbitas as world
    world.DUR = 5.0
    rng = np.random.default_rng(seed)
    camera = world.camera(rng)
    all_features, check_values = [], []
    for episode in range(8):
        appearance = "A" if episode < 6 else "B"
        frames, _times, truth = world.episodio(rng, camera, appearance, True)
        positions = truth["pix"]
        for t in TIMES:
            now = positions[:, t]
            velocity = now - positions[:, t - 1]
            ordered = np.argsort(now[:, 0], kind="stable")
            features = np.concatenate((now[ordered] / 128.0,
                                       velocity[ordered] / 4.0), axis=1).ravel()
            all_features.append(features)
            check_values.append(_sensor_values(frames[t + 45]))
    bank_values = np.load(bank)["values"]
    if not np.array_equal(np.asarray(check_values, np.float32), bank_values):
        raise RuntimeError("reprodução do mundo divergiu: oráculo não alinhado ao banco")
    return np.asarray(all_features, np.float32)


def equalize_groups(pixel, privileged, fit):
    def median_nonzero(x):
        d = KernelRidge.distances(x[fit], x[fit])
        positive = d[np.triu_indices(len(fit), 1)]
        positive = positive[positive > 1e-12]
        return float(np.median(positive)) if len(positive) else 1.0
    scale = np.sqrt(median_nonzero(pixel) / max(median_nonzero(privileged), 1e-12))
    return np.concatenate((pixel, privileged * scale), axis=1), float(scale)


def metrics(values, forecast, episode, test):
    result = {"overall": decision(values, forecast, test),
              "value_mse": float(np.mean((values[test] - forecast) ** 2)),
              "per_episode": {}}
    for g in np.unique(episode[test]):
        local_mask = episode[test] == g
        local_indices = test[local_mask]
        local_prediction = forecast[local_mask]
        result["per_episode"][str(int(g))] = decision(values, local_prediction, local_indices)
    return result


def run(bank: Path, nulls=32, seed=53):
    data = np.load(bank)
    if str(data["domain"]) != "orbitas":
        raise ValueError("oráculo físico apenas no mundo de órbitas")
    h = data["histories"].astype(np.float32)
    pixel = np.concatenate((h[:, -1], h[:, -1] - h[:, -2]), axis=1)
    truth = reproduce_oracle(bank, seed=seed)
    values = data["values"]
    episode = data["episode"]
    tr, val, test = group_split(episode)
    fit = np.sort(np.concatenate((tr, val)))
    combined, scale = equalize_groups(pixel, truth, fit)
    score = {}
    for name, features in (("pixels", pixel), ("percepcao_oracular", truth),
                           ("combinado", combined)):
        score[name] = metrics(values, predict(features, values, fit, test), episode, test)
    rng = np.random.default_rng(2701)
    null_regrets = []
    for _ in range(nulls):
        altered = truth.copy()
        for split in (fit, test):
            altered[split] = truth[rng.permutation(split)]
        null = predict(altered, values, fit, test)
        null_regrets.append(metrics(values, null, episode, test)["overall"].get("normalized_regret"))
    observed = score["percepcao_oracular"]["overall"].get("normalized_regret")
    return {"bank": str(bank), "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
            "oracle_dim": truth.shape[1], "combined_scale": scale,
            "reproduced_answers": len(values), "scores": score,
            "shuffled_oracle_regrets": null_regrets,
            "shuffled_equal_or_better": sum(x <= observed for x in null_regrets),
            "null_count": nulls}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()
    result = run(args.bank, seed=args.seed)
    target = args.bank.with_name(args.bank.stem + "_oraculo.json")
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"reproduced_answers": result["reproduced_answers"],
                      "scores": result["scores"],
                      "shuffled_equal_or_better": result["shuffled_equal_or_better"],
                      "null_count": result["null_count"]},
                     indent=2, ensure_ascii=False))
