"""Avalia hipóteses de identidade geradas apenas dos quadros observados."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from avaliar import group_split
from geral import _sensor_values
from hipoteses_estado import describe_beam, filter_times
from oraculo import HERE, TIMES, metrics
from prefixo import predict


def replay(bank, seed):
    sys.path.insert(0, str(HERE.parent / "representacao"))
    import mundo_orbitas as world

    world.DUR = 5.0
    rng = np.random.default_rng(seed)
    camera = world.camera(rng)
    best, mean, uncertainty, check = [], [], [], []
    for episode in range(8):
        frames, _, _hidden = world.episodio(rng, camera, "A" if episode < 6 else "B", True)
        beams = filter_times(frames, TIMES)
        for t in TIMES:
            top, average, spread = describe_beam(beams[t], t)
            best.append(top)
            mean.append(average)
            uncertainty.append(spread)
            check.append(_sensor_values(frames[t + 45]))
    data = np.load(bank)
    if not np.array_equal(np.asarray(check, np.float32), data["values"]):
        raise RuntimeError("replay não coincide com o banco; não avaliar")
    return np.asarray(best), np.asarray(mean), np.asarray(uncertainty)


def run(bank, seed, nulls=32):
    data = np.load(bank)
    best, mean, uncertainty = replay(bank, seed)
    h, values, episode = data["histories"], data["values"], data["episode"]
    pixels = np.concatenate((h[:, -1], h[:, -1] - h[:, -2]), axis=1)
    tr, val, test = group_split(episode)
    fit = np.sort(np.concatenate((tr, val)))
    scores = {name: metrics(values, predict(x, values, fit, test), episode, test)
              for name, x in (("pixels", pixels), ("melhor_hipotese", best),
                              ("media_hipoteses", mean))}
    rng = np.random.default_rng(3051)
    null_scores = {"melhor_hipotese": [], "media_hipoteses": []}
    for _ in range(nulls):
        permuted = {}
        for name, source in (("melhor_hipotese", best), ("media_hipoteses", mean)):
            altered = source.copy()
            for split in (fit, test):
                altered[split] = source[rng.permutation(split)]
            permuted[name] = metrics(values, predict(altered, values, fit, test),
                                     episode, test)["overall"]["normalized_regret"]
        for name, value in permuted.items():
            null_scores[name].append(value)
    return {"seed": seed, "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
            "scores": scores,
            "uncertainty": {"fit_mean": float(uncertainty[fit].mean()),
                            "test_mean": float(uncertainty[test].mean()),
                            "test_by_episode": {str(int(g)): float(uncertainty[test][episode[test] == g].mean())
                                                for g in np.unique(episode[test])}},
            "null_regrets": null_scores,
            "null_equal_or_better": {name: sum(v <= scores[name]["overall"]["normalized_regret"]
                                               for v in values_null)
                                     for name, values_null in null_scores.items()},
            "null_count": nulls}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()
    output = run(args.bank, args.seed)
    target = args.bank.with_name(args.bank.stem + "_hipoteses.json")
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "null_regrets"},
                     ensure_ascii=False, indent=2))
