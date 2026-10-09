"""Busca contraexemplos de suficiência da assinatura visual de oito passos."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from avaliar_composicao import load

HERE = Path(__file__).resolve().parent
PAIRS = list(itertools.combinations(range(4), 2))


def signatures(future, index):
    end = future[:, :, index, 1]
    return np.stack([end[:, a] - end[:, b] for a, b in PAIRS], axis=1).reshape(len(end), -1)


def distances(x):
    matrix = x.astype(np.float64)
    norms = (matrix * matrix).sum(1)
    sq = np.maximum(norms[:, None] + norms[None, :] - 2 * matrix @ matrix.T, 0)
    return np.sqrt(sq)


def examine(game):
    banks = [load(game, seed) for seed in (97, 103)]
    paths = [bank[0] for bank in banks]
    initial = np.concatenate([bank[1] for bank in banks])
    future = np.concatenate([bank[2] for bank in banks])
    episode = np.concatenate([bank[3] for bank in banks])
    short = signatures(future, 0)
    long = signatures(future, 15)
    sd, ld = distances(short), distances(long)
    pairs = [(i, j) for i in range(len(short)) for j in range(i + 1, len(short))]
    short_values = np.asarray([sd[i, j] for i, j in pairs])
    long_values = np.asarray([ld[i, j] for i, j in pairs])
    short_median = float(np.median(short_values))
    long_median = float(np.median(long_values))
    exact = []
    for k, (i, j) in enumerate(pairs):
        if np.array_equal(short[i], short[j]) and not np.array_equal(long[i], long[j]):
            exact.append((i, j, float(long_values[k])))
    exact.sort(key=lambda row: -row[2])
    short_full = (future[:, :, 0, 1] - initial[:, None, 1]).reshape(len(future), -1)
    initial_pair = initial.reshape(len(future), -1)
    representations = {
        "contrast_only": short,
        "all_four_short_effects": short_full,
        "initial_two_frames": initial_pair,
        "initial_frames_and_four_effects": np.concatenate((initial_pair, short_full), axis=1),
    }
    collision_counts = {}
    collision_examples = {}
    for name, representation in representations.items():
        counterexamples = [(i, j) for i, j in pairs
                           if np.array_equal(representation[i], representation[j])
                           and not np.array_equal(long[i], long[j])]
        collision_counts[name] = len(counterexamples)
        collision_examples[name] = [{"a": i, "b": j,
                                     "long_distance": float(ld[i, j])}
                                    for i, j in counterexamples[:3]]
    # Near pairs are descriptive only; no proof of identical state.
    normalized_short = short_values / max(short_median, 1e-12)
    normalized_long = long_values / max(long_median, 1e-12)
    eligible = np.flatnonzero(normalized_short <= 0.1)
    close = sorted(eligible, key=lambda k: -normalized_long[k])[:10]
    def describe(index):
        i, j = pairs[index]
        return {"a": {"seed": 97 if i < 64 else 103,
                      "root": i % 64, "episode": int(episode[i])},
                "b": {"seed": 97 if j < 64 else 103,
                      "root": j % 64, "episode": int(episode[j])},
                "short_distance_over_median": float(normalized_short[index]),
                "long_distance_over_median": float(normalized_long[index])}
    report = {"status": "diagnostico_de_aliasing_na_representacao_8_passos",
              "game": game, "states": len(short), "state_pairs": len(pairs),
              "source_bank_sha256": [hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in paths],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "median_short_distance": short_median,
              "median_long_distance": long_median,
              "exact_collision_long_divergence_by_representation": collision_counts,
              "exact_collision_examples_by_representation": collision_examples,
              "contrast_only_zero_signature_counterexamples": int(sum(
                  not np.any(short[i]) for i, _, _ in exact)),
              "exact_short_same_long_different_count": len(exact),
              "top_exact_counterexamples": [
                  {"a": row[0], "b": row[1], "long_distance": row[2]}
                  for row in exact[:10]],
              "near_short_pairs_under_tenth_median": len(eligible),
              "top_near_pairs": [describe(k) for k in close]}
    output = HERE / "resultados" / f"{game}_aliasing_assinatura.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    report = examine(args.game)
    print(json.dumps(report, ensure_ascii=False, indent=2))
