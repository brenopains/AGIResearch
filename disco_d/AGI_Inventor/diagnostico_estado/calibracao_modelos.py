"""Audita se o comitê sustenta testemunhas contrafactuais de ação."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import group_split, views
from desacordo_ativo import committee
from prefixo import decision

HERE = Path(__file__).resolve().parent


def run(game, nulls=32):
    bank = HERE / "resultados" / f"{game}_ciclo.npz"
    data = np.load(bank)
    episode = data["episode"]
    y = data["returns"].reshape(-1, 4, 4).max(axis=2)
    x = views(data["histories"])
    members, source_hash = committee(game)
    tr, val, test = group_split(episode)
    fit = np.sort(np.concatenate((tr, val)))
    # Calibração fora do próprio episódio de cada exemplo 0–5.
    oof = []
    for i in fit:
        own = int(episode[i])
        rows = [members[own], members[6 + own]]
        oof.append(np.mean([model.predict(x[name][i:i + 1])[0]
                            for name, model in rows], axis=0))
    oof = np.asarray(oof)
    radius = np.quantile(np.abs(y[fit] - oof), 0.9, axis=0)

    predictions = np.stack([model.predict(x[name][test]) for name, model in members])
    center = predictions.mean(axis=0)
    lower, upper = center - radius, center + radius
    envelope_lower, envelope_upper = predictions.min(axis=0), predictions.max(axis=0)
    truth = y[test]
    actual_spread = np.ptp(truth, axis=1)
    actual_unique = np.sort(truth, axis=1)[:, -1] > np.sort(truth, axis=1)[:, -2] + 1e-8
    votes = predictions.argmax(axis=2)
    nonunanimous = np.asarray([len(np.unique(votes[:, i])) > 1 for i in range(len(test))])
    contained = (truth >= lower) & (truth <= upper)
    envelope_contained = (truth >= envelope_lower) & (truth <= envelope_upper)
    possible = upper >= lower.max(axis=1, keepdims=True)
    actual_optimal = truth == truth.max(axis=1, keepdims=True)
    possible_covers = np.any(possible & actual_optimal, axis=1)
    score = decision(y, center, test)
    rng = np.random.default_rng(4121)
    null_regret = []
    for _ in range(nulls):
        shuffled = center[rng.permutation(len(test))]
        null_regret.append(decision(y, shuffled, test)["normalized_regret"])
    return {"game": game, "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
            "committee_bank_sha256": source_hash,
            "fit_episodes": sorted(set(map(int, episode[fit]))),
            "test_episodes": sorted(set(map(int, episode[test]))),
            "calibration_abs_error_q90": radius.tolist(),
            "test_roots": len(test),
            "nonunanimous_roots": int(nonunanimous.sum()),
            "nonunanimous_without_true_contrast": int(np.sum(nonunanimous & (actual_spread <= 1e-8))),
            "true_contrast_roots": int(np.sum(actual_spread > 1e-8)),
            "unique_best_roots": int(actual_unique.sum()),
            "all_four_covered_by_calibrated_intervals": int(np.all(contained, axis=1).sum()),
            "all_four_covered_by_committee_envelope": int(np.all(envelope_contained, axis=1).sum()),
            "mean_interval_width": float(np.mean(upper - lower)),
            "mean_true_action_span": float(np.mean(actual_spread)),
            "possible_optimal_set_mean_size": float(np.mean(possible.sum(axis=1))),
            "possible_optimal_set_covers_truth": int(possible_covers.sum()),
            "decision": score,
            "null_regrets": null_regret,
            "null_equal_or_better": int(sum(r <= score["normalized_regret"] for r in null_regret)),
            "null_count": nulls,
            "per_root": [{"episode": int(episode[i]), "actual": truth[j].tolist(),
                          "predicted": center[j].tolist(),
                          "votes": np.bincount(votes[:, j], minlength=4).tolist(),
                          "interval_covers_all": bool(np.all(contained[j])),
                          "possible_actions": np.flatnonzero(possible[j]).tolist()}
                         for j, i in enumerate(test)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    result = run(args.game)
    target = HERE / "resultados" / f"{args.game}_calibracao_modelos.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_root", "null_regrets")},
                     ensure_ascii=False, indent=2))
