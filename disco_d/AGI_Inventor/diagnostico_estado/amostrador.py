"""Seleciona quais estados sondar antes de conhecer seus 16 retornos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge, group_split, views


def run(bank: Path):
    data = np.load(bank)
    episode = data["episode"]
    tr, val, test = group_split(episode)
    state = views(data["histories"])["r0_21x16_2"]
    value = data["returns"].reshape(-1, 4, 4).max(axis=2)
    sorted_value = np.sort(value, axis=1)
    label = (sorted_value[:, -1] - sorted_value[:, -2] > 1e-8).astype(np.float64)
    score_val = KernelRidge().fit(state[tr], label[tr, None]).predict(state[val]).ravel()
    fit = np.sort(np.concatenate([tr, val]))
    score_test = KernelRidge().fit(state[fit], label[fit, None]).predict(state[test]).ravel()
    selected = test[np.argsort(-score_test, kind="stable")[:4]]
    observed = int(label[selected].sum())
    rng = np.random.default_rng(1307)
    random_outcomes = [int(label[rng.choice(test, 4, replace=False)].sum())
                       for _ in range(1000)]
    return {"bank": str(bank), "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
            "train_positive": int(label[tr].sum()),
            "validation_positive": int(label[val].sum()),
            "test_positive_total": int(label[test].sum()),
            "validation_top4_positive": int(label[val[np.argsort(-score_val)[:4]]].sum()),
            "selected_top4_positive": observed,
            "random_mean": float(np.mean(random_outcomes)),
            "random_equal_or_better": int(sum(n >= observed for n in random_outcomes)),
            "random_total": len(random_outcomes),
            "gate_pass": bool(observed >= 3 and
                              sum(n >= observed for n in random_outcomes) < 50)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    args = parser.parse_args()
    report = run(args.bank)
    args.bank.with_name(args.bank.stem + "_amostrador.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
