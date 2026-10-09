"""Verifica se a melhora no banco ativo exige ler o estado visual."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from aquisicao import frozen_model
from avaliar import KernelRidge, group_split
from prefixo import decision

HERE = Path(__file__).resolve().parent


def normalize_regret(values, action, indices):
    selected = values[indices]
    spread = np.ptp(selected, axis=1)
    relevant = spread > 1e-8
    if not relevant.any():
        return None
    return float(np.mean((selected.max(axis=1) - selected[:, action])[relevant]
                         / spread[relevant]))


def run(game, nulls=32):
    bank = HERE / "resultados" / f"{game}_ativo_geral.npz"
    data = np.load(bank)
    history, y, episode = data["histories"], data["values"], data["episode"]
    tr, val, test = group_split(episode)
    fit = np.sort(np.concatenate((tr, val)))
    x = np.concatenate((history[:, -1], history[:, -1] - history[:, -2]), axis=1)
    constant = min(range(4), key=lambda action: (normalize_regret(y, action, fit), action))
    constant_prediction = np.zeros((len(test), 4))
    constant_prediction[:, constant] = 1
    new_model = KernelRidge().fit(x[fit], y[fit])
    old_model, old_hash = frozen_model(game)
    scores = {"constant": decision(y, constant_prediction, test),
              "new_model": decision(y, new_model.predict(x[test]), test),
              "old_model": decision(y, old_model.predict(x[test]), test)}
    rng = np.random.default_rng(6201)
    shuffled = []
    for _ in range(nulls):
        labels = y[fit][rng.permutation(len(fit))]
        prediction = KernelRidge().fit(x[fit], labels).predict(x[test])
        shuffled.append(decision(y, prediction, test)["normalized_regret"])
    observed = scores["new_model"]["normalized_regret"]
    def action_counts(indices):
        yy = y[indices]
        return {"informative": int(np.sum(np.ptp(yy, axis=1) > 1e-8)),
                "unique_best": int(np.sum(np.sort(yy, axis=1)[:, -1]
                                           > np.sort(yy, axis=1)[:, -2] + 1e-8)),
                "argmax_counts": np.bincount(yy.argmax(axis=1), minlength=4).tolist(),
                "mean_spread": float(np.mean(np.ptp(yy, axis=1)))}
    return {"game": game, "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
            "old_bank_sha256": old_hash,
            "constant_selected_on_fit": constant,
            "train": action_counts(fit), "test": action_counts(test),
            "scores": scores, "shuffled_regrets": shuffled,
            "shuffled_equal_or_better": int(sum(value <= observed for value in shuffled)),
            "null_count": nulls}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    result = run(args.game)
    target = HERE / "resultados" / f"{args.game}_ativo_constante.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "shuffled_regrets"},
                     ensure_ascii=False, indent=2))
