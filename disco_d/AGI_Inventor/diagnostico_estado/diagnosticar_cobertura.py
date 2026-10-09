"""Diagnóstico exploratório de cobertura de treino versus representação temporal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from auditar_constante import normalize_regret
from avaliar import KernelRidge, group_split, views
from transferencia_ale import regret

HERE = Path(__file__).resolve().parent
NAMES = ("r0_21x16_2", "hist_21x16_16", "res_42x32_2",
         "hist_42x32_16", "native_160x210_2")


def score(y, actions, groups, indices):
    rr = [regret(y[i], int(actions[j])) for j, i in enumerate(indices)]
    per_episode = []
    for episode in sorted(set(map(int, groups[indices]))):
        used = [r for r, i in zip(rr, indices) if groups[i] == episode and r is not None]
        per_episode.append(float(np.mean(used)) if used else None)
    good = [r for r in per_episode if r is not None]
    return {"mean": float(np.mean(good)) if good else None,
            "per_episode": per_episode,
            "informative_roots": sum(r is not None for r in rr),
            "choices": {str(a): int(np.sum(actions == a)) for a in range(4)}}


def run(game):
    path = HERE / "resultados" / f"{game}_trajetoria_independente.npz"
    bank = np.load(path)
    histories, y, group = bank["histories"], bank["values"], bank["episode"]
    tr, val, test = group_split(group)
    fit = np.sort(np.concatenate((tr, val)))
    constant = min(range(4), key=lambda a: (normalize_regret(y, a, fit), a))
    representation = views(histories)
    scores = {}
    for name in NAMES:
        pred = KernelRidge().fit(representation[name][fit], y[fit]).predict(
            representation[name][test])
        scores[name] = score(y, pred.argmax(1), group, test)
    scores["constant_fit"] = score(y, np.full(len(test), constant), group, test)
    # Rótulos dos episódios de construção são destruídos; o teste permanece intacto.
    rng = np.random.default_rng(9703)
    null = []
    for _ in range(32):
        corrupted = y.copy()
        for episode in range(6):
            ix = fit[group[fit] == episode]
            corrupted[ix] = y[rng.permutation(ix)]
        pred = KernelRidge().fit(representation["r0_21x16_2"][fit],
                                 corrupted[fit]).predict(representation["r0_21x16_2"][test])
        null.append(score(y, pred.argmax(1), group, test)["mean"])
    actual = scores["r0_21x16_2"]["mean"]
    report = {"status": "exploratorio_apos_ver_banco",
              "game": game, "seed": 97,
              "bank_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "fit_episodes": list(range(6)), "test_episodes": [6, 7],
              "constant_fit": constant, "scores": scores,
              "null_mean_regrets": null,
              "null_equal_or_better": int(sum(v <= actual for v in null))}
    output = HERE / "resultados" / f"{game}_diagnostico_cobertura.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    result = run(args.game)
    print(json.dumps({"game": args.game, "scores": result["scores"],
                      "null_equal_or_better": result["null_equal_or_better"]},
                     ensure_ascii=False, indent=2))
