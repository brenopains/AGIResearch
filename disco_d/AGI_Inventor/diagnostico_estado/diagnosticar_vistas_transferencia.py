"""Diagnóstico pós-hoc: janela temporal versus resolução no desvio da semente 97."""

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


def run(game):
    train_path = HERE / "resultados" / f"{game}_estados_ativos.npz"
    test_path = HERE / "resultados" / f"{game}_trajetoria_independente.npz"
    train = np.load(train_path)
    test = np.load(test_path)
    tr, val, _ = group_split(train["episode"])
    fit = np.sort(np.concatenate((tr, val)))
    xtrain = views(train["histories"])
    xtest = views(test["histories"])
    ytrain, ytest = train["values"], test["values"]
    constant = min(range(4), key=lambda action: (normalize_regret(ytrain, action, fit), action))
    episode = test["episode"]
    results = {}
    for name in NAMES:
        pred = KernelRidge().fit(xtrain[name][fit], ytrain[fit]).predict(xtest[name])
        choice = np.argmax(pred, axis=1)
        root_regret = [regret(ytest[i], int(choice[i])) for i in range(len(choice))]
        informative = [i for i, r in enumerate(root_regret) if r is not None]
        per_episode = []
        for group in range(8):
            rr = [root_regret[i] for i in informative if episode[i] == group]
            per_episode.append(float(np.mean(rr)) if rr else None)
        used = [r for r in per_episode if r is not None]
        results[name] = {
            "mean_per_informative_episode": float(np.mean(used)) if used else None,
            "per_episode": per_episode,
            "mean_per_informative_root": float(np.mean([root_regret[i]
                                                         for i in informative])) if informative else None,
            "choices": {str(a): int(np.sum(choice == a)) for a in range(4)},
            "informative_roots": len(informative),
            "optimal_on_challenge": int(sum(root_regret[i] == 0 for i in informative
                                           if regret(ytest[i], constant) > 0)),
        }
    report = {"status": "exploratorio_pos_hoc",
              "game": game,
              "constant_action": constant,
              "train_seed": 59, "test_seed": 97,
              "train_bank_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
              "test_bank_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "results": results}
    output = HERE / "resultados" / f"{game}_vistas_transferencia.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    result = run(args.game)
    print(json.dumps(result, ensure_ascii=False, indent=2))
