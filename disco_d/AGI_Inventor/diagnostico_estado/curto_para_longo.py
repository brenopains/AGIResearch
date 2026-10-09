"""Oráculo diagnóstico: efeitos reais de 8 passos informam valor de 128?"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from auditar_constante import normalize_regret
from avaliar import KernelRidge, views
from avaliar_efeitos_locais import downsample
from transferencia_ale import regret

HERE = Path(__file__).resolve().parent


def score(values, actions, indices):
    regrets = [regret(values[i], int(a)) for i, a in zip(indices, actions)]
    valid = [r for r in regrets if r is not None]
    return float(np.mean(valid)) if valid else None


def run(game):
    long_path = HERE / "resultados" / f"{game}_trajetoria_independente.npz"
    short_path = HERE / "resultados" / f"{game}_efeitos_locais_97.npz"
    long = np.load(long_path)
    short = np.load(short_path)
    if not np.array_equal(long["histories"], short["histories"]):
        raise RuntimeError("bancos de horizonte curto/longo não estão alinhados")
    values, groups = long["values"], long["episode"]
    history = long["histories"]
    now = downsample(history[:, -1, None], 21)[:, 0]
    future = downsample(short["outcomes"], 21) - now[:, None]
    features = {"pixels_atuais": views(history)["r0_21x16_2"],
                "efeitos_reais_8_passos": future.reshape(len(future), -1),
                "recompensas_reais_8_passos": short["rewards"].astype(np.float32)}
    scores = {name: [] for name in (*features, "constante")}
    for episode in range(8):
        fit = np.flatnonzero(groups != episode)
        test = np.flatnonzero(groups == episode)
        constant = min(range(4), key=lambda a: (normalize_regret(values, a, fit), a))
        scores["constante"].append(score(values, np.full(len(test), constant), test))
        for name, x in features.items():
            pred = KernelRidge().fit(x[fit], values[fit]).predict(x[test])
            scores[name].append(score(values, pred.argmax(1), test))
    report = {"status": "oraculo_exploratorio_na_semente_97", "game": game,
              "long_bank_sha256": hashlib.sha256(long_path.read_bytes()).hexdigest(),
              "short_bank_sha256": hashlib.sha256(short_path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scores": {name: {"per_episode": row,
                                 "mean": float(np.mean([v for v in row if v is not None]))}
                         for name, row in scores.items()}}
    output = HERE / "resultados" / f"{game}_curto_para_longo.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    print(json.dumps(run(args.game), ensure_ascii=False, indent=2))
