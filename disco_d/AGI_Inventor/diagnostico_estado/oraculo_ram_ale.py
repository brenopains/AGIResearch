"""Reconstitui RAM ALE para diagnosticar percepção versus previsão de efeitos.

RAM nunca entra no agente visual; cada checkpoint é certificado contra a
história já salva antes de ser usado como oráculo de avaliação.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from auditar_constante import normalize_regret
from avaliar import KernelRidge
from coleta import gray, make_ale
from transferencia_ale import regret

HERE = Path(__file__).resolve().parent
SEED = 97


def rebuild(game, histories):
    ale = make_ale(game, SEED)
    actions = list(ale.getMinimalActionSet()[:4])
    rng = np.random.default_rng(SEED)
    ram = []
    resets = 0
    for episode in range(8):
        ale.reset_game()
        current = gray(ale)
        history = deque([current.copy() for _ in range(16)], maxlen=16)
        for checkpoint in range(8):
            for _ in range(128):
                ale.act(actions[int(rng.integers(0, 4))])
                if ale.game_over():
                    ale.reset_game()
                    current = gray(ale)
                    history = deque([current.copy() for _ in range(16)], maxlen=16)
                    resets += 1
                else:
                    history.append(gray(ale))
            index = episode * 8 + checkpoint
            if not np.array_equal(np.stack(history), histories[index]):
                raise RuntimeError(f"replay divergiu no estado {index}")
            ram.append(np.asarray(ale.getRAM(), np.uint8).copy())
    return np.stack(ram), resets


def metric(values, indices, actions):
    rr = [regret(values[index], int(action)) for index, action in zip(indices, actions)]
    good = [r for r in rr if r is not None]
    return float(np.mean(good)) if good else None


def evaluate(game):
    bank_path = HERE / "resultados" / f"{game}_trajetoria_independente.npz"
    bank = np.load(bank_path)
    histories, values, groups = bank["histories"], bank["values"], bank["episode"]
    ram, resets = rebuild(game, histories)
    scores = {"ram": [], "constant": []}
    for episode in range(8):
        fit = np.flatnonzero(groups != episode)
        test = np.flatnonzero(groups == episode)
        constant = min(range(4), key=lambda a: (normalize_regret(values, a, fit), a))
        predictions = KernelRidge().fit(ram[fit].astype(np.float32) / 255,
                                        values[fit]).predict(ram[test].astype(np.float32) / 255)
        scores["ram"].append(metric(values, test, predictions.argmax(1)))
        scores["constant"].append(metric(values, test, np.full(len(test), constant)))
    output = {"game": game, "seed": SEED, "status": "oraculo_diagnostico_interno",
              "bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "checkpoints_certified": len(ram), "resets": resets,
              "ram_size": ram.shape[1],
              "scores": {name: {"per_episode": row,
                                 "mean": float(np.mean([v for v in row if v is not None]))}
                         for name, row in scores.items()}}
    result = HERE / "resultados" / f"{game}_oraculo_ram_ale.json"
    result.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.game), ensure_ascii=False, indent=2))
