"""Converte o replay certificado ao contrato comum sem mudar rótulos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def convert(game):
    source = HERE / "resultados" / f"{game}_estados_ativos.npz"
    metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["bank_sha256"]:
        raise RuntimeError("hash do replay não confere")
    if metadata["verified_events"] != 64:
        raise RuntimeError("replay incompleto")
    data = np.load(source)
    raw = data["histories"].astype(np.float32) / 255.0
    n, t = raw.shape[:2]
    history = raw.reshape(n, t, 21, 10, 16, 10).mean((3, 5)).reshape(n, t, -1)
    values = data["values"].astype(np.float32)
    if values.shape != (n, 4) or history.shape[1:] != (16, 336):
        raise RuntimeError("contrato inválido")
    output = HERE / "resultados" / f"{game}_ativo_geral.npz"
    np.savez_compressed(output, histories=history, values=values,
                        episode=data["episode"], domain=np.array(game))
    return {"game": game, "source_sha256": metadata["bank_sha256"],
            "bank": str(output), "bank_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "roots": n}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    print(json.dumps(convert(args.game), ensure_ascii=False, indent=2))
