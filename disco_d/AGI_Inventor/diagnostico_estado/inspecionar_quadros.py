"""Folha de contato para inspecionar histórias visuais de estados contrastivos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent


def run(game):
    bank = np.load(HERE / "resultados" / f"{game}_trajetoria_independente.npz")
    histories, values = bank["histories"], bank["values"]
    constant = 1 if game == "breakout" else 2
    challenging = np.flatnonzero(values.max(1) > values[:, constant])[:4]
    canvas = Image.new("RGB", (16 * 160, len(challenging) * (210 + 28)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, index in enumerate(challenging):
        draw.text((4, row * 238 + 3),
                  f"state {index}; values {values[index].tolist()}", fill="black")
        for t, frame in enumerate(histories[index]):
            canvas.paste(Image.fromarray(frame, mode="L").convert("RGB"),
                         (t * 160, row * 238 + 28))
    output = HERE / "resultados" / f"{game}_contato_temporal.png"
    canvas.save(output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    run(args.game)
