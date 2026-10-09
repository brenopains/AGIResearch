"""Mesmo compilador de contraexemplos para observações e valores de qualquer domínio."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from avaliar import group_split
from contraexemplos import assess, synthesize

HERE = Path(__file__).resolve().parent


def save_atari_case(game: str, output: Path):
    source = HERE / "resultados" / f"{game}_ciclo.npz"
    data = np.load(source)
    raw = data["histories"].astype(np.float32) / 255.0
    n, t = raw.shape[:2]
    history = raw.reshape(n, t, 21, 10, 16, 10).mean((3, 5)).reshape(n, t, -1)
    values = data["returns"].reshape(n, 4, 4).max(axis=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, histories=history, values=values,
                        episode=data["episode"], domain=np.array(game))
    return hashlib.sha256(output.read_bytes()).hexdigest()


def _foreground(frame: np.ndarray) -> np.ndarray:
    color = frame.astype(np.int16)
    high = color.max(axis=2)
    low = color.min(axis=2)
    return ((high > 150) & ((high - low) > 25)).astype(np.float32)


def _sensor_values(frame: np.ndarray):
    foreground = _foreground(frame)
    yy, xx = np.mgrid[0:128, 0:128]
    centers = ((32, 32), (32, 96), (96, 32), (96, 96))
    return np.array([np.sum(foreground * np.exp(
        -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 40**2)))
        for cy, cx in centers], dtype=np.float32)


def save_orbit_case(output: Path, seed: int = 53):
    sys.path.insert(0, str(HERE.parent / "representacao"))
    import mundo_orbitas as world

    # A física é a original. Só reduzimos a duração da renderização não observada.
    world.DUR = 5.0
    rng = np.random.default_rng(seed)
    cam = world.camera(rng)
    histories, values, groups = [], [], []
    time_points = (20, 28, 36, 44, 52, 60, 68, 76)
    for episode in range(8):
        appearance = "A" if episode < 6 else "B"
        frames, _time, _hidden_truth = world.episodio(rng, cam, appearance, True)
        foreground = np.stack([_foreground(frame) for frame in frames])
        coarse = foreground.reshape(len(frames), 16, 8, 16, 8).mean((2, 4))
        for t in time_points:
            histories.append(coarse[t - 15:t + 1].reshape(16, -1))
            values.append(_sensor_values(frames[t + 45]))
            groups.append(episode)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, histories=np.asarray(histories, np.float32),
                        values=np.asarray(values, np.float32),
                        episode=np.asarray(groups), domain=np.array("orbitas"))
    metadata = {"seed": seed, "episodes": 8, "windows_per_episode": 8,
                "future_offset_frames": 45,
                "learning_inputs": "16 binary foreground frames, 16x16; values for four sensor directions",
                "test_appearance": "B", "source_world": str(HERE.parent / "representacao" / "mundo_orbitas.py"),
                "bank_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def evaluate(bank: Path, nulls: int = 32):
    data = np.load(bank)
    history = np.asarray(data["histories"], np.float32)
    values = np.asarray(data["values"], np.float32)
    episode = data["episode"]
    if history.ndim != 3 or history.shape[1] != 16 or values.shape != (len(history), 4):
        raise ValueError("contrato requer N×16×D de observações e N×4 de valores")
    tr, val, test = group_split(episode)
    base = np.concatenate((history[:, -1], history[:, -1] - history[:, -2]), axis=1)
    chosen = synthesize(base, history, values, episode, tr, val, "history")
    actual = assess(base, history, values, chosen, tr, val, test)
    present = synthesize(base, history, values, episode, tr, val, "current")
    present_outcome = assess(base, history, values, present, tr, val, test)
    rng = np.random.default_rng(1061)
    placebo = []
    for _ in range(nulls):
        shuffled = history.copy()
        for split in (tr, val, test):
            shuffled[split] = history[rng.permutation(split)]
        null_choice = synthesize(base, shuffled, values, episode, tr, val, "history")
        null_result = assess(base, shuffled, values, null_choice, tr, val, test)
        placebo.append(null_result.get("regret_gain"))
    observed = actual.get("regret_gain")
    at_least = (sum(p >= observed for p in placebo if p is not None)
                if observed is not None else None)
    return {"bank": str(bank), "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
            "domain": str(data["domain"]), "input_shape": list(history.shape),
            "actual": actual, "present_only": present_outcome,
            "null_regret_gains": placebo, "null_at_least_as_good": at_least,
            "null_count": nulls,
            "gate_local": bool(observed is not None and observed > 0
                               and chosen.get("validation_gain", 0) > 0
                               and chosen.get("train_counterexamples", 0) >= 4
                               and at_least is not None and at_least <= 1
                               and observed > present_outcome.get("regret_gain", 0))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("make-atari", "make-orbit", "evaluate"))
    parser.add_argument("argument", type=str)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()
    if args.action == "make-atari":
        if args.argument not in ("breakout", "pong"):
            raise ValueError("apenas jogos de desenvolvimento")
        path = args.out or HERE / "resultados" / f"{args.argument}_geral.npz"
        print(json.dumps({"bank": str(path), "sha256": save_atari_case(args.argument, path)}))
    elif args.action == "make-orbit":
        path = args.out or HERE / "resultados" / "orbitas_geral.npz"
        print(json.dumps(save_orbit_case(path, seed=args.seed), indent=2))
    else:
        path = Path(args.argument)
        result = evaluate(path)
        path.with_name(path.stem + "_diagnostico.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        def compact(item):
            return {k: v for k, v in item.items() if k != "definition"}
        print(json.dumps({"domain": result["domain"], "actual": compact(result["actual"]),
                          "present_only": compact(result["present_only"]),
                          "null_at_least_as_good": result["null_at_least_as_good"],
                          "gate_local": result["gate_local"]},
                         indent=2, ensure_ascii=False))
