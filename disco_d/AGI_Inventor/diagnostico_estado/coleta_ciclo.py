"""Coleta pareada para buscar conjuntamente perguntas e estado histórico."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from coleta import gray, make_ale
from perguntas import PROGRAMS, run_program

HERE = Path(__file__).resolve().parent


def collect(game: str, path: Path, seed: int = 29, episodes: int = 8,
            roots: int = 8, max_steps: int = 160000):
    ale = make_ale(game, seed)
    actions = list(ale.getMinimalActionSet()[:4])
    rng = np.random.default_rng(seed)
    budget = [max_steps]
    histories, returns, images, groups = [], [], [], []
    replay_checks = 0
    for episode in range(episodes):
        ale.reset_game()
        frame = gray(ale)
        history = deque([frame.copy() for _ in range(16)], maxlen=16)
        step = 0
        done = 0
        while done < roots:
            if budget[0] < len(PROGRAMS) * 128 + 128 + 33:
                raise RuntimeError("orçamento insuficiente para terminar uma situação pareada")
            if ale.game_over():
                ale.reset_game()
                frame = gray(ale)
                history = deque([frame.copy() for _ in range(16)], maxlen=16)
            ale.act(actions[int(rng.integers(4))])
            budget[0] -= 1
            frame = gray(ale)
            history.append(frame.copy())
            step += 1
            if step < 32 or step % 24:
                continue
            state = ale.cloneSystemState()
            r, im = [], []
            for program in PROGRAMS:
                ale.restoreSystemState(state)
                reward, outcome = run_program(ale, actions, program, budget)
                r.append(reward)
                im.append(outcome)
            if done == 0:
                ale.restoreSystemState(state)
                repeated_r, repeated_im = run_program(ale, actions, PROGRAMS[0], budget)
                if repeated_r != r[0] or not np.array_equal(repeated_im, im[0]):
                    raise RuntimeError("restauração não reproduziu a consequência")
                replay_checks += 1
            ale.restoreSystemState(state)
            histories.append(np.stack(history).astype(np.uint8))
            returns.append(r)
            images.append(im)
            groups.append(episode)
            done += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, histories=np.stack(histories),
                        returns=np.asarray(returns, np.float32),
                        images=np.asarray(images, np.float32),
                        episode=np.asarray(groups), programs=np.asarray(PROGRAMS),
                        actions=np.asarray([a.value for a in actions]))
    meta = {"game": game, "seed": seed, "roots": len(groups),
            "steps_used": max_steps - budget[0], "max_steps": max_steps,
            "replay_checks": replay_checks,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    target = args.out or HERE / "resultados" / f"{args.game}_ciclo.npz"
    print(json.dumps(collect(args.game, target), indent=2))
