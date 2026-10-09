"""Busca de perguntas contrafactuais compostas, com ramos ALE pareados."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import deque
from pathlib import Path

import numpy as np

from coleta import gray, make_ale, small
from avaliar import group_split

HERE = Path(__file__).resolve().parent
PROGRAMS = tuple(itertools.product(range(4), repeat=2))
CONSTANTS = (0, 5, 10, 15)


def run_program(ale, actions, program, budget):
    total = 0.0
    for t in range(128):
        if budget[0] <= 0:
            raise RuntimeError("orçamento de passos esgotado")
        total += float(ale.act(actions[program[0] if t < 32 else program[1]]))
        budget[0] -= 1
        if ale.game_over():
            break
    return total, small(gray(ale), 21)


def collect(game, path, seed=19, episodes=8, roots=4, max_steps=80000):
    ale = make_ale(game, seed)
    actions = list(ale.getMinimalActionSet()[:4])
    rng = np.random.default_rng(seed)
    budget = [max_steps]
    all_returns, all_images, groups = [], [], []
    replay_failures = 0
    for episode in range(episodes):
        ale.reset_game()
        step = 0
        collected = 0
        while collected < roots:
            if budget[0] < len(PROGRAMS) * 128 + 33:
                raise RuntimeError("orçamento insuficiente para uma raiz pareada")
            if ale.game_over():
                ale.reset_game()
            ale.act(actions[int(rng.integers(4))])
            budget[0] -= 1
            step += 1
            if step < 32 or step % 24:
                continue
            state = ale.cloneSystemState()
            result, image = [], []
            for p in PROGRAMS:
                ale.restoreSystemState(state)
                ret, im = run_program(ale, actions, p, budget)
                result.append(ret)
                image.append(im)
            if collected == 0:
                ale.restoreSystemState(state)
                ret2, im2 = run_program(ale, actions, PROGRAMS[0], budget)
                if ret2 != result[0] or not np.array_equal(im2, image[0]):
                    replay_failures += 1
                    raise RuntimeError("snapshot não reproduz programa")
            ale.restoreSystemState(state)
            all_returns.append(result)
            all_images.append(image)
            groups.append(episode)
            collected += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, returns=np.asarray(all_returns, np.float32),
                        images=np.asarray(all_images, np.float32),
                        episode=np.asarray(groups),
                        programs=np.asarray(PROGRAMS),
                        actions=np.asarray([a.value for a in actions]))
    meta = {"game": game, "seed": seed, "roots": len(groups),
            "steps_used": max_steps - budget[0], "max_steps": max_steps,
            "replay_failures": replay_failures,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def quality(returns, indices, subset):
    selected = returns[np.ix_(indices, subset)]
    gap = np.ptp(selected, axis=1)
    return int(np.sum(gap > 1e-8)), float(gap.mean())


def evaluate(bank):
    data = np.load(bank)
    episode = data["episode"]
    tr, val, test = group_split(episode)
    ret = data["returns"]
    best = max(itertools.combinations(range(16), 4),
               key=lambda s: (*quality(ret, tr, s), tuple(-i for i in s)))
    rng = np.random.default_rng(901)
    nulls = [tuple(sorted(rng.choice(16, 4, replace=False))) for _ in range(128)]
    return {
        "bank": str(bank), "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "selected_programs": [PROGRAMS[i] for i in best],
        "constant_programs": [PROGRAMS[i] for i in CONSTANTS],
        "construction": {"selected": quality(ret, tr, best),
                         "constant": quality(ret, tr, CONSTANTS)},
        "selection": {"selected": quality(ret, val, best),
                      "constant": quality(ret, val, CONSTANTS)},
        "evaluation": {"selected": quality(ret, test, best),
                       "constant": quality(ret, test, CONSTANTS),
                       "random_128_informative": [quality(ret, test, n)[0] for n in nulls]},
        "validacao_passou": quality(ret, val, best)[0] > quality(ret, val, CONSTANTS)[0],
        "test_groups": sorted(map(int, np.unique(episode[test]))),
        "scope": "biblioteca de 16 programas de dois segmentos; seleção offline"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    target = args.out or HERE / "resultados" / f"{args.game}_perguntas.npz"
    meta = collect(args.game, target)
    report = evaluate(target)
    target.with_name(target.stem + "_diagnostico.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"coleta": meta, "diagnostico": report}, ensure_ascii=False, indent=2))
