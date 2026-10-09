"""Coleta efeitos visuais de intervenções curtas em estados clonados."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from coleta import gray, make_ale

HERE = Path(__file__).resolve().parent
EPISODES = 8
CHECKPOINTS = 8
INTERVAL = 128
HORIZON = 8
MAX_STEPS = 20000


def branch(ale, action, budget):
    gain = 0.0
    steps = 0
    for _ in range(HORIZON):
        if budget[0] <= 0:
            raise RuntimeError("orçamento ALE excedido")
        gain += float(ale.act(action))
        budget[0] -= 1
        steps += 1
        if ale.game_over():
            break
    return gray(ale), gain, steps, bool(ale.game_over())


def collect(game, seed=97):
    ale = make_ale(game, seed)
    actions = list(ale.getMinimalActionSet()[:4])
    rng = np.random.default_rng(seed)
    budget = [MAX_STEPS]
    histories, outcomes, rewards, lengths, terminals, episodes = [], [], [], [], [], []
    events = []
    resets = 0
    expected_path = HERE / "resultados" / f"{game}_trajetoria_independente.npz"
    expected = np.load(expected_path)["histories"] if seed == 97 else None
    replay_checked = False
    for episode in range(EPISODES):
        ale.reset_game()
        initial = gray(ale)
        history = deque([initial.copy() for _ in range(16)], maxlen=16)
        for checkpoint in range(CHECKPOINTS):
            for _ in range(INTERVAL):
                if budget[0] <= 4 * HORIZON + 2 * HORIZON:
                    raise RuntimeError("orçamento insuficiente para estado completo")
                ale.act(actions[int(rng.integers(0, 4))])
                budget[0] -= 1
                if ale.game_over():
                    ale.reset_game()
                    initial = gray(ale)
                    history = deque([initial.copy() for _ in range(16)], maxlen=16)
                    resets += 1
                else:
                    history.append(gray(ale))
            index = episode * CHECKPOINTS + checkpoint
            current = np.stack(history).astype(np.uint8)
            if expected is not None and not np.array_equal(current, expected[index]):
                raise RuntimeError(f"história não coincide com banco no checkpoint {index}")
            state = ale.cloneSystemState()
            if not replay_checked:
                twice = []
                for _ in range(2):
                    ale.restoreSystemState(state)
                    twice.append(branch(ale, actions[0], budget))
                if not (np.array_equal(twice[0][0], twice[1][0])
                        and twice[0][1:] == twice[1][1:]):
                    raise RuntimeError("replay de intervenção divergiu")
                replay_checked = True
            ends, gains, counts, done = [], [], [], []
            for command in range(4):
                ale.restoreSystemState(state)
                image, gain, steps, terminal = branch(ale, actions[command], budget)
                ends.append(image)
                gains.append(gain)
                counts.append(steps)
                done.append(terminal)
            ale.restoreSystemState(state)
            histories.append(current)
            outcomes.append(np.stack(ends))
            rewards.append(gains)
            lengths.append(counts)
            terminals.append(done)
            episodes.append(episode)
            events.append({"episode": episode, "checkpoint": checkpoint,
                           "rewards": gains, "steps": counts, "terminal": done})
        print(json.dumps({"game": game, "episode": episode,
                          "checkpoints": len(histories), "resets": resets,
                          "steps": MAX_STEPS - budget[0]}), flush=True)
    stem = HERE / "resultados" / f"{game}_efeitos_locais_{seed}"
    bank_path = stem.with_suffix(".npz")
    np.savez_compressed(bank_path, histories=np.stack(histories),
                        outcomes=np.stack(outcomes), rewards=np.asarray(rewards),
                        lengths=np.asarray(lengths), terminal=np.asarray(terminals),
                        episode=np.asarray(episodes))
    events_path = stem.with_suffix(".jsonl")
    with events_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    report = {"game": game, "seed": seed, "checkpoints": len(histories),
              "certified_against_previous_bank": expected is not None,
              "replay_checked": replay_checked, "resets": resets,
              "steps_used": MAX_STEPS - budget[0],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
              "events_sha256": hashlib.sha256(events_path.read_bytes()).hexdigest()}
    stem.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("--seed", type=int, default=97)
    args = parser.parse_args()
    print(json.dumps(collect(args.game, args.seed), ensure_ascii=False, indent=2))
