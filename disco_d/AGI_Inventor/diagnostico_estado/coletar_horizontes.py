"""Ramos pareados de 128 passos com estado visual a cada oito ações."""

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
HORIZONS = tuple(range(8, 129, 8))
MAX_STEPS = 45000


def branch(ale, action, initial_pair, budget):
    recent = deque((frame.copy() for frame in initial_pair), maxlen=2)
    pairs, rewards, lengths, terminals = [], [], [], []
    gain = 0.0
    ended = False
    elapsed = 0
    for horizon in HORIZONS:
        while elapsed < horizon and not ended:
            if budget[0] <= 0:
                raise RuntimeError("orçamento ALE excedido")
            gain += float(ale.act(action))
            budget[0] -= 1
            elapsed += 1
            recent.append(gray(ale))
            ended = bool(ale.game_over())
        pairs.append(np.stack(recent))
        rewards.append(gain)
        lengths.append(elapsed)
        terminals.append(ended)
    return np.stack(pairs), np.asarray(rewards, np.float32), np.asarray(lengths), np.asarray(terminals)


def collect(game, seed):
    expected_path = HERE / "resultados" / f"{game}_efeitos_locais_{seed}.npz"
    expected = np.load(expected_path) if expected_path.exists() else None
    ale = make_ale(game, seed)
    actions = list(ale.getMinimalActionSet()[:4])
    rng = np.random.default_rng(seed)
    budget = [MAX_STEPS]
    histories, outcomes, gains, lengths, terminals, groups = [], [], [], [], [], []
    events = []
    resets = 0
    certified = 0
    for episode in range(EPISODES):
        ale.reset_game()
        initial = gray(ale)
        history = deque([initial.copy() for _ in range(16)], maxlen=16)
        for checkpoint in range(CHECKPOINTS):
            for _ in range(INTERVAL):
                if budget[0] <= 4 * HORIZONS[-1]:
                    raise RuntimeError("orçamento insuficiente para ramos completos")
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
            if expected is not None and not np.array_equal(
                    current, expected["histories"][index]):
                raise RuntimeError(f"histórico divergiu no checkpoint {index}")
            state = ale.cloneSystemState()
            root_pairs, root_gains, root_lengths, root_terminals = [], [], [], []
            for command in range(4):
                ale.restoreSystemState(state)
                pair, reward, elapsed, terminal = branch(
                    ale, actions[command], current[-2:], budget)
                if expected is not None:
                    if not (np.array_equal(pair[0, -1], expected["outcomes"][index, command])
                            and reward[0] == expected["rewards"][index, command]
                            and elapsed[0] == expected["lengths"][index, command]
                            and terminal[0] == expected["terminal"][index, command]):
                        raise RuntimeError(f"ramo de oito passos divergiu: {index}, {command}")
                    certified += 1
                root_pairs.append(pair)
                root_gains.append(reward)
                root_lengths.append(elapsed)
                root_terminals.append(terminal)
            ale.restoreSystemState(state)
            histories.append(current)
            outcomes.append(np.stack(root_pairs))
            gains.append(np.stack(root_gains))
            lengths.append(np.stack(root_lengths))
            terminals.append(np.stack(root_terminals))
            groups.append(episode)
            events.append({"episode": episode, "checkpoint": checkpoint,
                           "rewards": np.stack(root_gains).tolist(),
                           "lengths": np.stack(root_lengths).tolist(),
                           "terminal": np.stack(root_terminals).tolist()})
        print(json.dumps({"game": game, "seed": seed, "episode": episode,
                          "roots": len(histories), "steps": MAX_STEPS - budget[0],
                          "h8_certified": certified}), flush=True)
    stem = HERE / "resultados" / f"{game}_horizontes_{seed}"
    bank_path = stem.with_suffix(".npz")
    np.savez_compressed(bank_path, histories=np.stack(histories),
                        pairs=np.stack(outcomes), rewards=np.stack(gains),
                        lengths=np.stack(lengths), terminal=np.stack(terminals),
                        episode=np.asarray(groups), horizons=np.asarray(HORIZONS))
    event_path = stem.with_suffix(".jsonl")
    with event_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    report = {"game": game, "seed": seed, "roots": len(histories),
              "h8_certified": certified, "resets": resets,
              "steps_used": MAX_STEPS - budget[0],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
              "events_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
              "short_bank_sha256": hashlib.sha256(expected_path.read_bytes()).hexdigest()
              if expected is not None else None}
    stem.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("--seed", type=int, choices=(97, 103, 109), required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.game, args.seed), ensure_ascii=False, indent=2))
