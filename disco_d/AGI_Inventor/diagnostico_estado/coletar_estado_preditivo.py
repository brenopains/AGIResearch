"""Observa baterias contrafactuais ao longo de trajetórias de 64 passos."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from coleta import gray, make_ale, small

HERE = Path(__file__).resolve().parent
EPISODES = 8
CHECKPOINTS = 8
INTERVAL = 128
STAGES = 9  # raiz, 8, 16, ... 64
MAX_STEPS = 100000


def advance(ale, action, pair, budget):
    previous = pair[-1]
    latest = pair[-1]
    reward = 0.0
    elapsed = 0
    for _ in range(8):
        if budget[0] <= 0:
            raise RuntimeError("orçamento ALE esgotado")
        reward += float(ale.act(action))
        budget[0] -= 1
        elapsed += 1
        if elapsed == 7:
            previous = gray(ale)
        if ale.game_over():
            latest = gray(ale)
            break
        if elapsed == 8:
            latest = gray(ale)
    # Quando o jogo termina antes do sétimo passo, o quadro anterior é
    # irrelevante: o estado não treinará atualização. O quadro final é exato.
    return np.stack((previous, latest)), reward, elapsed, bool(ale.game_over())


def probe(ale, actions, state, pair, budget):
    current = small(pair[-1], 21)
    effects, rewards, lengths, terminal = [], [], [], []
    for action in actions:
        ale.restoreSystemState(state)
        outcome, gain, elapsed, done = advance(ale, action, pair, budget)
        effects.append(small(outcome[-1], 21) - current)
        rewards.append(gain)
        lengths.append(elapsed)
        terminal.append(done)
    ale.restoreSystemState(state)
    return np.stack(effects), np.asarray(rewards, np.float32), np.asarray(lengths), np.asarray(terminal)


def collect(game, seed):
    horizon_path = HERE / "resultados" / f"{game}_horizontes_{seed}.npz"
    short_path = HERE / "resultados" / f"{game}_efeitos_locais_{seed}.npz"
    expected = np.load(horizon_path)
    short = np.load(short_path)
    expected_histories = expected["histories"]
    expected_pairs = expected["pairs"]
    short_outcomes = short["outcomes"]
    short_rewards = short["rewards"]
    short_lengths = short["lengths"]
    short_terminal = short["terminal"]
    ale = make_ale(game, seed)
    actions = list(ale.getMinimalActionSet()[:4])
    rng = np.random.default_rng(seed)
    budget = [MAX_STEPS]
    observed_pairs, signatures, valid, groups, events = [], [], [], [], []
    certified_roots, certified_tests, certified_waypoints = 0, 0, 0
    resets = 0
    for episode in range(EPISODES):
        ale.reset_game()
        frame = gray(ale)
        history = deque([frame.copy() for _ in range(16)], maxlen=16)
        for checkpoint in range(CHECKPOINTS):
            for _ in range(INTERVAL):
                if budget[0] <= 4 * 8 * STAGES:
                    raise RuntimeError("orçamento insuficiente")
                ale.act(actions[int(rng.integers(0, 4))])
                budget[0] -= 1
                if ale.game_over():
                    ale.reset_game()
                    frame = gray(ale)
                    history = deque([frame.copy() for _ in range(16)], maxlen=16)
                    resets += 1
                else:
                    history.append(gray(ale))
            index = episode * CHECKPOINTS + checkpoint
            current = np.stack(history).astype(np.uint8)
            if not np.array_equal(current, expected_histories[index]):
                raise RuntimeError(f"histórico divergiu: {index}")
            certified_roots += 1
            state = ale.cloneSystemState()
            initial_pair = current[-2:]
            root_signature, rewards, lengths, terminal = probe(
                ale, actions, state, initial_pair, budget)
            true_root_effects = np.stack([
                small(short_outcomes[index, a], 21) - small(initial_pair[-1], 21)
                for a in range(4)])
            if not (np.allclose(root_signature, true_root_effects, atol=0, rtol=0)
                    and np.array_equal(rewards, short_rewards[index])
                    and np.array_equal(lengths, short_lengths[index])
                    and np.array_equal(terminal, short_terminal[index])):
                raise RuntimeError(f"teste inicial divergiu: {index}")
            certified_tests += 4
            root_pair_small = np.stack([small(frame, 21) for frame in initial_pair])
            path_pairs, path_signatures, path_valid = [], [], []
            for primary in range(4):
                pair = initial_pair.copy()
                path_state = state
                ended = False
                one_pairs = [root_pair_small]
                one_signatures = [root_signature]
                one_valid = [True]
                for stage in range(1, STAGES):
                    if not ended:
                        ale.restoreSystemState(path_state)
                        pair, _, _, ended = advance(ale, actions[primary], pair, budget)
                        path_state = ale.cloneSystemState()
                    reference = expected_pairs[index, primary, stage - 1]
                    matched = (np.array_equal(pair[-1], reference[-1])
                               if ended else np.array_equal(pair, reference))
                    if not matched:
                        raise RuntimeError(f"estado intermediário divergiu: {index},{primary},{stage}")
                    certified_waypoints += 1
                    one_pairs.append(np.stack([small(frame, 21) for frame in pair]))
                    if ended:
                        one_signatures.append(np.zeros((4, 21, 16), np.float32))
                        one_valid.append(False)
                    else:
                        signature, _, _, _ = probe(ale, actions, path_state, pair, budget)
                        one_signatures.append(signature)
                        one_valid.append(True)
                path_pairs.append(np.stack(one_pairs))
                path_signatures.append(np.stack(one_signatures))
                path_valid.append(one_valid)
            ale.restoreSystemState(state)
            observed_pairs.append(np.stack(path_pairs))
            signatures.append(np.stack(path_signatures))
            valid.append(np.stack(path_valid))
            groups.append(episode)
            events.append({"episode": episode, "checkpoint": checkpoint,
                           "valid_states": np.stack(path_valid).astype(int).tolist()})
        print(json.dumps({"game": game, "seed": seed, "episode": episode,
                          "roots": certified_roots, "waypoints": certified_waypoints,
                          "steps": MAX_STEPS - budget[0]}), flush=True)
    stem = HERE / "resultados" / f"{game}_estado_preditivo_{seed}"
    bank_path = stem.with_suffix(".npz")
    np.savez_compressed(bank_path, pairs=np.stack(observed_pairs),
                        signatures=np.stack(signatures), valid=np.stack(valid),
                        episode=np.asarray(groups))
    event_path = stem.with_suffix(".jsonl")
    with event_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    report = {"game": game, "seed": seed,
              "certified_roots": certified_roots,
              "certified_initial_tests": certified_tests,
              "certified_waypoints": certified_waypoints,
              "valid_predictive_states": int(np.sum(valid)),
              "resets": resets, "steps_used": MAX_STEPS - budget[0],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_horizon_bank_sha256": hashlib.sha256(horizon_path.read_bytes()).hexdigest(),
              "source_short_bank_sha256": hashlib.sha256(short_path.read_bytes()).hexdigest(),
              "bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
              "events_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest()}
    stem.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("--seed", type=int, choices=(97, 103), required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.game, args.seed), ensure_ascii=False, indent=2))
