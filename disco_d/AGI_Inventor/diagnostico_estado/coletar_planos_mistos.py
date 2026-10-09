"""Avaliação de composição: ações alternadas nunca presentes no treino.

Os planos são fixados no código antes da coleta. Cada raiz é confrontada com
o banco de históricos independente; os estados posteriores só servem para
avaliar, nunca para ajustar o modelo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from coleta import gray, make_ale, small
from coletar_estado_preditivo import advance, probe

HERE = Path(__file__).resolve().parent
PLANS = np.asarray([
    [0, 1, 2, 3, 0, 1, 2, 3],
    [3, 2, 1, 0, 3, 2, 1, 0],
    [0, 2, 3, 1, 2, 0, 1, 3],
], dtype=np.int8)
EPISODES = 8
CHECKPOINTS = 8
INTERVAL = 128
STAGES = 9
MAX_STEPS = 120000


def collect(game, seed):
    horizon_path = HERE / "resultados" / f"{game}_horizontes_{seed}.npz"
    short_path = HERE / "resultados" / f"{game}_efeitos_locais_{seed}.npz"
    horizon = np.load(horizon_path)
    short = np.load(short_path)
    expected_histories = horizon["histories"]
    short_outcomes = short["outcomes"]
    short_rewards = short["rewards"]
    short_lengths = short["lengths"]
    short_terminal = short["terminal"]
    ale = make_ale(game, seed)
    actions = list(ale.getMinimalActionSet()[:4])
    rng = np.random.default_rng(seed)
    budget = [MAX_STEPS]
    bank_signature, bank_valid, groups = [], [], []
    certified_roots, certified_tests = 0, 0
    resets = 0
    for episode in range(EPISODES):
        ale.reset_game()
        frame = gray(ale)
        history = deque([frame.copy() for _ in range(16)], maxlen=16)
        for checkpoint in range(CHECKPOINTS):
            for _ in range(INTERVAL):
                if budget[0] <= 4 * 8 * STAGES:
                    raise RuntimeError("orçamento ALE esgotado")
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
            root_signatures, root_valid = [], []
            for plan in PLANS:
                path_state = state
                pair = initial_pair.copy()
                ended = False
                signatures = [root_signature]
                valid = [True]
                for code in plan:
                    if not ended:
                        ale.restoreSystemState(path_state)
                        pair, _, _, ended = advance(ale, actions[int(code)], pair, budget)
                        path_state = ale.cloneSystemState()
                    if ended:
                        signatures.append(np.zeros((4, 21, 16), np.float32))
                        valid.append(False)
                    else:
                        signature, _, _, _ = probe(ale, actions, path_state, pair, budget)
                        signatures.append(signature)
                        valid.append(True)
                root_signatures.append(np.stack(signatures))
                root_valid.append(valid)
            ale.restoreSystemState(state)
            bank_signature.append(np.stack(root_signatures))
            bank_valid.append(np.stack(root_valid))
            groups.append(episode)
        print(json.dumps({"game": game, "seed": seed, "episode": episode,
                          "roots": certified_roots,
                          "steps": MAX_STEPS - budget[0]}), flush=True)
    stem = HERE / "resultados" / f"{game}_planos_mistos_{seed}"
    bank_path = stem.with_suffix(".npz")
    np.savez_compressed(bank_path, signatures=np.stack(bank_signature),
                        valid=np.stack(bank_valid), episode=np.asarray(groups),
                        plans=PLANS)
    report = {"status": "avaliacao_apenas", "game": game, "seed": seed,
              "certified_roots": certified_roots,
              "certified_initial_tests": certified_tests,
              "valid_states": int(np.sum(bank_valid)),
              "resets": resets, "steps_used": MAX_STEPS - budget[0],
              "plans": PLANS.tolist(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_horizon_bank_sha256": hashlib.sha256(horizon_path.read_bytes()).hexdigest(),
              "source_short_bank_sha256": hashlib.sha256(short_path.read_bytes()).hexdigest(),
              "bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest()}
    stem.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("--seed", type=int, choices=(103, 109), required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.game, args.seed), ensure_ascii=False, indent=2))
