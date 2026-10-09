"""Compõe distribuições de experiências, preservando crédito e sucessor.

Treino 97; 103 é desenvolvimento já visto. Sem objeto ou regra de jogo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from avaliar_composicao import features, reduce_frames
from avaliar_estado_preditivo import encode, load
from diagnosticar_credito import summarize
from extrator_experiencias import ExperienceExtractor
from valor_composto import increments

HERE = Path(__file__).resolve().parent
NEIGHBORS = 8


def weights(query, support, neighbors=NEIGHBORS):
    distance = KernelRidge.distances(np.asarray(query, float), np.asarray(support, float))
    k = min(neighbors, len(support))
    ids = np.argpartition(distance, k - 1, axis=1)[:, :k]
    chosen = np.take_along_axis(distance, ids, axis=1)
    scale = np.maximum(chosen.max(1, keepdims=True), 1e-12)
    probability = np.exp(-chosen / scale)
    probability /= probability.sum(1, keepdims=True)
    out = np.zeros_like(distance)
    np.put_along_axis(out, ids, probability, axis=1)
    return out


def value_table(source, successor, reward, continuation, stages=16, neighbors=NEIGHBORS):
    transition = weights(successor, source, neighbors)
    value = np.zeros(len(source))
    values = []
    for _ in range(stages):
        value = reward + continuation * (transition @ value)
        values.append(value.copy())
    return np.stack(values, axis=-1)


def mean_rollout(query, source, successor, reward, continuation, stages=16, frozen=False,
                 neighbors=NEIGHBORS):
    state = np.asarray(query, float).copy()
    alive, total = np.ones(len(state)), np.zeros(len(state))
    out = []
    for _ in range(stages):
        probability = weights(state, source, neighbors)
        total += alive * (probability @ reward)
        alive *= probability @ continuation
        if not frozen:
            state = probability @ successor
        out.append(total.copy())
    return np.stack(out, axis=-1)


def make_edges(z, valid, cumulative):
    reward = increments(cumulative)
    edges = []
    for action in range(4):
        mask = valid[:, action, :-1].reshape(-1)
        edges.append({"source": z[:, action, :-1].reshape(-1, 32)[mask],
                      "successor": z[:, action, 1:].reshape(-1, 32)[mask],
                      "reward": reward[:, action, :8].reshape(-1)[mask],
                      "continuation": valid[:, action, 1:].reshape(-1)[mask].astype(float)})
    return edges


def predictions(root, edges, altered=None):
    results = []
    rng = np.random.default_rng(9753)
    for action in range(4):
        e = edges[(action + 1) % 4 if altered == "action" else action]
        source, successor = e["source"], e["successor"].copy()
        reward, continuation = e["reward"].copy(), e["continuation"].copy()
        if altered == "successor":
            permutation = rng.permutation(len(source))
            successor, continuation = successor[permutation], continuation[permutation]
        if altered == "reward":
            reward = reward[rng.permutation(len(source))]
        if altered in ("mean", "frozen"):
            values = mean_rollout(root, source, successor, reward, continuation,
                                  frozen=altered == "frozen")
        else:
            values = weights(root, source) @ value_table(source, successor, reward, continuation)
        results.append(values)
    return np.stack(results, axis=1)


def evaluate(game):
    start = time.process_time()
    path, _, signature, valid, _ = load(game, 97)
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    z = encode(signature, basis["mean"], basis["basis"])
    train_h = np.load(HERE / "resultados" / f"{game}_horizontes_97.npz")
    test_path = HERE / "resultados" / f"{game}_horizontes_103.npz"
    test_h = np.load(test_path)
    source = ExperienceExtractor(game)
    initial_pair = reduce_frames(test_h["histories"][:, -2:])
    root, _, _, _ = source.predict(features(initial_pair), set(range(8)))
    short = reduce_frames(test_h["pairs"][:, :, 0, 1]) - initial_pair[:, None, 1]
    oracle_root = encode(short, basis["mean"], basis["basis"])
    edges = make_edges(z, valid, train_h["rewards"])
    forecasts = {"distributed": predictions(root, edges),
                 "mean_collapsed": predictions(root, edges, "mean"),
                 "state_frozen": predictions(root, edges, "frozen"),
                 "successor_shuffled": predictions(root, edges, "successor"),
                 "reward_shuffled": predictions(root, edges, "reward"),
                 "action_permuted": predictions(root, edges, "action"),
                 "oracle_initial_distributed": predictions(oracle_root, edges)}
    curves = {}
    for horizon in (8, 32, 64, 128):
        index = horizon // 8 - 1
        truth = test_h["rewards"][:, :, index]
        row = {name: summarize(v[:, :, index], truth, test_h["episode"])
               for name, v in forecasts.items()}
        action = int(train_h["rewards"][:, :, index].mean(0).argmax())
        constant_values = np.zeros_like(truth)
        constant_values[:, action] = 1
        row["constant"] = summarize(constant_values, truth, test_h["episode"])
        comparisons = {name: float(row[name]["mean_regret"] - row["distributed"]["mean_regret"])
                       for name in ("constant", "mean_collapsed", "successor_shuffled",
                                    "reward_shuffled", "action_permuted")}
        curves[str(horizon)] = {"arms": row, "gain_vs_controls": comparisons,
                                "better_than_all_controls": all(g > 1e-12 for g in comparisons.values())}
    report = {"status": "exploratory_not_confirmation", "game": game, "train_seed": 97,
              "test_seed": 103, "neighbors": NEIGHBORS, "curves": curves,
              "edges_per_action": [len(e["source"]) for e in edges],
              "new_ale_steps": 0, "cpu_seconds": time.process_time() - start,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "train_bank_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "test_bank_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest()}
    output = HERE / "resultados" / f"{game}_futuros_distribuidos.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    r = evaluate(args.game)
    print(json.dumps({"game": args.game, "cpu": r["cpu_seconds"],
                      "curves": {h: {n: a["mean_regret"] for n, a in row["arms"].items()}
                                 for h, row in r["curves"].items()}}, indent=2))
