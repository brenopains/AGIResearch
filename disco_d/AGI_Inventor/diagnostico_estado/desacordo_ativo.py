"""Busca estados por desacordo entre preditores, verifica decisões por rollback."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from aquisicao import branch_reward, frozen_model, prepare_episode
from avaliar import KernelRidge, views
from coleta import gray, make_ale, small

HERE = Path(__file__).resolve().parent
EPISODES = 8
POOL = 24
PROBES = 4
SEGMENTS = 4
SEGMENT_STEPS = 32
SHARED_BUDGET = 40000
ARM_BUDGET = 100000


def committee(game):
    bank = HERE / "resultados" / f"{game}_ciclo.npz"
    data = np.load(bank)
    groups = data["episode"]
    y = data["returns"].reshape(-1, 4, 4).max(axis=2)
    features = views(data["histories"])
    members = []
    for name in ("r0_21x16_2", "hist_21x16_16"):
        x = features[name]
        for omitted in range(6):
            fit = np.flatnonzero((groups < 6) & (groups != omitted))
            members.append((name, KernelRidge().fit(x[fit], y[fit])))
    return members, hashlib.sha256(bank.read_bytes()).hexdigest()


def features_from_history(history):
    h = np.stack([small(frame, 21) for frame in history]).astype(np.float32)
    last = h[-1].ravel()
    before = h[-2].ravel()
    return {"r0_21x16_2": np.concatenate((last, last - before)),
            "hist_21x16_16": h.ravel()}


def candidate(ale, actions, initial, initial_history, macro, budget):
    ale.restoreSystemState(initial)
    history = deque((frame.copy() for frame in initial_history), maxlen=16)
    before = budget[0]
    for command in macro:
        for _ in range(SEGMENT_STEPS):
            if budget[0] <= 0:
                raise RuntimeError("orçamento comum esgotado")
            ale.act(actions[command])
            budget[0] -= 1
            history.append(gray(ale))
            if ale.game_over():
                return {"macro": macro, "terminal": True, "steps": before - budget[0]}
    return {"macro": macro, "terminal": False, "steps": before - budget[0],
            "snapshot": ale.cloneSystemState(),
            "features": features_from_history(history)}


def vote_score(members, observation):
    votes = np.asarray([int(np.argmax(model.predict(observation[name][None])[0]))
                        for name, model in members], dtype=np.int8)
    counts = np.bincount(votes, minlength=4)
    return float(1 - counts.max() / len(votes)), counts.tolist()


def verify(ale, actions, item, decisive, budget):
    before = budget[0]
    chosen = int(np.argmax(decisive.predict(item["features"]["r0_21x16_2"][None])[0]))
    returns = np.zeros((4, 4), np.float32)
    for first in range(4):
        for second in range(4):
            ale.restoreSystemState(item["snapshot"])
            returns[first, second] = branch_reward(ale, actions, first, second, budget)
    values = returns.max(axis=1)
    span = float(np.ptp(values))
    missed = float(values.max() - values[chosen])
    return {"macro": item["macro"], "votes": item["votes"],
            "disagreement": item["disagreement"], "chosen": chosen,
            "best": int(values.argmax()), "values": values.tolist(),
            "counterexample": bool(missed > 1e-8),
            "regret": missed / span if span > 1e-8 else 0.0,
            "unique_best": bool(np.sort(values)[-1] > np.sort(values)[-2] + 1e-8),
            "steps": before - budget[0]}


def check_replay(ale, actions, item, budget):
    observations = []
    for _ in range(2):
        ale.restoreSystemState(item["snapshot"])
        reward = branch_reward(ale, actions, 0, 0, budget)
        observations.append((reward, gray(ale)))
    if not (observations[0][0] == observations[1][0]
            and np.array_equal(observations[0][1], observations[1][1])):
        raise RuntimeError("snapshot ALE não produz replay idêntico")


def run(game, output, seed=59):
    decisive, bank_hash = frozen_model(game)
    members, committee_hash = committee(game)
    if bank_hash != committee_hash:
        raise RuntimeError("decisor e comitê usam bancos diferentes")
    ale = make_ale(game, seed)
    actions = list(ale.getMinimalActionSet()[:4])
    shared = [SHARED_BUDGET]
    budgets = {"disagreement": [ARM_BUDGET], "random": [ARM_BUDGET]}
    warm_rng = np.random.default_rng(seed)
    events, rows = [], []
    replay_checked = False
    for episode in range(EPISODES):
        initial, history = prepare_episode(ale, actions, warm_rng, shared)
        pool_rng = np.random.default_rng(seed * 1000 + episode * 11 + 1)
        macros = [tuple(map(int, macro)) for macro in
                  pool_rng.choice(4, size=(POOL, SEGMENTS), replace=True)]
        # Pool não é adaptado ao retorno; repetir macro desperdica orçamento.
        seen = set()
        for i, macro in enumerate(macros):
            while macro in seen:
                macro = tuple(int(a) for a in pool_rng.integers(0, 4, SEGMENTS))
            seen.add(macro)
            macros[i] = macro
        pool_before = shared[0]
        items = [candidate(ale, actions, initial, history, macro, shared) for macro in macros]
        valid = [item for item in items if not item["terminal"]]
        if len(valid) < PROBES:
            rows.append({"episode": episode, "status": "candidatos_insuficientes",
                         "valid": len(valid), "pool_steps": pool_before - shared[0]})
            continue
        if not replay_checked:
            check_replay(ale, actions, valid[0], shared)
            replay_checked = True
        for item in valid:
            item["disagreement"], item["votes"] = vote_score(members, item["features"])
        random_rng = np.random.default_rng(seed * 1000 + episode * 11 + 2)
        tie_order = random_rng.permutation(len(valid))
        tie_rank = np.empty(len(valid), int)
        tie_rank[tie_order] = np.arange(len(valid))
        ranked = sorted(range(len(valid)), key=lambda i: (-valid[i]["disagreement"], tie_rank[i]))
        selections = {"disagreement": ranked[:PROBES],
                      "random": random_rng.choice(len(valid), PROBES, replace=False).tolist()}
        row = {"episode": episode, "status": "ok", "pool_steps": pool_before - shared[0],
               "valid": len(valid),
               "nonunanimous": int(sum(item["disagreement"] > 0 for item in valid)),
               "max_disagreement": max(item["disagreement"] for item in valid)}
        for arm, chosen in selections.items():
            arm_before = budgets[arm][0]
            records = []
            for j, index in enumerate(chosen):
                event = verify(ale, actions, valid[index], decisive, budgets[arm])
                event.update({"game": game, "episode": episode, "arm": arm,
                              "probe": j, "pool_index": index})
                records.append(event)
                events.append(event)
            row[arm] = {"found": bool(any(x["counterexample"] for x in records)),
                        "counterexamples": int(sum(x["counterexample"] for x in records)),
                        "unique_best": int(sum(x["unique_best"] for x in records)),
                        "steps": arm_before - budgets[arm][0],
                        "max_regret": max(x["regret"] for x in records),
                        "mean_disagreement": float(np.mean([x["disagreement"] for x in records]))}
        rows.append(row)
        print(json.dumps({"game": game, "episode": episode, "valid": len(valid),
                          "nonunanimous": row["nonunanimous"],
                          "active_found": row["disagreement"]["found"],
                          "random_found": row["random"]["found"]}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    events_path = output.with_suffix(".jsonl")
    with events_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    complete = [row for row in rows if row["status"] == "ok"]
    active = sum(row["disagreement"]["found"] for row in complete)
    random = sum(row["random"]["found"] for row in complete)
    steps = SHARED_BUDGET - shared[0] + sum(ARM_BUDGET - budget[0]
                                            for budget in budgets.values())
    report = {"game": game, "seed": seed, "reference_bank_sha256": bank_hash,
              "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                for name in ("desacordo_ativo.py", "aquisicao.py", "avaliar.py", "coleta.py")},
              "episodes": rows, "complete_episodes": len(complete),
              "disagreement_found": active, "random_found": random,
              "net_episodes": active - random,
              "nonunanimous_candidates": sum(row["nonunanimous"] for row in complete),
              "valid_candidates": sum(row["valid"] for row in complete),
              "shared_steps": SHARED_BUDGET - shared[0],
              "active_steps": ARM_BUDGET - budgets["disagreement"][0],
              "random_steps": ARM_BUDGET - budgets["random"][0],
              "total_steps": steps, "replay_checked": replay_checked,
              "gate_local": bool(len(complete) == EPISODES and active - random >= 3 and
                                 all(row["disagreement"]["steps"] <= row["random"]["steps"]
                                     for row in complete)),
              "events_sha256": hashlib.sha256(events_path.read_bytes()).hexdigest()}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=59)
    args = parser.parse_args()
    path = args.out or HERE / "resultados" / f"{args.game}_desacordo_ativo.json"
    report = run(args.game, path, seed=args.seed)
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"},
                     indent=2, ensure_ascii=False))
