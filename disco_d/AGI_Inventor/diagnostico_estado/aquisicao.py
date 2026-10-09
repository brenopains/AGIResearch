"""Busca ativa por estados que falsificam o decisor congelado; custo pareado."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from avaliar import KernelRidge, views
from coleta import gray, make_ale, small

HERE = Path(__file__).resolve().parent
SEGMENTS = 4
SEGMENT_STEPS = 16
BRANCH_PREFIX = 32
BRANCH_SUFFIX = 96
BRANCHES = 16
EPISODES = 8
PROBES = 8
MAX_STEPS_PER_ARM = 160000


def frozen_model(game: str):
    bank = HERE / "resultados" / f"{game}_ciclo.npz"
    data = np.load(bank)
    fit = np.flatnonzero(data["episode"] < 6)
    x = views(data["histories"])["r0_21x16_2"]
    y = data["returns"].reshape(-1, 4, 4).max(axis=2)
    return KernelRidge().fit(x[fit], y[fit]), hashlib.sha256(bank.read_bytes()).hexdigest()


def base_from_history(history):
    now = small(history[-1], 21).ravel()
    before = small(history[-2], 21).ravel()
    return np.concatenate((now, now - before)).astype(np.float32)


def new_macro(rng, seen):
    for _ in range(1024):
        macro = tuple(int(x) for x in rng.integers(0, 4, SEGMENTS))
        if macro not in seen:
            return macro
    raise RuntimeError("não foi possível gerar macro novo")


def mutate(rng, parent, seen):
    for _ in range(1024):
        macro = list(parent)
        position = int(rng.integers(SEGMENTS))
        options = [a for a in range(4) if a != macro[position]]
        macro[position] = int(options[int(rng.integers(len(options)))])
        if tuple(macro) not in seen:
            return tuple(macro)
    return new_macro(rng, seen)


def branch_reward(ale, actions, prefix, suffix, budget):
    total = 0.0
    for t in range(BRANCH_PREFIX + BRANCH_SUFFIX):
        if budget[0] <= 0:
            raise RuntimeError("orçamento ALE esgotado")
        action = prefix if t < BRANCH_PREFIX else suffix
        total += float(ale.act(actions[action]))
        budget[0] -= 1
        if ale.game_over():
            break
    return total


def probe(ale, actions, initial, initial_history, macro, model, budget):
    used_before = budget[0]
    ale.restoreSystemState(initial)
    history = deque((frame.copy() for frame in initial_history), maxlen=16)
    for action in macro:
        for _ in range(SEGMENT_STEPS):
            if budget[0] <= 0:
                raise RuntimeError("orçamento ALE esgotado")
            ale.act(actions[action])
            budget[0] -= 1
            history.append(gray(ale))
            if ale.game_over():
                return {"macro": macro, "terminal": True,
                        "steps": used_before - budget[0], "counterexample": False,
                        "regret": 0.0, "unique_best": False}
    state = ale.cloneSystemState()
    x = base_from_history(history)
    predicted = model.predict(x[None])[0]
    chosen = int(np.argmax(predicted))
    outcomes = np.zeros((4, 4), np.float32)
    for first in range(4):
        for last in range(4):
            ale.restoreSystemState(state)
            outcomes[first, last] = branch_reward(ale, actions, first, last, budget)
    values = outcomes.max(axis=1)
    spread = float(np.ptp(values))
    missed = float(values.max() - values[chosen])
    sorted_values = np.sort(values)
    return {"macro": macro, "terminal": False,
            "steps": used_before - budget[0],
            "chosen_prefix": chosen, "best_prefix": int(np.argmax(values)),
            "value_by_prefix": values.tolist(),
            "counterexample": bool(missed > 1e-8),
            "regret": missed / spread if spread > 1e-8 else 0.0,
            "unique_best": bool(sorted_values[-1] > sorted_values[-2] + 1e-8)}


def prepare_episode(ale, actions, rng, shared_budget):
    ale.reset_game()
    frame = gray(ale)
    history = deque([frame.copy() for _ in range(16)], maxlen=16)
    for _ in range(32):
        if shared_budget[0] <= 0:
            raise RuntimeError("orçamento comum esgotado")
        ale.act(actions[int(rng.integers(4))])
        shared_budget[0] -= 1
        history.append(gray(ale))
        if ale.game_over():
            ale.reset_game()
            frame = gray(ale)
            history = deque([frame.copy() for _ in range(16)], maxlen=16)
    return ale.cloneSystemState(), tuple(history)


def run_game(game: str, output: Path, seed=43):
    model, reference_hash = frozen_model(game)
    ale = make_ale(game, seed)
    actions = list(ale.getMinimalActionSet()[:4])
    shared_rng = np.random.default_rng(seed)
    shared_budget = [EPISODES * 32]
    budgets = {"active": [MAX_STEPS_PER_ARM],
               "random": [MAX_STEPS_PER_ARM]}
    events, episodes = [], []
    for episode in range(EPISODES):
        initial, history = prepare_episode(ale, actions, shared_rng, shared_budget)
        row = {"episode": episode}
        for arm in ("active", "random"):
            rng = np.random.default_rng(seed * 1000 + episode * 7 + (1 if arm == "active" else 2))
            seen = set()
            records = []
            spent_before = budgets[arm][0]
            first_cost = None
            for trial in range(PROBES):
                if arm == "active" and trial >= 2:
                    counterexamples = [r for r in records if r["counterexample"]]
                    macro = (mutate(rng, max(counterexamples, key=lambda r: r["regret"])["macro"], seen)
                             if counterexamples else new_macro(rng, seen))
                else:
                    macro = new_macro(rng, seen)
                seen.add(macro)
                result = probe(ale, actions, initial, history, macro, model, budgets[arm])
                result.update({"game": game, "episode": episode,
                               "arm": arm, "trial": trial})
                events.append(result)
                records.append(result)
                if result["counterexample"] and first_cost is None:
                    first_cost = spent_before - budgets[arm][0]
            cost = spent_before - budgets[arm][0]
            row[arm] = {"found": bool(any(r["counterexample"] for r in records)),
                        "counterexample_probes": int(sum(r["counterexample"] for r in records)),
                        "unique_best_probes": int(sum(r["unique_best"] for r in records)),
                        "max_regret": float(max(r["regret"] for r in records)),
                        "steps": cost, "first_cost": first_cost}
        episodes.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    active = sum(r["active"]["found"] for r in episodes)
    random = sum(r["random"]["found"] for r in episodes)
    report = {"game": game, "seed": seed, "reference_bank_sha256": reference_hash,
              "episodes": episodes, "active_found_episodes": active,
              "random_found_episodes": random,
              "net_episodes": active - random,
              "active_steps": MAX_STEPS_PER_ARM - budgets["active"][0],
              "random_steps": MAX_STEPS_PER_ARM - budgets["random"][0],
              "shared_warmup_steps": EPISODES * 32 - shared_budget[0],
              "gate_local": bool(active - random >= 3 and
                                 all(r["active"]["steps"] <= r["random"]["steps"]
                                     for r in episodes)),
              "events_sha256": hashlib.sha256(output.with_suffix(".jsonl").read_bytes()).hexdigest()}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    path = args.out or HERE / "resultados" / f"{args.game}_aquisicao.json"
    report = run_game(args.game, path)
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"},
                     indent=2, ensure_ascii=False))
