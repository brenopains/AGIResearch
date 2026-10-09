"""Avalia modelos congelados em estados ALE novos escolhidos independentemente."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from aquisicao import frozen_model, prepare_episode
from auditar_constante import normalize_regret
from avaliar import KernelRidge, group_split
from coleta import make_ale
from desacordo_ativo import check_replay, verify
from reconstruir_desacordo import candidate_with_history

HERE = Path(__file__).resolve().parent
SEED = 73
EPISODES = 8
PROBES = 8
MAX_STEPS = 170000


def fit_models(game):
    bank = HERE / "resultados" / f"{game}_ativo_geral.npz"
    data = np.load(bank)
    history, values, episode = data["histories"], data["values"], data["episode"]
    tr, val, _test = group_split(episode)
    fit = np.sort(np.concatenate((tr, val)))
    x = np.concatenate((history[:, -1], history[:, -1] - history[:, -2]), axis=1)
    model = KernelRidge().fit(x[fit], values[fit])
    constant = min(range(4), key=lambda action: (normalize_regret(values, action, fit), action))
    old, old_hash = frozen_model(game)
    return model, old, constant, hashlib.sha256(bank.read_bytes()).hexdigest(), old_hash


def regret(values, action):
    spread = float(np.ptp(values))
    return (float(values.max() - values[action]) / spread) if spread > 1e-8 else None


def aggregate(rows, name):
    per_episode = []
    for episode in range(EPISODES):
        rr = [row["regret"][name] for row in rows
              if row["episode"] == episode and row["regret"][name] is not None]
        per_episode.append(float(np.mean(rr)) if rr else None)
    nonempty = [r for r in per_episode if r is not None]
    return {"per_episode": per_episode,
            "mean_of_informative_episodes": float(np.mean(nonempty)) if nonempty else None,
            "informative_episodes": len(nonempty)}


def run(game, seed=SEED):
    new, old, constant, new_hash, old_hash = fit_models(game)
    ale = make_ale(game, seed)
    actions = list(ale.getMinimalActionSet()[:4])
    warm_rng = np.random.default_rng(seed)
    budget = [MAX_STEPS]
    events, rows, inputs = [], [], []
    checked = False
    for episode in range(EPISODES):
        initial, history = prepare_episode(ale, actions, warm_rng, budget)
        rng = np.random.default_rng(seed * 1000 + episode * 13)
        seen = set()
        for trial in range(PROBES):
            macro = tuple(int(a) for a in rng.integers(0, 4, 4))
            while macro in seen:
                macro = tuple(int(a) for a in rng.integers(0, 4, 4))
            seen.add(macro)
            item = candidate_with_history(ale, actions, initial, history, macro, budget)
            if item["terminal"]:
                events.append({"game": game, "episode": episode, "trial": trial,
                               "macro": macro, "terminal": True, "steps": item["steps"]})
                continue
            if not checked:
                check_replay(ale, actions, item, budget)
                checked = True
            x = item["features"]["r0_21x16_2"][None]
            new_prediction = new.predict(x)[0]
            old_prediction = old.predict(x)[0]
            item["votes"], item["disagreement"] = [0, 0, 0, 0], 0.0
            before = budget[0]
            verified = verify(ale, actions, item, old, budget)
            values = np.asarray(verified["values"], np.float32)
            choice = {"new": int(new_prediction.argmax()),
                      "old": int(old_prediction.argmax()), "constant": constant}
            row = {"game": game, "episode": episode, "trial": trial,
                   "macro": macro, "values": values.tolist(),
                   "choice": choice,
                   "regret": {name: regret(values, action) for name, action in choice.items()},
                   "branch_steps": before - budget[0],
                   "terminal": False}
            rows.append(row)
            inputs.append(item["features"]["r0_21x16_2"])
            events.append(row)
        print(json.dumps({"game": game, "episode": episode,
                          "verified_roots": sum(r["episode"] == episode for r in rows),
                          "steps_used": MAX_STEPS - budget[0]}), flush=True)
    if not rows:
        raise RuntimeError("nenhum estado verificado")
    outcomes = {name: aggregate(rows, name) for name in ("new", "old", "constant")}
    rng = np.random.default_rng(7301)
    x = np.stack(inputs)
    shuffled_regrets = []
    for _ in range(32):
        altered = x.copy()
        for episode in range(EPISODES):
            indices = np.asarray([i for i, row in enumerate(rows) if row["episode"] == episode])
            altered[indices] = x[rng.permutation(indices)]
        forecasts = new.predict(altered)
        regrets = [regret(np.asarray(row["values"]), int(forecasts[i].argmax()))
                   for i, row in enumerate(rows)]
        valid = [r for r in regrets if r is not None]
        shuffled_regrets.append(float(np.mean(valid)) if valid else None)
    actual_valid = [row["regret"]["new"] for row in rows if row["regret"]["new"] is not None]
    actual_root_mean = float(np.mean(actual_valid))
    comparable = [ep for ep in range(EPISODES)
                  if all(outcomes[name]["per_episode"][ep] is not None
                         for name in ("new", "old", "constant"))]
    wins = sum(outcomes["new"]["per_episode"][ep]
               < min(outcomes["old"]["per_episode"][ep],
                     outcomes["constant"]["per_episode"][ep]) for ep in comparable)
    null_equal = sum(value is not None and value <= actual_root_mean
                     for value in shuffled_regrets)
    n_mean = outcomes["new"]["mean_of_informative_episodes"]
    o_mean = outcomes["old"]["mean_of_informative_episodes"]
    c_mean = outcomes["constant"]["mean_of_informative_episodes"]
    gate_breakout = bool(game == "breakout" and len(comparable) == EPISODES
                         and wins >= 6 and c_mean - n_mean >= 0.10
                         and o_mean - n_mean >= 0.10 and null_equal <= 1)
    pong_loss = bool(game == "pong" and n_mean - c_mean > 0.05)
    output = HERE / "resultados" / f"{game}_transferencia_ale.json"
    event_path = output.with_suffix(".jsonl")
    with event_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    report = {"game": game, "seed": seed,
              "new_training_bank_sha256": new_hash, "old_training_bank_sha256": old_hash,
              "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                for name in ("transferencia_ale.py", "reconstruir_desacordo.py",
                                             "avaliar.py", "coleta.py")},
              "verified_roots": len(rows), "terminal_candidates": int(sum(e["terminal"] for e in events)),
              "total_steps": MAX_STEPS - budget[0], "replay_checked": checked,
              "scores": outcomes, "wins_over_both_episodes": wins,
              "shuffled_regrets": shuffled_regrets,
              "shuffled_equal_or_better": null_equal,
              "actual_root_mean_regret": actual_root_mean,
              "gate_breakout": gate_breakout,
              "pong_loss": pong_loss,
              "events_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest()}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    result = run(args.game)
    print(json.dumps({k: v for k, v in result.items() if k != "shuffled_regrets"},
                     ensure_ascii=False, indent=2))
