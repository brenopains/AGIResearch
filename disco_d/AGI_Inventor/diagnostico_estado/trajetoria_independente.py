"""Confirma modelos congelados em checkpoints temporais sem seleção por modelo."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from coleta import gray, make_ale
from desacordo_ativo import check_replay, features_from_history, verify
from transferencia_ale import fit_models, regret

HERE = Path(__file__).resolve().parent
SEED = 97
EPISODES = 8
CHECKPOINTS = 8
INTERVAL = 128
MAX_STEPS = 170000
NULLS = 32


def summarize(rows, key):
    per_episode = []
    for episode in range(EPISODES):
        valid = [row["regret"][key] for row in rows
                 if row["episode"] == episode and row["regret"][key] is not None]
        per_episode.append(float(np.mean(valid)) if valid else None)
    used = [r for r in per_episode if r is not None]
    return {"per_episode": per_episode,
            "mean": float(np.mean(used)) if used else None,
            "informative_episodes": len(used)}


def run(game):
    model, old, constant, training_hash, old_hash = fit_models(game)
    ale = make_ale(game, SEED)
    actions = list(ale.getMinimalActionSet()[:4])
    rng = np.random.default_rng(SEED)
    budget = [MAX_STEPS]
    events, rows, inputs, histories, episodes, all_values = [], [], [], [], [], []
    replay_checked = False
    resets = 0

    for episode in range(EPISODES):
        ale.reset_game()
        start = gray(ale)
        history = deque([start.copy() for _ in range(16)], maxlen=16)
        for checkpoint in range(CHECKPOINTS):
            for _ in range(INTERVAL):
                if budget[0] <= 16 * 128 + 16:
                    raise RuntimeError("orçamento ALE insuficiente antes da verificação")
                ale.act(actions[int(rng.integers(0, 4))])
                budget[0] -= 1
                if ale.game_over():
                    ale.reset_game()
                    start = gray(ale)
                    history = deque([start.copy() for _ in range(16)], maxlen=16)
                    resets += 1
                else:
                    history.append(gray(ale))
            item = {"snapshot": ale.cloneSystemState(),
                    "features": features_from_history(history),
                    "votes": [0, 0, 0, 0], "disagreement": 0.0,
                    "macro": (episode, checkpoint)}
            if not replay_checked:
                check_replay(ale, actions, item, budget)
                replay_checked = True
            x = item["features"]["r0_21x16_2"]
            choices = {"new": int(model.predict(x[None])[0].argmax()),
                       "old": int(old.predict(x[None])[0].argmax()),
                       "constant": constant}
            before = budget[0]
            measured = verify(ale, actions, item, old, budget)
            # As ramificações são consultas; a trajetória principal continua
            # exatamente do snapshot anterior à consulta.
            ale.restoreSystemState(item["snapshot"])
            values = np.asarray(measured["values"], np.float32)
            row = {"game": game, "episode": episode, "checkpoint": checkpoint,
                   "values": values.tolist(), "choice": choices,
                   "regret": {key: regret(values, action)
                              for key, action in choices.items()},
                   "branch_steps": before - budget[0]}
            rows.append(row)
            events.append(row)
            inputs.append(x)
            histories.append(np.stack(history).astype(np.uint8))
            episodes.append(episode)
            all_values.append(values)
        print(json.dumps({"game": game, "episode": episode,
                          "roots": len(rows), "resets": resets,
                          "steps": MAX_STEPS - budget[0]}), flush=True)

    scores = {key: summarize(rows, key) for key in ("new", "old", "constant")}
    informative = [row for row in rows if row["regret"]["new"] is not None]
    challenge = [row for row in informative if row["regret"]["constant"] > 0]
    challenge_episodes = len(set(row["episode"] for row in challenge))
    x = np.stack(inputs)
    null_rng = np.random.default_rng(9701)
    null_scores = []
    for _ in range(NULLS):
        permuted = x.copy()
        for episode in range(EPISODES):
            ix = np.flatnonzero(np.asarray(episodes) == episode)
            permuted[ix] = x[null_rng.permutation(ix)]
        pred = model.predict(permuted)
        outcome = [regret(np.asarray(row["values"]), int(pred[i].argmax()))
                   for i, row in enumerate(rows)]
        usable = [v for v in outcome if v is not None]
        null_scores.append(float(np.mean(usable)))
    actual = float(np.mean([row["regret"]["new"] for row in informative]))
    null_equal_or_better = sum(score <= actual for score in null_scores)
    wins = sum(scores["new"]["per_episode"][episode]
               < scores["constant"]["per_episode"][episode]
               for episode in range(EPISODES))
    margin_adequate = bool(len(challenge) >= 12 and challenge_episodes >= 4)
    gate = bool(margin_adequate and scores["constant"]["mean"]
                - scores["new"]["mean"] >= 0.10 and wins >= 5
                and null_equal_or_better <= 1)

    stem = HERE / "resultados" / f"{game}_trajetoria_independente"
    bank = stem.with_suffix(".npz")
    np.savez_compressed(bank, histories=np.stack(histories),
                        values=np.stack(all_values), episode=np.asarray(episodes))
    event_path = stem.with_suffix(".jsonl")
    with event_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    report = {"game": game, "seed": SEED,
              "training_bank_sha256": training_hash,
              "old_bank_sha256": old_hash,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
              "events_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
              "roots": len(rows), "informative_roots": len(informative),
              "constant_fails": len(challenge), "challenge_episodes": challenge_episodes,
              "action_counts": {str(a): sum(row["choice"]["new"] == a for row in rows)
                                for a in range(4)},
              "resets": resets, "replay_checked": replay_checked,
              "steps": MAX_STEPS - budget[0], "scores": scores,
              "wins_over_constant_episodes": wins,
              "actual_root_mean_regret": actual,
              "null_root_mean_regrets": null_scores,
              "null_equal_or_better": null_equal_or_better,
              "margin_adequate": margin_adequate, "gate_local": gate}
    stem.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    result = run(args.game)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "null_root_mean_regrets"}, ensure_ascii=False, indent=2))
