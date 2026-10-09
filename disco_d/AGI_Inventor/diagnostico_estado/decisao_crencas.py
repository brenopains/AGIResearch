"""Calcula E[Q(estado)] sobre hipóteses visuais em vez de Q(E[estado])."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from avaliar import KernelRidge, group_split
from geral import _sensor_values
from hipoteses_estado import describe_hypothesis, filter_times
from oraculo import HERE, TIMES, metrics
from prefixo import decision


def replay(bank, seed):
    sys.path.insert(0, str(HERE.parent / "representacao"))
    import mundo_orbitas as world

    world.DUR = 5.0
    rng = np.random.default_rng(seed)
    camera = world.camera(rng)
    states, weights, check = [], [], []
    for episode in range(8):
        frames, _, _truth = world.episodio(rng, camera, "A" if episode < 6 else "B", True)
        beams = filter_times(frames, TIMES)
        for t in TIMES:
            beam = beams[t]
            x = np.stack([describe_hypothesis(hypothesis, t) for hypothesis in beam])
            log_weights = -np.asarray([hypothesis.cost for hypothesis in beam]) / 2
            w = np.exp(log_weights - log_weights.max())
            w /= w.sum()
            states.append(x)
            weights.append(w)
            check.append(_sensor_values(frames[t + 45]))
    data = np.load(bank)
    if not np.array_equal(np.asarray(check, np.float32), data["values"]):
        raise RuntimeError("replay divergiu do banco congelado")
    return states, weights


def predict_belief(model, states, weights, test, permute=False, rng=None):
    predictions = []
    for i in test:
        q = model.predict(states[i])
        w = weights[i]
        if permute:
            w = w[rng.permutation(len(w))]
        predictions.append(w @ q)
    return np.asarray(predictions)


def run(bank, seed, nulls=32):
    data = np.load(bank)
    states, weights = replay(bank, seed)
    values, episode = data["values"], data["episode"]
    h = data["histories"]
    pixels = np.concatenate((h[:, -1], h[:, -1] - h[:, -2]), axis=1)
    best = np.stack([x[0] for x in states])
    mean = np.stack([w @ x for w, x in zip(weights, states)])
    tr, val, test = group_split(episode)
    fit = np.sort(np.concatenate((tr, val)))
    best_model = KernelRidge().fit(best[fit], values[fit])
    mean_model = KernelRidge().fit(mean[fit], values[fit])
    pixel_model = KernelRidge().fit(pixels[fit], values[fit])
    predicted = {"pixels": pixel_model.predict(pixels[test]),
                 "melhor_estado": best_model.predict(best[test]),
                 "estado_medio": mean_model.predict(mean[test]),
                 "valor_medio": predict_belief(best_model, states, weights, test)}
    scores = {name: metrics(values, forecast, episode, test)
              for name, forecast in predicted.items()}
    action_best = predicted["melhor_estado"].argmax(axis=1)
    action_belief = predicted["valor_medio"].argmax(axis=1)
    changed = np.flatnonzero(action_best != action_belief)
    truth = values[test]
    spread = np.maximum(np.ptp(truth, axis=1), 1e-8)
    regret_best = (truth.max(axis=1) - truth[np.arange(len(test)), action_best]) / spread
    regret_belief = (truth.max(axis=1) - truth[np.arange(len(test)), action_belief]) / spread
    dissent = []
    for i in test:
        actions = best_model.predict(states[i]).argmax(axis=1)
        vote = np.bincount(actions, weights=weights[i], minlength=4)
        dissent.append(1 - float(vote.max()))
    dissent = np.asarray(dissent)
    rng = np.random.default_rng(3101)
    null_regret = []
    for _ in range(nulls):
        forecast = predict_belief(best_model, states, weights, test, True, rng)
        null_regret.append(decision(values, forecast, test)["normalized_regret"])
    observed = scores["valor_medio"]["overall"]["normalized_regret"]
    return {"seed": seed, "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
            "scores": scores, "actions_changed": int(len(changed)),
            "changed_improved": int(np.sum(regret_belief[changed] < regret_best[changed])),
            "changed_worsened": int(np.sum(regret_belief[changed] > regret_best[changed])),
            "decision_disagreement": {
                "mean_dissent_mass": float(dissent.mean()),
                "roots_over_tenth": int(np.sum(dissent > 0.1)),
                "roots_over_quarter": int(np.sum(dissent > 0.25)),
                "by_episode": {str(int(g)): float(dissent[episode[test] == g].mean())
                               for g in np.unique(episode[test])}},
            "null_weight_permutation_regret": null_regret,
            "null_equal_or_better": sum(x <= observed for x in null_regret),
            "null_count": nulls}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    result = run(args.bank, args.seed)
    target = args.bank.with_name(args.bank.stem + "_decisao_crencas.json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "null_weight_permutation_regret"},
                     ensure_ascii=False, indent=2))
