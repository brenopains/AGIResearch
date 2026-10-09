"""Diagnóstico exploratório LOSO de trilhas de mudanças versus pixels crus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from auditar_constante import normalize_regret
from avaliar import KernelRidge, views
from transferencia_ale import regret
from variaveis_movimento import batch

HERE = Path(__file__).resolve().parent


def score_fold(y, episode, choices):
    rr = [regret(y[i], int(choices[j])) for j, i in enumerate(episode)]
    good = [v for v in rr if v is not None]
    return float(np.mean(good)) if good else None


def evaluate(game):
    path = HERE / "resultados" / f"{game}_trajetoria_independente.npz"
    data = np.load(path)
    history, values, groups = data["histories"], data["values"], data["episode"]
    pixel = views(history)["r0_21x16_2"]
    motion = batch(history)
    slot_indices = np.r_[0:36, 60:96]
    representations = {"pixels": pixel, "trilhas": motion,
                       "trilhas_sem_relacoes": motion[:, slot_indices]}
    scores = {name: [] for name in (*representations, "constante")}
    predictions = {name: np.zeros(len(values), np.int8) for name in scores}
    models = []
    for held in range(8):
        fit = np.flatnonzero(groups != held)
        test = np.flatnonzero(groups == held)
        constant = min(range(4), key=lambda action:
                       (normalize_regret(values, action, fit), action))
        baseline = np.full(len(test), constant, np.int8)
        scores["constante"].append(score_fold(values, test, baseline))
        predictions["constante"][test] = baseline
        for name, x in representations.items():
            model = KernelRidge().fit(x[fit], values[fit])
            choice = model.predict(x[test]).argmax(1)
            scores[name].append(score_fold(values, test, choice))
            predictions[name][test] = choice
            if name == "trilhas":
                models.append((test, model))
    rng = np.random.default_rng(9707)
    null_means = []
    for _ in range(32):
        episode_regrets = []
        for test, model in models:
            permuted = rng.permutation(test)
            choice = model.predict(motion[permuted]).argmax(1)
            episode_regrets.append(score_fold(values, test, choice))
        valid = [v for v in episode_regrets if v is not None]
        null_means.append(float(np.mean(valid)))
    summary = {}
    for name, per_episode in scores.items():
        valid = [v for v in per_episode if v is not None]
        summary[name] = {"per_episode": per_episode,
                         "mean": float(np.mean(valid)) if valid else None,
                         "actions": {str(a): int(np.sum(predictions[name] == a))
                                     for a in range(4)}}
    motion_mean = summary["trilhas"]["mean"]
    beats_baselines = sum(
        scores["trilhas"][i] is not None
        and scores["trilhas"][i] < min(scores["pixels"][i], scores["constante"][i])
        for i in range(8))
    report = {"status": "exploratorio_banco_ja_inspecionado",
              "game": game, "seed": 97,
              "bank_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                for name in ("avaliar_movimento.py", "variaveis_movimento.py")},
              "motion_feature_dimension": int(motion.shape[1]),
              "scores": summary,
              "beats_both_episodes": beats_baselines,
              "shuffled_test_motion_mean": null_means,
              "shuffled_equal_or_better": sum(v <= motion_mean for v in null_means)}
    output = HERE / "resultados" / f"{game}_avaliacao_movimento.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    result = evaluate(args.game)
    print(json.dumps({"game": args.game,
                      "scores": result["scores"],
                      "beats_both_episodes": result["beats_both_episodes"],
                      "shuffled_equal_or_better": result["shuffled_equal_or_better"]},
                     ensure_ascii=False, indent=2))
