"""Busca conjunta de uma pergunta causal e uma leitura de memória executável."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge, append_memory, group_split, memory_candidates, views


def candidates(history):
    found = list(memory_candidates(history))
    for early, late in ((2, 4), (4, 8), (8, 15)):
        for y in range(0, 21, 3):
            for x in range(0, 16, 4):
                a = history[:, -1 - early, y:y + 3, x:x + 4].mean((1, 2))
                b = history[:, -1 - late, y:y + 3, x:x + 4].mean((1, 2))
                yield ("diferenca", early, late, y, x), a - b
    for (lag, y, x), value in found:
        yield ("media", lag, y, x), value


def mse(actual, predicted):
    return float(np.mean((actual - predicted) ** 2))


def predict(x, target, tr, held):
    return KernelRidge().fit(x[tr], target[tr, None]).predict(x[held]).ravel()


def oof_residual(base, target, episode, tr):
    residual = np.zeros(len(tr), np.float64)
    for group in np.unique(episode[tr]):
        held = tr[episode[tr] == group]
        other = tr[episode[tr] != group]
        residual[np.searchsorted(tr, held)] = target[held] - predict(base, target, other, held)
    return residual


def target_for_question(images, q, tr):
    # A consequência é a diferença entre dois programas na MESMA situação.
    difference = images[:, q].reshape(len(images), -1) - images[:, 0].reshape(len(images), -1)
    centered = difference[tr] - difference[tr].mean(0)
    if float(np.linalg.norm(centered)) < 1e-8:
        return None
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    target = difference @ direction
    if float(target[tr].std()) < 1e-8:
        return None
    return target


def choose(base, history, images, episode, tr, val):
    """A seleção inteira usa somente construção e seleção; não lê episódios finais."""
    readings = list(candidates(history))
    best = None
    for q in range(1, 16):
        target = target_for_question(images, q, tr)
        if target is None:
            continue
        residual = oof_residual(base, target, episode, tr)
        if float(residual.std()) < 1e-10:
            continue
        ranked = []
        for spec, value in readings:
            if float(value[tr].std()) < 1e-8:
                continue
            correlation = float(np.corrcoef(value[tr], residual)[0, 1])
            if np.isfinite(correlation):
                ranked.append((abs(correlation), spec, value))
        ranked.sort(key=lambda row: -row[0])
        baseline = mse(target[val], predict(base, target, tr, val))
        if baseline < 1e-12:
            continue
        for _correlation, spec, value in ranked[:6]:
            augmented = append_memory(base, value, tr)
            altered = mse(target[val], predict(augmented, target, tr, val))
            improvement = (baseline - altered) / max(float(np.var(target[tr])), 1e-12)
            if best is None or improvement > best["selection_gain_scaled"]:
                best = {"question_id": q, "program": spec,
                        "selection_gain_scaled": float(improvement),
                        "selection_base_mse": baseline,
                        "selection_memory_mse": altered}
    return best


def evaluate_choice(data, selected, histories, tr, val, test):
    if selected is None:
        return {"status": "nenhum_par_pergunta_memoria"}
    base = views(data["histories"])["r0_21x16_2"]
    images = data["images"]
    q = selected["question_id"]
    target = target_for_question(images, q, tr)
    dictionary = dict(candidates(histories))
    value = dictionary[tuple(selected["program"])]
    fit = np.sort(np.concatenate([tr, val]))
    augmented = append_memory(base, value, fit)
    base_mse = mse(target[test], predict(base, target, fit, test))
    augmented_mse = mse(target[test], predict(augmented, target, fit, test))
    reward_difference = data["returns"][:, q] - data["returns"][:, 0]
    relevant = np.abs(reward_difference[test]) > 1e-8
    outcome = {"question_id": q, "program": selected["program"],
               "validation_gain_scaled": selected["selection_gain_scaled"],
               "test_base_mse": base_mse,
               "test_memory_mse": augmented_mse,
               "test_relative_gain": 1.0 - augmented_mse / max(base_mse, 1e-12),
               "reward_informative_roots": int(relevant.sum())}
    if relevant.any():
        old = predict(base, reward_difference, fit, test)
        new = predict(augmented, reward_difference, fit, test)
        truth = reward_difference[test][relevant] > 0
        outcome["reward_pair_choice"] = {
            "base_accuracy": float(np.mean((old[relevant] > 0) == truth)),
            "memory_accuracy": float(np.mean((new[relevant] > 0) == truth)),
            "count": int(relevant.sum())}
    else:
        outcome["reward_pair_choice"] = {"status": "sem_contraste_de_retorno"}
    return outcome


def run(bank: Path, nulls=32):
    data = np.load(bank)
    episode = data["episode"]
    tr, val, test = group_split(episode)
    f = views(data["histories"])
    base, history = f["r0_21x16_2"], f["c21_history"]
    chosen = choose(base, history, data["images"], episode, tr, val)
    actual = evaluate_choice(data, chosen, history, tr, val, test)
    rng = np.random.default_rng(811)
    placebo = []
    for _ in range(nulls):
        shuffled = history.copy()
        for split in (tr, val, test):
            shuffled[split] = history[rng.permutation(split)]
        null_choice = choose(base, shuffled, data["images"], episode, tr, val)
        null_result = evaluate_choice(data, null_choice, shuffled, tr, val, test)
        placebo.append(null_result.get("test_relative_gain"))
    finite = [p for p in placebo if p is not None and np.isfinite(p)]
    actual_gain = actual.get("test_relative_gain")
    return {"bank": str(bank), "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
            "split": {"construction": [0, 1, 2, 3], "selection": [4, 5],
                      "evaluation": [6, 7]},
            "selected": actual,
            "null": {"count": nulls, "test_relative_gains": placebo,
                     "mean": float(np.mean(finite)) if finite else None,
                     "at_least_as_good": (sum(p >= actual_gain for p in finite)
                                          if actual_gain is not None else None)},
            "interpretation_gate": bool(actual_gain is not None and actual_gain > 0
                                        and len(finite) == nulls
                                        and sum(p >= actual_gain for p in finite) <= 1
                                        and actual.get("reward_informative_roots", 0) >= 4
                                        and actual.get("reward_pair_choice", {}).get("memory_accuracy", 0)
                                        > actual.get("reward_pair_choice", {}).get("base_accuracy", 0))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    parser.add_argument("--nulls", type=int, default=32)
    args = parser.parse_args()
    result = run(args.bank, args.nulls)
    target = args.bank.with_name(args.bank.stem + "_diagnostico.json")
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected": result["selected"],
                      "null_at_least_as_good": result["null"]["at_least_as_good"],
                      "null_count": args.nulls,
                      "interpretation_gate": result["interpretation_gate"]},
                     indent=2, ensure_ascii=False))
