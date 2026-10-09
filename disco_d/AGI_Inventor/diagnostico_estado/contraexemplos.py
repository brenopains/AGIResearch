"""Compila um operador histórico a partir de erros de escolha de ação."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import append_memory, group_split, views
from prefixo import decision, predict

LAGS = (2, 4, 8, 15)
SPARSITIES = (4, 16, 64)


def source(history, spec):
    if spec[0] == "frame":
        return history[:, -1 - spec[1]].reshape(len(history), -1)
    if spec[0] == "difference":
        return (history[:, -1 - spec[1]] - history[:, -1 - spec[2]]).reshape(len(history), -1)
    if spec[0] == "current":
        return history[:, -1].reshape(len(history), -1)
    if spec[0] == "current_delta":
        return (history[:, -1] - history[:, -2]).reshape(len(history), -1)
    raise ValueError(spec)


def programs(mode):
    if mode == "current":
        return [("current",), ("current_delta",)]
    return [("frame", lag) for lag in LAGS] + [
        ("difference", a, b) for i, a in enumerate(LAGS) for b in LAGS[i + 1:]]


def training_errors(base, values, episode, tr):
    predicted = np.zeros((len(tr), 4), np.float64)
    for group in np.unique(episode[tr]):
        held = tr[episode[tr] == group]
        other = tr[episode[tr] != group]
        predicted[np.searchsorted(tr, held)] = predict(base, values, other, held)
    truth = values[tr]
    chosen = predicted.argmax(1)
    spread = np.ptp(truth, axis=1)
    regret = truth.max(1) - truth[np.arange(len(tr)), chosen]
    errors = (regret > 1e-8) & (spread > 1e-8)
    if int(errors.sum()) < 4:
        return predicted, regret, None
    residual = (truth[errors] - predicted[errors])
    _, _, vt = np.linalg.svd(residual, full_matrices=False)
    direction = vt[0]
    weights = np.zeros(len(tr), np.float64)
    weights[errors] = regret[errors] / spread[errors]
    target = ((truth - predicted) @ direction) * weights
    return predicted, regret, target


def synthesize(base, history, values, episode, tr, val, mode):
    _predicted, regret, target = training_errors(base, values, episode, tr)
    count = int(np.sum(regret > 1e-8))
    if target is None:
        return {"status": "contraexemplos_insuficientes", "train_counterexamples": count}
    baseline_prediction = predict(base, values, tr, val)
    base_decision = decision(values, baseline_prediction, val)
    if "normalized_regret" not in base_decision:
        return {"status": "validacao_sem_contraste", "train_counterexamples": count}
    baseline_regret = base_decision["normalized_regret"]
    best = None
    for operator in programs(mode):
        matrix = source(history, operator)
        mean = matrix[tr].mean(0)
        centered = matrix[tr] - mean
        covariance = centered.T @ target
        scale = (centered * centered).sum(0) + 1e-6
        scores = covariance / np.sqrt(scale)
        for k in SPARSITIES:
            indices = np.argsort(-np.abs(scores), kind="stable")[:k]
            coefficient = covariance[indices] / scale[indices]
            norm = float(np.linalg.norm(coefficient))
            if norm < 1e-10:
                continue
            coefficient = coefficient / norm
            z = (matrix[:, indices] - mean[indices]) @ coefficient
            if z[tr].std() < 1e-8:
                continue
            augmented = append_memory(base, z, tr)
            forecast = predict(augmented, values, tr, val)
            regret_val = decision(values, forecast, val)["normalized_regret"]
            mse_val = float(np.mean((values[val] - forecast) ** 2))
            rank = (regret_val, mse_val, k)
            if best is None or rank < best["rank"]:
                best = {"rank": rank,
                        "definition": {"operator": operator,
                                       "indices": indices.tolist(),
                                       "weights": coefficient.tolist(),
                                       "center": mean[indices].tolist()},
                        "validation_regret": regret_val,
                        "validation_mse": mse_val}
    if best is None:
        return {"status": "nenhum_operador_executavel", "train_counterexamples": count}
    return {"status": "candidata", "train_counterexamples": count,
            "validation_base_regret": baseline_regret,
            "validation_memory_regret": best["validation_regret"],
            "validation_gain": baseline_regret - best["validation_regret"],
            "definition": best["definition"]}


def execute(history, definition):
    matrix = source(history, definition["operator"])
    index = np.asarray(definition["indices"], dtype=int)
    weights = np.asarray(definition["weights"], dtype=np.float64)
    center = np.asarray(definition["center"], dtype=np.float64)
    return (matrix[:, index] - center) @ weights


def assess(base, history, values, chosen, tr, val, test):
    fit = np.sort(np.concatenate([tr, val]))
    plain = predict(base, values, fit, test)
    base_decision = decision(values, plain, test)
    out = {"status": chosen["status"],
           "train_counterexamples": chosen.get("train_counterexamples"),
           "base_decision": base_decision,
           "validation_gain": chosen.get("validation_gain")}
    if chosen["status"] != "candidata":
        return out
    z = execute(history, chosen["definition"])
    future = predict(append_memory(base, z, fit), values, fit, test)
    memory_decision = decision(values, future, test)
    out.update({"definition": chosen["definition"],
                "memory_decision": memory_decision,
                "base_value_mse": float(np.mean((values[test] - plain) ** 2)),
                "memory_value_mse": float(np.mean((values[test] - future) ** 2))})
    if "normalized_regret" in base_decision:
        out["regret_gain"] = (base_decision["normalized_regret"]
                              - memory_decision["normalized_regret"])
    return out


def run(bank: Path, nulls=32):
    data = np.load(bank)
    episode = data["episode"]
    tr, val, test = group_split(episode)
    f = views(data["histories"])
    base, history = f["r0_21x16_2"], f["c21_history"]
    values = data["returns"].reshape(-1, 4, 4).max(2)
    chosen = synthesize(base, history, values, episode, tr, val, "history")
    actual = assess(base, history, values, chosen, tr, val, test)
    current_choice = synthesize(base, history, values, episode, tr, val, "current")
    current = assess(base, history, values, current_choice, tr, val, test)
    rng = np.random.default_rng(1061)
    placebo = []
    for _ in range(nulls):
        shuffled = history.copy()
        for split in (tr, val, test):
            shuffled[split] = history[rng.permutation(split)]
        null_choice = synthesize(base, shuffled, values, episode, tr, val, "history")
        null_result = assess(base, shuffled, values, null_choice, tr, val, test)
        placebo.append(null_result.get("regret_gain"))
    gain = actual.get("regret_gain")
    at_least = (sum(x >= gain for x in placebo if x is not None)
                if gain is not None else None)
    return {"bank": str(bank), "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
            "actual": actual, "current_only_control": current,
            "null_regret_gains": placebo, "null_at_least_as_good": at_least,
            "null_count": nulls,
            "gate_local": bool(gain is not None and gain > 0
                               and chosen.get("validation_gain", 0) > 0
                               and chosen.get("train_counterexamples", 0) >= 4
                               and actual.get("base_decision", {}).get("informative_roots", 0) >= 4
                               and at_least is not None and at_least <= 1
                               and gain > current.get("regret_gain", 0))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    parser.add_argument("--nulls", type=int, default=32)
    args = parser.parse_args()
    output = run(args.bank, args.nulls)
    target = args.bank.with_name(args.bank.stem + "_contraexemplos.json")
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    def compact(item):
        return {k: v for k, v in item.items() if k != "definition"}
    print(json.dumps({"actual": compact(output["actual"]),
                      "current_only_control": compact(output["current_only_control"]),
                      "null_at_least_as_good": output["null_at_least_as_good"],
                      "null_count": output["null_count"],
                      "gate_local": output["gate_local"]},
                     indent=2, ensure_ascii=False))
