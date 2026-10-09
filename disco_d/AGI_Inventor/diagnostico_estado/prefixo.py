"""Diagnóstico de valor: adivinhar prefixo, resolver sufixo finito."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge, append_memory, group_split, views
from ciclo import candidates


def predict(x, values, fit, held):
    return KernelRidge().fit(x[fit], values[fit]).predict(x[held])


def oof(base, values, episode, tr):
    residual = np.zeros((len(tr), 4), np.float64)
    for group in np.unique(episode[tr]):
        held = tr[episode[tr] == group]
        other = tr[episode[tr] != group]
        residual[np.searchsorted(tr, held)] = values[held] - predict(base, values, other, held)
    return residual


def choose(base, history, values, episode, tr, val):
    residual = oof(base, values, episode, tr)
    ranked = []
    for spec, z in candidates(history):
        if z[tr].std() < 1e-8:
            continue
        correlations = [abs(float(np.corrcoef(z[tr], residual[:, p])[0, 1]))
                        for p in range(4) if residual[:, p].std() > 1e-8]
        if correlations and np.isfinite(correlations).all():
            ranked.append((max(correlations), spec, z))
    ranked.sort(key=lambda row: -row[0])
    if not ranked:
        return {"status": "sem_leitura_historica"}
    base_error = float(np.mean((values[val] - predict(base, values, tr, val)) ** 2))
    checked = []
    for _corr, spec, z in ranked[:8]:
        augmented = append_memory(base, z, tr)
        error = float(np.mean((values[val] - predict(augmented, values, tr, val)) ** 2))
        checked.append((error, spec))
    error, spec = min(checked, key=lambda row: row[0])
    return {"status": "candidata", "program": spec,
            "selection_base_mse": base_error,
            "selection_memory_mse": error,
            "selection_gain": 1 - error / max(base_error, 1e-12)}


def decision(values, predicted, subset):
    truth = values[subset]
    spread = np.ptp(truth, axis=1)
    relevant = spread > 1e-8
    if not relevant.any():
        return {"informative_roots": 0, "status": "sem_contraste_de_valor"}
    chosen = predicted.argmax(1)
    received = truth[np.arange(len(subset)), chosen]
    regret = ((truth.max(1) - received) / np.maximum(spread, 1e-8))[relevant]
    random = ((truth.max(1) - truth.mean(1)) / np.maximum(spread, 1e-8))[relevant]
    return {"informative_roots": int(relevant.sum()),
            "normalized_regret": float(regret.mean()),
            "uniform_regret": float(random.mean())}


def evaluate(data, history, chosen, tr, val, test):
    values = data["returns"].reshape(-1, 4, 4).max(axis=2)
    base = views(data["histories"])["r0_21x16_2"]
    fit = np.sort(np.concatenate([tr, val]))
    baseline = predict(base, values, fit, test)
    out = {"program": chosen.get("program"),
           "selection_gain": chosen.get("selection_gain"),
           "base_value_mse": float(np.mean((values[test] - baseline) ** 2)),
           "base_decision": decision(values, baseline, test)}
    if chosen.get("status") != "candidata":
        return out
    z = dict(candidates(history))[tuple(chosen["program"])]
    augmented = append_memory(base, z, fit)
    forecast = predict(augmented, values, fit, test)
    out["memory_value_mse"] = float(np.mean((values[test] - forecast) ** 2))
    out["memory_decision"] = decision(values, forecast, test)
    out["value_mse_gain"] = 1 - out["memory_value_mse"] / max(out["base_value_mse"], 1e-12)
    if "normalized_regret" in out["base_decision"]:
        out["regret_gain"] = (out["base_decision"]["normalized_regret"]
                              - out["memory_decision"]["normalized_regret"])
    return out


def run(bank: Path, nulls=32):
    data = np.load(bank)
    episode = data["episode"]
    tr, val, test = group_split(episode)
    features = views(data["histories"])
    base, history = features["r0_21x16_2"], features["c21_history"]
    values = data["returns"].reshape(-1, 4, 4).max(axis=2)
    chosen = choose(base, history, values, episode, tr, val)
    actual = evaluate(data, history, chosen, tr, val, test)
    rng = np.random.default_rng(933)
    null_results = []
    for _ in range(nulls):
        shuffled = history.copy()
        for split in (tr, val, test):
            shuffled[split] = history[rng.permutation(split)]
        null_choice = choose(base, shuffled, values, episode, tr, val)
        null_results.append(evaluate(data, shuffled, null_choice, tr, val, test))
    placebo = [item.get("regret_gain") for item in null_results]
    observed = actual.get("regret_gain")
    at_least = (sum(x >= observed for x in placebo if x is not None)
                if observed is not None else None)
    return {"bank": str(bank), "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
            "actual": actual, "null_regret_gains": placebo,
            "null_at_least_as_good": at_least,
            "null_count": nulls,
            "gate_pass": bool(observed is not None and observed > 0
                              and at_least is not None and at_least <= 1
                              and chosen.get("selection_gain", 0) > 0)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    parser.add_argument("--nulls", type=int, default=32)
    args = parser.parse_args()
    output = run(args.bank, args.nulls)
    target = args.bank.with_name(args.bank.stem + "_prefixo.json")
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"actual": output["actual"],
                      "null_at_least_as_good": output["null_at_least_as_good"],
                      "null_count": output["null_count"],
                      "gate_pass": output["gate_pass"]},
                     indent=2, ensure_ascii=False))
