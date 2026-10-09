"""Audita se estado histórico multidimensional muda decisões fora da amostra."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge, group_split
from prefixo import decision, predict

SIZES = (1, 2, 4, 8, 16)


def raw_features(history, mode):
    if mode == "history":
        # Exclui os últimos dois quadros já entregues ao estado base.
        older = history[:, :-2]
        return np.diff(older, axis=1).reshape(len(history), -1)
    if mode == "present":
        return np.concatenate((history[:, -1], history[:, -1] - history[:, -2]), axis=1)
    raise ValueError(mode)


def pca(raw, tr):
    center = raw[tr].mean(0)
    _, _, direction = np.linalg.svd(raw[tr] - center, full_matrices=False)
    return (raw - center) @ direction[:max(SIZES)].T


def augment(base, latent, tr, k):
    z = latent[:, :k]
    mean = z[tr].mean(0)
    std = np.maximum(z[tr].std(0), 1e-8)
    distances = KernelRidge.distances(base[tr], base[tr])
    scale = 0.5 * np.sqrt(max(float(np.median(distances)), 1e-8)) / np.sqrt(k)
    return np.concatenate((base, (z - mean) / std * scale), axis=1)


def select(base, raw, values, tr, val):
    baseline = predict(base, values, tr, val)
    baseline_decision = decision(values, baseline, val)
    if "normalized_regret" not in baseline_decision:
        return {"status": "validacao_sem_contraste"}, None
    latent = pca(raw, tr)
    possibilities = []
    for k in SIZES:
        representation = augment(base, latent, tr, k)
        prediction = predict(representation, values, tr, val)
        regret = decision(values, prediction, val)["normalized_regret"]
        mse = float(np.mean((values[val] - prediction) ** 2))
        possibilities.append((regret, mse, k))
    regret, mse, k = min(possibilities)
    return {"status": "selected", "dimension": k,
            "validation_base_regret": baseline_decision["normalized_regret"],
            "validation_memory_regret": regret,
            "validation_gain": baseline_decision["normalized_regret"] - regret,
            "all_validation": [{"k": kk, "regret": rr, "mse": ee}
                               for rr, ee, kk in possibilities]}, latent


def assess(base, latent, values, selection, tr, val, test):
    fit = np.sort(np.concatenate((tr, val)))
    baseline = predict(base, values, fit, test)
    output = {"selection": selection,
              "base_decision": decision(values, baseline, test),
              "base_mse": float(np.mean((values[test] - baseline) ** 2))}
    if selection["status"] != "selected":
        return output
    representation = augment(base, latent, tr, selection["dimension"])
    forecast = predict(representation, values, fit, test)
    output["augmented_decision"] = decision(values, forecast, test)
    output["augmented_mse"] = float(np.mean((values[test] - forecast) ** 2))
    if "normalized_regret" in output["base_decision"]:
        output["regret_gain"] = (output["base_decision"]["normalized_regret"]
                                 - output["augmented_decision"]["normalized_regret"])
    return output


def run(bank: Path, nulls=32):
    data = np.load(bank)
    history = np.asarray(data["histories"], np.float32)
    values = np.asarray(data["values"], np.float32)
    tr, val, test = group_split(data["episode"])
    base = raw_features(history, "present")
    results = {}
    for mode in ("history", "present"):
        choice, latent = select(base, raw_features(history, mode), values, tr, val)
        results[mode] = assess(base, latent, values, choice, tr, val, test)
    rng = np.random.default_rng(731)
    placebo = []
    for _ in range(nulls):
        shuffled = history.copy()
        for split in (tr, val, test):
            shuffled[split] = history[rng.permutation(split)]
        choice, latent = select(base, raw_features(shuffled, "history"), values, tr, val)
        placebo.append(assess(base, latent, values, choice, tr, val, test).get("regret_gain"))
    actual_gain = results["history"].get("regret_gain")
    at_least = (sum(p >= actual_gain for p in placebo if p is not None)
                if actual_gain is not None else None)
    return {"bank": str(bank), "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
            "domain": str(data["domain"]), "results": results,
            "null_regret_gains": placebo, "null_at_least_as_good": at_least,
            "null_count": nulls,
            "gate_local": bool(actual_gain is not None and actual_gain > 0
                               and results["history"]["selection"].get("validation_gain", 0) > 0
                               and actual_gain > results["present"].get("regret_gain", 0)
                               and at_least is not None and at_least <= 1)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    args = parser.parse_args()
    report = run(args.bank)
    args.bank.with_name(args.bank.stem + "_capacidade.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"domain": report["domain"], "results": report["results"],
                      "null_at_least_as_good": report["null_at_least_as_good"],
                      "gate_local": report["gate_local"]},
                     indent=2, ensure_ascii=False))
