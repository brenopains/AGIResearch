"""Seleciona memória por melhoria de decisão, não por perda visual ou MSE."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import append_memory, group_split, views
from ciclo import candidates
from prefixo import decision, evaluate, oof, predict


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
    baseline_pred = predict(base, values, tr, val)
    baseline_decision = decision(values, baseline_pred, val)
    if "normalized_regret" not in baseline_decision:
        return {"status": "sem_contraste_de_valor_na_validacao"}
    scored = []
    for _corr, spec, z in ranked[:32]:
        pred = predict(append_memory(base, z, tr), values, tr, val)
        regret = decision(values, pred, val)["normalized_regret"]
        error = float(np.mean((values[val] - pred) ** 2))
        scored.append((regret, error, spec))
    if not scored:
        return {"status": "nenhuma_variavel_candidata"}
    regret, _error, spec = min(scored, key=lambda row: (row[0], row[1]))
    return {"status": "candidata", "program": spec,
            "validation_base_regret": baseline_decision["normalized_regret"],
            "validation_memory_regret": regret,
            "validation_gain": baseline_decision["normalized_regret"] - regret}


def run(bank: Path, nulls=32):
    data = np.load(bank)
    episode = data["episode"]
    tr, val, test = group_split(episode)
    feature = views(data["histories"])
    base, history = feature["r0_21x16_2"], feature["c21_history"]
    values = data["returns"].reshape(-1, 4, 4).max(axis=2)
    selected = choose(base, history, values, episode, tr, val)
    actual = evaluate(data, history, selected, tr, val, test)
    actual["validation_gain"] = selected.get("validation_gain")
    rng = np.random.default_rng(934)
    placebo = []
    for _ in range(nulls):
        shuffled = history.copy()
        for split in (tr, val, test):
            shuffled[split] = history[rng.permutation(split)]
        null_choice = choose(base, shuffled, values, episode, tr, val)
        placebo.append(evaluate(data, shuffled, null_choice, tr, val, test).get("regret_gain"))
    observed = actual.get("regret_gain")
    at_least = (sum(x >= observed for x in placebo if x is not None)
                if observed is not None else None)
    return {"bank": str(bank), "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
            "actual": actual, "null_regret_gains": placebo,
            "null_at_least_as_good": at_least,
            "null_count": nulls,
            "gate_pass": bool(observed is not None and observed > 0
                              and selected.get("validation_gain", 0) > 0
                              and at_least is not None and at_least <= 1)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    parser.add_argument("--nulls", type=int, default=32)
    args = parser.parse_args()
    output = run(args.bank, args.nulls)
    target = args.bank.with_name(args.bank.stem + "_decisao.json")
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"actual": output["actual"],
                      "null_at_least_as_good": output["null_at_least_as_good"],
                      "null_count": output["null_count"],
                      "gate_pass": output["gate_pass"]},
                     indent=2, ensure_ascii=False))
