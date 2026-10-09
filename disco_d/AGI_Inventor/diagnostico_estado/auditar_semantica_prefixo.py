"""Recontagem independente: seleção, cancelamento, corpos, curvas e métricas."""

import os
import platform
import sys
if sys.platform == "win32":
    platform.machine = platform._get_machine_win32
    platform.system = lambda: "Windows"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import json
from pathlib import Path
import numpy as np
from diagnosticar_semantica_prefixo import (ARMS, EVENTS, FREEZE, HERE, QUERIES,
    RESULT, bank_path, fitted_model, require, save_new, sha, verify_freeze)


def prefix(a, b):
    differing = np.flatnonzero(a != b)
    return int(differing[0]) if len(differing) else len(a)


def encode(words):
    # No call to the experiment's word_features implementation.
    result = np.zeros((len(words), 16, 5))
    for i, word in enumerate(words):
        for t in range(16):
            result[i, t, int(word[t]) if t < len(word) else 4] = 1
    return result.reshape(len(words), -1)


def distance(a, b):
    return np.maximum(np.einsum("ij,ij->i", a, a)[:, None]
                      + np.einsum("ij,ij->i", b, b)[None, :] - 2 * a @ b.T, 0)


def reexecute(model, arm, h, left, right):
    if arm == "zero":
        return np.zeros(len(h))
    if arm != "raw_rbf":
        # Evaluate the dual kernel, not the emitted ReadingBody used by the runner.
        return model.predict_dual(h, left, right)
    x = h.reshape(len(h), -1).astype(float)
    p, q = encode(left), encode(right)
    kp = np.exp(-distance(p, model.p) / model.word_scale)
    kq = np.exp(-distance(p, model.q) / model.word_scale)
    rp = np.exp(-distance(q, model.p) / model.word_scale)
    rq = np.exp(-distance(q, model.q) / model.word_scale)
    state = np.exp(-distance(x, model.x) / model.state_scale)
    n = len(model.x)
    return (state * (kp - kq - rp + rq)) @ (model.coeff[:n] - model.coeff[n:])


def scores(target, predicted):
    errors, costs, fair, accuracy = [], [], [], []
    ties = real = 0
    for actual, prediction in zip(target, predicted):
        tied = abs(prediction) <= 1e-12
        ties += int(tied)
        cost = max(-actual, 0) if prediction >= -1e-12 else max(actual, 0)
        costs.append(cost)
        fair.append(abs(actual) / 2 if tied else cost)
        errors.append((prediction - actual) ** 2)
        if abs(actual) > 1e-12:
            real += 1
            accuracy.append(.5 if tied else float((prediction > 0) == (actual > 0)))
    return {"count": len(target), "real_nonzero": real, "prediction_ties": ties,
            "mse": float(np.mean(errors)), "deterministic_left_tie_regret": float(np.mean(costs)),
            "fair_tie_regret": float(np.mean(fair)),
            "directional_accuracy_on_real_nonzero": float(np.mean(accuracy)) if real else None}


def summary(rows):
    target = [r["actual_delta"] for r in rows]
    predictions = {q: [r["predictions"][q] for r in rows] for q in QUERIES}
    defects = [abs(a - b) for a, b in zip(predictions[QUERIES[0]], predictions[QUERIES[2]])]
    metrics = {q: scores(target, p) for q, p in predictions.items()}
    return {"queries": metrics, "mean_absolute_consistency_defect": float(np.mean(defects)),
            "maximum_absolute_consistency_defect": float(max(defects)),
            "choice_disagreement_full_vs_oracle": sum((a >= -1e-12) != (b >= -1e-12)
                for a, b in zip(predictions[QUERIES[0]], predictions[QUERIES[2]])),
            "oracle_fair_tie_gain_over_full": metrics[QUERIES[0]]["fair_tie_regret"] - metrics[QUERIES[2]]["fair_tie_regret"],
            "oracle_fair_tie_gain_over_suffix_initial": metrics[QUERIES[1]]["fair_tie_regret"] - metrics[QUERIES[2]]["fair_tie_regret"]}


def equal_report(a, b, label):
    require(type(a) == type(b) or isinstance(a, (int, float)) and isinstance(b, (int, float)), f"tipo diverge: {label}")
    if isinstance(a, dict):
        require(a.keys() == b.keys(), f"campos divergem: {label}")
        for key in a:
            equal_report(a[key], b[key], f"{label}/{key}")
    elif isinstance(a, list):
        require(len(a) == len(b), f"tamanho diverge: {label}")
        for i, (x, y) in enumerate(zip(a, b)):
            equal_report(x, y, f"{label}/{i}")
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        require(bool(np.isclose(a, b, rtol=1e-9, atol=1e-10)), f"valor diverge: {label}")
    else:
        require(a == b, f"valor diverge: {label}")


def audit():
    freeze = verify_freeze()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines()]
    require(result["freeze_sha256"] == sha(FREEZE) and result["events_sha256"] == sha(EVENTS), "hash de resultado diverge")
    require(len(events) == result["event_count"], "contagem de eventos diverge")
    require(result["ale_calls"] == result["fit_calls"] == 0 and not result["human_level_goal_achieved"], "escopo diverge")
    maximum, evaluated, physical, excluded = 0., 0, 0, 0
    for game in ("breakout", "pong"):
        models = {a: fitted_model(HERE / freeze["models"][game][a], a) for a in ARMS}
        for seed in (97, 103):
            key = f"{game}_{seed}"
            partition = result["partitions"][key]
            with np.load(bank_path(game, seed), allow_pickle=False) as archive:
                bank = {n: archive[n] for n in ("actions", "histories", "valid", "rewards", "roots", "episode")}
            lengths = [prefix(a, b) for a, b in bank["actions"]]
            selected = [i for i, k in enumerate(lengths) if 0 < k < 16]
            require(selected == freeze["selections"][key]["rows"], "seleção por sintaxe diverge")
            require(lengths == freeze["selections"][key]["all_prefix_lengths"], "prefixos divergem")
            pe = [e for e in events if e["partition"] == key]
            expected_cases = []
            for i, k in enumerate(lengths):
                if i not in selected:
                    matching = [e for e in pe if e["event"] == "excluded_by_syntax" and e["row"] == i]
                    require(len(matching) == 1 and matching[0]["prefix_length"] == k, "exclusão diverge")
                    excluded += 1
                    continue
                h, r, v = bank["histories"][i], bank["rewards"][i], bank["valid"][i]
                require(np.array_equal(h[0, :k + 1], h[1, :k + 1]), "história física de prefixo diverge")
                require(np.array_equal(v[0, :k + 1], v[1, :k + 1]) and np.array_equal(r[0, :k], r[1, :k]), "prefixo físico diverge")
                for side in (0, 1):
                    require(not any(v[side, t + 1] and not v[side, t] for t in range(16)), "validade revive")
                    require(not any(r[side, t] != 0 and not v[side, t] for t in range(16)), "recompensa após término")
                delta = float(sum(map(float, r[0])) - sum(map(float, r[1])))
                suffix = float(sum(map(float, r[0, k:])) - sum(map(float, r[1, k:])))
                require(delta == suffix, "cancelamento físico diverge")
                case = {"row": i, "root": int(bank["roots"][i]), "episode": int(bank["episode"][i]),
                        "prefix_length": k, "left": bank["actions"][i, 0].tolist(), "right": bank["actions"][i, 1].tolist(),
                        "actual_delta": delta, "actual_suffix_delta": suffix,
                        "terminal_at_midpoint": not bool(v[0, k]), "terminal_at_end": [not bool(x) for x in v[:, -1]],
                        "physical_identity_error": 0.}
                matching = [e for e in pe if e["event"] == "physical_identity_verified" and e["row"] == i]
                require(len(matching) == 1, "evento físico ausente/duplicado")
                equal_report(case, {k: matching[0][k] for k in case}, "caso/evento")
                expected_cases.append(case)
                physical += 1
            equal_report(expected_cases, partition["cases"], "casos")
            require(partition["eligible_roots"] == len(selected) and partition["excluded_roots"] == 32 - len(selected), "contagem física diverge")
            require(partition["real_nonzero_at_16"] == sum(abs(c["actual_delta"]) > 1e-12 for c in expected_cases), "contrastes divergem")
            require(partition["prefix_lengths"] == sorted(set(lengths[i] for i in selected)), "prefixos resumidos divergem")
            require(partition["episodes"] == sorted(set(c["episode"] for c in expected_cases)), "episódios divergem")
            require(partition["terminal_midpoints"] == sum(c["terminal_at_midpoint"] for c in expected_cases), "términos divergem")
            for arm in (*ARMS, "zero"):
                rows = [e for e in pe if e["event"] == "frozen_prediction" and e["arm"] == arm]
                require(len(rows) == sum(16 - lengths[i] for i in selected), "previsões ausentes/duplicadas")
                require(len({(e["row"], e["horizon_blocks"]) for e in rows}) == len(rows), "consulta duplicada")
                for t in sorted(set(e["horizon_blocks"] for e in rows)):
                    group = [e for e in rows if e["horizon_blocks"] == t]
                    ids = [e["row"] for e in group]
                    require(ids == [i for i in selected if lengths[i] < t], "consulta indevida/fora de ordem")
                    initial = np.stack([bank["histories"][i, 0, 0] for i in ids])
                    mid = np.stack([bank["histories"][i, 0, lengths[i]] for i in ids])
                    left = [bank["actions"][i, 0, :t].tolist() for i in ids]
                    right = [bank["actions"][i, 1, :t].tolist() for i in ids]
                    sl = [p[lengths[i]:] for p, i in zip(left, ids)]
                    sr = [p[lengths[i]:] for p, i in zip(right, ids)]
                    for q, (h, l, r) in zip(QUERIES, ((initial, left, right), (initial, sl, sr), (mid, sl, sr))):
                        prediction = reexecute(models.get(arm), arm, h, l, r)
                        recorded = np.asarray([e["predictions"][q] for e in group])
                        error = float(np.max(np.abs(prediction - recorded)))
                        maximum = max(maximum, error)
                        require(error < 1e-8, "forma independente do corpo diverge")
                        evaluated += len(group)
                    for e in group:
                        i, k = e["row"], lengths[e["row"]]
                        target = float(sum(map(float, bank["rewards"][i, 0, :t])) - sum(map(float, bank["rewards"][i, 1, :t])))
                        require(e["actual_delta"] == e["actual_suffix_delta"] == target and e["prefix_length"] == k, "alvo da curva diverge")
                    equal_report(summary(group), partition["arms"][arm]["curve"][str(t)], "curva")
                final = [e for e in rows if e["horizon_blocks"] == 16]
                equal_report(summary(final), partition["arms"][arm]["final"], "final")
                for episode in partition["episodes"]:
                    equal_report(summary([e for e in final if e["episode"] == episode]),
                                 partition["arms"][arm]["per_episode"][str(episode)], "episódio")
    require(len(events) == physical + excluded + evaluated // 3, "eventos extras/incompletos")
    output = {"status": "audit_passed", "physical_cases": physical, "syntax_exclusions": excluded,
              "scalar_predictions_reexecuted": evaluated, "maximum_independent_body_error": maximum,
              "results_sha256": sha(RESULT), "freeze_sha256": sha(FREEZE), "development_only": True,
              "human_level_goal_achieved": False}
    save_new(HERE / "resultados" / "semantica_prefixo_comum_auditoria.json", output)
    return output


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
