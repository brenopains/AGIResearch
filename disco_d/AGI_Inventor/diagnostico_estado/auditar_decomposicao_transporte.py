"""Auditoria por diferenças entre valores de sufixos e produtos matriciais."""

import json
import numpy as np
from decompor_erros_transporte import (EVENTS, FROZEN_BODIES, HERE, MANIFEST, PRIOR_RESULT,
    QUERIES, RESULT, FunctionTransport, load_bank, require, save_new, sha, verify_manifest)
from auditar_semantica_prefixo import equal_report, scores


def summary(rows):
    target = [r["actual_delta"] for r in rows]
    result = {q: scores(target, [r["predictions"][q] for r in rows]) for q in QUERIES}
    parts = np.asarray([[r["reward_error"], r["transport_error"], r["boundary_error"]] for r in rows])
    gram = parts.T @ parts / len(parts)
    terms = {"reward_squared": float(gram[0, 0]), "transport_squared": float(gram[1, 1]),
             "boundary_squared": float(gram[2, 2]), "reward_transport_cross": float(2 * gram[0, 1]),
             "reward_boundary_cross": float(2 * gram[0, 2]), "transport_boundary_cross": float(2 * gram[1, 2])}
    return {"queries": result, "mse_decomposition": terms,
            "maximum_identity_error": max(r["identity_error"] for r in rows)}


def audit():
    manifest = verify_manifest()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_BODIES.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines()]
    require(result["manifest_sha256"] == sha(MANIFEST) and result["events_sha256"] == sha(EVENTS), "hash de resultado mudou")
    require(result["event_count"] == len(records) and result["fit_calls"] == result["ale_calls"] == 0, "contagem/escopo diverge")
    require(result["no_promotion"] and not result["human_level_goal_achieved"] and result["previous_verdict"] == prior["status"], "veredito anterior diverge")
    paired = [r for r in records if r["event"] == "paired_attribution"]
    steps = [r for r in records if r["event"] == "transition_error_attributed"]
    require(len(paired) == 2 * 5 * 3 * 32 and len(records) == len(paired) + len(steps), "consultas extras/ausentes")
    step_index = {(r["game"], r["arm"], r["horizon_blocks"], r["row"], r["side"], r["step"]): r for r in steps}
    require(len(step_index) == len(steps), "transição duplicada")
    maximum, step_count = 0., 0
    for game in ("breakout", "pong"):
        bank = load_bank(game, 103)
        for arm in manifest["arms"]:
            body = FunctionTransport.load(HERE / frozen["bodies"][game][arm])
            shape = bank["histories"].shape
            phi = body.features(bank["histories"].reshape(-1, *shape[3:])).reshape(*shape[:3], -1)
            if body.mode == "no_context":
                phi = np.broadcast_to(body.initial_prior, phi.shape)
            rows = [r for r in paired if r["game"] == game and r["arm"] == arm]
            for horizon in (4, 8, 16):
                group = [r for r in rows if r["horizon_blocks"] == horizon]
                require([r["row"] for r in group] == list(range(32)), "ordem/raiz de consulta diverge")
                for row in group:
                    i = row["row"]
                    require(row["root"] == int(bank["roots"][i]) and row["episode"] == int(bank["episode"][i]), "origem diverge")
                    branches = []
                    for side in (0, 1):
                        actions = bank["actions"][i, side, :horizon].tolist()
                        validity = bank["valid"][i, side, :horizon + 1]
                        rewards = bank["rewards"][i, side, :horizon]
                        require(not np.any((~validity[:-1]) & validity[1:]) and not np.any(rewards[~validity[:-1]] != 0), "física de término diverge")
                        values = [float(phi[i, side, t] @ body.augmented(actions[t:])[1:, 0]) for t in range(horizon + 1)]
                        raw_reward, t_sum, fixed = 0., 0., 0.
                        for t, action in enumerate(actions):
                            if not validity[t]:
                                continue
                            stored = step_index[(game, arm, horizon, i, side, t)]
                            reward_estimate = float(phi[i, side, t] @ body.reward[action])
                            # Different arithmetic: difference of adjacent scalar future values.
                            transport = values[t] - reward_estimate - float(validity[t + 1]) * values[t + 1]
                            raw_reward += reward_estimate
                            t_sum += transport
                            fixed += float(phi[i, side, 0] @ body.reward[action])
                            matrix = np.eye(len(body.initial_prior)) if body.mode == "no_transport" else body.transport[action]
                            residual = phi[i, side, t] @ matrix - float(validity[t + 1]) * phi[i, side, t + 1]
                            expected = {"step": t, "action": action, "suffix": actions[t + 1:],
                                        "actual_reward": float(rewards[t]), "predicted_reward": reward_estimate,
                                        "continuation": bool(validity[t + 1]), "reward_error": reward_estimate - float(rewards[t]),
                                        "transport_error_weighted": transport,
                                        "raw_transport_residual_norm": float(np.linalg.norm(residual)),
                                        "future_function_norm": float(np.linalg.norm(body.augmented(actions[t + 1:])[1:, 0]))}
                            equal_report(expected, {k: stored[k] for k in expected}, "transição")
                            maximum = max(maximum, abs(transport - stored["transport_error_weighted"]))
                            step_count += 1
                        actual = float(sum(map(float, rewards)))
                        branch = {"predicted": values[0], "actual": actual, "reward_error": raw_reward - actual,
                                  "transport_error": t_sum, "boundary_error": values[0] if not validity[0] else 0.,
                                  "real_readout": raw_reward, "fixed_readout": fixed}
                        require(abs(branch["predicted"] - actual - branch["reward_error"] - branch["transport_error"] - branch["boundary_error"]) < 1e-8, "identidade independente diverge")
                        branches.append(branch)
                    delta = {k: branches[0][k] - branches[1][k] for k in branches[0]}
                    expected = {"actual_delta": delta["actual"], "reward_error": delta["reward_error"],
                                "transport_error": delta["transport_error"], "boundary_error": delta["boundary_error"],
                                "predictions": {"composed": delta["predicted"], "real_histories_learned_reward_oracle": delta["real_readout"],
                                                "reward_error_removed_oracle": delta["predicted"] - delta["reward_error"],
                                                "initial_history_real_termination_oracle": delta["fixed_readout"], "zero": 0.}}
                    equal_report(expected, {k: row[k] for k in expected}, "par")
                equal_report(summary(group), result["games"][game][arm]["curve"][str(horizon)], "curva")
                for metric in ("mse", "fair_tie_regret"):
                    require(np.isclose(summary(group)["queries"]["composed"][metric], prior["games"][game]["horizons"][str(horizon)][arm][metric], atol=1e-10), "capacidade anterior mudou")
            for episode in range(8):
                group = [r for r in rows if r["horizon_blocks"] == 16 and r["episode"] == episode]
                equal_report(summary(group), result["games"][game][arm]["final_per_episode"][str(episode)], "episódio")
    require(step_count == len(steps), "transição extra/incompleta")
    output = {"status": "audit_passed", "paired_comparisons": len(paired), "attributed_steps": step_count,
              "maximum_adjacent_value_vs_vector_error": maximum, "result_sha256": sha(RESULT),
              "previous_gate_preserved": True, "development_only": True, "human_level_goal_achieved": False}
    save_new(HERE / "resultados" / "decomposicao_transporte_auditoria.json", output)
    return output


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
