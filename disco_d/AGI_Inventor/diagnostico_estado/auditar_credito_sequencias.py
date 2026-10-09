"""Crédito refeito por produtos de matrizes, sistema linear e controles."""

import json
import numpy as np
from piloto_credito_sequencias import (ARMS, BODIES, EVENTS, FITTED, HERE, PRIOR_BODIES, PROPOSAL, RESULT,
    FunctionTransport, load_bank, require, save_new, sha, training, transitions, verify_proposal)
from auditar_transporte_funcoes import execute_matrix
from auditar_semantica_prefixo import equal_report, scores


def credit_matrix(body, history, words):
    phi = body.features(history)
    if body.mode == "no_context":
        phi = np.broadcast_to(body.initial_prior, phi.shape)
    width = phi.shape[1]
    result = np.zeros((len(phi), body.actions, width))
    for i, word in enumerate(words):
        prefix = np.eye(width)
        for action in word:
            result[i, action] += phi[i] @ prefix
            if body.mode != "no_transport":
                prefix = prefix @ body.transport[action]
    return result.reshape(len(phi), -1)


def audit():
    proposal = verify_proposal()
    frozen = json.loads(BODIES.read_text(encoding="utf-8"))
    old = json.loads(PRIOR_BODIES.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines()]
    for path, expected in frozen["source_hashes"].items():
        require(sha(HERE / path) == expected, "corpo/sistema mudou")
    require(result["proposal_sha256"] == sha(PROPOSAL) and result["body_manifest_sha256"] == sha(BODIES)
            and result["source_body_manifest_sha256"] == sha(PRIOR_BODIES) and result["events_sha256"] == sha(EVENTS), "hash de resultado diverge")
    require(len(records) == result["event_count"] == 1354 and result["fit_calls"] == 10 and result["ale_calls"] == 0, "contagem/orçamento diverge")
    require(not result["human_level_goal_achieved"] and result["development_only"] and result["wall_seconds"] < proposal["maximum_main_wall_seconds"], "escopo/teto diverge")
    maximum_design = maximum_system = maximum_value = 0.
    for game in ("breakout", "pong"):
        base = FunctionTransport.load(HERE / old["bodies"][game]["transport"])
        train = training(load_bank(game, 97))
        bodies = {a: FunctionTransport.load(HERE / p) for a, p in frozen["bodies"][game].items()}
        bodies["frozen_local_weights"] = base
        for arm in FITTED:
            body = bodies[arm]
            require(np.array_equal(body.support, base.support) and np.array_equal(body.transport, base.transport)
                    and body.scale == base.scale and np.array_equal(body.initial_prior, base.initial_prior), "mudou metade congelada")
            h, left, right, target, lengths = train
            target = target.copy()
            if arm == "shuffled_histories":
                order = np.random.default_rng(9873).permutation(32)
                h = h.reshape(32, 16, *h.shape[1:])[order].reshape(h.shape)
            if arm == "shuffled_targets":
                rng = np.random.default_rng(9871)
                identical = np.asarray([a == b for a, b in zip(left, right)])
                for length in range(1, 17):
                    for same in (False, True):
                        selected = np.flatnonzero((lengths == length) & (identical == same))
                        target[selected] = target[rng.permutation(selected)]
            x = credit_matrix(body, h, left) - credit_matrix(body, h, right)
            with np.load(HERE / frozen["fitting_systems"][game][arm], allow_pickle=False) as system:
                recorded, used_target, prior = system["design"], system["target"], system["prior_weights"]
                scale, alpha = float(system["scale"]), float(system["alpha"])
            error = float(np.max(np.abs(recorded - x)))
            maximum_design = max(maximum_design, error)
            require(error < 1e-8 and np.array_equal(target, used_target) and np.array_equal(prior, base.reward.reshape(-1)), "crédito/alvo/prior diverge")
            correction = (body.reward.reshape(-1) - prior) * scale
            residual = (2 * recorded.T @ recorded / scale ** 2 + alpha * np.eye(recorded.shape[1])) @ correction
            residual -= 2 * recorded.T @ (used_target - recorded @ prior) / scale
            equation_error = float(np.max(np.abs(residual)))
            maximum_system = max(maximum_system, equation_error)
            require(equation_error < 1e-8, "equação normal do sistema não fecha")
            report = result["train_reports"][game][arm]
            require(np.isclose(np.mean((x @ body.reward.reshape(-1) - target) ** 2), report["training_delta_mse"]), "erro de treino diverge")
            minimum, _, rank, _ = np.linalg.lstsq(recorded, used_target, rcond=1e-10)
            require(int(rank) == report["numerical_rank_rcond_1e_10"] and
                    np.isclose(np.mean((recorded @ minimum - used_target) ** 2), report["unregularized_training_delta_mse_rcond_1e_10"]), "diagnóstico de família diverge")
        del train
        bank = load_bank(game, 103)
        source, _, actions, real_reward, _ = transitions(bank)
        for arm, body in bodies.items():
            phi = body.features(source)
            if body.mode == "no_context":
                phi = np.broadcast_to(body.initial_prior, phi.shape)
            estimated = np.asarray([float(p @ body.reward[int(a)]) for p, a in zip(phi, actions)])
            require(np.isclose(np.mean((estimated - real_reward) ** 2), result["games"][game]["real_state_local_reward_mse"][arm]), "leitura local diverge")
        for horizon in (4, 8, 16):
            actual = bank["rewards"][:, 0, :horizon].sum(1) - bank["rewards"][:, 1, :horizon].sum(1)
            left, right = bank["actions"][:, 0, :horizon].tolist(), bank["actions"][:, 1, :horizon].tolist()
            score_rows = result["games"][game]["horizons"][str(horizon)]
            for arm in ARMS:
                prediction = np.zeros(32) if arm == "zero" else execute_matrix(bodies[arm], bank["histories"][:, 0, 0], left) - execute_matrix(bodies[arm], bank["histories"][:, 0, 0], right)
                group = [r for r in records if r["event"] == "development_comparison" and r["game"] == game
                         and r["arm"] == arm and r["horizon_blocks"] == horizon]
                require([r["row"] for r in group] == list(range(32)), "comparações incompletas/duplicadas")
                recorded = np.asarray([r["predicted_delta"] for r in group])
                error = float(np.max(np.abs(prediction - recorded)))
                maximum_value = max(maximum_value, error)
                require(error < 1e-8 and np.array_equal(actual, [r["actual_delta"] for r in group]), "previsão/resultado diverge")
                equal_report(scores(actual, recorded), score_rows[arm], "métricas")
            if horizon == 16:
                checks = {f"{metric}_vs_{arm}": bool(score_rows["sequence_credit"][metric] < score_rows[arm][metric] - 1e-12)
                          for arm in ARMS if arm != "sequence_credit" for metric in ("mse", "fair_tie_regret")}
                require(checks == result["games"][game]["gate_checks"] and all(checks.values()) == result["games"][game]["development_gate_passed"], "portão diverge")
        del bank
    fit_events = [r for r in records if r["event"] == "return_weight_body_fitted"]
    require(len(fit_events) == 10 and all(r["test_targets_used"] == 0 for r in fit_events), "eventos de ajuste divergem")
    passed = all(g["development_gate_passed"] for g in result["games"].values())
    require(result["status"] == ("development_gate_passed_not_agent" if passed else "failed_development_gate"), "veredito conjunto diverge")
    output = {"status": "audit_passed", "event_count": len(records), "maximum_independent_credit_error": maximum_design,
              "maximum_normal_equation_residual": maximum_system, "maximum_matrix_body_error": maximum_value,
              "verdict_preserved": result["status"], "result_sha256": sha(RESULT),
              "development_only": True, "human_level_goal_achieved": False}
    save_new(HERE / "resultados" / "credito_sequencias_auditoria.json", output)
    return output


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
