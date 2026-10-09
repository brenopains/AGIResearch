"""Reexecuta corpos pela forma de matriz aumentada e reconcilia métricas."""

import json
import numpy as np
from piloto_transporte_funcoes import (ARMS, EVENTS, FROZEN_BODIES, HERE, PROPOSAL, RESULT,
                                      load_bank, require, save_new, sha, transitions, verify)
from transporte_funcoes import FunctionTransport
from auditar_semantica_prefixo import equal_report, scores


def execute_matrix(body, history, words):
    phi = body.features(history)
    if body.mode == "no_context":
        phi = np.broadcast_to(body.initial_prior, phi.shape)
    result = []
    for state, word in zip(phi, words):
        accumulator = np.concatenate(([0.], state))
        result.append(float((accumulator @ body.augmented(word))[0]))
    return np.asarray(result)


def audit():
    proposal = verify()
    frozen = json.loads(FROZEN_BODIES.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    for p, expected in frozen["source_hashes"].items():
        require(sha(HERE / p) == expected, "corpo/proposta congelado mudou")
    require(result["frozen_bodies_sha256"] == sha(FROZEN_BODIES) and result["proposal_sha256"] == sha(PROPOSAL), "hash de manifesto mudou")
    require(result["events_sha256"] == sha(EVENTS), "eventos mudaram")
    rows = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines()]
    require(len(rows) == result["event_count"] == 1152, "contagem de eventos diverge")
    require(result["fit_calls"] == 4 and result["ale_calls"] == 0 and not result["human_level_goal_achieved"], "escopo diverge")
    require(result["wall_seconds"] < proposal["maximum_main_wall_seconds"], "teto excedido")
    maximum = 0.
    for game in ("breakout", "pong"):
        bodies = {arm: FunctionTransport.load(HERE / p) for arm, p in frozen["bodies"][game].items()}
        bank = load_bank(game, 103)
        source, successor, actions, reward, continuation = transitions(bank)
        body = bodies["transport"]
        phi, successor_phi = body.features(source), body.features(successor)
        reward_errors = np.einsum("ij,ij->i", phi, body.reward[actions]) - reward
        transport_errors = np.einsum("ni,nij->nj", phi, body.transport[actions]) - continuation[:, None] * successor_phi
        require(np.isclose(np.mean(reward_errors ** 2), result["games"][game]["local_real_predecessor_reward_mse"]), "erro local de recompensa diverge")
        require(np.isclose(np.mean(transport_errors ** 2), result["games"][game]["local_real_predecessor_transport_mse"]), "erro local de transporte diverge")
        primary = bodies["transport"]
        for other in ("no_context", "no_transport"):
            require(np.array_equal(primary.reward, bodies[other].reward) and np.array_equal(primary.transport, bodies[other].transport), "controle alterou treino")
        require(np.array_equal(primary.reward[[2, 3, 0, 1]], bodies["permuted_actions"].reward) and
                np.array_equal(primary.transport[[2, 3, 0, 1]], bodies["permuted_actions"].transport), "permutação diverge")
        computed_gate = {}
        for horizon in (4, 8, 16):
            recorded_scores = result["games"][game]["horizons"][str(horizon)]
            actual = np.asarray([float(sum(map(float, pair[0, :horizon])) - sum(map(float, pair[1, :horizon]))) for pair in bank["rewards"]])
            left, right = bank["actions"][:, 0, :horizon].tolist(), bank["actions"][:, 1, :horizon].tolist()
            history = bank["histories"][:, 0, 0]
            for arm in ARMS:
                predicted = np.zeros(len(history)) if arm == "zero" else execute_matrix(bodies[arm], history, left) - execute_matrix(bodies[arm], history, right)
                group = [r for r in rows if r["game"] == game and r["horizon_blocks"] == horizon and r["arm"] == arm]
                require([r["row"] for r in group] == list(range(32)), "consultas ausentes/duplicadas")
                recorded = np.asarray([r["predicted_delta"] for r in group])
                require(np.array_equal(actual, [r["actual_delta"] for r in group]), "alvos divergem")
                error = float(np.max(np.abs(predicted - recorded)))
                maximum = max(maximum, error)
                require(error < 1e-8, "produto aumentado e recursão divergem")
                # Score recorded equivalent values, avoiding machine-roundoff tie flips.
                equal_report(scores(actual, recorded), recorded_scores[arm], "métricas")
            if horizon == 16:
                computed_gate = {f"{metric}_vs_{arm}": bool(recorded_scores["transport"][metric] < recorded_scores[arm][metric] - 1e-12)
                                 for arm in ARMS if arm != "transport" for metric in ("fair_tie_regret", "mse")}
        require(computed_gate == result["games"][game]["gate_checks"] and
                all(computed_gate.values()) == result["games"][game]["development_gate_passed"], "portão diverge")
    passed = all(g["development_gate_passed"] for g in result["games"].values())
    require(result["status"] == ("development_gate_passed_not_agent" if passed else "failed_development_gate"), "veredito conjunto diverge")
    report = {"status": "audit_passed", "event_count": len(rows), "maximum_matrix_recursion_error": maximum,
              "verdict_preserved": result["status"], "result_sha256": sha(RESULT),
              "development_only": True, "human_level_goal_achieved": False}
    save_new(HERE / "resultados" / "transporte_funcoes_auditoria.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
