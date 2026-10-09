"""Atribuição exata do erro dos corpos congelados; intervenções são oráculos."""

import os
import platform
import sys
if sys.platform == "win32":
    platform.machine = platform._get_machine_win32
    platform.system = lambda: "Windows"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import argparse
import json
import time
import numpy as np
from piloto_transporte_funcoes import (ARMS, FROZEN_BODIES, HERE, RESULT as PRIOR_RESULT,
                                      load_bank, require, save_new, sha, verify)
from diagnosticar_semantica_prefixo import metrics
from transporte_funcoes import FunctionTransport

STEM = HERE / "resultados" / "decomposicao_transporte"
MANIFEST = STEM.with_name(STEM.name + "_manifesto.json")
RESULT, EVENTS = STEM.with_suffix(".json"), STEM.with_suffix(".jsonl")
QUERIES = ("composed", "real_histories_learned_reward_oracle",
           "reward_error_removed_oracle", "initial_history_real_termination_oracle", "zero")


def trajectory(body, phi, actions, rewards, valid):
    n = len(actions)
    require(phi.shape == (n + 1, len(body.initial_prior)) and rewards.shape == (n,) and valid.shape == (n + 1,), "trajetória incompatível")
    require(np.isfinite(phi).all() and np.isfinite(rewards).all(), "trajetória não finita")
    require(not np.any((~valid[:-1]) & valid[1:]) and not np.any(rewards[~valid[:-1]] != 0), "padding/término físico incoerente")
    if body.mode == "no_context":
        phi = np.broadcast_to(body.initial_prior, phi.shape)
    word = actions.tolist()
    predicted = float(phi[0] @ body.word_value(word))
    actual = float(rewards.sum())
    reward_error = transport_error = observed_readout = fixed_readout = 0.
    boundary = 0. if valid[0] else predicted
    steps = []
    for t, action in enumerate(word):
        if not valid[t]:
            continue
        matrix = np.eye(len(body.initial_prior)) if body.mode == "no_transport" else body.transport[action]
        reward_estimate = float(phi[t] @ body.reward[action])
        local_reward = reward_estimate - float(rewards[t])
        residual = phi[t] @ matrix - float(valid[t + 1]) * phi[t + 1]
        suffix = word[t + 1:]
        direction = body.word_value(suffix)
        local_transport = float(residual @ direction)
        reward_error += local_reward
        transport_error += local_transport
        observed_readout += reward_estimate
        fixed_readout += float(phi[0] @ body.reward[action])
        steps.append({"step": t, "action": action, "suffix": suffix,
                      "actual_reward": float(rewards[t]), "predicted_reward": reward_estimate,
                      "continuation": bool(valid[t + 1]), "reward_error": local_reward,
                      "transport_error_weighted": local_transport,
                      "raw_transport_residual_norm": float(np.linalg.norm(residual)),
                      "future_function_norm": float(np.linalg.norm(direction))})
    discrepancy = predicted - actual - reward_error - transport_error - boundary
    require(abs(discrepancy) < 1e-8, "decomposição telescópica diverge")
    require(abs(observed_readout - actual - reward_error) < 1e-8, "oráculo de leitura diverge")
    return {"predicted": predicted, "actual": actual, "reward_error": reward_error,
            "transport_error": transport_error, "boundary_error": boundary,
            "real_readout": observed_readout, "fixed_readout": fixed_readout,
            "identity_error": float(abs(discrepancy)), "steps": steps}


def summarize(rows):
    actual = [r["actual_delta"] for r in rows]
    scores = {q: metrics(actual, [r["predictions"][q] for r in rows]) for q in QUERIES}
    r, t, b = (np.asarray([row[f"{name}_error"] for row in rows]) for name in ("reward", "transport", "boundary"))
    terms = {"reward_squared": float(np.mean(r * r)), "transport_squared": float(np.mean(t * t)),
             "boundary_squared": float(np.mean(b * b)), "reward_transport_cross": float(np.mean(2 * r * t)),
             "reward_boundary_cross": float(np.mean(2 * r * b)), "transport_boundary_cross": float(np.mean(2 * t * b))}
    require(abs(sum(terms.values()) - scores["composed"]["mse"]) < 1e-8, "termos quadráticos não reconciliamMSE")
    return {"queries": scores, "mse_decomposition": terms,
            "maximum_identity_error": max(r["identity_error"] for r in rows)}


def prepare():
    require(not any(p.exists() for p in (MANIFEST, RESULT, EVENTS)), "diagnóstico existe; preservar")
    verify()
    frozen = json.loads(FROZEN_BODIES.read_text(encoding="utf-8"))
    sources = ["decompor_erros_transporte.py", "auditar_decomposicao_transporte.py",
               "testes_decomposicao_transporte.py", "DECOMPOSICAO_TRANSPORTE.md",
               "piloto_transporte_funcoes.py", "transporte_funcoes.py", "diagnosticar_semantica_prefixo.py",
               FROZEN_BODIES.relative_to(HERE).as_posix(), PRIOR_RESULT.relative_to(HERE).as_posix(),
               "resultados/transporte_funcoes_auditoria.json"]
    for p, expected in frozen["source_hashes"].items():
        require(sha(HERE / p) == expected, "corpo anterior mudou")
        sources.append(p)
    for game in ("breakout", "pong"):
        for ext in (".npz", ".json", ".jsonl"):
            sources.append(f"resultados/{game}_transicoes_mistas_103{ext}")
    value = {"status": "frozen_before_attribution", "development_only": True,
             "source_hashes": {p: sha(HERE / p) for p in sorted(set(sources))},
             "arms": [a for a in ARMS if a != "zero"], "horizons": [4, 8, 16],
             "queries": list(QUERIES), "all_roots": True, "no_fit": True, "no_ale": True,
             "previous_gate_unchanged": True}
    save_new(MANIFEST, value)
    return {"status": value["status"], "manifest_sha256": sha(MANIFEST)}


def verify_manifest():
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for p, expected in value["source_hashes"].items():
        require(sha(HERE / p) == expected, f"hash mudou: {p}")
    verify()
    return value


def run():
    manifest = verify_manifest()
    require(not RESULT.exists() and not EVENTS.exists(), "cálculo existe; preservar")
    wall, cpu = time.perf_counter(), time.process_time()
    frozen = json.loads(FROZEN_BODIES.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    events, reports, maximum = [], {}, 0.
    for game in ("breakout", "pong"):
        bank = load_bank(game, 103)
        reports[game] = {}
        for arm in manifest["arms"]:
            body = FunctionTransport.load(HERE / frozen["bodies"][game][arm])
            shape = bank["histories"].shape
            phi = body.features(bank["histories"].reshape(-1, *shape[3:])).reshape(*shape[:3], -1)
            rows = []
            for horizon in manifest["horizons"]:
                for i in range(32):
                    branches = [trajectory(body, phi[i, side, :horizon + 1], bank["actions"][i, side, :horizon],
                                           bank["rewards"][i, side, :horizon], bank["valid"][i, side, :horizon + 1]) for side in (0, 1)]
                    for side, branch in enumerate(branches):
                        maximum = max(maximum, branch["identity_error"])
                        for step in branch["steps"]:
                            events.append({"event": "transition_error_attributed", "game": game, "arm": arm,
                                           "horizon_blocks": horizon, "row": i, "side": side,
                                           "episode": int(bank["episode"][i]), **step})
                    delta = {field: branches[0][field] - branches[1][field] for field in
                             ("predicted", "actual", "reward_error", "transport_error", "boundary_error", "real_readout", "fixed_readout")}
                    predictions = {"composed": delta["predicted"], "real_histories_learned_reward_oracle": delta["real_readout"],
                                   "reward_error_removed_oracle": delta["predicted"] - delta["reward_error"],
                                   "initial_history_real_termination_oracle": delta["fixed_readout"], "zero": 0.}
                    error = abs(delta["predicted"] - delta["actual"] - delta["reward_error"] - delta["transport_error"] - delta["boundary_error"])
                    require(error < 1e-8, "erro de contraste não reconciliado")
                    row = {"event": "paired_attribution", "game": game, "arm": arm, "horizon_blocks": horizon,
                           "row": i, "root": int(bank["roots"][i]), "episode": int(bank["episode"][i]),
                           "actual_delta": delta["actual"], "reward_error": delta["reward_error"],
                           "transport_error": delta["transport_error"], "boundary_error": delta["boundary_error"],
                           "identity_error": error, "predictions": predictions}
                    rows.append(row)
                    events.append(row)
            curve = {str(t): summarize([r for r in rows if r["horizon_blocks"] == t]) for t in manifest["horizons"]}
            # The new analysis must preserve the original forecasts and metrics.
            for t in manifest["horizons"]:
                original = prior["games"][game]["horizons"][str(t)][arm]
                require(all(np.isclose(curve[str(t)]["queries"]["composed"][metric], original[metric], atol=1e-10, rtol=1e-9)
                            for metric in ("mse", "fair_tie_regret")), "predição anterior mudou")
            reports[game][arm] = {"curve": curve, "final_per_episode":
                {str(e): summarize([r for r in rows if r["horizon_blocks"] == 16 and r["episode"] == e]) for e in range(8)}}
        del bank
    with EVENTS.open("x", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    result = {"status": "development_attribution_completed_not_agent", "games": reports,
              "maximum_branch_identity_error": maximum, "event_count": len(events),
              "manifest_sha256": sha(MANIFEST), "events_sha256": sha(EVENTS),
              "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - wall,
              "previous_verdict": prior["status"], "no_promotion": True,
              "fit_calls": 0, "ale_calls": 0, "human_level_goal_achieved": False}
    save_new(RESULT, result)
    return {"status": result["status"], "wall_seconds": result["wall_seconds"], "event_count": len(events),
            "primary": {g: r["transport"]["curve"]["16"] for g, r in reports.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--congelar", action="store_true")
    parser.add_argument("--executar", action="store_true")
    args = parser.parse_args()
    require(args.congelar != args.executar, "escolher congelar ou executar")
    print(json.dumps(prepare() if args.congelar else run(), ensure_ascii=False, indent=2))
