"""Aprender pesos de retorno mantendo funções e transportes congelados."""

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
from pathlib import Path
import numpy as np
from piloto_transporte_funcoes import FROZEN_BODIES as PRIOR_BODIES, HERE, load_bank, require, save_new, sha, transitions, verify
from diagnosticar_semantica_prefixo import metrics
from credito_sequencias_transportadas import fit_credit, save_body
from transporte_funcoes import FunctionTransport

STEM = HERE / "resultados" / "credito_sequencias"
PROPOSAL = STEM.with_name(STEM.name + "_proposta.json")
BODIES = STEM.with_name(STEM.name + "_corpos_manifesto.json")
RESULT, EVENTS = STEM.with_suffix(".json"), STEM.with_suffix(".jsonl")
FITTED = ("sequence_credit", "shuffled_targets", "no_context", "no_transport", "shuffled_histories")
ARMS = (*FITTED, "frozen_local_weights", "zero")


def prepare():
    require(not any(p.exists() for p in (PROPOSAL, BODIES, RESULT, EVENTS)), "piloto existe; preservar")
    verify()
    old = json.loads(PRIOR_BODIES.read_text(encoding="utf-8"))
    sources = ["credito_sequencias_transportadas.py", "piloto_credito_sequencias.py", "auditar_credito_sequencias.py",
               "testes_credito_sequencias.py", "CREDITO_SEQUENCIAS_TRANSPORTADAS.md",
               "transporte_funcoes.py", "piloto_transporte_funcoes.py", "diagnosticar_semantica_prefixo.py",
               PRIOR_BODIES.relative_to(HERE).as_posix(), "resultados/decomposicao_transporte.json",
               "resultados/decomposicao_transporte_auditoria.json"]
    for path, expected in old["source_hashes"].items():
        require(sha(HERE / path) == expected, "corpo anterior mudou")
        sources.append(path)
    for game in ("breakout", "pong"):
        for seed in (97, 103):
            for ext in (".npz", ".json", ".jsonl"):
                sources.append(f"resultados/{game}_transicoes_mistas_{seed}{ext}")
    value = {"status": "frozen_before_sequence_credit_fit", "development_only": True,
             "arms": list(ARMS), "primary": "sequence_credit", "alpha": .05,
             "train_prefixes": list(range(1, 17)), "evaluation_horizons": [4, 8, 16],
             "gate": "both_games_primary_lower_regret_and_delta_mse_than_each_other_arm_at16",
             "maximum_main_wall_seconds": 120, "no_ale": True,
             "source_hashes": {p: sha(HERE / p) for p in sorted(set(sources))}}
    save_new(PROPOSAL, value)
    return {"status": value["status"], "proposal_sha256": sha(PROPOSAL)}


def verify_proposal():
    value = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    for p, expected in value["source_hashes"].items():
        require(sha(HERE / p) == expected, f"hash mudou: {p}")
    verify()
    return value


def training(bank):
    h, left, right, target, length = [], [], [], [], []
    returns = bank["rewards"].cumsum(-1)
    for i in range(32):
        for t in range(1, 17):
            h.append(bank["histories"][i, 0, 0])
            left.append(bank["actions"][i, 0, :t].tolist())
            right.append(bank["actions"][i, 1, :t].tolist())
            target.append(float(returns[i, 0, t - 1] - returns[i, 1, t - 1]))
            length.append(t)
    return np.asarray(h), left, right, np.asarray(target), np.asarray(length)


def path_for(game, arm):
    return STEM.with_name(STEM.name + f"_{game}_97_{arm}.npz")


def run():
    proposal = verify_proposal()
    require(not any(p.exists() for p in (BODIES, RESULT, EVENTS)), "ajuste existe; preservar")
    wall, cpu = time.perf_counter(), time.process_time()
    old = json.loads(PRIOR_BODIES.read_text(encoding="utf-8"))
    models, paths, reports, fits, records = {}, {}, {}, {}, []
    for game in ("breakout", "pong"):
        bank = load_bank(game, 97)
        base = FunctionTransport.load(HERE / old["bodies"][game]["transport"])
        data = training(bank)
        models[game] = {"frozen_local_weights": base}
        paths[game], reports[game], fits[game] = {}, {}, {}
        for arm in FITTED:
            started = time.perf_counter()
            body, report, x, used_target = fit_credit(base, *data, arm=arm, alpha=proposal["alpha"])
            require(report["feature_body_unchanged"] and report["transport_body_unchanged"], "alterou metade congelada")
            path = path_for(game, arm)
            save_body(path, body, arm, report)
            fitting = path.with_name(path.stem + "_sistema.npz")
            require(not fitting.exists(), "sistema existe")
            np.savez_compressed(fitting, design=x, target=used_target, prior_weights=base.reward.reshape(-1), scale=report["feature_scale"], alpha=proposal["alpha"])
            restored = FunctionTransport.load(path)
            require(np.array_equal(body.comparison(data[0][:8], data[1][:8], data[2][:8]),
                                   restored.comparison(data[0][:8], data[1][:8], data[2][:8])), "exportação mudou resultados")
            report["wall_seconds"] = time.perf_counter() - started
            models[game][arm], paths[game][arm], fits[game][arm], reports[game][arm] = restored, path.relative_to(HERE).as_posix(), fitting.relative_to(HERE).as_posix(), report
            records.append({"event": "return_weight_body_fitted", "game": game, "arm": arm,
                            "parameters": paths[game][arm], "test_targets_used": 0, "training": report})
        del bank, data
    files = [PROPOSAL.relative_to(HERE).as_posix()]
    for game in paths:
        for arm in paths[game]:
            p = paths[game][arm]
            files.extend((p, Path(p).with_suffix(".json").as_posix(), fits[game][arm]))
    frozen = {"status": "new_bodies_frozen_before_this_run_loads103_targets", "development_only": True,
              "bodies": paths, "fitting_systems": fits, "train_reports": reports,
              "source_hashes": {p: sha(HERE / p) for p in files}}
    save_new(BODIES, frozen)
    games = {}
    for game in ("breakout", "pong"):
        require(time.perf_counter() - wall < proposal["maximum_main_wall_seconds"], "teto de piloto atingido; sem aprovação")
        bank = load_bank(game, 103)
        h = bank["histories"][:, 0, 0]
        report = {"horizons": {}, "real_state_local_reward_mse": {}}
        source, _, actions, real_reward, _ = transitions(bank)
        for arm, body in models[game].items():
            phi = body.features(source)
            if body.mode == "no_context":
                phi = np.broadcast_to(body.initial_prior, phi.shape)
            estimate = np.einsum("ij,ij->i", phi, body.reward[actions])
            report["real_state_local_reward_mse"][arm] = float(np.mean((estimate - real_reward) ** 2))
        for horizon in (4, 8, 16):
            left, right = bank["actions"][:, 0, :horizon].tolist(), bank["actions"][:, 1, :horizon].tolist()
            actual = bank["rewards"][:, 0, :horizon].sum(1) - bank["rewards"][:, 1, :horizon].sum(1)
            scores = {}
            for arm in ARMS:
                prediction = np.zeros(32) if arm == "zero" else models[game][arm].comparison(h, left, right)
                require(np.isfinite(prediction).all(), "previsão não finita")
                scores[arm] = metrics(actual, prediction)
                for i in range(32):
                    records.append({"event": "development_comparison", "game": game, "arm": arm,
                                    "horizon_blocks": horizon, "row": i, "root": int(bank["roots"][i]),
                                    "episode": int(bank["episode"][i]), "actual_delta": float(actual[i]), "predicted_delta": float(prediction[i])})
            report["horizons"][str(horizon)] = scores
        final = report["horizons"]["16"]
        report["gate_checks"] = {f"{metric}_vs_{arm}": bool(final["sequence_credit"][metric] < final[arm][metric] - 1e-12)
                                 for arm in ARMS if arm != "sequence_credit" for metric in ("mse", "fair_tie_regret")}
        report["development_gate_passed"] = all(report["gate_checks"].values())
        games[game] = report
        del bank, source
    with EVENTS.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    result = {"status": "development_gate_passed_not_agent" if all(g["development_gate_passed"] for g in games.values()) else "failed_development_gate",
              "games": games, "train_reports": reports, "source_body_manifest_sha256": sha(PRIOR_BODIES),
              "proposal_sha256": sha(PROPOSAL), "body_manifest_sha256": sha(BODIES), "events_sha256": sha(EVENTS),
              "event_count": len(records), "wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
              "fit_calls": 10, "ale_calls": 0, "human_level_goal_achieved": False, "development_only": True}
    require(result["wall_seconds"] < proposal["maximum_main_wall_seconds"], "teto excedido; sem aprovação")
    save_new(RESULT, result)
    return {"status": result["status"], "wall_seconds": result["wall_seconds"],
            "games": {g: {"gate": r["development_gate_passed"], "horizon16": r["horizons"]["16"],
                          "local_reward_mse": r["real_state_local_reward_mse"]} for g, r in games.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--congelar", action="store_true")
    parser.add_argument("--executar", action="store_true")
    args = parser.parse_args()
    require(args.congelar != args.executar, "escolher congelar ou executar")
    print(json.dumps(prepare() if args.congelar else run(), ensure_ascii=False, indent=2))
