"""Piloto delimitado de transporte de funções; treino97, desenvolvimento103."""

import os
import platform
import sys
if sys.platform == "win32":
    platform.machine = platform._get_machine_win32
    platform.system = lambda: "Windows"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
from diagnosticar_semantica_prefixo import HERE, bank_path, metrics, require, save_new, sha
from experiencia_cruzada_v2 import validate_dependencies, verify_proposal
from transporte_funcoes import FunctionTransport

STEM = HERE / "resultados" / "transporte_funcoes"
PROPOSAL = STEM.with_name(STEM.name + "_proposta.json")
FROZEN_BODIES = STEM.with_name(STEM.name + "_corpos_manifesto.json")
RESULT = STEM.with_suffix(".json")
EVENTS = STEM.with_suffix(".jsonl")
ARMS = ("transport", "shuffled_targets", "no_context", "no_transport", "permuted_actions", "zero")


def verify():
    value = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    for p, expected in value["source_hashes"].items():
        require(sha(HERE / p) == expected, f"proposta/fonte mudou: {p}")
    verify_proposal()
    for game in ("breakout", "pong"):
        validate_dependencies(game)
    return value


def prepare():
    require(not any(p.exists() for p in (PROPOSAL, FROZEN_BODIES, RESULT, EVENTS)), "piloto existente; preservar")
    sources = ["transporte_funcoes.py", "piloto_transporte_funcoes.py", "auditar_transporte_funcoes.py",
               "testes_transporte_funcoes.py", "TRANSPORTE_FUNCOES.md", "diagnosticar_semantica_prefixo.py"]
    for game in ("breakout", "pong"):
        validate_dependencies(game)
        for seed in (97, 103):
            for ext in (".npz", ".json", ".jsonl"):
                sources.append(bank_path(game, seed).with_suffix(ext).relative_to(HERE).as_posix())
    value = {"status": "frozen_before_new_model_fit", "development_only": True,
             "train_seed": 97, "development_seed": 103, "primary": "transport",
             "arms": list(ARMS), "supports": 32, "alpha": .05, "actions": 4,
             "horizons": [4, 8, 16], "primary_horizon": 16, "maximum_main_wall_seconds": 120,
             "gate": "both_games_lower_fair_tie_regret_and_delta_mse_than_all_controls",
             "source_hashes": {s: sha(HERE / s) for s in sources}}
    save_new(PROPOSAL, value)
    return {"status": value["status"], "proposal_sha256": sha(PROPOSAL)}


def load_bank(game, seed):
    with np.load(bank_path(game, seed), allow_pickle=False) as archive:
        return {key: archive[key] for key in ("histories", "valid", "rewards", "actions", "roots", "episode")}


def transitions(bank):
    h, valid = bank["histories"], bank["valid"]
    select = valid[:, :, :-1]
    return (h[:, :, :-1][select], h[:, :, 1:][select], bank["actions"][select],
            bank["rewards"][select], valid[:, :, 1:][select].astype(float))


def body_path(game, arm):
    return STEM.with_name(STEM.name + f"_{game}_97_{arm}.npz")


def run():
    proposal = verify()
    require(not any(p.exists() for p in (FROZEN_BODIES, RESULT, EVENTS)), "piloto já iniciado; preservar")
    started, cpu = time.perf_counter(), time.process_time()
    models, train_reports, paths = {}, {}, {}
    # Both games finish training before any103 outcome is opened by this run.
    for game in ("breakout", "pong"):
        train = transitions(load_bank(game, 97))
        before = time.perf_counter()
        primary = FunctionTransport().fit(*train)
        shuffled = FunctionTransport().fit(*train, shuffled=True)
        model = {"transport": primary, "shuffled_targets": shuffled}
        for arm in ("no_context", "no_transport", "permuted_actions"):
            model[arm] = copy.deepcopy(primary)
            if arm == "permuted_actions":
                model[arm].reward = model[arm].reward[[2, 3, 0, 1]].copy()
                model[arm].transport = model[arm].transport[[2, 3, 0, 1]].copy()
                model[arm].action_counts = np.asarray(model[arm].action_counts)[[2, 3, 0, 1]].tolist()
                model[arm].local_residual_mse = np.asarray(model[arm].local_residual_mse)[[2, 3, 0, 1]].tolist()
            else:
                model[arm].mode = arm
        paths[game] = {}
        for arm, body in model.items():
            path = body_path(game, arm)
            body.save(path)
            restored = FunctionTransport.load(path)
            require(np.array_equal(body.predict(train[0][:4], [[0, 1]] * 4),
                                   restored.predict(train[0][:4], [[0, 1]] * 4)), "exportação mudou resultado")
            paths[game][arm] = path.relative_to(HERE).as_posix()
        models[game] = model
        train_reports[game] = {"observed_transitions": len(train[0]), "action_counts": primary.action_counts,
            "supports": len(primary.support), "feature_dimension": len(primary.initial_prior),
            "fit_calls": 2, "wall_seconds": time.perf_counter() - before,
            "training_local_joint_mse": primary.local_residual_mse,
            "operator_spectral_norms": [float(np.linalg.norm(m, 2)) for m in primary.transport]}
        del train
    frozen_files = [PROPOSAL.relative_to(HERE).as_posix()]
    for models_for_game in paths.values():
        for p in models_for_game.values():
            frozen_files.extend((p, Path(p).with_suffix(".json").as_posix()))
    frozen = {"status": "bodies_frozen_before_this_run_loads103_targets", "development_only": True,
              "source_hashes": {p: sha(HERE / p) for p in frozen_files}, "bodies": paths,
              "train_reports": train_reports}
    save_new(FROZEN_BODIES, frozen)
    rows, game_reports = [], {}
    for game in ("breakout", "pong"):
        require(time.perf_counter() - started < proposal["maximum_main_wall_seconds"], "teto de piloto atingido; nenhuma aprovação")
        bank = load_bank(game, 103)
        history, programs = bank["histories"][:, 0, 0], bank["actions"]
        require(np.array_equal(history, bank["histories"][:, 1, 0]), "histórias iniciais dos pares divergem")
        source, successor, actions, reward, continuation = transitions(bank)
        phi = models[game]["transport"].features(source)
        next_phi = models[game]["transport"].features(successor)
        local_reward, local_transport = [], []
        for i, action in enumerate(actions):
            local_reward.append(float(phi[i] @ models[game]["transport"].reward[action] - reward[i]))
            local_transport.append(phi[i] @ models[game]["transport"].transport[action] - continuation[i] * next_phi[i])
        report = {"local_real_predecessor_reward_mse": float(np.mean(np.asarray(local_reward) ** 2)),
                  "local_real_predecessor_transport_mse": float(np.mean(np.asarray(local_transport) ** 2)),
                  "local_transitions": len(actions), "horizons": {}}
        for horizon in proposal["horizons"]:
            left, right = programs[:, 0, :horizon].tolist(), programs[:, 1, :horizon].tolist()
            actual = bank["rewards"][:, 0, :horizon].sum(1) - bank["rewards"][:, 1, :horizon].sum(1)
            scores = {}
            for arm in ARMS:
                predicted = np.zeros(len(history)) if arm == "zero" else models[game][arm].comparison(history, left, right)
                require(np.isfinite(predicted).all(), "composição não finita; nenhuma aprovação")
                scores[arm] = metrics(actual, predicted)
                for i in range(len(history)):
                    rows.append({"event": "development_prediction", "game": game, "arm": arm,
                                 "horizon_blocks": horizon, "row": i, "root": int(bank["roots"][i]),
                                 "episode": int(bank["episode"][i]), "actual_delta": float(actual[i]),
                                 "predicted_delta": float(predicted[i])})
            report["horizons"][str(horizon)] = scores
        final = report["horizons"]["16"]
        report["gate_checks"] = {f"{metric}_vs_{arm}": bool(final["transport"][metric] < final[arm][metric] - 1e-12)
                                 for arm in ARMS if arm != "transport" for metric in ("fair_tie_regret", "mse")}
        report["development_gate_passed"] = all(report["gate_checks"].values())
        game_reports[game] = report
        del bank, source, successor, phi, next_phi
    with EVENTS.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    result = {"status": "development_gate_passed_not_agent" if all(v["development_gate_passed"] for v in game_reports.values()) else "failed_development_gate",
              "games": game_reports, "train_reports": train_reports, "development_only": True,
              "ale_calls": 0, "fit_calls": 4, "human_level_goal_achieved": False,
              "proposal_sha256": sha(PROPOSAL), "frozen_bodies_sha256": sha(FROZEN_BODIES),
              "events_sha256": sha(EVENTS), "event_count": len(rows),
              "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - started}
    require(result["wall_seconds"] < proposal["maximum_main_wall_seconds"], "piloto excedeu teto; preservar corpos, sem aprovação")
    save_new(RESULT, result)
    return {"status": result["status"], "wall_seconds": result["wall_seconds"], "fit_calls": result["fit_calls"],
            "games": {game: {"gate": body["development_gate_passed"], "horizon16": body["horizons"]["16"],
                             "local_reward_mse": body["local_real_predecessor_reward_mse"],
                             "local_transport_mse": body["local_real_predecessor_transport_mse"]} for game, body in game_reports.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--congelar", action="store_true")
    parser.add_argument("--executar", action="store_true")
    args = parser.parse_args()
    require(args.congelar != args.executar, "escolher congelar ou executar")
    print(json.dumps(prepare() if args.congelar else run(), ensure_ascii=False, indent=2))
