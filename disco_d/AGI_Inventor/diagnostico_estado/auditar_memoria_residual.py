"""Reconciliar histórias causais, custos, corpos e portão diagnóstico."""

import json

import numpy as np

from avaliar_memoria_residual import load_data, manifest_path, paired_gain, stem
from coletar_transicoes_mistas import stem as bank_stem
from confirmar_futuros_distribuidos import verify
from confirmar_leituras_variaveis import verify_manifest
from memoria_residual import MODES, MemoryDynamics
from piloto_planos_propostos import HERE, sha


def audit_collection(game, seed):
    path = bank_stem(game, seed)
    # NpzFile decompresses an array on every subscription; materialize once
    # before checking overlaps so audit work stays linear in bank size.
    with np.load(path.with_suffix(".npz")) as archive:
        bank = {name: archive[name] for name in archive.files}
    report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    for suffix, field in ((".npz", "bank_sha256"), (".jsonl", "events_sha256")):
        if sha(path.with_suffix(suffix)) != report[field]:
            raise RuntimeError("banco/eventos da coleta mudaram")
    if (sha(HERE / "coletar_transicoes_mistas.py") != report["source_sha256"]
            or sha(HERE / "MEMORIA_RESIDUAL.md") != report["protocol_sha256"]):
        raise RuntimeError("coletor/protocolo mudaram")
    plans_path = path.with_name(path.name + "_planos_v2.json")
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    if sha(plans_path) != report["plans_sha256"] or plans["source_sha256"] != report["source_sha256"]:
        raise RuntimeError("planos/coletor não correspondem")
    if not np.array_equal(np.asarray(plans["plans"])[bank["roots"]], bank["actions"]):
        raise RuntimeError("sequências medidas divergem das propostas")
    first_attempt = path.with_name(path.name + "_planos.json")
    preserved = None
    if first_attempt.exists():
        old = json.loads(first_attempt.read_text(encoding="utf-8"))
        if old["plans"] != plans["plans"] or sha(HERE / "coletar_transicoes_mistas_tentativa1.py") != old["source_sha256"]:
            raise RuntimeError("planos ou código da tentativa interrompida mudaram")
        preserved = True
    events = [json.loads(line) for line in path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    if len(events) != 104 or len(bank["roots"]) != 32:
        raise RuntimeError("coleta incompleta")
    roots = {int(root): i for i, root in enumerate(bank["roots"])}
    seen_paths, seen_roots, replay_roots = set(), set(), set()
    short = np.load(HERE / "resultados" / f"{game}_efeitos_locais_{seed}.npz")
    checked_overlaps = 0
    for event in events:
        root = event["root"]
        if event["event"] == "root_probes":
            if root in seen_roots or event["physical_steps"] != int(short["lengths"][root].sum()):
                raise RuntimeError("sonda inicial duplicada/custo divergente")
            seen_roots.add(root)
        elif event["event"] == "trajectory_replay":
            if root in replay_roots or root % 8:
                raise RuntimeError("replay duplicado ou raiz errada")
            replay_roots.add(root)
        elif event["event"] == "mixed_trajectory":
            i, j = roots[root], event["path"]
            if (root, j) in seen_paths or event["history_includes_probe_frames"]:
                raise RuntimeError("trajetória duplicada/contaminada")
            seen_paths.add((root, j))
            if (not np.array_equal(event["valid"], bank["valid"][i, j])
                    or not np.array_equal(event["rewards_by_block"], bank["rewards"][i, j])
                    or not np.array_equal(event["plan"], bank["actions"][i, j])):
                raise RuntimeError("trajetória medida diverge do banco")
            if event["physical_steps"] != event["trajectory_steps"] + event["probe_steps"]:
                raise RuntimeError("custo de sondas não contabilizado")
            valid = bank["valid"][i, j]
            if not valid[0] or np.any(valid[1:] > valid[:-1]):
                raise RuntimeError("trajetória reiniciada depois de término")
            alive_blocks = int(valid[:-1].sum())
            if valid[-1]:
                if event["trajectory_steps"] != 128:
                    raise RuntimeError("comprimento sem término incorreto")
            else:
                final_length = event["trajectory_steps"] - 8 * (alive_blocks - 1)
                if not 1 <= final_length <= 8:
                    raise RuntimeError("bloco terminal com comprimento inválido")
            for t in range(16):
                current, following = bank["histories"][i, j, t], bank["histories"][i, j, t + 1]
                if valid[t]:
                    used = 8 if valid[t + 1] else final_length
                    if not np.array_equal(following[:-used], current[used:]):
                        raise RuntimeError("história não desloca somente quadros reais passados")
                    checked_overlaps += 1
                elif (bank["rewards"][i, j, t] != 0 or not np.array_equal(current, following)
                      or np.any(bank["signatures"][i, j, t + 1])):
                    raise RuntimeError("estado após término não absorvido")
        else:
            raise RuntimeError("evento desconhecido")
    if len(seen_paths) != 64 or len(seen_roots) != 32 or len(replay_roots) != 8:
        raise RuntimeError("trajetórias/raízes/replays faltantes")
    recomputed = 8192 + sum(e["physical_steps"] for e in events)
    if recomputed != report["steps_used"] or recomputed > 70000:
        raise RuntimeError("custo total divergente")
    for name, expected in report["existing_bank_hashes"].items():
        if sha(HERE / "resultados" / name) != expected:
            raise RuntimeError("banco anterior mudou")
    if (report["certified_walk_roots"], report["certified_root_probes"], report["certified_replays"]) != (64, 128, 8):
        raise RuntimeError("certificações da coleta incompletas")
    if checked_overlaps != report["observed_valid_transitions"]:
        raise RuntimeError("contagem de transições divergente")
    return {"game": game, "seed": seed, "events": len(events), "steps_recomputed": recomputed,
            "history_overlaps_checked": checked_overlaps, "aborted_first_attempt_preserved": preserved}


def audit_evaluation(game):
    manifest = json.loads(manifest_path(game).read_text(encoding="utf-8"))
    for path, expected in manifest["source_hashes"].items():
        if sha(HERE / path) != expected:
            raise RuntimeError(f"corpo/fonte/treino mudou: {path}")
    path = stem(game).with_name(stem(game).name + "_103.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    if sha(manifest_path(game)) != report["manifest_sha256"]:
        raise RuntimeError("manifesto mudou")
    bank, z, ids, collection = load_data(game, 103)
    episode, observed = bank["episode"], bank["rewards"].cumsum(-1)
    events = [json.loads(line) for line in path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    if len(events) != 192:
        raise RuntimeError("avaliação incompleta")
    seen, forecast = set(), {mode: np.zeros((32, 2, 16)) for mode in MODES}
    root_index = {int(root): i for i, root in enumerate(bank["roots"])}
    for event in events:
        mode, root = event["mode"], event["root"]
        key = (mode, root)
        if key in seen or event["future_inputs_during_rollout"]:
            raise RuntimeError("evento duplicado ou futuro usado na composição")
        seen.add(key)
        i = root_index[root]
        if not np.array_equal(event["observed_returns"], observed[i]):
            raise RuntimeError("retorno observado diverge do banco físico")
        forecast[mode][i] = event["predicted_returns"]
    errors, regrets, max_reexecution_error = {}, {}, 0.0
    for mode in MODES:
        model = MemoryDynamics.load(HERE / manifest["bodies"][mode])
        if model.counts != manifest["cases_by_action"][mode] or sum(model.counts) != manifest["training_cases"]:
            raise RuntimeError("contagens do corpo e treino divergem")
        root = model.state(z[:, :, 0].reshape(64, 32), bank["histories"][:, :, 0].reshape(64, 16, 336),
                           ids[:, :, 0].reshape(64))
        reexecuted = model.rollout(root, bank["actions"].reshape(64, 16))["returns"].reshape(32, 2, 16)
        difference = float(np.max(np.abs(reexecuted - forecast[mode])))
        max_reexecution_error = max(max_reexecution_error, difference)
        if difference > 1e-10:
            raise RuntimeError("corpo salvo não reproduz eventos da avaliação")
        for horizon in (32, 64, 128):
            t, row = horizon // 8 - 1, report["curves"][mode][str(horizon)]
            pred, real = forecast[mode][:, :, t], observed[:, :, t]
            contrast_error = ((pred[:, 0] - pred[:, 1]) - (real[:, 0] - real[:, 1])) ** 2
            choice = pred.argmax(1)
            regret = real.max(1) - real[np.arange(32), choice]
            recomputed = {"return_mse": float(np.mean((pred - real) ** 2)), "return_bias": float((pred - real).mean()),
                          "contrast_mse": float(contrast_error.mean()), "mean_regret": float(regret.mean())}
            if any(not np.isclose(row[k], v, atol=1e-12, rtol=0) for k, v in recomputed.items()):
                raise RuntimeError("métricas de retorno divergem")
            if row["choices"] != choice.tolist() or row["per_root_regret"] != regret.tolist():
                raise RuntimeError("decisão diverge dos retornos")
            if horizon == 128:
                errors[mode], regrets[mode] = contrast_error, regret
    checks = {}
    for mode in MODES:
        if mode == "residual_history":
            continue
        contrast = paired_gain(errors["residual_history"], errors[mode], episode)
        decision = paired_gain(regrets["residual_history"], regrets[mode], episode)
        if report["comparisons"][mode] != {"contrast": contrast, "decision": decision}:
            raise RuntimeError("comparação pareada divergente")
        checks[f"contrast_mean_vs_{mode}"] = contrast["mean"] > 1e-12
        checks[f"contrast_lower_vs_{mode}"] = contrast["bootstrap_95"][0] > 1e-12
        checks[f"contrast_two_episodes_vs_{mode}"] = contrast["positive_episodes"] >= 2
        checks[f"decision_mean_vs_{mode}"] = decision["mean"] > 1e-12
    checks["no_regression_vs_state_only"] = report["comparisons"]["state_only"]["decision"]["regressed_episodes"] == 0
    for mode in ("state_only", "shuffled_residuals"):
        checks[f"local_reward_vs_{mode}"] = report["local"]["residual_history"]["reward_mse"] < report["local"][mode]["reward_mse"] - 1e-12
    if checks != report["checks"] or bool(all(checks.values())) != report["positive_diagnostic"]:
        raise RuntimeError("portão diverge das medidas")
    if report["promoted"] or report["human_level_goal_achieved"] or report["status"] != "failed_diagnostic_gate":
        raise RuntimeError("ensaio negativo não pode ser promovido")
    return {"game": game, "events": len(events), "checks_recomputed": len(checks),
            "saved_body_reexecution_max_error": max_reexecution_error, "negative_verdict_preserved": True}


if __name__ == "__main__":
    verify()
    verify_manifest()
    report = {"status": "audit_passed", "collections": [audit_collection(g, s) for g in ("breakout", "pong") for s in (97, 103)],
              "evaluations": [audit_evaluation(g) for g in ("breakout", "pong")],
              "prior_confirmations_preserved": True, "human_level_goal_achieved": False,
              "cost_scope": "completed_collections_exact_aborted_attempts_partial_unknown",
              "audit_source_sha256": sha(HERE / "auditar_memoria_residual.py")}
    (HERE / "resultados" / "auditoria_memoria_residual.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
