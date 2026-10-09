"""Reexecutar corpos, conferir origem crua e recalcular portões sem promovê-los."""

import json

import numpy as np

from avaliar_leituras_por_programa import ARMS, HORIZONS, bank_path, fitted_model, manifest_path, reference_path, stem, validate_dependencies
from contrastes_programas import squared_scale, word_features
from leituras_por_programa import ReadingBody, flatten_history
from piloto_planos_propostos import HERE, sha


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def paired(primary, other, episode):
    values = np.asarray([(other - primary)[episode == e].mean() for e in range(8)])
    rng = np.random.default_rng(9827)
    boot = values[rng.integers(0, 8, (10000, 8))].mean(1)
    return {"mean": float(values.mean()), "bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
            "positive_episodes": int(np.sum(values > 1e-12)), "regressed_episodes": int(np.sum(values < -1e-12)),
            "per_episode": values.tolist()}


def audit(game):
    validate_dependencies(game)
    manifest = json.loads(manifest_path(game).read_text(encoding="utf-8"))
    for p, expected in manifest["source_hashes"].items():
        require(sha(HERE / p) == expected, f"fonte/corpo/banco mudou: {p}")
    require(set(manifest["bodies"]) == set(ARMS), "braços incompletos")
    require(not manifest["reserved_outcomes_used_as_fit_input"] and not manifest["probe_signatures_used_as_fit_input"], "escopo de ajuste divergente")
    with np.load(bank_path(game, True)) as archive:
        initial, programs, rewards = archive["histories"][:, 0, 0], archive["actions"], archive["rewards"].cumsum(-1)
    h = np.repeat(initial, 16, axis=0)
    left = [programs[i, 0, :n].tolist() for i in range(32) for n in range(1, 17)]
    right = [programs[i, 1, :n].tolist() for i in range(32) for n in range(1, 17)]
    target = (rewards[:, 0] - rewards[:, 1]).reshape(-1)
    p, q = word_features(left), word_features(right)
    require(programs.reshape(64, 16).tolist() == manifest["training_complete_programs"], "palavras treinadas divergentes")
    models, export_error, dual_error, fixed_error, fit_residual = {}, 0., 0., 0., {}
    for arm, path in manifest["bodies"].items():
        model = fitted_model(HERE / path, arm)
        models[arm] = model
        require(model.alpha == .05 and model.actions == 4 and model.maximum == 16, "hiperparâmetros/linguagem alterados")
        require(model.system_size == 1024 and manifest["costs"][arm]["actual_scalar_observations"] == 1024
                and manifest["costs"][arm]["correlated_prefix_pairs"] == 512, "orçamento/observações divergentes")
        require(np.array_equal(model.p, p) and np.array_equal(model.q, q), "palavras não vêm do treino")
        require(model.word_scale == squared_scale(np.concatenate((p, q))), "escala de palavras não deriva do treino")
        actual = model.training_difference if arm == "raw_rbf" else model.actual_difference
        require(np.array_equal(actual, target), "alvo real não vem do treino")
        if arm == "raw_rbf":
            require(np.array_equal(model.x, h.reshape(512, -1)), "RBF não recebe apenas a história crua")
            base = model._pair_kernel(model.x, model.p, model.q)
            require(np.array_equal(model.fit_target, target), "alvo RBF alterado")
        else:
            require(model.mode == arm and model.degree == (1 if arm == "affine" else 2), "família sensorial alterada")
            raw = flatten_history(h, arm == "two_frames")
            mean = raw.mean(0)
            scale = float(np.sqrt(np.mean(np.sum((raw - mean) ** 2, axis=1))))
            scale = 1. if scale < 1e-12 else scale
            require(np.array_equal(model.mean, mean) and model.scale == scale, "normalização não deriva do passado97")
            if arm == "shuffled_histories":
                order = np.random.default_rng(9823).permutation(32)
                raw = np.repeat(raw[::16][order], 16, axis=0)
            expected = (raw - mean) / scale
            require(np.array_equal(model.support, expected), "suportes não derivam das histórias/permutações declaradas")
            templates, origin = np.unique(expected, axis=0, return_inverse=True)
            require(np.array_equal(model.templates, templates) and np.array_equal(model.origin, origin), "templates divergentes")
            if arm == "shuffled_targets":
                expected_target, rng = target.copy(), np.random.default_rng(9823)
                lengths, same = np.asarray([len(word) for word in left]), np.all(p == q, axis=1)
                for length in range(1, 17):
                    for identical in (False, True):
                        ids = np.flatnonzero((lengths == length) & (same == identical))
                        expected_target[ids] = expected_target[rng.permutation(ids)]
                        require(np.array_equal(np.sort(model.fit_target[ids]), np.sort(target[ids])), "marginal do nulo alterada")
                require(np.array_equal(model.fit_target, expected_target), "nulo não é a permutação pré-registrada")
            else:
                require(np.array_equal(model.fit_target, target), "alvo de braço não nulo alterado")
            # Compute Gram from stored support, independently of .construct/.apply.
            u = model.support @ model.support.T
            sensory = np.ones((512, 512)) if arm == "word_only" else 1 + u + (u ** 2 if model.degree == 2 else 0)
            words = (model._word_kernel(p, p) - model._word_kernel(p, q)
                     - model._word_kernel(q, p) + model._word_kernel(q, q))
            base = sensory * words
            for example in manifest["readings"][arm]:
                saved = ReadingBody.load(HERE / example["parameters"])
                generated = model.construct([example["word"]])
                e = float(np.max(np.abs(saved.apply(initial) - generated.apply(initial))))
                export_error = max(export_error, e)
                x = (flatten_history(initial, saved.two_frames) - saved.mean) / saved.scale
                features = np.column_stack((np.ones(32), x, (x @ saved.templates.T) ** 2))
                weights = np.concatenate((saved.bias, saved.linear.reshape(-1), saved.quadratic.reshape(-1)))
                fixed_error = max(fixed_error, float(np.max(np.abs(features @ weights - saved.apply(initial)))))
        gram = np.block([[base, -base], [-base, base]])
        residual = float(np.max(np.abs((gram + model.alpha * np.eye(1024)) @ model.coeff
                                       - np.concatenate((model.fit_target, -model.fit_target)))))
        require(residual < 1e-8, "coeficientes não resolvem o ajuste declarado")
        fit_residual[arm] = residual
    with np.load(bank_path(game)) as archive:
        history, plans, real, episode, roots = (archive[k] for k in ("histories", "actions", "rewards", "episode", "roots"))
    observed = real.cumsum(-1)
    report_path = stem(game).with_name(stem(game).name + "_novas_103.json")
    require(manifest_path(game).stat().st_mtime < report_path.stat().st_mtime, "avaliação anterior ao congelamento")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path(game).read_text(encoding="utf-8"))
    require(report["manifest_sha256"] == sha(manifest_path(game)) and report["test_bank_sha256"] == sha(bank_path(game))
            and report["old_reference_sha256"] == sha(reference_path(game)), "cadeia de evidências divergente")
    train_words = {tuple(word) for word in programs.reshape(64, 16)}
    overlap = sum(tuple(word) in train_words for word in plans.reshape(64, 16))
    require(overlap == 0 == report["overlapping_complete_words"], "sobreposição de palavras completas")
    reexecution_error, decisions, regenerated = 0., {}, {}
    for horizon in HORIZONS:
        n, t = horizon // 8, horizon // 8 - 1
        left, right = plans[:, 0, :n].tolist(), plans[:, 1, :n].tolist()
        for arm, model in models.items():
            value = model.predict(history.reshape(32, -1) if arm == "raw_rbf" else history, left, right)
            if arm != "raw_rbf":
                dual_error = max(dual_error, float(np.max(np.abs(value - model.predict_dual(history, left, right)))))
                require(np.allclose(model.predict(history, left, left), 0, atol=1e-10), "identidade violada")
                require(np.allclose(value, -model.predict(history, right, left), atol=1e-10), "troca de sinal violada")
            row = report["curves"][str(horizon)][arm]
            e = float(np.max(np.abs(value - row["predicted_difference"])))
            reexecution_error = max(reexecution_error, e)
            choice = (value < 0).astype(int)
            regret = observed[:, :, t].max(1) - observed[np.arange(32), choice, t]
            error = (value - (observed[:, 0, t] - observed[:, 1, t])) ** 2
            require(np.array_equal(choice, row["choices"]) and np.array_equal(regret, row["per_root_regret"]), "decisão/arrependimento divergente")
            require(np.allclose(error, row["per_root_contrast_error"], atol=1e-10)
                    and np.isclose(error.mean(), row["contrast_mse"], atol=1e-10), "MSE divergente")
            regenerated[(horizon, arm)], decisions[(horizon, arm)] = value, (choice, regret)
        for name, old in reference["curves"][str(horizon)].items():
            require(report["curves"][str(horizon)]["old_" + name] == old, "referência antiga alterada")
            old_choice = np.asarray(old["choices"])
            old_regret = observed[:, :, t].max(1) - observed[np.arange(32), old_choice, t]
            require(np.array_equal(old_regret, old["per_root_regret"]), "controle pertence a outro resultado")
    require(max(reexecution_error, dual_error, fixed_error, export_error) < 1e-10, "reexecução/corpo/espaço fixo divergiu")
    primary = decisions[(128, "quadratic")][1]
    comparisons, checks = {}, {}
    for name, row in report["curves"]["128"].items():
        if name == "quadratic":
            continue
        comp = paired(primary, np.asarray(row["per_root_regret"]), episode)
        comparisons[name] = comp
        checks[f"mean_gain_vs_{name}"] = comp["mean"] > 1e-12
        checks[f"positive_lower_vs_{name}"] = comp["bootstrap_95"][0] > 1e-12
        checks[f"two_episodes_vs_{name}"] = comp["positive_episodes"] >= 2
    checks["no_regression_vs_old_constant"] = comparisons["old_constant_program_index"]["regressed_episodes"] == 0
    checks["contrast_mse_vs_affine"] = report["curves"]["128"]["quadratic"]["contrast_mse"] < report["curves"]["128"]["affine"]["contrast_mse"] - 1e-12
    checks["contrast_mse_vs_shuffled_targets"] = report["curves"]["128"]["quadratic"]["contrast_mse"] < report["curves"]["128"]["shuffled_targets"]["contrast_mse"] - 1e-12
    checks["real_contrast_exists"] = bool(np.any(observed[:, 0, -1] != observed[:, 1, -1]))
    require(comparisons == report["comparisons"] and checks == report["checks"]
            and bool(all(checks.values())) == report["positive_pilot"], "portão pareado/IC/veredito incorreto")
    expected_status = ("positive_raw_development_needs_confirmation" if all(checks.values()) else "failed_development_gate")
    if not checks["real_contrast_exists"]:
        expected_status = "inconclusive_no_real_program_contrast"
    require(expected_status == report["status"] and not report["promoted"] and not report["human_level_goal_achieved"]
            and not report["independent_confirmation"] and not report["probe_signatures_used_as_input"] and report["new_ale_steps"] == 0,
            "alcance/rotulagem incorretos")
    events = [json.loads(line) for line in report_path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    require(len(events) == 7 * 3 * 33 + 1, "eventos de avaliação incompletos")
    seen, batches = set(), set()
    root_index = {int(root): i for i, root in enumerate(roots)}
    for event in events:
        if event["event"] == "reserved_program_decision_measured":
            key = event["horizon"], event["arm"]
            i = root_index[event["root"]]
            require((*key, i) not in seen, "evento duplicado")
            seen.add((*key, i))
            require(event["episode"] == int(episode[i]) and event["observed_returns"] == observed[i, :, key[0] // 8 - 1].tolist()
                    and event["choice"] == int(decisions[key][0][i]) and event["regret"] == float(decisions[key][1][i])
                    and abs(event["predicted_difference"] - regenerated[key][i]) < 1e-10
                    and not event["reserved_outcomes_used_as_input"], "evento não corresponde à execução")
        elif event["event"] in ("program_readings_executed", "raw_rbf_comparison_executed"):
            key = event["horizon"], event["arm"]
            require(key not in batches and event["reading_count"] == 64 and not event["reserved_outcomes_used_as_input"], "lote de leitura incorreto")
            batches.add(key)
        elif event["event"] == "development_gate_decided":
            require(event["positive"] == report["positive_pilot"] and not event["promoted"]
                    and event["failed_checks"] == [k for k, v in checks.items() if not v], "evento de portão incorreto")
        else:
            raise RuntimeError("evento inesperado")
    require(len(seen) == 672 and len(batches) == 21, "faltam decisões/leituras")
    return {"game": game, "status": "audit_passed", "body_reexecution_max_error": reexecution_error,
            "body_vs_dual_max_error": dual_error, "fixed_feature_equivalence_max_error": fixed_error,
            "exported_reading_max_error": export_error, "fit_equation_max_errors": fit_residual,
            "evaluation_events": len(events), "checks_recomputed": len(checks), "new_complete_words": 64,
            "primary_positive": report["positive_pilot"], "verdict_preserved": True, "raw_input_provenance_checked": True,
            "report_sha256": sha(report_path), "events_sha256": sha(report_path.with_suffix(".jsonl"))}


if __name__ == "__main__":
    games = [audit(game) for game in ("breakout", "pong")]
    result = {"status": "audit_passed", "games": games, "prior_confirmations_preserved": True,
              "human_level_goal_achieved": False, "promoted": False}
    output = HERE / "resultados" / "auditoria_leituras_por_programa.json"
    if output.exists():
        require(json.loads(output.read_text(encoding="utf-8")) == result, "auditoria existente diverge; preservar")
    else:
        output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
