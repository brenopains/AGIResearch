"""Reexecução de funções, seleção, controles e corpos sem promover resultado."""

import json
import numpy as np
from piloto_funcoes_contraexemplos import (ARMS, BODIES, CONSTRUCTION, EVENTS, GENERATORS, HERE, RESULT,
    SLOTS, FunctionTransport, GeneratedTransport, PairFunctions, append_slots, data_for_episodes,
    load_bank, require, root_view, save_new, sha, verify_proposal)
from funcoes_contraexemplos import propose
from auditar_semantica_prefixo import equal_report, scores


def independent_functions(functions, history, source):
    h = np.asarray(history, float).reshape(len(history), -1)
    raw = np.asarray(source, float).reshape(len(source), -1)
    units = []
    for i, origin in enumerate(functions.origins):
        if origin["kind"] == "observed_pair":
            a, b = raw[origin["rows"]]
            delta = a - b
            denominator = float(delta @ delta)
            require(denominator > 0, "par idêntico criou função")
            direction = 4 * delta / denominator
            require(np.array_equal(direction, functions.directions[i]) and
                    np.isclose(float((a + b) @ direction / 2), functions.offsets[i]), "definição não vem do par observado")
            # Difference of squared distances, factored to avoid cancellation.
            logit = 2 * np.sum(((h - a) + (h - b)) * ((h - b) - (h - a)), axis=1) / denominator
        else:
            logit = np.sum(h * functions.directions[i], axis=1) - functions.offsets[i]
        units.append(np.tanh(logit))
    u = np.column_stack(units) if units else np.empty((len(h), 0))
    output = [np.prod(u[:, e["units"]], axis=1) for e in functions.expressions]
    return np.column_stack(output) if output else np.empty((len(h), 0))


def coefficients(phi, pn, data, alpha):
    rewards, matrices = [], []
    for action in range(4):
        ids = data["actions"] == action
        x = phi[ids]
        target = np.column_stack((data["reward"][ids], data["continuation"][ids, None] * pn[ids]))
        lower = np.linalg.cholesky(x.T @ x + alpha * np.eye(x.shape[1]))
        result = np.linalg.solve(lower.T, np.linalg.solve(lower, x.T @ target))
        rewards.append(result[:, 0])
        matrices.append(result[:, 1:])
    return np.asarray(rewards), np.asarray(matrices)


def forward(phi, reward, matrices, left, right):
    result = []
    for state, a, b in zip(phi, left, right):
        sides = []
        for word in (a, b):
            z, total = state.copy(), 0.
            for action in word:
                total += float(z @ reward[action])
                z = z @ matrices[action]
            sides.append(total)
        result.append(sides[0] - sides[1])
    return np.asarray(result)


def regret(target, prediction):
    return np.asarray([abs(y) / 2 if abs(p) <= 1e-12 else max(-y, 0) if p >= -1e-12 else max(y, 0)
                       for y, p in zip(target, prediction)])


def audit():
    proposal = verify_proposal()
    construction = json.loads(CONSTRUCTION.read_text(encoding="utf-8"))
    frozen = json.loads(BODIES.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines()]
    for manifest in (construction, frozen):
        for path, expected in manifest["source_hashes"].items():
            require(sha(HERE / path) == expected, "fonte/corpo/parâmetro congelado mudou")
    require(result["events_sha256"] == sha(EVENTS) and result["construction_manifest_sha256"] == sha(CONSTRUCTION)
            and result["body_manifest_sha256"] == sha(BODIES), "hash de resultado mudou")
    require(result["event_count"] == len(records) and result["ale_calls"] == 0 and not result["human_level_goal_achieved"], "escopo/contagem diverge")
    require(result["wall_seconds"] < proposal["maximum_main_wall_seconds"], "teto excedido")
    selected_events = [r for r in records if r["event"] == "function_selected_in_development"]
    evaluations = [r for r in records if r["event"] == "candidate_evaluated"]
    held = [r for r in records if r["event"] == "held_comparison"]
    dictionaries_events = [r for r in records if r["event"] == "dictionary_generated"]
    require(len(held) == 2 * 40 * 3 * len(ARMS) and len(dictionaries_events) == 8, "eventos finais incompletos")
    require(len(records) == len(selected_events) + len(evaluations) + len(held) + len(dictionaries_events), "tipo de evento desconhecido")
    max_function, max_value, checked = 0., 0., 0
    for game in ("breakout", "pong"):
        bank = load_bank(game, 97)
        data = data_for_episodes(bank, [0, 1, 2, 3])
        construct = construction["construction"][game]
        require(data["bank_indices"].tolist() == construct["source_bank_indices"] and len(data["reward"]) == construct["observed_transitions"], "origem dos dados diverge")
        base = FunctionTransport.load(HERE / construct["base"])
        reproduced = FunctionTransport().fit(data["source"], data["successor"], data["actions"], data["reward"], data["continuation"])
        require(np.array_equal(base.support, reproduced.support) and np.allclose(base.transport, reproduced.transport, atol=1e-10)
                and np.allclose(base.reward, reproduced.reward, atol=1e-10), "base não reproduz construção0–3")
        pv, p, pn = base.features(root_view(bank, [4, 5])["history"]), base.features(data["source"]), base.features(data["successor"])
        validation = root_view(bank, [4, 5])
        models = {a: GeneratedTransport.load(HERE / path) for a, path in frozen["bodies"][game].items()}
        for arm in GENERATORS:
            functions = PairFunctions.load(HERE / construct["dictionaries"][arm])
            again, stats = propose(data["source"], data["actions"], data["reward"], data["episodes"], p, arm)
            require(np.array_equal(functions.directions, again.directions) and np.array_equal(functions.offsets, again.offsets)
                    and functions.expressions == again.expressions and functions.origins == again.origins
                    and stats == construct["generation"][arm], "geração/origem não reproduz")
            arrays = []
            for h in (data["source"], data["successor"], validation["history"]):
                actual = functions.apply(h)
                independent = independent_functions(functions, h, data["source"])
                err = float(np.max(np.abs(actual - independent))) if actual.size else 0.
                max_function = max(max_function, err)
                require(err < 1e-8, "comparação por distâncias difere do corpo afim")
                # Remaining replay uses exported arithmetic to avoid borderline
                # selection changes caused solely by equivalent rounding.
                arrays.append(actual)
            g, gn, gv = arrays
            chosen = []
            report = result["selection"][game][arm]
            base_r, base_m = coefficients(append_slots(p, g, []), append_slots(pn, gn, []), data, .05)
            current_prediction = forward(append_slots(pv, gv, []), base_r, base_m, validation["left"], validation["right"])
            original_score = scores(validation["actual"], current_prediction)
            equal_report(original_score, report["baseline_validation"], "base seleção")
            arm_events = [r for r in evaluations if r["game"] == game and r["arm"] == arm]
            if original_score["fair_tie_regret"] <= 1e-12:
                require(not arm_events and not report["selected"] and report["status"] == "inconclusive_no_validation_decision_error", "ausência de margem não registrada")
            for iteration in sorted(set(r["iteration"] for r in arm_events)):
                group = [r for r in arm_events if r["iteration"] == iteration]
                remaining = [i for i in range(len(functions.expressions)) if i not in chosen]
                require([r["candidate"] for r in group] == remaining, "candidatos omitidos/duplicados")
                phi = append_slots(p, g, chosen)
                projected = phi @ np.linalg.lstsq(phi, g[:, remaining], rcond=1e-10)[0]
                variances = np.var(g[:, remaining], axis=0)
                ratios = np.mean((projected - g[:, remaining]) ** 2, axis=0) / np.maximum(variances, 1e-12)
                before = scores(validation["actual"], current_prediction)
                old_regret = regret(validation["actual"], current_prediction)
                eligible = []
                for record, ratio, var in zip(group, ratios, variances):
                    require(record["selected_before"] == chosen and np.isclose(record["novelty_ratio_training"], ratio)
                            and np.isclose(record["training_variance"], var), "novidade/estado difere")
                    index = record["candidate"]
                    if var < 1e-12 or ratio <= 1e-4:
                        require(record["validation"] is None and not record["eligible"], "redundância passou")
                        continue
                    trial = chosen + [index]
                    r, m = coefficients(append_slots(p, g, trial), append_slots(pn, gn, trial), data, .05)
                    prediction = forward(append_slots(pv, gv, trial), r, m, validation["left"], validation["right"])
                    score = scores(validation["actual"], prediction)
                    improved = int(np.sum(regret(validation["actual"], prediction) < old_regret - 1e-12))
                    qualifies = bool(score["fair_tie_regret"] < before["fair_tie_regret"] - 1e-12 and score["mse"] < before["mse"] - 1e-12 and improved >= 2)
                    equal_report(score, record["validation"], "candidata")
                    require(qualifies == record["eligible"] and improved == record["positive_validation_roots"], "seleção difere")
                    if qualifies:
                        rank = (score["fair_tie_regret"], score["mse"], len(functions.expressions[index]["units"]), index)
                        eligible.append((rank, index, prediction))
                    checked += 1
                accepted = [r for r in selected_events if r["game"] == game and r["arm"] == arm and r["iteration"] == iteration]
                if eligible:
                    _, index, current_prediction = min(eligible, key=lambda x: x[0])
                    chosen.append(index)
                    require(len(accepted) == 1 and accepted[0]["candidate"] == index and accepted[0]["selected_after"] == chosen
                            and not accepted[0]["is_general_library_promotion"], "escolha não reproduz")
                else:
                    require(not accepted, "candidata sem ganho foi escolhida")
            require(chosen == report["selected"] == [i for i in models[arm].selected if i is not None], "corpo não é o escolhido")
            equal_report(scores(validation["actual"], current_prediction), report["final_validation"], "seleção final")
        primary = models["counterexample_pairs"]
        removed = models["remove_generated"]
        width = len(base.initial_prior)
        require(removed.selected == [None] * SLOTS and not np.any(removed.reward[:, width:])
                and not np.any(removed.transport[:, width:, :]) and not np.any(removed.transport[:, :, width:]), "ablação não remove coordenadas")
        require(np.array_equal(primary.reward[:, :width], removed.reward[:, :width]) and
                np.array_equal(primary.transport[:, :width, :width], removed.transport[:, :width, :width]), "ablação modificou bloco antigo")
        context = models["remove_context"]
        require(context.mode == "no_context" and np.array_equal(context.reward, primary.reward) and
                np.array_equal(context.transport, primary.transport), "controle de contexto alterou corpo")
        for seed, eps in ((97, [6, 7]), (103, list(range(8)))):
            test_bank = bank if seed == 97 else load_bank(game, seed)
            for horizon in (4, 8, 16):
                view = root_view(test_bank, eps, horizon)
                for arm in ARMS:
                    if arm == "zero":
                        prediction = np.zeros(len(view["ids"]))
                    else:
                        model = models[arm]
                        phi = model.features(view["history"])
                        if model.mode == "no_context":
                            phi = np.broadcast_to(model.initial_prior, phi.shape)
                        prediction = forward(phi, model.reward, model.transport, view["left"], view["right"])
                    group = [r for r in held if r["game"] == game and r["seed"] == seed and r["arm"] == arm and r["horizon_blocks"] == horizon]
                    require([r["row"] for r in group] == view["ids"].tolist(), "casos finais ausentes/duplicados")
                    recorded = np.asarray([r["predicted_delta"] for r in group])
                    error = float(np.max(np.abs(recorded - prediction)))
                    max_value = max(max_value, error)
                    require(error < 1e-8 and np.array_equal(view["actual"], [r["actual_delta"] for r in group]), "previsão física/corpo difere")
                    equal_report(scores(view["actual"], recorded), result["games"][game][str(seed)][str(horizon)][arm], "final")
        final = result["games"][game]["103"]["16"]
        checks = {"generated_function_selected": bool(result["selection"][game]["counterexample_pairs"]["selected"])}
        for arm in ARMS:
            if arm != "counterexample_pairs":
                for metric in ("mse", "fair_tie_regret"):
                    checks[f"{metric}_vs_{arm}"] = bool(final["counterexample_pairs"][metric] < final[arm][metric] - 1e-12)
        require(checks == result["games"][game]["gate_checks"] and all(checks.values()) == result["games"][game]["development_gate_passed"], "portão difere")
        del bank
    passed = all(g["development_gate_passed"] for g in result["games"].values())
    require(result["status"] == ("development_gate_passed_not_agent" if passed else "failed_development_gate"), "veredito conjunto difere")
    output = {"status": "audit_passed", "candidate_fits_checked": checked, "held_comparisons": len(held),
              "maximum_distance_vs_affine_function_error": max_function, "maximum_forward_vs_backward_body_error": max_value,
              "verdict_preserved": result["status"], "result_sha256": sha(RESULT), "development_only": True,
              "human_level_goal_achieved": False}
    save_new(HERE / "resultados" / "funcoes_contraexemplos_auditoria.json", output)
    return output


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
