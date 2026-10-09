"""Reexecutar geração, seleção e previsões; conferir a contabilidade física."""
from piloto_prospectivo import (np, json, HERE, PROPOSAL, DICTIONARIES, BODIES, FIT_EVENTS,
    RESULT, EVENTS, GENERATORS, ARMS, SEEDS, FunctionTransport, PairFunctions, GeneratedTransport,
    append_slots, bank_path, coverage, load_bank, transitions, view, witness_indices,
    require, save_new, sha, verify_bodies, propose, propose_future)
from auditar_funcoes_contraexemplos import coefficients, forward, independent_functions, regret
from auditar_semantica_prefixo import scores, equal_report


def audit():
    frozen = verify_bodies()
    dictionaries = json.loads(DICTIONARIES.read_text(encoding="utf-8"))
    for path, expected in dictionaries["source_hashes"].items():
        require(sha(HERE / path) == expected, "dicionário/base mudou")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    held = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines()]
    fit_events = [json.loads(line) for line in FIT_EVENTS.read_text(encoding="utf-8").splitlines()]
    require(result["events_sha256"] == sha(EVENTS) and result["bodies_sha256"] == sha(BODIES)
        and result["proposal_sha256"] == sha(PROPOSAL), "hash de resultado diverge")
    require(len(held) == result["event_count"] == 2 * 2 * 3 * len(ARMS) * 64, "eventos finais incompletos")
    require(not result["human_level_goal_achieved"] and not result["trained_function_cross_domain_transfer_demonstrated"], "escopo exagerado")
    evaluations = [r for r in fit_events if r["event"] == "candidate_evaluated"]
    selected_events = [r for r in fit_events if r["event"] == "function_selected_in_development"]
    generated = [r for r in fit_events if r["event"] == "prospective_dictionary_generated"]
    require(len(generated) == 8 and len(fit_events) == len(evaluations) + len(selected_events) + len(generated), "evento de ajuste desconhecido")
    maximum, function_error, checked, physical_calls, statuses = 0., 0., 0, 0, []
    for game in ("breakout", "pong"):
        banks = {}
        for seed in SEEDS:
            bank = load_bank(game, seed)
            banks[seed] = bank
            stem = bank_path(game, seed)
            certificate = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
            records = [json.loads(line) for line in stem.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
            require(sum(r["ale_calls"] for r in records) == certificate["ale_calls"] <= 50000, "orçamento físico diverge")
            require(certificate["wall_seconds"] < 120 and certificate["maximum_sampled_rss_bytes"] < 6 * 1024 ** 3, "teto excedido")
            require(sum(r["event"] == "full_program_replay_certified" for r in records) == certificate["certified_replays"] == 8, "replays ausentes")
            require(sum(r["event"] == "walk_checkpoint" for r in records) == 64, "caminhada incompleta")
            rows = [r for r in records if r["event"] == "common_program_measured"]
            require(len(rows) == 32 * len(bank["programs"]), "programas ausentes")
            for i, root in enumerate(bank["roots"]):
                group = [r for r in rows if r["root"] == int(root)]
                require([r["program"] for r in group] == list(range(len(bank["programs"]))), "programas duplicados/omitidos")
                for j, row in enumerate(group):
                    require(np.array_equal(row["word"], bank["programs"][j]) and np.array_equal(row["rewards"], bank["rewards"][i, j])
                        and np.array_equal(row["valid"], bank["valid"][i, j]), "banco/eventos divergem")
                chunk = stem.with_name(stem.name + f"_root_{root:02d}.npz")
                with np.load(chunk, allow_pickle=False) as data:
                    require(np.array_equal(data["histories"], bank["histories"][i]) and np.array_equal(data["rewards"], bank["rewards"][i])
                        and np.array_equal(data["valid"], bank["valid"][i]), "checkpoint por raiz diverge")
            if seed == 157:
                require(certificate["frozen_bodies_sha256_before_collection"] == sha(BODIES), "avaliação não precedida por congelamento")
            physical_calls += certificate["ale_calls"]
        data = transitions(banks[149])
        construct = dictionaries["construction"][game]
        base = FunctionTransport.load(HERE / construct["base"])
        again_base = FunctionTransport().fit(data["source"], data["successor"], data["actions"], data["reward"], data["continuation"])
        require(np.array_equal(base.support, again_base.support) and np.allclose(base.reward, again_base.reward, atol=1e-10)
            and np.allclose(base.transport, again_base.transport, atol=1e-10), "base não reproduz")
        ids = witness_indices(data["actions"], data["episodes"])
        require(ids.tolist() == construct["witness_indices"] and len(ids) <= 512, "testemunhas mudaram")
        sample = {key: value[ids] for key, value in data.items()}
        sp, sn = base.features(sample["source"]), base.features(sample["successor"])
        p, pn = base.features(data["source"]), base.features(data["successor"])
        validation = view(banks[151])
        cov = coverage(validation["actual"], validation["episodes"])
        require(cov == frozen["validation"][game]["coverage"], "cobertura de validação diverge")
        pv = base.features(validation["history"])
        models = {arm: GeneratedTransport.load(HERE / path) for arm, path in frozen["bodies"][game].items()}
        for arm in GENERATORS:
            functions = PairFunctions.load(HERE / construct["dictionaries"][arm])
            if arm in ("future_pairs", "shuffled_future"):
                regenerated, stats = propose_future(sample["source"], sample["actions"], sn, sample["continuation"], sample["episodes"], sp, arm == "shuffled_future")
            else:
                regenerated, stats = propose(sample["source"], sample["actions"], sample["reward"], sample["episodes"], sp,
                    "counterexample_pairs" if arm == "reward_pairs" else "random_directions")
            require(np.array_equal(functions.directions, regenerated.directions) and np.array_equal(functions.offsets, regenerated.offsets)
                and functions.expressions == regenerated.expressions and functions.origins == regenerated.origins
                and stats == construct["generation"][arm], "geração prospectiva não reproduz")
            direct = functions.apply(validation["history"])
            independent = independent_functions(functions, validation["history"], sample["source"])
            error = float(np.max(np.abs(direct - independent))) if direct.size else 0.
            function_error = max(function_error, error)
            require(error < 1e-8, "função não deriva dos pares declarados")
            g, gn, gv = functions.apply(data["source"]), functions.apply(data["successor"]), direct
            r, m = coefficients(append_slots(p, g, []), append_slots(pn, gn, []), data, .05)
            current = forward(append_slots(pv, gv, []), r, m, validation["left"], validation["right"])
            report = frozen["validation"][game]["selection"][arm]
            equal_report(scores(validation["actual"], current), report["baseline_validation"], "seleção base")
            arm_events = [e for e in evaluations if e["game"] == game and e["arm"] == arm]
            chosen = []
            if not cov["sufficient"]:
                require(not arm_events and report["status"] == "inconclusive_validation_coverage" and not report["selected"], "instrumento sem cobertura aprovou função")
            for iteration in sorted(set(e["iteration"] for e in arm_events)):
                group = [e for e in arm_events if e["iteration"] == iteration]
                remaining = [i for i in range(len(functions.expressions)) if i not in chosen]
                require([e["candidate"] for e in group] == remaining, "candidatos omitidos")
                phi = append_slots(p, g, chosen)
                projection = phi @ np.linalg.lstsq(phi, g[:, remaining], rcond=1e-10)[0]
                variance = np.var(g[:, remaining], axis=0)
                novelty = np.mean((projection - g[:, remaining]) ** 2, axis=0) / np.maximum(variance, 1e-12)
                before = scores(validation["actual"], current)
                old_cost = regret(validation["actual"], current)
                choices = []
                for event, ratio, var in zip(group, novelty, variance):
                    require(event["selected_before"] == chosen and np.isclose(event["novelty_ratio_training"], ratio)
                        and np.isclose(event["training_variance"], var), "novidade/estado divergem")
                    if var < 1e-12 or ratio <= 1e-4:
                        require(event["validation"] is None and not event["eligible"], "redundância aprovada")
                        continue
                    i = event["candidate"]
                    rr, mm = coefficients(append_slots(p, g, chosen + [i]), append_slots(pn, gn, chosen + [i]), data, .05)
                    prediction = forward(append_slots(pv, gv, chosen + [i]), rr, mm, validation["left"], validation["right"])
                    score = scores(validation["actual"], prediction)
                    improved = int(np.sum(regret(validation["actual"], prediction) < old_cost - 1e-12))
                    eligible = bool(score["fair_tie_regret"] < before["fair_tie_regret"] - 1e-12 and score["mse"] < before["mse"] - 1e-12 and improved >= 2)
                    equal_report(score, event["validation"], "candidata")
                    require(eligible == event["eligible"] and improved == event["positive_validation_roots"], "critério de seleção diverge")
                    if eligible:
                        choices.append(((score["fair_tie_regret"], score["mse"], len(functions.expressions[i]["units"]), i), i, prediction, rr, mm))
                    checked += 1
                selections = [e for e in selected_events if e["game"] == game and e["arm"] == arm and e["iteration"] == iteration]
                if choices:
                    _, i, current, r, m = min(choices, key=lambda x: x[0])
                    chosen.append(i)
                    require(len(selections) == 1 and selections[0]["candidate"] == i and selections[0]["selected_after"] == chosen, "escolha diverge")
                else:
                    require(not selections, "escolha sem ganho")
            require(chosen == report["selected"] == [i for i in models[arm].selected if i is not None], "corpo não é o escolhido")
            require(np.allclose(models[arm].reward, r, atol=1e-9) and np.allclose(models[arm].transport, m, atol=1e-9), "corpo não reproduz ajuste")
            equal_report(scores(validation["actual"], current), report["final_validation"], "seleção final")
        primary, removed, context = (models[a] for a in ("future_pairs", "remove_generated", "remove_context"))
        width = len(base.initial_prior)
        require(removed.selected == [None] * 4 and not np.any(removed.reward[:, width:]) and not np.any(removed.transport[:, width:, :])
            and not np.any(removed.transport[:, :, width:]) and not np.any(removed.initial_prior[width:]), "ablação incompleta")
        require(np.array_equal(removed.reward[:, :width], primary.reward[:, :width]) and np.array_equal(removed.transport[:, :width, :width], primary.transport[:, :width, :width]), "ablação mudou bloco antigo")
        require(context.mode == "no_context" and np.array_equal(context.reward, primary.reward) and np.array_equal(context.transport, primary.transport), "controle mudou operadores")
        for partition, offset in (("known", 0), ("new", 4)):
            cell = result["games"][game][partition]
            for horizon in (4, 8, 16):
                cases = view(banks[157], offset, horizon)
                for arm in ARMS:
                    if arm == "zero":
                        prediction = np.zeros(64)
                    else:
                        model = models[arm]
                        phi = model.features(cases["history"])
                        if model.mode == "no_context":
                            phi = np.broadcast_to(model.initial_prior, phi.shape)
                        prediction = forward(phi, model.reward, model.transport, cases["left"], cases["right"])
                    group = [e for e in held if e["game"] == game and e["partition"] == partition and e["horizon"] == horizon and e["arm"] == arm]
                    require([(e["bank_row"], e["left"], e["right"]) for e in group] == cases["bank_cases"], "casos ausentes/duplicados")
                    recorded = np.asarray([e["predicted_delta"] for e in group])
                    error = float(np.max(np.abs(recorded - prediction)))
                    maximum = max(maximum, error)
                    require(error < 1e-8 and np.array_equal([e["actual_delta"] for e in group], cases["actual"]), "previsão/física difere")
                    equal_report(scores(cases["actual"], recorded), cell["curves"][str(horizon)][arm], "final")
            cases = view(banks[157], offset)
            cov_held = coverage(cases["actual"], cases["episodes"])
            require(cov_held == cell["coverage"], "cobertura final difere")
            checks = {"selected_function": bool(frozen["validation"][game]["selection"]["future_pairs"]["selected"]),
                "validation_coverage": cov["sufficient"], "held_coverage": cov_held["sufficient"]}
            primary_score = cell["curves"]["16"]["future_pairs"]
            for arm in ARMS:
                if arm != "future_pairs":
                    for metric in ("mse", "fair_tie_regret"):
                        checks[f"{metric}_vs_{arm}"] = bool(primary_score[metric] < cell["curves"]["16"][arm][metric] - 1e-12)
            status = "inconclusive_coverage" if not cov["sufficient"] or not cov_held["sufficient"] else "passed" if all(checks.values()) else "failed"
            require(checks == cell["gate_checks"] and status == cell["status"], "portão difere")
            statuses.append(status)
        del banks
    overall = "capability_gate_passed_not_agi" if all(s == "passed" for s in statuses) else "gate_not_passed_with_inconclusive_cells" if "inconclusive_coverage" in statuses else "capability_gate_failed"
    require(result["status"] == overall, "veredito difere")
    output = {"status": "audit_passed", "candidate_fits_checked": checked, "held_comparisons": len(held),
        "physical_ale_calls_reconciled": physical_calls, "maximum_function_error": function_error,
        "maximum_forward_vs_backward_prediction_error": maximum, "verdict_preserved": overall,
        "result_sha256": sha(RESULT), "human_level_goal_achieved": False}
    save_new(HERE / "resultados" / "prospectiva_auditoria.json", output)
    return output


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
