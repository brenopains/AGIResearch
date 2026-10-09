"""Auditoria de cobertura cruzada, origem de ajuste e decisões em corpos salvos."""

import json

import numpy as np

from contrastes_programas import word_features
from experiencia_cruzada import ARMS, CHECKPOINTS, PAIRS, PARTITIONS, PREFIXES, PROPOSAL, TEST_SEED, TRAIN_SEED, base, fitted_model, manifest_path, predict, result_path, save_json, validate_dependencies, verify_proposal
from leituras_por_programa import flatten_history
from piloto_planos_propostos import HERE, sha
from coletar_experiencia_cruzada import digest


def require(condition, why):
    if not condition:
        raise RuntimeError(why)


def bank_audit(game, seed):
    output = base(game, seed)
    with np.load(output.with_suffix(".npz")) as archive:
        bank = {k: archive[k] for k in archive.files}
    report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    require(report["bank_sha256"] == sha(output.with_suffix(".npz")) and report["events_sha256"] == sha(output.with_suffix(".jsonl")), "banco/eventos mudaram")
    require(report["proposal_sha256"] == sha(PROPOSAL) and PROPOSAL.stat().st_mtime < output.with_suffix(".npz").stat().st_mtime, "proposta não antecedeu coleta")
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    words = np.asarray(proposal["programs"][:4] if seed == TRAIN_SEED else proposal["programs"])
    require(np.array_equal(bank["programs"], words), "programas não são os propostos")
    expected_roots = [8 * e + c for e in range(8) for c in CHECKPOINTS]
    require(np.array_equal(bank["roots"], expected_roots) and np.array_equal(bank["episode"], np.repeat(np.arange(8), 4)), "raízes/episódios divergem")
    require(bank["histories"].shape == (32, 16, 336) and bank["rewards"].shape == (32, len(words), 16), "formas incompletas")
    events = [json.loads(line) for line in output.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    index = {int(root): i for i, root in enumerate(bank["roots"])}
    measured, walks, replays = set(), set(), set()
    for event in events:
        if event["event"] == "walk_checkpoint":
            require(event["root"] not in walks and event["ale_calls"] == 128
                    and event["selected"] == (event["root"] % 8 in CHECKPOINTS), "caminhada/checkpoint incorreto")
            walks.add(event["root"])
        elif event["event"] == "common_program_executed":
            i, j = index[event["root"]], event["program_index"]
            require((i, j) not in measured, "programa medido duplicado")
            measured.add((i, j))
            require(event["word"] == words[j].tolist() and event["rewards_by_block"] == bank["rewards"][i, j].tolist()
                    and event["valid"] == bank["valid"][i, j].tolist() and event["ale_calls"] == int(bank["lengths"][i, j])
                    and event["final_frame_sha256"] == str(bank["final_frame_sha256"][i, j])
                    and event["initial_history_sha256"] == digest(bank["histories"][i])
                    and event["raw_initial_history_sha256"] == str(bank["raw_history_sha256"][i]), "trajetória/evento não confere")
            valid, reward, length = bank["valid"][i, j], bank["rewards"][i, j], bank["lengths"][i, j]
            require(bool(valid[0]) and not np.any(valid[1:] > valid[:-1]) and not np.any(reward[~valid[:-1]]), "término não absorvido")
            if valid[-1]:
                require(length == 128, "comprimento não terminal incorreto")
            else:
                n = int(valid[:-1].sum())
                require(1 <= length - 8 * (n - 1) <= 8, "comprimento terminal incorreto")
        elif event["event"] == "full_program_replay_certified":
            require(event["root"] not in replays and event["root"] % 8 == 0
                    and event["ale_calls"] == int(bank["lengths"][index[event["root"]], 0]), "replay/custo incorreto")
            replays.add(event["root"])
        else:
            raise RuntimeError("evento desconhecido")
    require(len(measured) == 32 * len(words) and walks == set(range(64)) and len(replays) == 8, "faltam eventos físicos")
    calls = sum(e["ale_calls"] for e in events)
    require(calls == report["ale_calls"] <= 50000 and report["walk_calls"] == 8192 and report["certified_replays"] == 8, "orçamento físico incorreto")
    if seed == TEST_SEED:
        require(report["fit_manifest_sha256_before_test_collection"] == sha(manifest_path(game))
                and manifest_path(game).stat().st_mtime < output.with_suffix(".npz").stat().st_mtime, "ajuste não congelado antes do teste")
    return bank, {"seed": seed, "ale_calls": calls, "physical_events": len(events), "certified_replays": len(replays), "common_program_coverage_complete": True}


def independent_gate(rows, real, episodes):
    delta = real[:, :, 0] - real[:, :, 1]
    opposite = [bool(np.any(delta[:, j] < -1e-12) and np.any(delta[:, j] > 1e-12)) for j in range(2)]
    active = [e for e in range(8) if np.any(np.abs(delta[episodes == e]) > 1e-12)]
    identification = {"opposite_preferences_per_repeated_pair": opposite, "episodes_with_real_contrast": active,
                      "non_tied_comparisons": int(np.sum(np.abs(delta) > 1e-12)), "sufficient_contextual_contrast": bool(all(opposite) and len(active) >= 3)}
    primary = np.asarray(rows["quadratic"]["per_root_regret"])
    gains, checks = {}, {"sufficient_contextual_contrast": identification["sufficient_contextual_contrast"]}
    for arm in ARMS:
        if arm == "quadratic":
            continue
        other = np.asarray(rows[arm]["per_root_regret"])
        values = np.asarray([(other - primary)[episodes == e].mean() for e in range(8)])
        boot = values[np.random.default_rng(9837).integers(0, 8, (10000, 8))].mean(1)
        gain = {"mean": float(values.mean()), "bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
                "positive_episodes": int(np.sum(values > 1e-12)), "per_episode": values.tolist()}
        gains[arm] = gain
        checks[f"mean_gain_vs_{arm}"] = gain["mean"] > 1e-12
        checks[f"positive_lower_vs_{arm}"] = gain["bootstrap_95"][0] > 1e-12
        checks[f"two_episodes_vs_{arm}"] = gain["positive_episodes"] >= 2
    passed = bool(all(checks.values()))
    status = "positive_reserved_contextual_test" if passed else "failed_capability_gate"
    if not identification["sufficient_contextual_contrast"]:
        status = "inconclusive_insufficient_contextual_contrast"
    return identification, gains, checks, passed, status


def audit(game):
    validate_dependencies(game)
    train, train_physical = bank_audit(game, TRAIN_SEED)
    test, test_physical = bank_audit(game, TEST_SEED)
    manifest = json.loads(manifest_path(game).read_text(encoding="utf-8"))
    for p, expected in manifest["source_hashes"].items():
        require(sha(HERE / p) == expected, "fonte/banco/corpo de ajuste mudou")
    require(manifest["scalar_returns_used_for_fit"] == 1024 and manifest["training_physical_scalar_returns_at_all_prefixes"] == 2048
            and manifest["reserved_outcomes_used_for_fit"] == 0, "contabilidade de treino incorreta")
    h = np.repeat(train["histories"], 16, axis=0)
    words, tr = train["programs"], train["rewards"].cumsum(-1)
    left = [words[p, :length].tolist() for i in range(32) for p, q in PAIRS for length in PREFIXES]
    right = [words[q, :length].tolist() for i in range(32) for p, q in PAIRS for length in PREFIXES]
    target = np.asarray([tr[i, p, length - 1] - tr[i, q, length - 1] for i in range(32) for p, q in PAIRS for length in PREFIXES])
    p, q = word_features(left), word_features(right)
    models, equation_errors = {}, {}
    for arm, path in manifest["bodies"].items():
        model = fitted_model(HERE / path, arm)
        models[arm] = model
        require(model.alpha == .05 and model.system_size == 1024 and np.array_equal(model.p, p) and np.array_equal(model.q, q), "linguagem/ajuste divergentes")
        actual = model.training_difference if arm == "raw_rbf" else model.actual_difference
        require(np.array_equal(actual, target), "alvos não vêm exclusivamente de131")
        if arm == "raw_rbf":
            require(np.array_equal(model.x, h.reshape(512, -1)), "RBF não recebe somente pixels131")
            base_gram = model._pair_kernel(model.x, p, q)
        else:
            raw = flatten_history(h, arm == "two_frames")
            mean = raw.mean(0)
            scale = float(np.sqrt(np.mean(np.sum((raw - mean) ** 2, axis=1))))
            if scale < 1e-12:
                scale = 1.
            require(np.array_equal(model.mean, mean) and model.scale == scale, "normalização não vem de131")
            if arm == "shuffled_histories":
                raw = np.repeat(raw[::16][np.random.default_rng(9823).permutation(32)], 16, axis=0)
            require(np.array_equal(model.support, (raw - mean) / scale), "história/suportes alterados")
            expected_target = target.copy()
            if arm == "shuffled_targets":
                lengths, same, rng = np.asarray([len(w) for w in left]), np.all(p == q, axis=1), np.random.default_rng(9823)
                for length in PREFIXES:
                    for identical in (False, True):
                        ids = np.flatnonzero((lengths == length) & (same == identical))
                        expected_target[ids] = expected_target[rng.permutation(ids)]
            require(np.array_equal(model.fit_target, expected_target), "alvo/nulo não é o pré-especificado")
            u = model.support @ model.support.T
            sensory = np.ones((512, 512)) if arm == "word_only" else 1 + u + (u ** 2 if model.degree == 2 else 0)
            wk = model._word_kernel
            base_gram = sensory * (wk(p, p) - wk(p, q) - wk(q, p) + wk(q, q))
        gram = np.block([[base_gram, -base_gram], [-base_gram, base_gram]])
        error = float(np.max(np.abs((gram + .05 * np.eye(1024)) @ model.coeff - np.concatenate((model.fit_target, -model.fit_target)))))
        require(error < 1e-8, "coeficientes não resolvem o sistema declarado")
        equation_errors[arm] = error
    report_path = result_path(game)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(report["manifest_sha256"] == sha(manifest_path(game)) and report["test_bank_sha256"] == sha(base(game, TEST_SEED).with_suffix(".npz"))
            and not report["human_level_goal_achieved"] and not report["promoted"], "cadeia de evidências/alcance mudou")
    observed, programs = test["rewards"].cumsum(-1), test["programs"]
    records, decisions, maximum, statuses = {}, {}, 0., {}
    for name, pairs in PARTITIONS.items():
        partition = report["partitions"][name]
        for horizon in (32, 64, 128):
            n, t = horizon // 8, horizon // 8 - 1
            left = [programs[p, :n].tolist() for _ in range(32) for p, q in pairs]
            right = [programs[q, :n].tolist() for _ in range(32) for p, q in pairs]
            real = np.stack([observed[:, [p, q], t] for p, q in pairs], axis=1)
            for arm, model in models.items():
                delta = predict(model, arm, np.repeat(test["histories"], 2, axis=0), left, right).reshape(32, 2)
                row = partition["curves"][str(horizon)][arm]
                maximum = max(maximum, float(np.max(np.abs(delta - row["predicted_difference"]))))
                choice = (delta < 0).astype(int)
                regret = real.max(2) - np.take_along_axis(real, choice[:, :, None], axis=2)[:, :, 0]
                require(np.array_equal(choice, row["choices"]) and np.array_equal(regret, row["per_comparison_regret"])
                        and np.array_equal(regret.mean(1), row["per_root_regret"]), "decisão/regret não confere")
                mse = float(np.mean((delta - (real[:, :, 0] - real[:, :, 1])) ** 2))
                require(abs(mse - row["contrast_mse"]) < 1e-10, "erro de contraste divergiu")
                records[(name, horizon, arm)], decisions[(name, horizon, arm)] = (delta, real), (choice, regret)
        real = np.stack([observed[:, [p, q], -1] for p, q in pairs], axis=1)
        ident, gains, checks, positive, status = independent_gate(partition["curves"]["128"], real, test["episode"])
        require(partition["identification"] == ident and partition["comparisons"] == gains and partition["checks"] == checks
                and partition["positive"] == positive and partition["status"] == status, "portão/contexto/IC não confere")
        statuses[name] = status
    require(maximum < 1e-10 and report["joint_positive"] == all(p["positive"] for p in report["partitions"].values()), "reexecução/veredito global mudou")
    events = [json.loads(line) for line in report_path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    require(len(events) == 2 * 7 * 3 * 64 + 2, "eventos de avaliação incompletos")
    seen, roots = set(), {int(root): i for i, root in enumerate(test["roots"])}
    for event in events:
        if event["event"] == "crossed_partition_gate_decided":
            p = report["partitions"][event["partition"]]
            require(event["status"] == p["status"] and event["positive"] == p["positive"] and not event["promoted"], "evento de portão divergente")
            continue
        key = event["partition"], event["horizon"], event["arm"]
        i, j = roots[event["root"]], event["pair"]
        require((*key, i, j) not in seen, "decisão duplicada")
        seen.add((*key, i, j))
        delta, real = records[key]
        choice, regret = decisions[key]
        require(event["program_indices"] == list(PARTITIONS[key[0]][j]) and event["observed_returns"] == real[i, j].tolist()
                and abs(event["predicted_difference"] - delta[i, j]) < 1e-10 and event["choice"] == int(choice[i, j])
                and event["regret"] == float(regret[i, j]) and not event["reserved_outcomes_used_as_input"], "evento não é a decisão medida")
    require(len(seen) == 2688, "faltam decisões")
    return {"game": game, "collection": [train_physical, test_physical], "partitions": statuses,
            "body_reexecution_max_error": maximum, "fit_equation_max_errors": equation_errors,
            "evaluation_events": len(events), "report_sha256": sha(report_path), "verdicts_preserved": True}


if __name__ == "__main__":
    verify_proposal()
    resources = json.loads((HERE / "resultados" / "experiencia_cruzada_recursos.json").read_text(encoding="utf-8"))
    require(resources["status"] == "pilot_finished_not_agent" and len(resources["stages"]) == 8
            and all(s["status"] == "finished" and s["exit_code"] == 0 for s in resources["stages"]), "piloto incompleto")
    require(resources["peak_sampled_aggregate_rss_bytes"] <= 6 * 1024 ** 3 and resources["maximum_processes"] <= 2
            and resources["elapsed_wall_seconds"] <= 1800, "limite operacional excedido")
    result = {"status": "audit_passed", "games": [audit(game) for game in ("breakout", "pong")],
              "resource_monitor_verified": True, "human_level_goal_achieved": False,
              "resources_sha256": sha(HERE / "resultados" / "experiencia_cruzada_recursos.json")}
    output = HERE / "resultados" / "auditoria_experiencia_cruzada.json"
    if output.exists():
        require(json.loads(output.read_text(encoding="utf-8")) == result, "auditoria anterior diverge; preservar")
    else:
        save_json(output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
