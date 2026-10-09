"""Evidências físicas, suporte treinado, palavras novas e portões pareados."""

import json

import numpy as np

from avaliar_contrastes_programas import ARMS, manifest_path, memory_path, paired_gain, stem
from avaliar_estado_preditivo import encode
from avaliar_memoria_residual import load_data
from coletar_programas_reservados import stem as test_stem
from contrastes_programas import ProgramComparison, word_features
from confirmar_futuros_distribuidos import verify
from confirmar_leituras_variaveis import verify_manifest
from memoria_residual import MODES, MemoryDynamics
from piloto_planos_propostos import HERE, sha


def collection(game):
    base = test_stem(game)
    report = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(base.with_suffix(".npz")) as archive:
        bank = {name: archive[name] for name in archive.files}
    for suffix, key in ((".npz", "bank_sha256"), (".jsonl", "events_sha256")):
        if sha(base.with_suffix(suffix)) != report[key]:
            raise RuntimeError("banco/eventos reservados mudaram")
    if (sha(HERE / "coletar_programas_reservados.py") != report["source_sha256"]
            or sha(HERE / "CONTRASTES_PROGRAMAS.md") != report["protocol_sha256"]):
        raise RuntimeError("coletor/protocolo mudaram")
    plan_path = base.with_name(base.name + "_planos.json")
    proposal = json.loads(plan_path.read_text(encoding="utf-8"))
    if sha(plan_path) != report["plans_sha256"]:
        raise RuntimeError("propostas mudaram")
    if not np.array_equal(np.asarray(proposal["plans"])[bank["roots"]], bank["actions"]):
        raise RuntimeError("programas medidos divergem dos propostos")
    if manifest_path(game).stat().st_mtime > plan_path.stat().st_mtime:
        raise RuntimeError("manifesto não precede proposta física armazenada")
    with np.load(HERE / "resultados" / f"{game}_transicoes_mistas_103.npz") as archive:
        past_h, past_s, past_roots = archive["histories"][:, 0, 0], archive["signatures"][:, 0, 0], archive["roots"]
    if (not np.array_equal(bank["histories"], past_h) or not np.array_equal(bank["signatures"], past_s)
            or not np.array_equal(bank["roots"], past_roots)):
        raise RuntimeError("programas novos não partem das mesmas observações iniciais")
    events = [json.loads(line) for line in base.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    if len(events) != 104:
        raise RuntimeError("eventos físicos incompletos")
    indexes, seen, probe_roots, replay_roots = {int(r): i for i, r in enumerate(bank["roots"])}, set(), set(), set()
    short = np.load(HERE / "resultados" / f"{game}_efeitos_locais_103.npz")
    for e in events:
        i = indexes[e["root"]]
        if e["event"] == "root_probes":
            if e["root"] in probe_roots or e["physical_steps"] != int(short["lengths"][e["root"]].sum()):
                raise RuntimeError("sondas duplicadas/custo inválido")
            probe_roots.add(e["root"])
        elif e["event"] == "program_executed":
            j = e["program_index"]
            if (i, j) in seen:
                raise RuntimeError("programa medido duplicado")
            seen.add((i, j))
            if (not np.array_equal(e["plan"], bank["actions"][i, j])
                    or not np.array_equal(e["rewards_by_block"], bank["rewards"][i, j])
                    or not np.array_equal(e["valid"], bank["valid"][i, j])
                    or e["physical_steps"] != e["program_steps"] or e["program_steps"] != bank["lengths"][i, j]):
                raise RuntimeError("resultado físico diverge do banco")
            valid = bank["valid"][i, j]
            if not valid[0] or np.any(valid[1:] > valid[:-1]) or np.any(bank["rewards"][i, j][~valid[:-1]]):
                raise RuntimeError("término não foi absorvido")
            n = int(valid[:-1].sum())
            if valid[-1]:
                if e["program_steps"] != 128:
                    raise RuntimeError("comprimento sem término incorreto")
            elif not 1 <= e["program_steps"] - 8 * (n - 1) <= 8:
                raise RuntimeError("comprimento terminal incorreto")
        elif e["event"] == "replay_certified":
            if e["root"] in replay_roots or e["root"] % 8 or e["physical_steps"] != bank["lengths"][i, 0]:
                raise RuntimeError("replay duplicado/comprimento incorreto")
            replay_roots.add(e["root"])
        else:
            raise RuntimeError("evento físico desconhecido")
    if len(seen) != 64 or len(probe_roots) != 32 or len(replay_roots) != 8:
        raise RuntimeError("faltam programas/sondas/replays")
    total = 8192 + sum(e["physical_steps"] for e in events)
    if total != report["steps_used"] or total > 24000:
        raise RuntimeError("contabilidade física incorreta")
    if (report["certified_roots"], report["certified_short_probes"], report["certified_replays"]) != (64, 128, 8):
        raise RuntimeError("certificações incompletas")
    return bank, {"game": game, "physical_events": len(events), "steps_recomputed": total,
                  "same_initial_observations": True, "manifest_before_collection_proposal": True}


def audit(game):
    manifest = json.loads(manifest_path(game).read_text(encoding="utf-8"))
    for p, expected in manifest["source_hashes"].items():
        if sha(HERE / p) != expected:
            raise RuntimeError(f"fonte/corpo/treino mudou: {p}")
    bank, physical = collection(game)
    path = stem(game).with_name(stem(game).name + "_novas_103.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    if sha(manifest_path(game)) != report["manifest_sha256"]:
        raise RuntimeError("manifesto mudou")
    serialization = json.loads(stem(game).with_name(stem(game).name + "_serializacao.json").read_text(encoding="utf-8"))
    if (serialization["unchanged_evaluation_report_sha256"] != sha(path)
            or serialization["source_frozen_sha256"] != sha(HERE / "avaliar_contrastes_programas.py")
            or serialization["adapter_source_sha256"] != sha(HERE / "avaliar_contrastes_serializacao.py")
            or serialization["models_retrained"] or serialization["gates_changed"]):
        raise RuntimeError("conversão JSON alterou evidências/modelos/portão")
    train, train_z, train_ids, _ = load_data(game, 97)
    reader = MemoryDynamics.load(memory_path(game, "residual_history"))
    root_train = reader.state(train_z[:, 0, 0], train["histories"][:, 0, 0], train_ids[:, 0, 0])
    train_returns = train["rewards"].cumsum(-1)
    actual_difference = (train_returns[:, 0] - train_returns[:, 1]).reshape(-1)
    lengths = np.tile(np.arange(1, 17), 32)
    train_left = [train["actions"][i, 0, :n].tolist() for i in range(32) for n in range(1, 17)]
    train_right = [train["actions"][i, 1, :n].tolist() for i in range(32) for n in range(1, 17)]
    expected_p, expected_q = word_features(train_left), word_features(train_right)
    same = np.all(expected_p == expected_q, axis=1)
    models = {arm: ProgramComparison.load(HERE / p) for arm, p in manifest["bodies"].items()}
    for arm, model in models.items():
        if model.system_size != 1024 or not np.array_equal(model.p, expected_p) or not np.array_equal(model.q, expected_q):
            raise RuntimeError("corpo foi ajustado com programas/orçamento diferentes")
        if not np.array_equal(model.training_difference, actual_difference):
            raise RuntimeError("alvo real do treino diverge das medições")
        if arm == "shuffled_targets":
            for n in range(1, 17):
                for identical in (False, True):
                    selected = (lengths == n) & (same == identical)
                    if not np.array_equal(np.sort(model.fit_target[selected]), np.sort(actual_difference[selected])):
                        raise RuntimeError("nulo não preserva marginais por comprimento/identidade")
        elif arm == "absolute_returns":
            target = np.concatenate((train_returns[:, 0].reshape(-1), train_returns[:, 1].reshape(-1)))
            if not np.array_equal(model.fit_target, target):
                raise RuntimeError("regressão absoluta usa outros dados")
        elif not np.array_equal(model.fit_target, actual_difference):
            raise RuntimeError("contraste não corresponde ao treino")
        if arm != "shuffled_states" and not np.allclose(model.x, np.repeat(root_train, 16, axis=0), atol=1e-12):
            raise RuntimeError("estado de treino diverge")
        if arm == "shuffled_states":
            permutation = np.random.default_rng(9813).permutation(32)
            if not np.allclose(model.x, np.repeat(root_train[permutation], 16, axis=0), atol=1e-12):
                raise RuntimeError("nulo de estado não preserva prefixos por raiz")
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    z = encode(bank["signatures"].reshape(32, 4, 336), basis["mean"], basis["basis"])
    x = reader.state(z, bank["histories"], np.arange(32) + 103 * 100000)
    observed, episode = bank["rewards"].cumsum(-1), bank["episode"]
    memory = {}
    for mode in MODES:
        model = MemoryDynamics.load(memory_path(game, mode))
        ids = np.arange(32 * 2 * 17).reshape(32, 2, 17)[:, 0, 0] + 103 * 100000
        initial = model.state(z, bank["histories"], ids)
        memory[f"memory_{mode}"] = model.rollout(np.repeat(initial, 2, axis=0), bank["actions"].reshape(64, 16))["returns"].reshape(32, 2, 16)
    events = [json.loads(line) for line in path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    expected_events = 32 * (len(ARMS) + len(MODES) + 1) * 3
    if len(events) != expected_events:
        raise RuntimeError("eventos da avaliação incompletos")
    indexes, seen = {int(root): i for i, root in enumerate(bank["roots"])}, set()
    for e in events:
        i, name, horizon = indexes[e["root"]], e["arm"], e["horizon"]
        key = (e["root"], name, horizon)
        if key in seen or e["reserved_outcomes_used_as_input"]:
            raise RuntimeError("evento duplicado ou resultado usado como entrada")
        seen.add(key)
        row = report["curves"][str(horizon)][name]
        if (e["predicted_difference"] != row["predicted_difference"][i]
                or not np.array_equal(e["observed_returns"], observed[i, :, horizon // 8 - 1])
                or e["choice"] != row["choices"][i]):
            raise RuntimeError("evento diverge do corpo/medição")
    regrets, errors, max_error = {}, {}, 0.0
    for horizon in (32, 64, 128):
        t, n = horizon // 8 - 1, horizon // 8
        left, right = bank["actions"][:, 0, :n].tolist(), bank["actions"][:, 1, :n].tolist()
        predictions = {arm: model.predict(x, left, right) for arm, model in models.items()}
        predictions.update({name: value[:, 0, t] - value[:, 1, t] for name, value in memory.items()})
        predictions["constant_program_index"] = np.full(32, 1. if manifest["constant_program_index"] == 0 else -1.)
        real, delta = observed[:, :, t], observed[:, 0, t] - observed[:, 1, t]
        for name, pred in predictions.items():
            row = report["curves"][str(horizon)][name]
            max_error = max(max_error, float(np.max(np.abs(pred - row["predicted_difference"]))))
            choice = (pred < 0).astype(int)
            regret = real.max(1) - real[np.arange(32), choice]
            error = (pred - delta) ** 2
            if (not np.isclose(regret.mean(), row["mean_regret"], atol=1e-12)
                    or not np.isclose(error.mean(), row["contrast_mse"], atol=1e-12)
                    or not np.array_equal(choice, row["choices"])):
                raise RuntimeError("métrica/decisão incorreta")
            if horizon == 128:
                regrets[name], errors[name] = regret, error
    if max_error > 1e-10:
        raise RuntimeError("corpos não reproduzem a avaliação")
    checks = {}
    for name, regret in regrets.items():
        if name == "conditional":
            continue
        comp = paired_gain(regrets["conditional"], regret, episode)
        if comp != report["comparisons"][name]:
            raise RuntimeError("intervalo/ganho pareado diverge")
        checks[f"mean_gain_vs_{name}"] = comp["mean"] > 1e-12
        checks[f"positive_lower_vs_{name}"] = comp["bootstrap_95"][0] > 1e-12
        checks[f"two_episodes_vs_{name}"] = comp["positive_episodes"] >= 2
    checks["no_regression_vs_constant"] = report["comparisons"]["constant_program_index"]["regressed_episodes"] == 0
    checks["contrast_mse_vs_absolute_returns"] = bool(errors["conditional"].mean() < errors["absolute_returns"].mean() - 1e-12)
    checks["contrast_mse_vs_shuffled_targets"] = bool(errors["conditional"].mean() < errors["shuffled_targets"].mean() - 1e-12)
    overlap = sum(tuple(p) in {tuple(w) for w in manifest["training_complete_programs"]} for p in bank["actions"].reshape(64, 16))
    checks["all_complete_words_unseen_by_comparison_fit"] = overlap == 0
    if checks != report["checks"] or bool(all(checks.values())) != report["positive_pilot"]:
        raise RuntimeError("portão diverge das evidências")
    if report["status"] != "failed_development_gate" or report["promoted"] or report["human_level_goal_achieved"]:
        raise RuntimeError("resultado negativo deve permanecer sem promoção")
    return {**physical, "evaluation_events": len(events), "checks_recomputed": len(checks),
            "body_reexecution_max_error": max_error, "new_complete_words": 64 - overlap,
            "negative_verdict_preserved": True, "serialization_only_fix_verified": True}


if __name__ == "__main__":
    verify()
    verify_manifest()
    result = {"status": "audit_passed", "games": [audit(g) for g in ("breakout", "pong")],
              "prior_confirmations_preserved": True, "human_level_goal_achieved": False,
              "audit_source_sha256": sha(HERE / "auditar_contrastes_programas.py")}
    (HERE / "resultados" / "auditoria_contrastes_programas.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
