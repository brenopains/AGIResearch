"""Auditar custos, corpos, eventos e portões do piloto de desfechos."""

import json

import numpy as np

from confirmar_futuros_distribuidos import verify
from confirmar_leituras_variaveis import verify_manifest
from piloto_desfechos_aprendidos import HERE, LIMIT, POLICIES, comparison, load_references, proposal_path, stem
from piloto_planos_propostos import sha


def audit(game):
    proposal = json.loads(proposal_path(game).read_text(encoding="utf-8"))
    for name, expected in proposal["source_hashes"].items():
        if sha(HERE / name) != expected:
            raise RuntimeError(f"fonte/corpo/banco mudou: {name}")
    path = stem(game).with_suffix(".json")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["proposals_sha256"] != sha(proposal_path(game)):
        raise RuntimeError("propostas mudaram")
    events = [json.loads(line) for line in path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    if len(events) != 64 * len(POLICIES):
        raise RuntimeError("eventos incompletos")
    seen, measured, physical, replay = set(), {}, 0, 0
    for row in events:
        i, name = row["root"], row["policy"]
        key = (i, name)
        if key in seen or name not in POLICIES or not 0 <= i < 64:
            raise RuntimeError("evento duplicado ou inválido")
        seen.add(key)
        plan_key = (i, tuple(row["plan"]))
        if (row["observed_return"] != report["returns"][name]["per_root"][i]
                or row["plan"] != proposal["policies"][name]["plans"][i]
                or row["predicted_return"] != proposal["policies"][name]["values"][i]):
            raise RuntimeError("evento diverge da proposta/retorno")
        outcome = (row["observed_return"], row["plan_steps"], row["terminal"])
        if row["event"] == "measurement_reused":
            if row["physical_steps"] != 0 or measured.get(plan_key) != outcome:
                raise RuntimeError("reuso sem medida original ou divergente")
        elif row["event"] == "plan_executed":
            if plan_key in measured or row["physical_steps"] != row["plan_steps"]:
                raise RuntimeError("custo físico incorreto")
            measured[plan_key] = outcome
        else:
            raise RuntimeError("tipo de evento desconhecido")
        physical += row["physical_steps"]
        if name == "conditional_outcomes" and i % 8 == 0:
            replay += row["plan_steps"]
    short_steps = int(np.load(HERE / "resultados" / f"{game}_efeitos_locais_103.npz")["lengths"].sum())
    recomputed = 8192 + short_steps + physical + replay
    if recomputed != report["steps_used"] or recomputed > LIMIT:
        raise RuntimeError("contabilidade de ações incorreta")
    references, prior_costs = load_references(game)
    if prior_costs != report["cached_reference_costs_separate"]:
        raise RuntimeError("custos das referências misturados")
    for name, value in references.items():
        if not np.array_equal(value, report["returns"][name]["per_root"]):
            raise RuntimeError("referência anterior foi alterada")
    primary = np.asarray(report["returns"]["conditional_outcomes"]["per_root"])
    episode = np.repeat(np.arange(8), 8)
    checks = {}
    for name, row in report["returns"].items():
        value = np.asarray(row["per_root"])
        if row["mean"] != float(value.mean()):
            raise RuntimeError("média de retorno incorreta")
        if name == "conditional_outcomes":
            continue
        comp = comparison(primary, value, episode)
        if comp != report["comparisons"][name]:
            raise RuntimeError("comparação pareada diverge")
        checks[f"positive_mean_vs_{name}"] = comp["mean_gain"] > 1e-12
        checks[f"positive_lower_vs_{name}"] = comp["bootstrap_95"][0] > 1e-12
        checks[f"two_episodes_vs_{name}"] = comp["positive_episodes"] >= 2
    checks["no_regression_vs_constant"] = report["comparisons"]["plans_constant"]["regressed_episodes"] == 0
    if checks != report["checks"] or bool(all(checks.values())) != report["positive_pilot"]:
        raise RuntimeError("portão reportado diverge do portão recalculado")
    if report["status"] != "failed_development_gate" or report["human_level_goal_achieved"]:
        raise RuntimeError("resultado negativo deve permanecer sem alegação de AGI")
    if (report["certified_roots"], report["certified_short_branches"], report["replays_certified"]) != (64, 256, 8):
        raise RuntimeError("certificações incompletas")
    return {"game": game, "status": "audit_passed", "events": len(events), "steps_recomputed": recomputed,
            "checks_recomputed": len(checks), "pilot_preserved": "negative", "hashes_intact": True}


if __name__ == "__main__":
    verify()
    verify_manifest()
    result = {"status": "audit_passed", "games": [audit(g) for g in ("breakout", "pong")],
              "human_level_goal_achieved": False, "prior_confirmations_preserved": True}
    output = HERE / "resultados" / "auditoria_desfechos_aprendidos.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
