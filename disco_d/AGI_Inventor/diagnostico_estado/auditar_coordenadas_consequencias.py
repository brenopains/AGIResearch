"""Conferir corpos, retornos, custos e preservação dos vereditos."""

import json

import numpy as np

from confirmar_futuros_distribuidos import verify
from confirmar_leituras_variaveis import verify_manifest
from piloto_coordenadas_consequencias import HERE, POLICIES, proposal_path, stem
from piloto_planos_propostos import sha


def audit(game):
    proposals = json.loads(proposal_path(game).read_text(encoding="utf-8"))
    for name, expected in proposals["source_hashes"].items():
        if sha(HERE / name) != expected:
            raise RuntimeError(f"fonte/corpo/banco mudou: {name}")
    path = stem(game).with_suffix(".json")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["proposals_sha256"] != sha(proposal_path(game)):
        raise RuntimeError("propostas mudaram")
    events = [json.loads(s) for s in path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    if len(events) != 64 * len(POLICIES):
        raise RuntimeError("eventos incompletos")
    seen = set()
    physical = 0
    replay = 0
    for row in events:
        key = (row["root"], row["policy"])
        if key in seen:
            raise RuntimeError("evento de política duplicado")
        seen.add(key)
        i, name = key
        if (row["observed_return"] != report["returns"][name]["per_root"][i]
                or row["plan"] != proposals["policies"][name]["plans"][i]
                or row["predicted_return"] != proposals["policies"][name]["values"][i]):
            raise RuntimeError("evento diverge da proposta/retorno")
        if row["event"] == "measurement_reused" and row["physical_steps"] != 0:
            raise RuntimeError("reuso contado como execução nova")
        physical += row["physical_steps"]
        if name == "consequence_geometry" and i % 8 == 0:
            replay += row["plan_steps"]
    short_steps = int(np.load(HERE / "resultados" / f"{game}_efeitos_locais_103.npz")["lengths"].sum())
    recomputed = 8 * 8 * 128 + short_steps + physical + replay
    if recomputed != report["steps_used"] or recomputed > 65000:
        raise RuntimeError("contabilidade de ações incorreta")
    if report["positive_pilot"] or report["status"] != "failed_development_gate":
        raise RuntimeError("veredito negativo desta rodada deve permanecer")
    return {"game": game, "status": "audit_passed", "events": len(events),
            "steps_recomputed": recomputed, "pilot_preserved": "negative",
            "corpus_parameter_hashes_intact": True}


if __name__ == "__main__":
    verify()
    verify_manifest()
    result = {"status": "audit_passed", "games": [audit(g) for g in ("breakout", "pong")],
              "previous_goal_turn_classification": "progress", "human_level_goal_achieved": False,
              "prior_confirmations_preserved": True}
    output = HERE / "resultados" / "auditoria_coordenadas_consequencias.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
