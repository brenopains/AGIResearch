"""Reconciliar custos, retornos e oráculos sem mudar seus vereditos."""

import json
from pathlib import Path

import numpy as np

from confirmar_futuros_distribuidos import verify
from confirmar_leituras_variaveis import verify_manifest
from piloto_planos_propostos import POLICIES, proposal_path, sha, stem

HERE = Path(__file__).resolve().parent


def audit(game):
    proposal = json.loads(proposal_path(game).read_text(encoding="utf-8"))
    for name, expected in proposal["source_hashes"].items():
        if sha(HERE / name) != expected:
            raise RuntimeError("fonte ou banco congelado divergiu")
    pilot_path = stem(game).with_suffix(".json")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if sha(proposal_path(game)) != pilot["source_proposals_sha256"]:
        raise RuntimeError("propostas mudaram")
    rows = [json.loads(s) for s in pilot_path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 64 * 5 or len({(r["root"], r["policy"]) for r in rows}) != 320:
        raise RuntimeError("eventos incompletos ou duplicados")
    for row in rows:
        i, policy = row["root"], row["policy"]
        if (row["plan"] != proposal["policies"][policy]["plans"][i]
                or row["observed_return"] != pilot["returns"][policy]["per_root"][i]
                or row["predicted_return"] != proposal["policies"][policy]["values"][i]):
            raise RuntimeError("evento difere de plano ou retorno")
    if pilot["positive_pilot"] or pilot["status"] != "failed_development_gate":
        raise RuntimeError("esta auditoria deve preservar os dois pilotos negativos")
    oracle_stem = HERE / "resultados" / f"{game}_oraculo_planos_propostos_103"
    oracle = json.loads(oracle_stem.with_suffix(".json").read_text(encoding="utf-8"))
    if oracle["pilot_sha256_unchanged"] != sha(pilot_path):
        raise RuntimeError("piloto mudou depois do diagnóstico")
    bank = np.load(oracle_stem.with_suffix(".npz"))
    if oracle["bank_sha256"] != sha(oracle_stem.with_suffix(".npz")):
        raise RuntimeError("banco oracular mudou")
    roots = bank["roots"]
    if list(roots) != [e * 8 + checkpoint for e in range(8) for checkpoint in range(4)]:
        raise RuntimeError("amostra diagnosticada divergiu do protocolo")
    for j, policy in enumerate(("distributed_search", "reward_search")):
        observed = bank["rewards"][:, j].sum(1)
        if not np.array_equal(observed, np.asarray(pilot["returns"][policy]["per_root"])[roots]):
            raise RuntimeError("incrementos não reproduzem retorno do piloto")
        if oracle["diagnostic"][policy]["estimators"]["true_reward_oracle"]["mse"] != 0:
            raise RuntimeError("controle positivo do oráculo falhou")
    if pilot["steps_used"] > 65000 or oracle["steps_used"] > 65000:
        raise RuntimeError("teto de passos excedido")
    return {"game": game, "status": "audit_passed", "pilot_verdict_preserved": "negative",
            "pilot_event_count": len(rows), "oracle_roots": len(roots),
            "new_ale_steps_pilot_and_diagnostic": pilot["steps_used"] + oracle["steps_used"],
            "limit": "real_data_localized_losses_no_human_level_evidence"}


if __name__ == "__main__":
    verify()
    verify_manifest()
    report = {"status": "audit_passed", "games": [audit(g) for g in ("breakout", "pong")],
              "previous_goal_turn_classification": "progress", "goal_human_level_achieved": False,
              "frozen_confirmations_unchanged": True}
    path = HERE / "resultados" / "auditoria_planos_propostos.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
