"""Recalcula o veredito a partir dos retornos e escolhas realmente registrados."""

import json
from pathlib import Path

import numpy as np

from confirmar_futuros_distribuidos import MANIFEST, STEM, gate, sha, verify
from confirmar_leituras_variaveis import verify_manifest as verify_previous


def audit():
    manifest = verify()
    verify_previous()
    report_path = STEM.with_suffix(".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    bank_path = Path(__file__).resolve().parent / "resultados" / "breakout_horizontes_127.npz"
    if sha(bank_path) != report["test_bank_sha256"] or sha(MANIFEST) != report["manifest_sha256"]:
        raise RuntimeError("proveniência da confirmação mudou")
    bank = np.load(bank_path)
    episode = bank["episode"]
    if not np.array_equal(np.bincount(episode), np.full(8, 8)):
        raise RuntimeError("partição de episódios divergiu")
    rows = [json.loads(line) for line in STEM.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 64 or [r["root"] for r in rows] != list(range(64)):
        raise RuntimeError("eventos não cobrem todas as raízes uma vez")
    truth = bank["rewards"][:, :, -1]
    for i, row in enumerate(rows):
        if not np.array_equal(row["actual_returns"], truth[i]) or row["episode"] != episode[i]:
            raise RuntimeError("evento não corresponde ao banco")
        if row["choices"]["distributed"] != int(np.argmax(row["predicted_distributed_values"])):
            raise RuntimeError("escolha não deriva da previsão")
    recorded = report["evaluations"]["128"]
    errors = {}
    for name, choices in recorded["choices"].items():
        if [r["choices"][name] for r in rows] != choices:
            raise RuntimeError("escolhas divergem entre JSON e eventos")
        errors[name] = truth.max(1) - truth[np.arange(64), np.asarray(choices)]
        if not np.array_equal(errors[name], recorded["regret"][name]["per_root"]):
            raise RuntimeError("arrependimento informado não corresponde aos retornos")
        if float(errors[name].mean()) != recorded["regret"][name]["mean"]:
            raise RuntimeError("média incorreta")
    passed, checks, comparisons = gate(errors, episode)
    if (passed != report["passed"] or checks != recorded["checks"]
            or comparisons != recorded["comparisons"]):
        raise RuntimeError("veredito informado difere do recalculado")
    if report["goal_human_level_achieved"]:
        raise RuntimeError("escopo local não comprova inteligência humana")
    result = {"status": "audit_passed", "local_confirmation_passed": passed,
              "raw_bank_and_events_reconciled": True, "both_manifests_intact": True,
              "previous_goal_turn_classification": "progress",
              "human_level_goal": "not_achieved_keep_active",
              "report_sha256": sha(report_path), "primary_mean_regret": float(errors["distributed"].mean()),
              "controls_count": len(manifest["controls"]),
              "limit": "single_game_four_fixed_programs_no_general_discovery_evidence"}
    output = STEM.with_name(STEM.name + "_auditoria.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
