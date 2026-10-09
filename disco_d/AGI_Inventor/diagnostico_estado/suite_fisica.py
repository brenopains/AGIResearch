"""Executa sementes novas com os mecanismos congelados de percepção física."""

from __future__ import annotations

import json
from pathlib import Path

from geral import save_orbit_case
from oraculo import run as oracle_run
from percepcao_pixels import run as perception_run

HERE = Path(__file__).resolve().parent
SEEDS = (97, 101, 107, 113)


def regret(row, name):
    return row["scores"][name]["overall"]["normalized_regret"]


def main():
    summary = []
    for seed in SEEDS:
        bank = HERE / "resultados" / f"orbitas_geral_s{seed}.npz"
        metadata = save_orbit_case(bank, seed=seed)
        oracle = oracle_run(bank, seed=seed)
        perception = perception_run(bank, seed=seed)
        bank.with_name(bank.stem + "_oraculo.json").write_text(
            json.dumps(oracle, ensure_ascii=False, indent=2), encoding="utf-8")
        bank.with_name(bank.stem + "_percepcao_pixels.json").write_text(
            json.dumps(perception, ensure_ascii=False, indent=2), encoding="utf-8")
        row = {"seed": seed, "sha256": metadata["bank_sha256"],
               "pixels": regret(oracle, "pixels"),
               "oraculo": regret(oracle, "percepcao_oracular"),
               "rastreador": regret(perception, "rastreador_temporal"),
               "componentes": regret(perception, "entidades_extraidas"),
               "nulos_oraculo_iguais_melhores": oracle["shuffled_equal_or_better"],
               "nulos_rastreador_iguais_melhores": perception["shuffled_tracking_equal_or_better"],
               "fracao_tres_entidades": perception["tracking_test"]["fraction_three_components"],
               "erro_posicao_px": perception["tracking_test"]["mean_matching_error_when_three_pixels"]}
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    result = {"seeds": list(SEEDS), "worlds": summary,
              "oraculo_melhor_pares": sum(r["oraculo"] < r["pixels"] for r in summary),
              "rastreador_melhor_pares": sum(r["rastreador"] < r["pixels"] for r in summary),
              "rastreador_metade_ganho_pares": sum(
                  r["pixels"] > r["oraculo"] and
                  r["pixels"] - r["rastreador"] >= (r["pixels"] - r["oraculo"]) / 2
                  for r in summary)}
    (HERE / "resultados" / "suite_fisica.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "worlds"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
