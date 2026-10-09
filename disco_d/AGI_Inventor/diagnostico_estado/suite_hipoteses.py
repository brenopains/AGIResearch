"""Confirma hipóteses perceptivas em seis mundos não usados no desenvolvimento."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from avaliar_hipoteses import run as hypothesis_run
from geral import save_orbit_case
from percepcao_pixels import run as greedy_run

HERE = Path(__file__).resolve().parent
SEEDS = (127, 131, 137, 139, 149, 151)
SOURCES = ("hipoteses_estado.py", "avaliar_hipoteses.py", "percepcao_pixels.py",
           "avaliar.py", "prefixo.py", "geral.py", "mundo_orbitas.py")


def file_hash(filename):
    path = HERE.parent / "representacao" / filename if filename == "mundo_orbitas.py" else HERE / filename
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regret(report, variant):
    return report["scores"][variant]["overall"]["normalized_regret"]


def main():
    hashes = {filename: file_hash(filename) for filename in SOURCES}
    result = {"seeds": list(SEEDS), "source_sha256_before": hashes, "worlds": []}
    output = HERE / "resultados" / "suite_hipoteses.json"
    for seed in SEEDS:
        start = time.perf_counter()
        bank = HERE / "resultados" / f"orbitas_hipoteses_s{seed}.npz"
        if bank.exists():
            raise RuntimeError(f"banco pré-existente; confirmação não pode sobrescrever: {bank}")
        metadata = save_orbit_case(bank, seed=seed)
        greedy = greedy_run(bank, seed=seed)
        hypotheses = hypothesis_run(bank, seed=seed)
        bank.with_name(bank.stem + "_guloso.json").write_text(
            json.dumps(greedy, ensure_ascii=False, indent=2), encoding="utf-8")
        bank.with_name(bank.stem + "_hipoteses.json").write_text(
            json.dumps(hypotheses, ensure_ascii=False, indent=2), encoding="utf-8")
        row = {"seed": seed, "bank_sha256": metadata["bank_sha256"],
               "pixels": regret(hypotheses, "pixels"),
               "guloso": regret(greedy, "rastreador_temporal"),
               "melhor_hipotese": regret(hypotheses, "melhor_hipotese"),
               "media_hipoteses": regret(hypotheses, "media_hipoteses"),
               "nulos": hypotheses["null_equal_or_better"],
               "seconds": time.perf_counter() - start}
        result["worlds"].append(row)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if hashes != {filename: file_hash(filename) for filename in SOURCES}:
        raise RuntimeError("código mudou durante confirmação")
    gate = {}
    for variant in ("melhor_hipotese", "media_hipoteses"):
        wins = sum(row[variant] < min(row["pixels"], row["guloso"])
                   for row in result["worlds"])
        null_passes = sum(row["nulos"][variant] <= 1 for row in result["worlds"])
        gate[variant] = {"wins_over_both": wins, "worlds_with_at_most_one_null": null_passes,
                         "pass": bool(wins >= 5 and null_passes >= 4)}
    result["gate"] = gate
    result["source_sha256_after"] = hashes
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gate": gate, "total_seconds": sum(row["seconds"] for row in result["worlds"])},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
