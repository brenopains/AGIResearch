"""Diagnóstico posterior à confirmação: respostas medidas não são um agente.

Programas, fragmentos e escolhas vêm do manifesto/piloto já congelados.
113 não seleciona, ajusta ou promove. Este diagnóstico não altera o veredito.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from avaliar_composicao import features, reduce_frames
from avaliar_sondas import ScaledRidge
from confirmar_leituras_variaveis import verify_manifest
from extrator_leituras_variaveis import VariableExtractor
from leituras_variaveis import execute
from selecionar_sondas_decisao import input_matrix, measure, preferences
from sondas_contraexemplo import AleRunner, Budget, MAX_STEPS

HERE = Path(__file__).resolve().parent


def diagnose():
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    manifest = verify_manifest()
    frozen_result = HERE / "resultados" / "pong_confirmacao_leituras_113.json"
    if not frozen_result.exists():
        raise RuntimeError("diagnóstico só depois de concluir a confirmação")
    confirmation_sha = hashlib.sha256(frozen_result.read_bytes()).hexdigest()
    horizon = np.load(HERE / "resultados" / "pong_horizontes_113.npz")
    expected = horizon["histories"]
    extractor = VariableExtractor("pong")
    programs = extractor.source.definition["programs"]
    output_bank = HERE / "resultados" / "pong_leituras_variaveis_oraculo_113.npz"
    if output_bank.exists():
        raise RuntimeError("banco diagnóstico já existe; ler resultado")
    budget = Budget(MAX_STEPS)
    runner = AleRunner("pong", 113, expected, budget)
    fields, events = [], []
    for program in programs:
        fields.append(np.stack([runner.observe(root, program) for root in range(64)]))
        runner.certify(0, program)
        event = {"type": "measured_counterfactual_program", "actions": program,
                 "roots": 64, "steps_used": budget.used}
        events.append(event)
        print(json.dumps(event), flush=True)
    fields = np.stack(fields)
    actual_q = np.zeros((64, extractor.n))
    for index, probe in enumerate(extractor.probes):
        actual_q[:, 4 * index:4 * index + 4] = execute(probe["readout_program"], fields[probe["program_index"]])
    np.savez_compressed(output_bank, effects=fields, responses=actual_q, episode=horizon["episode"])
    with output_bank.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    x = features(reduce_frames(expected[:, -2:]))
    z, _, _, _ = extractor.source.predict(x, set(range(8)))
    predicted_q, _, _ = extractor.predict(x, set(range(8)))
    train = np.load(HERE / "resultados" / "pong_leituras_variaveis.npz")
    prior = np.load(HERE / "resultados" / "pong_predicoes_sondas_experiencias.npz")
    reward = np.load(HERE / "resultados" / "pong_horizontes_97.npz")["rewards"][:, :, -1]
    actual = horizon["rewards"][:, :, -1]
    target, constant = preferences(reward), int(np.argmax(reward.mean(0)))
    pilot = json.loads((HERE / "resultados" / "pong_avaliacao_leituras_variaveis.json").read_text(encoding="utf-8"))
    arms = {}
    for label, selected, fit_q, test_q in (
        ("frozen_predicted", manifest["selected_parents"]["variable_readout"], train["cv"], predicted_q),
        ("measured_same_selected_probes", manifest["selected_parents"]["variable_readout"], train["real_train"], actual_q),
        ("frozen_oracle_selection", pilot["arms"]["real_responses_oracle"]["selected_parents"], train["real_train"], actual_q),
    ):
        columns = [4 * p + channel for p in selected for channel in range(4)]
        arms[label] = measure(input_matrix(prior["full_zcv"], fit_q, columns),
                              input_matrix(z, test_q, columns), target, actual, horizon["episode"], constant)
    reference = np.mean((actual_q - train["real_train"].mean(0)) ** 2, axis=0)
    error = np.mean((actual_q - predicted_q) ** 2, axis=0)
    ratios = np.divide(error, reference, out=np.full_like(error, np.nan), where=reference > 1e-12)
    selected_columns = [4 * p + channel for p in manifest["selected_parents"]["variable_readout"] for channel in range(4)]
    if hashlib.sha256(frozen_result.read_bytes()).hexdigest() != confirmation_sha:
        raise RuntimeError("resultado congelado mudou")
    result = {"status": "post_confirmation_oracle_diagnostic_not_promoted", "game": "pong", "seed": 113,
              "confirmation_sha256_unchanged": confirmation_sha, "arms": arms,
              "extraction_error_to_constant": [float(v) if np.isfinite(v) else None for v in ratios],
              "selected_probe_mean_extraction_error": float(np.nanmean(ratios[selected_columns])),
              "mean_extraction_error_by_channel": [float(v) if np.isfinite(v) else None for v in np.nanmean(ratios.reshape(-1, 4), axis=0)],
              "steps_used": budget.used, "replays_certified": runner.replay_count,
              "roots_certified": len(runner.states), "source_hashes_verified": True,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "bank_sha256": hashlib.sha256(output_bank.read_bytes()).hexdigest(),
              "cpu_seconds": time.process_time() - started_cpu, "wall_seconds": time.perf_counter() - started_wall}
    output_bank.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = diagnose()
    print(json.dumps({"status": r["status"],
                      "regret": {name: data["mean_regret"] for name, data in r["arms"].items()},
                      "constant": r["arms"]["frozen_predicted"]["constant_regret"],
                      "selected_extraction_error": r["selected_probe_mean_extraction_error"],
                      "channel_errors": r["mean_extraction_error_by_channel"],
                      "steps": r["steps_used"]}, ensure_ascii=False, indent=2))
