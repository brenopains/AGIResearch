"""Confirmação única: seleções de 103 congeladas, semente 113 intacta.

O modelo recebe só dois quadros. Retornos dos ramos entram apenas no avaliador.
Manifesto inclui fontes, programas, treino, seleções e limites antes da coleta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from avaliar_composicao import features, reduce_frames
from avaliar_sondas import ScaledRidge
from coletar_efeitos_locais import collect as collect_short
from coletar_horizontes import collect as collect_horizons
from extrator_leituras_variaveis import VariableExtractor
from selecionar_sondas_decisao import input_matrix, preferences, regrets

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "resultados" / "confirmacao_leituras_113_manifesto.json"
OUTPUT = HERE / "resultados" / "pong_confirmacao_leituras_113.json"
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 9743


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze():
    if MANIFEST.exists():
        raise RuntimeError("manifesto já existe; não sobrescrever pré-registro")
    if list((HERE / "resultados").glob("pong_*113*.npz")):
        raise RuntimeError("semente 113 já possui banco; não é confirmação intacta")
    selection = json.loads((HERE / "resultados" / "pong_avaliacao_leituras_variaveis.json").read_text(encoding="utf-8"))
    if not selection["positive_pilot"]:
        raise RuntimeError("piloto não tem vantagem contra todos os controles")
    files = [
        "confirmar_leituras_variaveis.py", "leituras_variaveis.py", "extrator_leituras_variaveis.py",
        "extrator_experiencias.py", "avaliar_leituras_variaveis.py", "avaliar.py", "avaliar_sondas.py",
        "avaliar_estado_preditivo.py", "avaliar_composicao.py", "selecionar_sondas_decisao.py",
        "coletar_efeitos_locais.py", "coletar_horizontes.py", "coleta.py",
        "resultados/pong_avaliacao_leituras_variaveis.json",
        "resultados/pong_leituras_variaveis.npz", "resultados/pong_leituras_variaveis_definicoes.json",
        "resultados/pong_predicoes_sondas_experiencias.npz", "resultados/pong_base_estado_preditivo.npz",
        "resultados/pong_estado_preditivo_97.npz", "resultados/pong_horizontes_97.npz",
        "resultados/pong_sondas_decisao_definicoes.json", "resultados/pong_sondas_decisao_97.npz",
        "CONFIRMACAO_LEITURAS_113.md",
    ]
    data = {"status": "frozen_before_113", "game": "pong", "train_seed": 97, "test_seed": 113,
            "source_hashes": {name: sha(HERE / name) for name in files},
            "selected_parents": {name: selection["arms"][name]["selected_parents"]
                                 for name in ("variable_readout", "random_kernels", "shuffled_association")},
            "bootstrap_samples": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
            "gate": {"mean_and_bootstrap_positive_vs": ["base", "random", "shuffle", "constant"],
                     "positive_episodes_vs_base_at_least": 2, "base_episode_regressions_allowed": 0},
            "scope": "offline_selection_among_four_constant_128_step_programs_not_human_intelligence"}
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def verify_manifest():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, expected in data["source_hashes"].items():
        if sha(HERE / name) != expected:
            raise RuntimeError(f"arquivo congelado mudou: {name}")
    if (data["bootstrap_samples"], data["bootstrap_seed"]) != (BOOTSTRAPS, BOOTSTRAP_SEED):
        raise RuntimeError("configuração mudou")
    return data


def comparison(proposed, reference, episode):
    paired = np.asarray([float((reference - proposed)[episode == e].mean()) for e in range(8)])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resampled = paired[rng.integers(0, 8, size=(BOOTSTRAPS, 8))].mean(1)
    interval = np.quantile(resampled, (0.025, 0.975))
    return {"paired_gain_each_episode": paired.tolist(),
            "mean_gain": float(paired.mean()), "bootstrap_95": interval.tolist(),
            "positive_episodes": int(np.sum(paired > 1e-12)),
            "regressed_episodes": int(np.sum(paired < -1e-12)),
            "mean_positive": bool(paired.mean() > 1e-12),
            "lower_above_zero": bool(interval[0] > 1e-12)}


def confirm():
    if OUTPUT.exists():
        raise RuntimeError("confirmação já foi executada; ler resultado existente")
    manifest = verify_manifest()
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    short_path = HERE / "resultados" / "pong_efeitos_locais_113.npz"
    horizon_path = HERE / "resultados" / "pong_horizontes_113.npz"
    if not short_path.exists():
        collect_short("pong", 113)
    short_report = json.loads(short_path.with_suffix(".json").read_text(encoding="utf-8"))
    if not (short_report["seed"] == 113 and short_report["replay_checked"]
            and short_report["bank_sha256"] == sha(short_path)
            and short_report["source_sha256"] == manifest["source_hashes"]["coletar_efeitos_locais.py"]):
        raise RuntimeError("banco curto não corresponde ao coletor congelado")
    if not horizon_path.exists():
        collect_horizons("pong", 113)
    horizon_report = json.loads(horizon_path.with_suffix(".json").read_text(encoding="utf-8"))
    if not (horizon_report["seed"] == 113 and horizon_report["h8_certified"] == 256
            and horizon_report["bank_sha256"] == sha(horizon_path)
            and horizon_report["source_sha256"] == manifest["source_hashes"]["coletar_horizontes.py"]):
        raise RuntimeError("banco de horizonte não corresponde ao coletor congelado")
    verify_manifest()
    horizons = np.load(horizon_path)
    x = features(reduce_frames(horizons["histories"][:, -2:]))
    actual, episode = horizons["rewards"][:, :, -1], horizons["episode"]
    reward = np.load(HERE / "resultados" / "pong_horizontes_97.npz")["rewards"][:, :, -1]
    target = preferences(reward)
    prior = np.load(HERE / "resultados" / "pong_predicoes_sondas_experiencias.npz")
    train = np.load(HERE / "resultados" / "pong_leituras_variaveis.npz")
    extractor = VariableExtractor("pong")
    stored = json.loads((HERE / "resultados" / "pong_leituras_variaveis_definicoes.json").read_text(encoding="utf-8"))
    if stored["probes"] != extractor.probes:
        raise RuntimeError("programas de leitura mudaram")
    z, _, _, _ = extractor.source.predict(x, set(range(8)))
    predicted, random_predicted, counts = extractor.predict(x, set(range(8)))
    base_fit = prior["full_zcv"]
    prediction = ScaledRidge().fit(base_fit, target).predict(z)
    errors = {"base": regrets(actual, prediction)}
    choices = {"base": np.argmax(prediction, axis=1).tolist()}
    constant = int(np.argmax(reward.mean(0)))
    errors["constant"] = actual.max(1) - actual[:, constant]
    choices["constant"] = [constant] * 64
    shuffled = train["cv"][np.random.default_rng(9741).permutation(64)]
    for label, selected_key, fit, test in (
        ("proposed", "variable_readout", train["cv"], predicted),
        ("random", "random_kernels", train["random_cv"], random_predicted),
        ("shuffle", "shuffled_association", shuffled, predicted),
    ):
        selected = manifest["selected_parents"][selected_key]
        columns = [4 * p + channel for p in selected for channel in range(4)]
        forecast = ScaledRidge().fit(input_matrix(base_fit, fit, columns), target).predict(input_matrix(z, test, columns))
        errors[label] = regrets(actual, forecast)
        choices[label] = np.argmax(forecast, axis=1).tolist()
    comparisons = {label: comparison(errors["proposed"], errors[label], episode)
                   for label in ("base", "random", "shuffle", "constant")}
    checks = {}
    for label, data in comparisons.items():
        checks[f"mean_positive_vs_{label}"] = data["mean_positive"]
        checks[f"bootstrap_lower_positive_vs_{label}"] = data["lower_above_zero"]
    checks["gain_at_least_two_episodes_vs_base"] = comparisons["base"]["positive_episodes"] >= 2
    checks["no_episode_regression_vs_base"] = comparisons["base"]["regressed_episodes"] == 0
    passed = bool(all(checks.values()))
    status = "confirmed_local_decision_gain" if passed else "failed_preregistered_confirmation"
    if not np.any(np.ptp(actual, axis=1) > 0):
        status = "inconclusive_no_reward_contrast"
    elif not np.any(errors["constant"] > 0):
        status = "inconclusive_no_margin_against_constant"
    result = {"status": status, "passed": passed, "human_level_goal_achieved": False,
              "game": "pong", "train_seed": 97, "test_seed": 113,
              "regret": {label: {"mean": float(error.mean()),
                                  "per_episode": [float(error[episode == e].mean()) for e in range(8)],
                                  "per_root": error.tolist()} for label, error in errors.items()},
              "comparisons": comparisons, "checks": checks, "choices": choices,
              "actual_returns": actual.tolist(), "roots_with_return_contrast": int(np.sum(np.ptp(actual, axis=1) > 0)),
              "selected_parents": manifest["selected_parents"], "training_counts": counts,
              "source_hashes_verified": True, "manifest_sha256": sha(MANIFEST),
              "test_bank_sha256": sha(horizon_path),
              "certified_short_branches": horizon_report["h8_certified"],
              "steps_used": short_report["steps_used"] + horizon_report["steps_used"],
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        result = freeze()
        print(json.dumps({"status": result["status"], "selected": result["selected_parents"]}, ensure_ascii=False, indent=2))
    else:
        result = confirm()
        print(json.dumps({"status": result["status"], "passed": result["passed"],
                          "regret": {label: data["mean"] for label, data in result["regret"].items()},
                          "comparisons": result["comparisons"], "checks": result["checks"],
                          "steps": result["steps_used"]}, ensure_ascii=False, indent=2))
