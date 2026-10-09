"""Portão congelado de decisão por futuros distribuídos; teste 127 intacto."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from avaliar_composicao import features, reduce_frames
from avaliar_estado_preditivo import encode, load
from avaliar_sondas import ScaledRidge
from coletar_efeitos_locais import collect as collect_short
from coletar_horizontes import collect as collect_horizons
from extrator_experiencias import ExperienceExtractor
from futuros_distribuidos import NEIGHBORS, make_edges, predictions
from selecionar_sondas_decisao import preferences

HERE = Path(__file__).resolve().parent
STEM = HERE / "resultados" / "breakout_confirmacao_futuros_127"
MANIFEST = STEM.with_name(STEM.name + "_manifesto.json")
CONTROLS = ("constant", "mean_collapsed", "successor_shuffled", "reward_shuffled",
            "action_permuted", "direct_preferences")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparison(proposed, reference, episode):
    paired = np.asarray([(reference - proposed)[episode == e].mean() for e in range(8)])
    rng = np.random.default_rng(9757)
    means = paired[rng.integers(0, 8, (10000, 8))].mean(1)
    interval = np.quantile(means, (.025, .975))
    return {"mean_gain": float(paired.mean()), "bootstrap_95": interval.tolist(),
            "per_episode": paired.tolist(), "positive_episodes": int(np.sum(paired > 1e-12)),
            "regressed_episodes": int(np.sum(paired < -1e-12))}


def gate(errors, episode):
    comparisons = {name: comparison(errors["distributed"], errors[name], episode) for name in CONTROLS}
    checks = {}
    for name, data in comparisons.items():
        checks[f"positive_mean_vs_{name}"] = data["mean_gain"] > 1e-12
        checks[f"positive_bootstrap_lower_vs_{name}"] = data["bootstrap_95"][0] > 1e-12
        checks[f"at_least_two_episodes_vs_{name}"] = data["positive_episodes"] >= 2
    checks["no_episode_regression_vs_constant"] = comparisons["constant"]["regressed_episodes"] == 0
    return bool(all(checks.values())), checks, comparisons


def freeze():
    if MANIFEST.exists() or list((HERE / "resultados").glob("breakout_*127*.npz")):
        raise RuntimeError("manifesto existente ou semente exposta; não refazer registro")
    pilot = json.loads((HERE / "resultados" / "breakout_futuros_distribuidos.json").read_text(encoding="utf-8"))
    if not pilot["curves"]["128"]["better_than_all_controls"]:
        raise RuntimeError("desenvolvimento não venceu todos os controles")
    sources = ["confirmar_futuros_distribuidos.py", "futuros_distribuidos.py", "diagnosticar_credito.py",
               "valor_composto.py", "avaliar.py", "avaliar_composicao.py", "avaliar_efeitos_locais.py",
               "variaveis_movimento.py",
               "avaliar_estado_preditivo.py", "avaliar_sondas.py", "selecionar_sondas_decisao.py",
               "extrator_experiencias.py", "coletar_efeitos_locais.py", "coletar_horizontes.py", "coleta.py",
               "CONFIRMACAO_FUTUROS_127.md"]
    artifacts = [f"resultados/breakout_{name}" for name in (
        "estado_preditivo_97.npz", "base_estado_preditivo.npz", "horizontes_97.npz",
        "sondas_decisao_definicoes.json", "sondas_decisao_97.npz",
        "predicoes_sondas_experiencias.npz", "futuros_distribuidos.json")]
    manifest = {"status": "frozen_before_breakout_127", "train_seed": 97, "test_seed": 127,
                "primary_horizon": 128, "neighbors": NEIGHBORS, "controls": list(CONTROLS),
                "bootstrap_seed": 9757, "bootstrap_samples": 10000,
                "source_hashes": {name: sha(HERE / name) for name in sources + artifacts},
                "total_step_limit": 65000, "goal_human_level_achieved": False}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def verify():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, expected in manifest["source_hashes"].items():
        if sha(HERE / name) != expected:
            raise RuntimeError(f"arquivo congelado mudou: {name}")
    return manifest


def confirm():
    output = STEM.with_suffix(".json")
    if output.exists():
        raise RuntimeError("confirmação concluída; não sobrescrever")
    manifest = verify()
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    short_path = HERE / "resultados" / "breakout_efeitos_locais_127.npz"
    horizon_path = HERE / "resultados" / "breakout_horizontes_127.npz"
    if not short_path.exists():
        collect_short("breakout", 127)
    if not horizon_path.exists():
        collect_horizons("breakout", 127)
    short_report = json.loads(short_path.with_suffix(".json").read_text(encoding="utf-8"))
    horizon_report = json.loads(horizon_path.with_suffix(".json").read_text(encoding="utf-8"))
    for path, report, collector in (
        (short_path, short_report, "coletar_efeitos_locais.py"),
        (horizon_path, horizon_report, "coletar_horizontes.py"),
    ):
        if (report["seed"] != 127 or report["bank_sha256"] != sha(path)
                or report["source_sha256"] != manifest["source_hashes"][collector]):
            raise RuntimeError("banco não corresponde ao coletor congelado")
    steps = short_report["steps_used"] + horizon_report["steps_used"]
    if not short_report["replay_checked"] or horizon_report["h8_certified"] != 256:
        raise RuntimeError("certificação incompleta")
    if steps > manifest["total_step_limit"]:
        raise RuntimeError("teto agregado de passos excedido")
    verify()
    horizon = np.load(horizon_path)
    initial = reduce_frames(horizon["histories"][:, -2:])
    # Forecasts depend solely on initial pixels and train97. No future or
    # reward field of this bank is accessed until forecasts are complete.
    root, _, _, counts = ExperienceExtractor("breakout").predict(features(initial), set(range(8)))
    _, _, signature, valid, _ = load("breakout", 97)
    basis = np.load(HERE / "resultados" / "breakout_base_estado_preditivo.npz")
    z = encode(signature, basis["mean"], basis["basis"])
    train = np.load(HERE / "resultados" / "breakout_horizontes_97.npz")
    edges = make_edges(z, valid, train["rewards"])
    forecasts = {"distributed": predictions(root, edges),
                 "mean_collapsed": predictions(root, edges, "mean"),
                 "successor_shuffled": predictions(root, edges, "successor"),
                 "reward_shuffled": predictions(root, edges, "reward"),
                 "action_permuted": predictions(root, edges, "action")}
    prior = np.load(HERE / "resultados" / "breakout_predicoes_sondas_experiencias.npz")
    choices_by_horizon = {}
    for h in (64, 128):
        index = h // 8 - 1
        choices = {name: np.argmax(value[:, :, index], 1) for name, value in forecasts.items()}
        choices["constant"] = np.full(64, int(train["rewards"][:, :, index].mean(0).argmax()))
        direct = ScaledRidge().fit(prior["full_zcv"], preferences(train["rewards"][:, :, index])).predict(root)
        choices["direct_preferences"] = np.argmax(direct, 1)
        choices_by_horizon[str(h)] = choices
    actual, episode = horizon["rewards"], horizon["episode"]
    evaluations = {}
    for h in (64, 128):
        truth = actual[:, :, h // 8 - 1]
        choices = choices_by_horizon[str(h)]
        errors = {name: truth.max(1) - truth[np.arange(64), action] for name, action in choices.items()}
        passed, checks, comparisons = gate(errors, episode)
        status = "confirmed_local_decision_gain" if passed else "failed_preregistered_confirmation"
        if not np.any(np.ptp(truth, axis=1) > 0) or not np.any(errors["constant"] > 0):
            status, passed = "inconclusive_no_reward_contrast_or_margin", False
        evaluations[str(h)] = {"status": status, "passed": passed, "checks": checks,
                               "comparisons": comparisons,
                               "regret": {name: {"mean": float(e.mean()), "per_root": e.tolist(),
                                                 "per_episode": [float(e[episode == n].mean()) for n in range(8)]}
                                          for name, e in errors.items()},
                               "choices": {name: c.tolist() for name, c in choices.items()},
                               "roots_with_reward_contrast": int(np.sum(np.ptp(truth, axis=1) > 0))}
    verify()
    result = {"status": evaluations["128"]["status"], "passed": evaluations["128"]["passed"],
              "goal_human_level_achieved": False, "train_seed": 97, "test_seed": 127,
              "primary_horizon": 128, "evaluations": evaluations, "training_counts": counts,
              "steps_used": steps, "certified_short_branches": horizon_report["h8_certified"],
              "manifest_sha256": sha(MANIFEST), "test_bank_sha256": sha(horizon_path),
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with STEM.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for root_id in range(64):
            event = {"event": "frozen_decision_evaluated", "root": root_id, "episode": int(episode[root_id]),
                     "choices": {k: int(c[root_id]) for k, c in choices_by_horizon["128"].items()},
                     "actual_returns": actual[root_id, :, -1].tolist(),
                     "predicted_distributed_values": forecasts["distributed"][root_id, :, -1].tolist()}
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    result = freeze() if args.freeze else confirm()
    print(json.dumps(result if args.freeze else {
        "status": result["status"], "passed": result["passed"], "steps": result["steps_used"],
        "curves": {h: {k: v["mean"] for k, v in row["regret"].items()}
                   for h, row in result["evaluations"].items()},
        "primary_comparisons": result["evaluations"]["128"]["comparisons"],
    }, ensure_ascii=False, indent=2))
