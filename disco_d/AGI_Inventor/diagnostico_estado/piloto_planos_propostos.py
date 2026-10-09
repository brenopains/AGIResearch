"""Proposição e execução separadas: nenhum retorno novo escolhe o plano."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from avaliar_composicao import features, reduce_frames
from avaliar_estado_preditivo import encode, load
from avaliar_sondas import ScaledRidge
from coleta import gray, make_ale
from confirmar_futuros_distribuidos import verify as verify_frozen
from confirmar_leituras_variaveis import verify_manifest as verify_previous
from extrator_experiencias import ExperienceExtractor
from futuros_distribuidos import make_edges, predictions
from planejador_distribuido import ExperiencePlanner
from selecionar_sondas_decisao import preferences

HERE = Path(__file__).resolve().parent
POLICIES = ("distributed_search", "mean_search", "successor_search", "reward_search", "action_search")
REFERENCES = ("constant", "distributed_constant", "direct_preferences")
MAX_STEPS = 65000


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def stem(game):
    return HERE / "resultados" / f"{game}_planos_propostos_103"


def proposal_path(game):
    return stem(game).with_name(stem(game).name + "_propostas.json")


def propose(game):
    path = proposal_path(game)
    if path.exists():
        raise RuntimeError("propostas já existem; não sobrescrever após observar")
    verify_frozen()
    verify_previous()
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    _, _, signature, valid, _ = load(game, 97)
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    z = encode(signature, basis["mean"], basis["basis"])
    train = np.load(HERE / "resultados" / f"{game}_horizontes_97.npz")
    initial = reduce_frames(np.load(HERE / "resultados" / f"{game}_horizontes_103.npz")["histories"][:, -2:])
    root, _, _, counts = ExperienceExtractor(game).predict(features(initial), set(range(8)))
    edges = make_edges(z, valid, train["rewards"])
    reports = {}
    for name, alteration, collapse in (
        ("distributed_search", None, False), ("mean_search", None, True),
        ("successor_search", "successor", False), ("reward_search", "reward", False),
        ("action_search", "action", False),
    ):
        reports[name] = ExperiencePlanner(edges, 8, alteration).search(root, 16, 8, collapse=collapse)
        print(json.dumps({"game": game, "policy_generated": name,
                          "cpu_seconds": time.process_time() - start_cpu}), flush=True)
    predicted_constant = predictions(root, edges)[:, :, -1]
    chosen = predicted_constant.argmax(1)
    reports["distributed_constant"] = {"plans": [[int(a)] * 16 for a in chosen],
                                       "values": predicted_constant[np.arange(64), chosen].tolist()}
    fixed = int(train["rewards"][:, :, -1].mean(0).argmax())
    reports["constant"] = {"plans": [[fixed] * 16 for _ in range(64)]}
    prior = np.load(HERE / "resultados" / f"{game}_predicoes_sondas_experiencias.npz")
    direct = ScaledRidge().fit(prior["full_zcv"], preferences(train["rewards"][:, :, -1])).predict(root)
    reports["direct_preferences"] = {"plans": [[int(a)] * 16 for a in direct.argmax(1)]}
    sources = ["piloto_planos_propostos.py", "planejador_distribuido.py", "PILOTO_PLANOS_PROPOSTOS.md",
               "coleta.py", "avaliar.py", "avaliar_composicao.py", "avaliar_efeitos_locais.py",
               "variaveis_movimento.py", "avaliar_estado_preditivo.py", "avaliar_sondas.py",
               "extrator_experiencias.py", "selecionar_sondas_decisao.py", "futuros_distribuidos.py",
               "valor_composto.py", "diagnosticar_credito.py"]
    artifacts = [f"resultados/{game}_{suffix}" for suffix in (
        "estado_preditivo_97.npz", "base_estado_preditivo.npz", "horizontes_97.npz", "horizontes_103.npz",
        "efeitos_locais_103.npz", "predicoes_sondas_experiencias.npz", "sondas_decisao_97.npz",
        "sondas_decisao_definicoes.json")]
    report = {"status": "plans_frozen_before_execution", "game": game, "train_seed": 97, "test_seed": 103,
              "policies": reports, "training_counts": counts, "horizon_blocks": 16, "block_steps": 8,
              "beam_width": 8, "steps_limit": MAX_STEPS,
              "source_hashes": {p: sha(HERE / p) for p in sources + artifacts},
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with path.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for name in POLICIES:
            for i, trace in enumerate(reports[name]["traces"]):
                handle.write(json.dumps({"event": "hypotheses_proposed", "root": i, "policy": name,
                                         "candidates": trace, "observed_return": None}) + "\n")
    return report


def execute_plan(ale, actions, plan, budget):
    total, elapsed = 0.0, 0
    for code in plan:
        for _ in range(8):
            if budget[0] <= 0:
                raise RuntimeError("orçamento ALE esgotado")
            total += float(ale.act(actions[code]))
            budget[0] -= 1
            elapsed += 1
            if ale.game_over():
                return total, elapsed, True, gray(ale)
    return total, elapsed, False, gray(ale)


def compare(proposed, reference, episode):
    paired = np.asarray([(proposed - reference)[episode == e].mean() for e in range(8)])
    rng = np.random.default_rng(9763)
    samples = paired[rng.integers(0, 8, (10000, 8))].mean(1)
    return {"mean_gain": float(paired.mean()), "per_episode": paired.tolist(),
            "bootstrap_95": np.quantile(samples, (.025, .975)).tolist(),
            "positive_episodes": int(np.sum(paired > 1e-12)),
            "regressed_episodes": int(np.sum(paired < -1e-12))}


def execute(game):
    output = stem(game).with_suffix(".json")
    if output.exists():
        raise RuntimeError("piloto concluído; não sobrescrever")
    proposal = json.loads(proposal_path(game).read_text(encoding="utf-8"))
    for p, expected in proposal["source_hashes"].items():
        if sha(HERE / p) != expected:
            raise RuntimeError(f"corpo/experiência mudou depois das propostas: {p}")
    verify_frozen()
    verify_previous()
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    existing = np.load(HERE / "resultados" / f"{game}_horizontes_103.npz")
    short = np.load(HERE / "resultados" / f"{game}_efeitos_locais_103.npz")
    expected_histories, previous_returns = existing["histories"], existing["rewards"][:, :, -1]
    episode_ids = existing["episode"]
    ale, rng, budget = make_ale(game, 103), np.random.default_rng(103), [MAX_STEPS]
    actions = list(ale.getMinimalActionSet()[:4])
    gains = {name: np.zeros(64) for name in POLICIES + REFERENCES}
    events, certified_roots, certified_short, certified_replays = [], 0, 0, 0
    for episode in range(8):
        ale.reset_game()
        frame = gray(ale)
        history = deque([frame.copy() for _ in range(16)], maxlen=16)
        for checkpoint in range(8):
            for _ in range(128):
                if budget[0] <= 0:
                    raise RuntimeError("orçamento ALE esgotado")
                ale.act(actions[int(rng.integers(0, 4))])
                budget[0] -= 1
                if ale.game_over():
                    ale.reset_game()
                    frame = gray(ale)
                    history = deque([frame.copy() for _ in range(16)], maxlen=16)
                else:
                    history.append(gray(ale))
            i = episode * 8 + checkpoint
            if not np.array_equal(np.stack(history), expected_histories[i]):
                raise RuntimeError(f"histórico divergiu: {i}")
            certified_roots += 1
            state = ale.cloneSystemState()
            for action in range(4):
                ale.restoreSystemState(state)
                gain, elapsed, terminal, frame = execute_plan(ale, actions, [action], budget)
                if (gain != short["rewards"][i, action] or elapsed != short["lengths"][i, action]
                        or terminal != short["terminal"][i, action]
                        or not np.array_equal(frame, short["outcomes"][i, action])):
                    raise RuntimeError("ramo curto divergiu")
                certified_short += 1
            # Execute unique plans once; duplicate policies share exactly the
            # same measured branch. Deduplication does not hide its action cost.
            measured = {}
            for name in POLICIES:
                plan = proposal["policies"][name]["plans"][i]
                key = tuple(plan)
                if key not in measured:
                    ale.restoreSystemState(state)
                    measured[key] = execute_plan(ale, actions, plan, budget)
                gain, elapsed, terminal, final = measured[key]
                gains[name][i] = gain
                if name == "distributed_search" and checkpoint == 0:
                    ale.restoreSystemState(state)
                    replay = execute_plan(ale, actions, plan, budget)
                    if replay[:3] != (gain, elapsed, terminal) or not np.array_equal(replay[3], final):
                        raise RuntimeError("replay do plano divergiu")
                    certified_replays += 1
                events.append({"event": "plan_executed", "root": i, "episode": episode, "policy": name,
                               "plan": plan, "predicted_return": proposal["policies"][name]["values"][i],
                               "observed_return": gain, "executed_steps": elapsed, "terminal": terminal})
            for name in REFERENCES:
                action = proposal["policies"][name]["plans"][i][0]
                gains[name][i] = previous_returns[i, action]
            ale.restoreSystemState(state)
        print(json.dumps({"game": game, "episode_evaluated": episode,
                          "steps_used": MAX_STEPS - budget[0]}), flush=True)
    comparisons = {name: compare(gains["distributed_search"], gains[name], episode_ids)
                   for name in POLICIES[1:] + REFERENCES}
    checks = {}
    for name, data in comparisons.items():
        checks[f"positive_mean_vs_{name}"] = data["mean_gain"] > 1e-12
        checks[f"positive_lower_vs_{name}"] = data["bootstrap_95"][0] > 1e-12
        checks[f"two_episodes_vs_{name}"] = data["positive_episodes"] >= 2
    checks["no_episode_regression_vs_constant"] = comparisons["constant"]["regressed_episodes"] == 0
    positive = bool(all(checks.values()))
    status = "positive_development_pilot_needs_confirmation" if positive else "failed_development_gate"
    all_returns = np.column_stack([*gains.values(), *previous_returns.T])
    if not np.any(all_returns.max(1) > gains["constant"]):
        status, positive = "inconclusive_no_margin_vs_constant", False
    report = {"status": status, "positive_pilot": positive, "human_level_goal_achieved": False,
              "game": game, "train_seed": 97, "test_seed": 103, "comparisons": comparisons, "checks": checks,
              "returns": {name: {"mean": float(r.mean()), "per_root": r.tolist(),
                                 "per_episode": [float(r[episode_ids == e].mean()) for e in range(8)]}
                          for name, r in gains.items()},
              "nonconstant_plans": {name: sum(len(set(p)) > 1 for p in proposal["policies"][name]["plans"])
                                    for name in POLICIES},
              "predicted_return_mse": {name: float(np.mean((np.asarray(proposal["policies"][name]["values"]) - gains[name]) ** 2))
                                       for name in POLICIES},
              "certified_roots": certified_roots, "certified_short_branches": certified_short,
              "certified_replays": certified_replays, "steps_used": MAX_STEPS - budget[0],
              "source_proposals_sha256": sha(proposal_path(game)),
              "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall,
              "reference_returns_previously_measured_not_new_steps": True}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with output.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--propor", action="store_true")
    mode.add_argument("--executar", action="store_true")
    args = parser.parse_args()
    report = propose(args.game) if args.propor else execute(args.game)
    print(json.dumps({"status": report["status"], "game": args.game,
                      **({"generation_cpu": report["cpu_seconds"]} if args.propor else {
                          "returns": {k: v["mean"] for k, v in report["returns"].items()},
                          "nonconstant_plans": report["nonconstant_plans"], "steps": report["steps_used"],
                          "positive_pilot": report["positive_pilot"]})}, ensure_ascii=False, indent=2))
