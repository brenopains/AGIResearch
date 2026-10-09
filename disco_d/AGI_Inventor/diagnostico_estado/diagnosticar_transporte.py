"""Comparar previsão de transição, suporte real e escolha oracular do suporte.

O próximo estado real entra somente no diagnóstico do suporte; não é dado
ao planejador. A unidade é a atualização de um bloco, não retorno distante.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from avaliar_estado_preditivo import encode, load
from coordenadas_consequencias import ConsequenceGeometry, ConsequenceReadout
from futuros_distribuidos import make_edges, weights
from piloto_planos_propostos import proposal_path, sha

HERE = Path(__file__).resolve().parent


def run(game):
    proposals = json.loads(proposal_path(game).read_text(encoding="utf-8"))
    for p, expected in proposals["source_hashes"].items():
        if sha(HERE / p) != expected:
            raise RuntimeError("corpo anterior mudou")
    _, _, signature, valid, _ = load(game, 97)
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    z = encode(signature, basis["mean"], basis["basis"])
    cumulative = np.load(HERE / "resultados" / f"{game}_horizontes_97.npz")["rewards"]
    edges = make_edges(z, valid, cumulative)
    path = HERE / "resultados" / f"{game}_oraculo_planos_propostos_103.npz"
    bank = np.load(path)
    real_z = encode(bank["signatures"].reshape(32, 2, 17, 4, 336), basis["mean"], basis["basis"])
    real_valid, roots = bank["valid"], bank["roots"]
    all_support = np.concatenate([e["source"] for e in edges])
    reader = ConsequenceReadout().fit(edges)
    geometry = ConsequenceGeometry(reader, all_support)
    cases = []
    for j, policy in enumerate(("distributed_search", "reward_search")):
        for k, root in enumerate(roots):
            plan = proposals["policies"][policy]["plans"][int(root)]
            for t, action in enumerate(plan):
                if real_valid[k, j, t] and real_valid[k, j, t + 1]:
                    cases.append((int(bank["episode"][k]), action, real_z[k, j, t], real_z[k, j, t + 1]))
    rows = []
    for action in range(4):
        selected = [c for c in cases if c[1] == action]
        if not selected:
            continue
        group = np.asarray([c[0] for c in selected])
        current, successor = np.stack([c[2] for c in selected]), np.stack([c[3] for c in selected])
        e = edges[action]
        surviving = e["continuation"] > 0
        source, target = e["source"][surviving], e["successor"][surviving]
        forecasts = {
            "persistence": current,
            "source_neighbors": weights(current, source) @ target,
            "source_neighbors_consequence_geometry": weights(geometry.transform(current), geometry.transform(source)) @ target,
            "kernel_mean_update": current + KernelRidge().fit(source, target - source).predict(current),
        }
        oracle_distance = KernelRidge.distances(successor, target)
        forecasts["oracle_nearest_successor"] = target[oracle_distance.argmin(1)]
        scores = {}
        denominator = np.mean((successor - current) ** 2)
        for name, predicted in forecasts.items():
            error = np.mean((predicted - successor) ** 2, axis=1)
            scores[name] = {"mse": float(error.mean()),
                            "ratio_to_persistence": float(error.mean() / denominator) if denominator > 0 else None,
                            "per_episode": {str(ep): float(error[group == ep].mean()) for ep in sorted(set(group))}}
        nearest_source = KernelRidge.distances(current, source).min(1)
        nearest_successor = oracle_distance.min(1)
        rows.append({"action": action, "cases": len(selected), "surviving_training_edges": len(source),
                     "scores": scores, "nearest_source_distance_mean": float(nearest_source.mean()),
                     "nearest_successor_distance_mean": float(nearest_successor.mean())})
    report = {"status": "post_pilot_transport_diagnostic_not_agent", "game": game,
              "train_seed": 97, "test_seed": 103, "valid_mixed_transitions": len(cases),
              "by_action": rows, "new_ale_steps": 0, "measured_bank_sha256": sha(path),
              "source_sha256": sha(Path(__file__)),
              "limit": "oracle_support_match_is_not_a_prediction_and_mean_latent_error_is_not_decision_quality"}
    out = HERE / "resultados" / f"{game}_diagnostico_transporte.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    r = run(args.game)
    print(json.dumps({"game": args.game, "valid_transitions": r["valid_mixed_transitions"],
                      "scores": [{"action": a["action"], "cases": a["cases"],
                                  "ratios": {n: v["ratio_to_persistence"] for n, v in a["scores"].items()}}
                                 for a in r["by_action"]]}, indent=2))
