"""Diagnóstico de atualizar contexto versus copiar sucessores de galeria."""

import argparse
import json
from pathlib import Path

import numpy as np

from avaliar_estado_preditivo import encode, load
from futuros_distribuidos import make_edges, weights
from operadores_locais import LocalOperators
from piloto_planos_propostos import proposal_path, sha

HERE = Path(__file__).resolve().parent


def evaluate(game):
    proposals = json.loads(proposal_path(game).read_text(encoding="utf-8"))
    _, _, signature, valid, _ = load(game, 97)
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    z = encode(signature, basis["mean"], basis["basis"])
    reward = np.load(HERE / "resultados" / f"{game}_horizontes_97.npz")["rewards"]
    edges = make_edges(z, valid, reward)
    bank_path = HERE / "resultados" / f"{game}_oraculo_planos_propostos_103.npz"
    bank = np.load(bank_path)
    real = encode(bank["signatures"].reshape(32, 2, 17, 4, 336), basis["mean"], basis["basis"])
    cases = []
    for j, policy in enumerate(("distributed_search", "reward_search")):
        for k, root in enumerate(bank["roots"]):
            plan = proposals["policies"][policy]["plans"][int(root)]
            for t, action in enumerate(plan):
                if bank["valid"][k, j, t] and bank["valid"][k, j, t + 1]:
                    cases.append((int(bank["episode"][k]), action, real[k, j, t], real[k, j, t + 1]))
    errors = {n: [] for n in ("transport", "gallery", "shuffled", "fixed_change", "persistence")}
    episode = []
    by_action = []
    for action in range(4):
        group = [c for c in cases if c[1] == action]
        if not group:
            continue
        x, y = np.stack([c[2] for c in group]), np.stack([c[3] for c in group])
        e = edges[action]
        survive = e["continuation"] > 0
        source, successor = e["source"][survive], e["successor"][survive]
        operator = LocalOperators(source, successor)
        predicted = {"transport": operator.predict(x), "gallery": weights(x, source) @ successor,
                     "shuffled": LocalOperators(source, successor, shuffled=True).predict(x),
                     "fixed_change": x + (successor - source).mean(0), "persistence": x}
        row = {"action": action, "count": len(x), "mse": {}}
        for name, value in predicted.items():
            error = np.mean((value - y) ** 2, axis=1)
            errors[name].extend(error.tolist())
            row["mse"][name] = float(error.mean())
        episode.extend(c[0] for c in group)
        by_action.append(row)
    episode = np.asarray(episode)
    aggregate = {name: float(np.mean(e)) for name, e in errors.items()}
    differences = {}
    for name in ("gallery", "shuffled", "fixed_change", "persistence"):
        paired = np.asarray(errors[name]) - np.asarray(errors["transport"])
        differences[name] = [float(paired[episode == i].mean()) for i in range(8)]
    result = {"status": "retrospective_local_operator_diagnostic_not_promoted", "game": game,
              "valid_transitions": len(cases), "mse": aggregate, "by_action": by_action,
              "paired_error_gain_per_episode": differences, "new_ale_steps": 0,
              "bank_sha256": sha(bank_path), "source_sha256": sha(Path(__file__)),
              "limit": "local_latent_transition_error_is_not_planning_or_human_level_intelligence"}
    out = HERE / "resultados" / f"{game}_avaliacao_operadores_locais.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    r = evaluate(args.game)
    print(json.dumps({"game": args.game, "mse": r["mse"], "by_action": r["by_action"],
                      "paired_gains": r["paired_error_gain_per_episode"]}, indent=2))
