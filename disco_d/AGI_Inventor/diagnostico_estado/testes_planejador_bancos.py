"""O valor da busca deve coincidir com composição independente da palavra."""

import json
from pathlib import Path

import numpy as np

from avaliar_composicao import features, reduce_frames
from avaliar_estado_preditivo import encode, load
from extrator_experiencias import ExperienceExtractor
from futuros_distribuidos import make_edges
from planejador_distribuido import ExperiencePlanner

HERE = Path(__file__).resolve().parent


def main():
    for game in ("breakout", "pong"):
        proposals = json.loads((HERE / "resultados" / f"{game}_planos_propostos_103_propostas.json").read_text(encoding="utf-8"))
        _, _, signature, valid, _ = load(game, 97)
        basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
        z = encode(signature, basis["mean"], basis["basis"])
        reward = np.load(HERE / "resultados" / f"{game}_horizontes_97.npz")["rewards"]
        initial = reduce_frames(np.load(HERE / "resultados" / f"{game}_horizontes_103.npz")["histories"][:, -2:])
        root, _, _, _ = ExperienceExtractor(game).predict(features(initial), set(range(8)))
        edges = make_edges(z, valid, reward)
        max_error = 0
        for name, alteration in (("distributed_search", None), ("successor_search", "successor"),
                                  ("reward_search", "reward"), ("action_search", "action")):
            planner = ExperiencePlanner(edges, 8, alteration)
            data = proposals["policies"][name]
            values = np.diag(planner.plan_values(root, data["plans"]))
            error = float(np.max(np.abs(values - data["values"])))
            max_error = max(error, max_error)
            assert error < 1e-8, (game, name, error)
            for trace in data["traces"]:
                assert len(trace) == 8 and all(len(row["plan"]) == 16 for row in trace)
                assert all(-1e-10 <= row["predicted_survival"] <= 1 + 1e-10 for row in trace)
        print(json.dumps({"game": game, "search_vs_composition_max_error": max_error,
                          "roots_checked": 64, "models_checked": 4}))


if __name__ == "__main__":
    main()
