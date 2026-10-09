"""Audita perda de busca no próprio modelo; planos novos não são executados.

Comparar com planos já medidos separa um limite da poda de um limite da
previsão. Aumentar valor previsto não significa aumentar retorno verdadeiro.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from avaliar_composicao import features, reduce_frames
from avaliar_estado_preditivo import encode, load
from extrator_experiencias import ExperienceExtractor
from futuros_distribuidos import make_edges
from piloto_planos_propostos import proposal_path, sha, stem
from planejador_distribuido import ExperiencePlanner

HERE = Path(__file__).resolve().parent


def main(game):
    proposals = json.loads(proposal_path(game).read_text(encoding="utf-8"))
    for p, expected in proposals["source_hashes"].items():
        if sha(HERE / p) != expected:
            raise RuntimeError("corpo da proposta mudou")
    _, _, signature, valid, _ = load(game, 97)
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    z = encode(signature, basis["mean"], basis["basis"])
    train = np.load(HERE / "resultados" / f"{game}_horizontes_97.npz")
    initial = reduce_frames(np.load(HERE / "resultados" / f"{game}_horizontes_103.npz")["histories"][:, -2:])
    root, _, _, _ = ExperienceExtractor(game).predict(features(initial), set(range(8)))
    planner = ExperiencePlanner(make_edges(z, valid, train["rewards"]))
    original = np.asarray(proposals["policies"]["distributed_search"]["values"])
    known = np.column_stack([np.diag(planner.plan_values(root, p["plans"])) for p in proposals["policies"].values()])
    missed = known.max(1) - original
    wide = planner.search(root, 16, 64)
    wide_value = np.asarray(wide["values"])
    report = {"status": "retrospective_search_diagnostic_not_promoted", "game": game,
              "root_count": 64, "roots_with_better_known_plan_in_same_model": int(np.sum(missed > 1e-8)),
              "mean_gap_to_known_model_plan": float(missed.mean()),
              "mean_predicted_value_width_8": float(original.mean()),
              "mean_predicted_value_width_64": float(wide_value.mean()),
              "roots_width_64_improved": int(np.sum(wide_value > original + 1e-8)),
              "roots_width_64_regressed": int(np.sum(wide_value < original - 1e-8)),
              "wide_plans": wide["plans"], "wide_predicted_values": wide["values"],
              "wide_observed_returns": None, "new_ale_steps": 0,
              "limit": "model_search_quality_only_unmeasured_plans_may_exploit_prediction_error"}
    out = stem(game).with_name(stem(game).name + "_poda.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("wide_plans", "wide_predicted_values")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    main(parser.parse_args().game)
