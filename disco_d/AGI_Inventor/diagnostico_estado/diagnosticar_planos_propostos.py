"""Separar cobertura do repertório, ordenação e erro de valor do modelo.

O seletor oracular vê retornos já medidos somente neste diagnóstico. Nenhuma
proposta, modelo ou veredito do piloto é alterado.
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
    report_path = stem(game).with_suffix(".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    proposals = json.loads(proposal_path(game).read_text(encoding="utf-8"))
    for p, expected in proposals["source_hashes"].items():
        if sha(HERE / p) != expected:
            raise RuntimeError("fontes congeladas mudaram")
    before_hash = sha(report_path)
    _, _, signature, valid, _ = load(game, 97)
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    z = encode(signature, basis["mean"], basis["basis"])
    train = np.load(HERE / "resultados" / f"{game}_horizontes_97.npz")
    test = np.load(HERE / "resultados" / f"{game}_horizontes_103.npz")
    initial = reduce_frames(test["histories"][:, -2:])
    root, _, _, _ = ExperienceExtractor(game).predict(features(initial), set(range(8)))
    planner = ExperiencePlanner(make_edges(z, valid, train["rewards"]), 8)
    names = list(report["returns"])
    actual = np.column_stack([report["returns"][n]["per_root"] for n in names])
    predicted = np.column_stack([np.diag(planner.plan_values(root, proposals["policies"][n]["plans"])) for n in names])
    episode = test["episode"]
    if not np.allclose(predicted[:, names.index("distributed_search")], proposals["policies"]["distributed_search"]["values"], atol=1e-8):
        raise RuntimeError("modelo normal não reproduz a busca")
    best_measured = actual.max(1)
    best_predicted = predicted.argmax(1)
    reselected = actual[np.arange(64), best_predicted]
    chosen = actual[:, names.index("distributed_search")]
    switches = np.asarray([sum(a != b for a, b in zip(p, p[1:]))
                           for p in proposals["policies"]["distributed_search"]["plans"]])
    rank_rows = []
    for name in ("constant", "distributed_constant", "reward_search"):
        other = names.index(name)
        proposed = names.index("distributed_search")
        predicted_gain = predicted[:, proposed] - predicted[:, other]
        actual_gain = actual[:, proposed] - actual[:, other]
        informative = np.abs(actual_gain) > 1e-8
        rank_rows.append({"reference": name, "informative_roots": int(informative.sum()),
                          "correct_gain_sign_fraction": float(np.mean((predicted_gain[informative] > 0) ==
                                                                       (actual_gain[informative] > 0)))
                          if informative.any() else None,
                          "predicted_mean_gain": float(predicted_gain.mean()), "actual_mean_gain": float(actual_gain.mean())})
    result = {"status": "post_pilot_oracle_diagnostic_not_promoted", "game": game,
              "pilot_sha256_unchanged": before_hash, "oracle_candidate_mean_return": float(best_measured.mean()),
              "proposed_mean_return": float(chosen.mean()), "normal_model_reselection_mean_return": float(reselected.mean()),
              "oracle_selection_gap": float((best_measured - chosen).mean()),
              "oracle_gap_per_episode": [float((best_measured - chosen)[episode == e].mean()) for e in range(8)],
              "roots_with_strictly_better_measured_plan": int(np.sum(best_measured > chosen)),
              "gain_ordering": rank_rows,
              "value_error_by_policy": {n: {"bias": float((predicted[:, j] - actual[:, j]).mean()),
                                            "mse": float(np.mean((predicted[:, j] - actual[:, j]) ** 2))}
                                         for j, n in enumerate(names)},
              "switches_mean": float(switches.mean()), "switches_per_root": switches.tolist(),
              "new_ale_steps": 0, "known_candidate_count": len(names),
              "limit": "oracle_repertoire_is_not_a_usable_policy_and_does_not_localize_every_cause"}
    if sha(report_path) != before_hash:
        raise RuntimeError("piloto foi alterado pelo diagnóstico")
    output = stem(game).with_name(stem(game).name + "_diagnostico.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    main(parser.parse_args().game)
