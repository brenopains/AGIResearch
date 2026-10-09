"""Substituir só a leitura inicial; preservar corpos, planos e retornos."""

import json

import numpy as np

from avaliar_composicao import reduce_frames
from avaliar_estado_preditivo import encode
from desfechos_aprendidos import OutcomeModel, OutcomePlanner
from piloto_desfechos_aprendidos import HERE, POLICIES, proposal_path, stem
from piloto_planos_propostos import sha


def run(game):
    proposal = json.loads(proposal_path(game).read_text(encoding="utf-8"))
    pilot = json.loads(stem(game).with_suffix(".json").read_text(encoding="utf-8"))
    for p, expected in proposal["source_hashes"].items():
        if sha(HERE / p) != expected:
            raise RuntimeError("fonte/corpo/banco congelado mudou")
    bank = np.load(HERE / "resultados" / f"{game}_horizontes_103.npz")
    initial = reduce_frames(bank["histories"][:, -2:])
    short_effect = reduce_frames(bank["pairs"][:, :, 0, 1]) - initial[:, None, 1]
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    real_initial = encode(short_effect, basis["mean"], basis["basis"])
    results = {}
    for name in POLICIES:
        model = OutcomeModel.load(stem(game).with_name(stem(game).name + f"_{name}.npz"))
        planner = OutcomePlanner(model)
        plans = proposal["policies"][name]["plans"]
        real_initial_values = np.diag(planner.plan_values(real_initial, plans))
        actual = np.asarray(pilot["returns"][name]["per_root"])
        frozen = np.asarray(proposal["policies"][name]["values"])
        scores = {}
        for label, value in (("frozen_root", frozen), ("real_initial_only", real_initial_values)):
            scores[label] = {"mse": float(np.mean((value - actual) ** 2)),
                             "bias": float((value - actual).mean()), "prediction_mean": float(value.mean()),
                             "per_root_prediction": value.tolist()}
        # Local first-action reward is separately measured in the existing bank.
        first_actions = np.asarray([p[0] for p in plans])
        truth_first = bank["rewards"][np.arange(64), first_actions, 0]
        predicted_first = np.asarray([float((planner.model.posterior(real_initial[i:i + 1], int(a))
                                           @ model.outcomes[int(a)]["reward"])[0]) for i, a in enumerate(first_actions)])
        scores["real_initial_first_block"] = {"mse": float(np.mean((predicted_first - truth_first) ** 2)),
                                               "bias": float((predicted_first - truth_first).mean())}
        results[name] = {"observed_mean": float(actual.mean()), "same_plans_same_bodies": True, "scores": scores}
    output = HERE / "resultados" / f"{game}_diagnostico_raiz_desfechos.json"
    if output.exists():
        raise RuntimeError("diagnóstico já existe; preservar")
    report = {"status": "post_pilot_initial_state_oracle_not_agent", "game": game, "new_ale_steps": 0,
              "policies": results, "proposals_sha256": sha(proposal_path(game)),
              "pilot_sha256": sha(stem(game).with_suffix(".json")), "source_sha256": sha(HERE / "diagnosticar_raiz_desfechos.py"),
              "limit": "initial_state_measured_by_interventions_is_not_available_to_original_policy",
              "human_level_goal_achieved": False}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    for game in ("breakout", "pong"):
        result = run(game)
        print(json.dumps({"game": game, "policies": {n: {"observed_mean": row["observed_mean"],
                         "scores": {s: {k: v for k, v in score.items() if k != "per_root_prediction"}
                                    for s, score in row["scores"].items()}} for n, row in result["policies"].items()}}, indent=2))
