"""Limites empíricos pós-resultado; não alterar escolha, ajuste ou portão."""

import json
from collections import defaultdict

import numpy as np

from experiencia_cruzada_v2 import PARTITIONS, TEST_SEED, TRAIN_SEED, base, result_path, save_json
from piloto_planos_propostos import HERE, sha


def bound(real, history=None):
    # Best constant choice per repeated pair, with access to all held-out outcomes.
    # This is an oracle diagnostic, never supplied to an agent.
    groups = defaultdict(list)
    for i in range(len(real)):
        groups[b"all" if history is None else np.ascontiguousarray(history[i]).tobytes()].append(i)
    costs, choices = [], []
    regret = real.max(2)[:, :, None] - real
    for indices in groups.values():
        aggregate = regret[indices].sum(0)
        costs.append(float(aggregate.min(1).sum()))
        choices.append(aggregate.argmin(1).tolist())
    return {"minimum_empirical_mean_regret": sum(costs) / (len(real) * real.shape[1]),
            "observation_groups": len(groups), "oracle_choices": choices if history is None else None,
            "test_returns_used_only_for_diagnostic": True, "is_a_generalization_guarantee": False}


def main():
    output = HERE / "resultados" / "limites_empiricos_cruzada.json"
    games, physical = {}, 0
    for game in ("breakout", "pong"):
        with np.load(base(game, TEST_SEED).with_suffix(".npz")) as archive:
            h, returns = archive["histories"], archive["rewards"].cumsum(-1)
        path = result_path(game)
        result = json.loads(path.read_text(encoding="utf-8"))
        parts = {}
        for name, pairs in PARTITIONS.items():
            real = np.stack([returns[:, [p, q], -1] for p, q in pairs], axis=1)
            rows = result["partitions"][name]["curves"]["128"]
            floor = bound(real)
            parts[name] = {"program_only_oracle_bound128": floor,
                           "two_frame_observation_oracle_bound128": bound(real, h[:, -2:]),
                           "full_history_observation_oracle_bound128": bound(real, h),
                           "measured128": {arm: {"regret": row["mean_regret"], "mse": row["contrast_mse"],
                                                  "below_empirical_program_only_bound": bool(row["mean_regret"] < floor["minimum_empirical_mean_regret"] - 1e-12)}
                                           for arm, row in rows.items()},
                           "primary_gains": result["partitions"][name]["comparisons"],
                           "scope": "post_result_diagnostic_cannot_promote_a_control"}
        games[game] = {"partitions": parts, "report_sha256": sha(path), "bank_sha256": sha(base(game, TEST_SEED).with_suffix(".npz"))}
        for seed in (TRAIN_SEED, TEST_SEED):
            physical += json.loads(base(game, seed).with_suffix(".json").read_text(encoding="utf-8"))["ale_calls"]
    calibration = json.loads((HERE / "resultados" / "calibracao_restauracao_ale.json").read_text(encoding="utf-8"))
    old = json.loads((HERE / "resultados" / "breakout_experiencia_cruzada_131_interrupcao.json").read_text(encoding="utf-8"))
    report = {"status": "post_result_empirical_bounds_not_a_changed_gate", "games": games,
              "ale_calls_completed_banks": physical, "ale_calls_failed_attempt": old["ale_calls"],
              "ale_calls_calibration": calibration["ale_calls"],
              "total_ale_calls_including_failed_attempt_and_calibration": physical + old["ale_calls"] + calibration["ale_calls"],
              "source_sha256": sha(HERE / "diagnosticar_cruzada_limites.py"), "human_level_goal_achieved": False}
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != report:
            raise RuntimeError("diagnóstico existe e diverge; preservar")
    else:
        save_json(output, report)
    compact = {game: {name: {"word_only_floor": part["program_only_oracle_bound128"]["minimum_empirical_mean_regret"],
                             "history_floor": part["full_history_observation_oracle_bound128"]["minimum_empirical_mean_regret"],
                             "measured": part["measured128"],
                             "primary_vs_words": part["primary_gains"]["word_only"],
                             "primary_vs_shuffled_history": part["primary_gains"]["shuffled_histories"]}
                     for name, part in item["partitions"].items()} for game, item in games.items()}
    print(json.dumps({"games": compact, "total_ale_calls": report["total_ale_calls_including_failed_attempt_and_calibration"]}, indent=2))


if __name__ == "__main__":
    main()
