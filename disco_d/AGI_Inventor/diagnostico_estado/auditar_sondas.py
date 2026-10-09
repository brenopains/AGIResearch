"""Veredito conservador: nenhuma candidata vira capacidade por ajuste local."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def audit():
    result = {"status": "human_level_goal_not_achieved", "games": {}}
    for game in ("breakout", "pong"):
        path = HERE / "resultados" / f"{game}_conjuntos_sondas.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        proposed = data["arms"]["witness_readout"]
        controls = {"base": data["baseline"],
                    "random": data["arms"]["random_readout"],
                    "shuffle": data["arms"]["shuffled_feature_rows"]}
        checks = {}
        for cohort in ("evaluation", "seed103"):
            for name, control in controls.items():
                checks[f"{cohort}_better_than_{name}"] = (
                    proposed[cohort]["mean_regret"] < control[cohort]["mean_regret"] - 1e-12)
        positive_development = bool(all(checks.values()))
        zero_margin = data["strict_gain_gate_zero_margin_episodes"]
        status = ("inconclusive_gate_zero_margin" if zero_margin else
                  "needs_fresh_confirmation" if positive_development else
                  "rejected_no_incremental_decision_utility")
        result["games"][game] = {
            "status": status, "selected": proposed["selected"],
            "positive_development": positive_development,
            "checks": checks, "zero_margin_episodes": zero_margin,
            "independent_confirmation_available": False,
            "promoted": False,
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    result["promoted_probes"] = 0
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = HERE / "resultados" / "auditoria_sondas.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
