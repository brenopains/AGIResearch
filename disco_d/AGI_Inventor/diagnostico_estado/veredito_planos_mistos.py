"""Aplica sem ajuste o pré-registro da semente 109."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN = {
    "coletar_efeitos_locais.py": "1afbc809e3e5b14ae1d823db236cb6e06eb1171ab95ca9f3c8446c371dda0392",
    "coletar_horizontes.py": "8d89851cdbe7d070510b96a9a3f41c7a312f2bb34effb82960ad0c7fb9c6d631",
    "coletar_planos_mistos.py": "e06f4786f12435c2a9822d29d6d69c7686d11a522e2c12f189b26b5b18d24daa",
    "avaliar_planos_mistos.py": "74ead3119eca78bffbfa8845243aa7a85e6196cbc5ac9ce2d531a5c62de4d9fd",
    "avaliar_estado_preditivo.py": "fdcde6494a1c97eba5453700b72ba9a079d43dc327b70da69f9c955b3b6a8d8f",
}


def verdict():
    hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
              for name in FROZEN}
    if hashes != FROZEN:
        raise RuntimeError("código difere do pré-registro")
    result = {"pre_registration": "PRE_REGISTRO_PLANOS_109.md",
              "seed": 109, "hashes_verified": True, "games": {}}
    for game in ("breakout", "pong"):
        source = HERE / "resultados" / f"{game}_avaliacao_planos_mistos_109.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        row = data["curves"]["64"]
        proposed = row["estado_previsto"]
        tests = {"ratio_below_0_75": proposed["mean"] < 0.75}
        controls = {}
        for name in ("acoes_permutadas", "assinatura_persistente"):
            control = row[name]
            paired = np.asarray(control["per_episode"], dtype=np.float64) - np.asarray(
                proposed["per_episode"], dtype=np.float64)
            if not np.all(np.isfinite(paired)):
                raise RuntimeError(f"episódio sem pontuação: {game}, {name}")
            rng = np.random.default_rng(7701)
            indexes = rng.integers(0, 8, size=(10000, 8))
            means = paired[indexes].mean(axis=1)
            ci = np.quantile(means, (0.025, 0.975))
            checks = {
                "mean_better": proposed["mean"] < control["mean"],
                "paired_positive_at_least_6_of_8": int(np.sum(paired > 0)) >= 6,
                "bootstrap_95_lower_above_zero": float(ci[0]) > 0,
            }
            controls[name] = {"ratio": control["mean"],
                              "paired_differences": paired.tolist(),
                              "wins": int(np.sum(paired > 0)),
                              "mean_difference": float(np.mean(paired)),
                              "bootstrap_95": ci.tolist(),
                              "checks": checks}
            tests.update({f"{name}_{key}": value for key, value in checks.items()})
        result["games"][game] = {
            "proposed_ratio": proposed["mean"],
            "valid_states_at_64": data["valid_by_stage"][8],
            "controls": controls, "checks": tests,
            "passed": bool(all(tests.values())),
            "evaluation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    result["confirmed_by_preregistered_gate"] = all(
        game["passed"] for game in result["games"].values())
    output = HERE / "resultados" / "veredito_planos_mistos_109.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(verdict(), ensure_ascii=False, indent=2))
