"""Conferir o limite de regras só de programa por enumeração independente."""

import itertools
import json

import numpy as np

from diagnosticar_cruzada_limites import bound
from experiencia_cruzada_v2 import PARTITIONS, base
from piloto_planos_propostos import HERE


def enumerate_policies(real):
    regrets = []
    for policy in itertools.product((0, 1), repeat=real.shape[1]):
        chosen = np.stack([real[:, p, side] for p, side in enumerate(policy)], axis=1)
        regrets.append(float((real.max(2) - chosen).mean()))
    return min(regrets)


def main():
    real = np.asarray([[[1., 0.], [1., 0.]], [[0., 1.], [1., 0.]]])
    assert bound(real)["minimum_empirical_mean_regret"] == enumerate_policies(real) == .25
    assert bound(real, np.zeros((2, 1)))["minimum_empirical_mean_regret"] == .25
    assert bound(real, np.arange(2)[:, None])["minimum_empirical_mean_regret"] == 0.
    report = json.loads((HERE / "resultados" / "limites_empiricos_cruzada.json").read_text(encoding="utf-8"))
    for game in ("breakout", "pong"):
        with np.load(base(game, 137).with_suffix(".npz")) as bank:
            observed = bank["rewards"].cumsum(-1)
        for name, pairs in PARTITIONS.items():
            real = np.stack([observed[:, [p, q], -1] for p, q in pairs], axis=1)
            expected = enumerate_policies(real)
            measured = report["games"][game]["partitions"][name]["program_only_oracle_bound128"]["minimum_empirical_mean_regret"]
            assert measured == expected
    print("limites_cruzada_ok: enumeracao_das_quatro_politicas_e_colisoes")


if __name__ == "__main__":
    main()
