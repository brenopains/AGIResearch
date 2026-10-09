"""Categorias mantêm eventos inteiros; composição e corpo são verificáveis."""

import itertools
import tempfile
from pathlib import Path

import numpy as np

from desfechos_aprendidos import OutcomeModel, OutcomePlanner, prototypes


def main():
    x = np.asarray([[0.], [1.], [2.], [3.]])
    edges = [{"source": x, "successor": x + 1, "reward": np.asarray([0, 1, 0, 1]),
              "continuation": np.asarray([1, 1, 0, 1])},
             {"source": x, "successor": x - 1, "reward": np.asarray([1, 0, 1, 0]),
              "continuation": np.ones(4)}]
    for e in edges:
        ids, labels = prototypes(e, 2)
        assert np.all(e["reward"][ids][labels] == e["reward"])
        assert np.all(e["continuation"][ids][labels] == e["continuation"])
        assert labels.min() == 0 and labels.max() < len(ids)
    model = OutcomeModel().fit(edges, 2)
    probabilities = model.posterior(x, 0)
    assert np.all(probabilities >= 0) and np.allclose(probabilities.sum(1), 1)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "body.npz"
        model.save(path)
        restored = OutcomeModel.load(path)
        assert np.allclose(restored.posterior(x + .1, 0), model.posterior(x + .1, 0), atol=1e-12)
    planner = OutcomePlanner(model)
    plans = list(itertools.product(range(2), repeat=4))
    truth = planner.plan_values(x, plans)
    found = planner.search(x, 4, 16)
    assert np.allclose(found["values"], truth.max(1))
    terminal = [{"source": [[0.]], "successor": [[10.]], "reward": [3.], "continuation": [0.]}]
    assert OutcomePlanner(OutcomeModel().fit(terminal)).search(np.asarray([[0.]]), 6, 1)["values"] == [3.]
    print("desfechos_aprendidos_ok: eventos_conjuntos, probabilidades, corpo_exportado, busca_exata_pequena, termino")


if __name__ == "__main__":
    main()
