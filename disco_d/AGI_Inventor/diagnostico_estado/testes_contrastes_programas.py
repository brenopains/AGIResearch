"""Simetrias, dependência de contexto, corpos e diagnósticos."""

import tempfile
from pathlib import Path

import numpy as np

from contrastes_programas import ProgramComparison


def main():
    x = np.asarray([[-1.], [1.], [-2.], [2.]])
    p, q, r = [[0]] * 4, [[1]] * 4, [[2]] * 4
    yl, yr, roots = x[:, 0], -x[:, 0], np.arange(4)
    for mode in ("conditional", "shuffled_targets", "shuffled_states", "word_only", "absolute_returns"):
        model = ProgramComparison().fit(x, p, q, yl, yr, roots, mode)
        assert model.system_size == 8
        pq, qp, pp = model.predict(x, p, q), model.predict(x, q, p), model.predict(x, p, p)
        assert np.allclose(pq, -qp, atol=1e-12) and np.allclose(pp, 0, atol=1e-12)
        assert np.allclose(pq + model.predict(x, q, r), model.predict(x, p, r), atol=1e-12)
        if mode == "conditional":
            assert np.array_equal(np.sign(pq), np.sign(yl - yr))
        if mode == "word_only":
            assert np.allclose(model.predict(x + 100, p, q), pq)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "comparison.npz"
            model.save(path)
            restored = ProgramComparison.load(path)
            assert np.allclose(restored.predict(x + .1, p, q), model.predict(x + .1, p, q), atol=1e-12)
    try:
        ProgramComparison().fit(x, p, p, yl, yr, roots)
    except ValueError as error:
        assert "contraditória" in str(error)
    else:
        raise AssertionError("não diagnosticou par determinístico contraditório")
    print("contrastes_programas_ok: reflexividade, antissimetria, aditividade, contexto, exportacao, contradicao")


if __name__ == "__main__":
    main()
