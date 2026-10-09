"""Corpo exportado, crédito observado e escala determinada só pelo treino."""

import tempfile
from pathlib import Path

import numpy as np

from coordenadas_consequencias import ConsequenceGeometry, ConsequenceReadout, prepare_edges


def main():
    x = np.arange(16, dtype=float).reshape(8, 2)
    e = [{"source": x, "successor": x + .25, "reward": np.asarray([0, 0, 0, 1, 1, 0, 0, 0]),
          "continuation": np.ones(8)}]
    readout = ConsequenceReadout().fit(e)
    query = x + .1
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "body.npz"
        readout.save(path)
        restored = ConsequenceReadout.load(path)
        assert np.allclose(readout.coordinates(query), restored.coordinates(query), atol=1e-12)
    geometry = ConsequenceGeometry(readout, x)
    original_scales = (geometry.scale_z, geometry.scale_q)
    transformed = geometry.transform(query)
    assert transformed.shape == (8, 4) and np.isfinite(transformed).all()
    assert (geometry.scale_z, geometry.scale_q) == original_scales
    altered = prepare_edges(e, readout, geometry)
    assert np.array_equal(e[0]["continuation"], altered[0]["continuation"])
    assert np.array_equal(e[0]["source"], x)
    assert np.all((readout.coordinates(query)[:, 0] >= 0) & (readout.coordinates(query)[:, 0] <= 1))
    assert np.all(readout.coordinates(query)[:, 1] == 1)
    print("coordenadas_consequencias_ok: corpo_reexecutavel, geometria_de_treino, limite, termino_preservado")


if __name__ == "__main__":
    main()
