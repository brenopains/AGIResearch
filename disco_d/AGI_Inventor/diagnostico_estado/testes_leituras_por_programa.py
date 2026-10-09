"""Produtos de leituras, execução do corpo e equivalência explícita."""

import tempfile
from pathlib import Path

import numpy as np

from leituras_por_programa import ConditionalReading, ReadingBody


def main():
    h = np.asarray([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])[:, None]
    left, right = [[0]] * 4, [[1]] * 4
    target, zero, origins = (h[:, 0, 0] * h[:, 0, 1]), np.zeros(4), np.arange(4)
    primary = ConditionalReading().fit(h, left, right, target, zero, origins)
    affine = ConditionalReading().fit(h, left, right, target, zero, origins, "affine")
    query = np.asarray([[-.5, -.5], [-1.5, 1.5], [1.5, -.5], [.5, 1.5]])[:, None]
    p, q, r = [[0]] * 4, [[1]] * 4, [[2]] * 4
    prediction = primary.predict(query, p, q)
    assert np.array_equal(np.sign(prediction), np.sign(query[:, 0, 0] * query[:, 0, 1]))
    assert np.mean((primary.predict(h, left, right) - target) ** 2) < .01
    assert np.mean((affine.predict(h, left, right) - target) ** 2) > .9
    assert np.allclose(prediction, primary.predict_dual(query, p, q), atol=1e-12)
    assert np.allclose(primary.predict(query, p, p), 0, atol=1e-12)
    assert np.allclose(prediction, -primary.predict(query, q, p), atol=1e-12)
    assert np.allclose(prediction + primary.predict(query, q, r), primary.predict(query, p, r), atol=1e-12)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "generator.npz"
        primary.save(path)
        loaded = ConditionalReading.load(path)
        assert np.allclose(loaded.predict(query, p, q), prediction, atol=1e-12)
        body_path = Path(folder) / "reading.npz"
        body = primary.construct([[0]])
        body.save(body_path)
        restored = ReadingBody.load(body_path)
        assert np.allclose(restored.apply(query), body.apply(query), atol=1e-12)
        # Make the fixed feature space explicit. Conditioning selects its weights.
        normalized = (query.reshape(4, -1) - body.mean) / body.scale
        features = np.column_stack((np.ones(4), normalized, (normalized @ body.templates.T) ** 2))
        weights = np.concatenate((body.bias, body.linear.reshape(-1), body.quadratic.reshape(-1)))
        assert np.allclose(features @ weights, body.apply(query), atol=1e-12)
    all_h = np.repeat(h, 4, axis=1)
    for mode in ("two_frames", "shuffled_targets", "shuffled_histories", "word_only"):
        model = ConditionalReading().fit(all_h, left, right, target, zero, origins, mode)
        assert model.system_size == 8
        assert np.allclose(model.predict(all_h, left, right), model.predict_dual(all_h, left, right), atol=1e-12)
    for action in (
        lambda: ConditionalReading(alpha=0),
        lambda: ConditionalReading().fit(h, left, left, target, zero, origins),
        lambda: primary.construct([[4]]),
        lambda: primary.predict(np.full_like(h, np.nan), left, right),
    ):
        try:
            action()
        except ValueError:
            pass
        else:
            raise AssertionError("entrada inválida/contraditória não foi diagnosticada")
    print("leituras_por_programa_ok: produto_expressivo, corpo_executado, simetrias, exportacao, espaco_fixo_equivalente")


if __name__ == "__main__":
    main()
