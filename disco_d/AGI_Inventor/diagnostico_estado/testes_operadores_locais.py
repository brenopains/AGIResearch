"""O corpo reproduz experiência e conserva contexto novo numa translação."""

import numpy as np

from operadores_locais import LocalOperators


def main():
    source = np.asarray([[0., 0.], [10., 10.]])
    successor = source + [1, -2]
    operators = LocalOperators(source, successor, 1)
    query = np.asarray([[2., 3.]])
    assert np.allclose(operators.predict(query), [[3, 1]])
    candidates, probability, ids = operators.candidates(query)
    assert candidates.shape == (1, 1, 2)
    assert np.allclose(probability.sum(1), 1)
    assert np.allclose(candidates[:, 0], operators.predict(query))
    for i in range(2):
        body = operators.definition(i)["body"]["add"][1]
        assert np.allclose(source[i] + body, successor[i])
    assert np.array_equal(source, [[0, 0], [10, 10]])
    print("operadores_locais_ok: corpo_extraido, contexto_novo, alternativas_inteiras, origem")


if __name__ == "__main__":
    main()
