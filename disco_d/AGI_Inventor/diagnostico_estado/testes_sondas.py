"""Controle positivo: o dado exige uma pergunta composta não cadastrada."""

import numpy as np

from sondas_contraexemplo import Budget, BudgetExhausted, discover_pair


def main():
    calls = []

    def observe(root, program):
        calls.append((root, program))
        # Um mundo de teste cuja identidade só aparece após uma sequência.
        image = np.zeros((4, 4))
        if tuple(program) == (2, 1):
            image[1, 3] = root
        return image

    found, events = discover_pair(observe, 0, 1, 4, 2)
    assert len(events) == 20 and len(calls) == 40
    assert len(found) == 1 and found[0]["actions"] == [2, 1]
    vector = np.asarray(found[0]["readout"])
    assert vector.shape == (16,) and np.isclose(np.linalg.norm(vector), 1)
    assert abs((observe(0, (2, 1)).ravel() - observe(1, (2, 1)).ravel()) @ vector) == 1
    none, _ = discover_pair(lambda root, program: np.zeros(3), 0, 1, 4, 2)
    assert not none
    budget = Budget(1)
    budget.take()
    try:
        budget.take()
    except BudgetExhausted:
        pass
    else:
        raise AssertionError("orçamento não foi aplicado")
    print("sondas_ok: sequencia_composta, leitura_do_residuo, ausencia_de_sinal, orcamento")


if __name__ == "__main__":
    main()
