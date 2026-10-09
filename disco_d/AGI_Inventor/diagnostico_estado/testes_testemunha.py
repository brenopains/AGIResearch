"""Contratos exatos e de falta de modelo para a seleção de experimentos."""

from testemunha import Hipotese, Intervalo, encontrar, incorporar


def main():
    a = Hipotese("linear", {1: Intervalo(1, 1), 2: Intervalo(2, 2)}, "programa", True)
    b = Hipotese("quadratico", {1: Intervalo(1, 1), 2: Intervalo(4, 4)}, "programa", True)
    witness = encontrar([a, b], [1, 2])
    assert witness["status"] == "testemunha" and witness["consulta"] == 2
    assert witness["alcance"] == "exato"
    assert incorporar([a, b], 2, 4)["sobreviventes"] == ["quadratico"]
    assert incorporar([a, b], 2, 7)["status"] == "modelos_inadequados"
    assert encontrar([a, b], [1])["status"] == "indistinguiveis_no_dominio"
    overlap = [Hipotese("a", {2: Intervalo(0, 3)}, "calibrado", False),
               Hipotese("b", {2: Intervalo(2, 5)}, "calibrado", False)]
    assert encontrar(overlap, [2])["status"] == "indistinguiveis_no_dominio"
    print("testemunhas_ok")


if __name__ == "__main__":
    main()
