# operador proposto pelo modelo local; fora do nucleo

import math

def distancia_euclidiana(valores, meu):
    """valores: lista de numeros — o mesmo observavel medido em TODOS os
                    objetos (ex.: o tamanho de cada objeto da cena).
           meu:     o numero deste objeto (esta em `valores`).
           devolve: um INTEIRO pequeno (0 a ~20), deterministico."""
    total = 0
    for valor in valores:
        total += (valor - meu) ** 2
    return int(math.sqrt(total))
