# operador proposto pelo modelo local; fora do nucleo

def posicao(valores, meu):
    """valores: lista de numeros — o mesmo observavel medido em TODOS os
                    objetos (ex.: o tamanho de cada objeto da cena).
           meu:     o numero deste objeto (esta em `valores`).
           devolve: um INTEIRO pequeno (0 a ~20), deterministico."""
    # Ordenar os valores em ordem inversa
    sorted_values = sorted(valores, reverse=True)
    # Encontrar o índice da posição do meu valor
    return sorted_values.index(meu)
