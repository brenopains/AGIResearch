# operador proposto pelo modelo local; fora do nucleo

def distancia_euclidiana(valores, meu):
    """valores: lista de numeros — o mesmo observável medido em TODOS os
                    objetos (ex.: o tamanho de cada objeto da cena).
           meu:     o numero deste objeto (esta em `valores`).
           devolve: um INTEIRO pequeno (0 a ~20), deterministico."""
    if len(valores) < 2:
        return 0  # Caso especial: se tiver apenas um objeto, consideramos o centro como 0.

    # Calcular a média dos valores
    media = sum(valores) / len(valores)

    # Calcular a distância do objeto para a média
    distancia = abs(meu - media)

    # Mapa a distância para um valor entre 0 e ~20
    return int(distancia * 20 / max(valores) - 1)
