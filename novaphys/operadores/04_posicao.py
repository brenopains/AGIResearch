# operador proposto pelo modelo local; fora do nucleo

def posicao(valores, meu):
    """valores: lista de numeros — o mesmo observavel medido em TODOS os
                    objetos (ex.: o tamanho de cada objeto da cena).
           meu:     o numero deste objeto (esta em `valores`).
           devolve: um INTEIRO pequeno (0 a ~20), deterministico."""
    if not valores:
        return 0
    min_val = min(valores)
    max_val = max(valores)
    range_val = max_val - min_val
    
    # Distribuição uniforme
    normalized_position = (meu - min_val) / range_val
    
    # Mapa para a posição relativa
    if normalized_position < 0.1:
        return 0
    elif normalized_position < 0.2:
        return 1
    elif normalized_position < 0.3:
        return 2
    elif normalized_position < 0.4:
        return 3
    elif normalized_position < 0.5:
        return 4
    elif normalized_position < 0.6:
        return 5
    elif normalized_position < 0.7:
        return 6
    elif normalized_position < 0.8:
        return 7
    elif normalized_position < 0.9:
        return 8
    else:
        return 9
