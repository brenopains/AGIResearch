# operador proposto pelo modelo local; fora do nucleo

def distancia_media_relativa(valores, meu):
    """valores: lista de numeros — o mesmo observavel medido em TODOS os
                    objetos (ex.: o tamanho de cada objeto da cena).
           meu:     o numero deste objeto (esta em `valores`).
           devolve: um INTEIRO pequeno (0 a ~20), deterministico."""
    total_distancias = 0
    for valor in valores:
        if valor != meu:
            total_distancias += abs(valor - meu)
    
    media = total_distancias // len(valores)  # Divisão inteira para garantir um inteiro
    return media
