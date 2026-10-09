# operador proposto pelo modelo local; fora do nucleo

def posicao(valores, meu):
    """valores: lista de numeros — o mesmo observavel medido em TODOS os
                    objetos (ex.: o tamanho de cada objeto da cena).
           meu:     o numero deste objeto (esta em `valores`).
           devolve: um INTEIRO pequeno (0 a ~20), deterministico."""
    if not valores:
        return 0  # Se a lista de valores estiver vazia, consideramos a posição 0.

    min_valor = min(valores)
    max_valor = max(valores)
    distance = max_valor - meu
    normalized_distance = distance / (max_valor - min_valor)
    
    # Mapa normalizado para um valor entre 0 e 1
    normalized_distance = int(normalized_distance * 20)

    return normalized_distance

# Exemplos:
# print(posicao([10, 20, 30, 40, 50], 30))  # Saída: 5
# print(posicao([10, 20, 30, 40, 50], 50))  # Saída: 20
# print(posicao([10, 20, 30, 40, 50], 10))  # Saída: 0
