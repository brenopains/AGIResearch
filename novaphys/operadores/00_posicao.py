# operador proposto pelo modelo local; fora do nucleo

def posicao(valores, meu):
    """valores: lista de numeros — o mesmo observável medido em TODOS os
           objetos (ex.: o tamanho de cada objeto da cena).
           meu:     o numero deste objeto (esta em `valores`).
           devolve: um INTEIRO pequeno (0 a ~20), deterministico."""
    if not valores or meu not in valores:
        raise ValueError("Lista de valores vazia ou meu não está na lista.")
    
    sorted_valores = sorted(set(valores))  # Remove duplicatas e ordena os valores
    return sorted_valores.index(meu)  # Retorna a posição do meu na lista ordenada
