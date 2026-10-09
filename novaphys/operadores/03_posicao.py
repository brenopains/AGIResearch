# operador proposto pelo modelo local; fora do nucleo

def posicao(valores, meu):
    """valores: lista de numeros — o mesmo observavel medido em TODOS os
                    objetos (ex.: o tamanho de cada objeto da cena).
           meu:     o numero deste objeto (esta em `valores`).
           devolve: um INTEIRO pequeno (0 a ~20), deterministico."""
    # Encontre o índice do objeto 'meu' na lista 'valores'
    index = valores.index(meu)
    # Encontre o número mínimo na lista 'valores'
    min_valor = min(valores)
    # Encontre o número máximo na lista 'valores'
    max_valor = max(valores)
    
    # Cálculo da posição relativa do objeto 'meu'
    # A posição é determinada pela distância do 'meu' ao mínimo e máximo, normalizada para 0 a 20
    position = int(((min_valor - meu) / (max_valor - min_valor)) * 20)
    
    return position
