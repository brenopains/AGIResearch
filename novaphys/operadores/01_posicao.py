# operador proposto pelo modelo local; fora do nucleo

def posicao(valores, meu):
    if not valores:
        return 0

    min_val = min(valores)
    max_val = max(valores)
    num_values = len(valores)

    # Calculate the range of the values
    range_values = max_val - min_val

    # Calculate the position of the object
    position = int((meu - min_val) / (range_values + 1) * (num_values - 1))

    # Adjust position to be within the range of valid indices
    position = max(0, min(position, num_values - 1))

    return position
