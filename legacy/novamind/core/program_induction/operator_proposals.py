"""
Operator Proposals — Transformacoes espaciais PROPOSTAS a partir de conceitos do CGE.

Nenhuma e primitiva valida ate ser VALIDADA por execucao.

Cada operador aqui e uma "hipotese de transformacao" que o CGE sugere
(baseado em conceitos como simetria, rotacao, etc.). O sistema entao
VALIDA por execucao: se aplicar ao input produz o output, vira primitiva.

As funcoes de execucao sao OPERACOES GERAIS DE GRID (numpy nativas).
Nao ha logica especifica de ARC.
"""
from typing import Dict, Optional, Callable
import numpy as np


# ─── Spatial Operators ─────────────────────────────────────

# Cada operador e uma tupla: (nome, descricao, function, param_schema)

SPATIAL_OPERATORS = []


def _register(name, description, fn, schema=None):
    SPATIAL_OPERATORS.append({
        "name": name,
        "description": description,
        "function": fn,
        "schema": schema or [],
    })


# Rotacoes (numpy rot90)
_register(
    "rotate_90", "rotate grid 90 degrees clockwise",
    lambda g: np.rot90(g, k=-1),
)
_register(
    "rotate_180", "rotate grid 180 degrees",
    lambda g: np.rot90(g, k=2),
)
_register(
    "rotate_270", "rotate grid 270 degrees clockwise",
    lambda g: np.rot90(g, k=1),
)

# Reflexoes (numpy flip)
_register(
    "flip_horizontal", "mirror grid horizontally (left-right)",
    lambda g: np.fliplr(g),
)
_register(
    "flip_vertical", "mirror grid vertically (up-down)",
    lambda g: np.flipud(g),
)

# Transposicao
_register(
    "transpose", "transpose grid (swap axes)",
    lambda g: np.transpose(g).copy(),
)

# Identidade
_register(
    "identity", "no transformation",
    lambda g: g.copy(),
)


def detect_applicable_operators(input_grid: np.ndarray,
                                 output_grid: np.ndarray) -> list:
    """Detecta quais operadores espaciais TRANSFORMAM input em output.

    Nao e busca. E verificacao direta: aplica operador, compara.

    Returns lista de operadores que produziram match exato.
    """
    results = []
    for op in SPATIAL_OPERATORS:
        try:
            result = op["function"](input_grid)
            if result.shape == output_grid.shape and np.array_equal(result, output_grid):
                results.append(op)
        except Exception:
            pass
    return results


def get_all_proposals() -> list:
    """Retorna todos os operadores propostos."""
    return list(SPATIAL_OPERATORS)


def get_proposal_names() -> list:
    return [op["name"] for op in SPATIAL_OPERATORS]
