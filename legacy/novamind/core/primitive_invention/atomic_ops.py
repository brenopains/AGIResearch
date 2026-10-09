"""
Atomic Operations — VERDADEIRAMENTE atomicas.

NAO contem rotacao, flip, transpose, ou qualquer operacao derivada.
Contem APENAS:
  - loop_x: itera sobre colunas
  - loop_y: itera sobre linhas
  - read(grid, x, y): le valor do pixel
  - write(result, x, y, val): escreve valor
  - coord_transform((x, y), mode): transforma coordenadas

O sistema descobre COMPOSICOES destes operandos para formar rotacao.
"""
from typing import Tuple, Callable
import numpy as np


# ─── Coordenadas ───────────────────────────────────────────

# Modos de transformacao de coordenadas (sao os UNICOS operandos de transformacao)
COORD_MODES = {
    "identity": lambda x, y, h, w: (x, y),
    "swap": lambda x, y, h, w: (y, x),
    "invert_x": lambda x, y, h, w: (w - 1 - x, y),
    "invert_y": lambda x, y, h, w: (x, h - 1 - y),
    "swap_invert_x": lambda x, y, h, w: (y, w - 1 - x),     # rot90 CW (usa w apos swap!)
    "swap_invert_y": lambda x, y, h, w: (h - 1 - y, x),     # rot90 CCW (usa h apos swap!)
    "invert_both": lambda x, y, h, w: (w - 1 - x, h - 1 - y), # rot180
    "swap_invert_both": lambda x, y, h, w: (h - 1 - y, w - 1 - x), # rot180 via swap
}

COORD_MODE_NAMES = list(COORD_MODES.keys())


# ─── Programas construidos de operandos atomicos ───────────

def make_loop_program(coord_mode: str) -> Callable:
    """Gera um programa que aplica uma transformacao de coordenadas.

    A estrutura do programa e:
      loop_y:
        loop_x:
          read(grid, x, y)
          transform(coord_mode, x, y)
          write(result, tx, ty, val)

    O que VARIA entre programas e o MODO DE COORDENADA, nada mais.
    """
    transform = COORD_MODES[coord_mode]

    def program(grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        # O shape do resultado depende da transformacao
        if "swap" in coord_mode:
            result = np.zeros((w, h), dtype=grid.dtype)
        else:
            result = np.zeros_like(grid)

        for y in range(h):
            for x in range(w):
                val = grid[y, x]
                tx, ty = transform(x, y, h, w)
                # Garante que esta dentro dos limites
                if 0 <= tx < result.shape[1] and 0 <= ty < result.shape[0]:
                    result[ty, tx] = val

        return result

    return program


# ─── Registro ──────────────────────────────────────────────

def get_atomic_programs() -> list:
    """Retorna todos os programas atomicos (loop + coord_transform)."""
    programs = []
    for mode_name in COORD_MODE_NAMES:
        fn = make_loop_program(mode_name)
        programs.append((mode_name, fn, f"loop with coord transform: {mode_name}"))
    return programs


def get_atomic_program_names() -> list:
    return list(COORD_MODE_NAMES)
