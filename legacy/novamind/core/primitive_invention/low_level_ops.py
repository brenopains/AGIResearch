"""
LowLevelOperations — Operacoes de pixel UNIVERSALS, nao primitives.

Diferenca de primitives:
  - Primitives sao OBSERVAVEIS: rotate_90, flip_h, recolor, remove...
  - Low-level ops sao CONSTRUTIVAS: copiar pixel, transladar, inverter eixo...

Estas operacoes sao os "tijolos" com os quais novas primitives serao construidas.
Qualquer transformacao de grid pode ser expressa como composicao destas ops.
"""
from typing import List, Callable, Tuple
import numpy as np


# ─── Definicão de uma operacao de baixo nivel ─────────────

LowLevelOp = Callable[[np.ndarray], np.ndarray]

# ─── Catalogo de operacoes ────────────────────────────────

OPS: List[Tuple[str, LowLevelOp, str]] = []


def _register(name: str, fn: LowLevelOp, desc: str = ""):
    OPS.append((name, fn, desc))


# Transformacoes geometricas basicas (NENHUMA e primitiva — sao construtivas)

_register("transpose", lambda g: np.transpose(g).copy(), "swap axes")

_register("flip_lr", lambda g: np.fliplr(g), "mirror left-right")

_register("flip_ud", lambda g: np.flipud(g), "mirror up-down")

_register("rot90_cw", lambda g: np.rot90(g, k=-1), "rotate 90 deg clockwise")

_register("rot90_ccw", lambda g: np.rot90(g, k=1), "rotate 90 deg counter-clockwise")

_register("rot180", lambda g: np.rot90(g, k=2), "rotate 180 deg")


def _shift_0(grid: np.ndarray) -> np.ndarray:
    return np.roll(grid, shift=1, axis=0)


def _shift_1(grid: np.ndarray) -> np.ndarray:
    return np.roll(grid, shift=-1, axis=0)


def _shift_2(grid: np.ndarray) -> np.ndarray:
    return np.roll(grid, shift=1, axis=1)


def _shift_3(grid: np.ndarray) -> np.ndarray:
    return np.roll(grid, shift=-1, axis=1)


_register("shift_down", _shift_0, "shift all rows down by 1")
_register("shift_up", _shift_1, "shift all rows up by 1")
_register("shift_right", _shift_2, "shift all cols right by 1")
_register("shift_left", _shift_3, "shift all cols left by 1")

_register("identity", lambda g: g.copy(), "no change")


def _tile_2x2(grid: np.ndarray) -> np.ndarray:
    return np.tile(grid, (2, 2))


def _tile_2x1(grid: np.ndarray) -> np.ndarray:
    return np.tile(grid, (2, 1))


def _tile_1x2(grid: np.ndarray) -> np.ndarray:
    return np.tile(grid, (1, 2))


_register("tile_2x2", _tile_2x2, "tile grid 2x2")
_register("tile_2x1", _tile_2x1, "tile grid vertically 2x")
_register("tile_1x2", _tile_1x2, "tile grid horizontally 2x")


def _crop_to_content(grid: np.ndarray) -> np.ndarray:
    rows, cols = np.where(grid > 0)
    if len(rows) == 0:
        return grid.copy()
    return grid[rows.min():rows.max()+1, cols.min():cols.max()+1].copy()


_register("crop_content", _crop_to_content, "crop to non-zero bounding box")


def _pad_1(grid: np.ndarray) -> np.ndarray:
    return np.pad(grid, 1, mode='constant', constant_values=0)


_register("pad_1", _pad_1, "add 1px zero padding")


def _invert_colors(grid: np.ndarray) -> np.ndarray:
    """Inverte cores (preserva background 0)."""
    result = grid.copy()
    mask = result > 0
    if mask.any():
        unique = np.unique(result[mask])
        mapping = {c: 10 - c for c in unique}
        for old, new in mapping.items():
            result[result == old] = new
    return result


_register("invert_colors", _invert_colors, "invert non-zero colors")


def _threshold_4(grid: np.ndarray) -> np.ndarray:
    """Mantem so cores > 4."""
    result = grid.copy()
    result[result <= 4] = 0
    return result


_register("threshold_4", _threshold_4, "keep only colors > 4")


def get_ops() -> List[Tuple[str, LowLevelOp, str]]:
    return list(OPS)


def get_op_names() -> List[str]:
    return [name for name, _, _ in OPS]
