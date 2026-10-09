"""
Pruning — Poda estrutural de composicoes impossiveis.

Nao sao regras ARC. Sao constraints GERAIS sobre qualquer grid:
  - Nao faz sentido recolorir uma cor que nao existe
  - Nao faz sentido remover uma cor que nao existe
  - Nao faz sentido mover se nao ha objetos
  - Nao faz sentido adicionar se o grid esta cheio
  - Nao faz sentido recolorir depois de remover a mesma cor

Estas constraints valem para QUALQUER grid, nao so ARC.
"""
from typing import List, Dict, Optional
import numpy as np


def _object_colors(grid: np.ndarray) -> set:
    return set(int(c) for c in np.unique(grid) if c != 0)


_pruning_rules = []
_pruning_rules_names = []


def pruning_rule(name: str):
    """Decorator para registrar regra de pruning."""
    def decorator(fn):
        _pruning_rules.append(fn)
        _pruning_rules_names.append(name)
        return fn
    return decorator


def check_pruning(history: List[Dict], current_grid: np.ndarray,
                  proposed_type: str, proposed_params: Dict) -> List[str]:
    """Verifica se uma composicao e estruturalmente valida.

    Args:
        history: lista de dicts com {"type": str, "params": dict} dos passos anteriores
        current_grid: grid atual (depois dos passos anteriores)
        proposed_type: structure_type da primitiva proposta
        proposed_params: parametros inferidos

    Returns:
        lista de razoes pelas quais a composicao e invalida (vazia = valida)
    """
    reasons = []
    for rule_fn in _pruning_rules:
        reason = rule_fn(history, current_grid, proposed_type, proposed_params)
        if reason:
            reasons.append(reason)
    return reasons


# ─── Regras de pruning ─────────────────────────────────────


@pruning_rule("recolor_color_exists")
def _recolor_color_exists(history, grid, ptype, params):
    """Recolor: from_color deve existir no grid."""
    if ptype != "recolor":
        return ""
    from_c = params.get("from_color", -1)
    colors = _object_colors(grid)
    if from_c not in colors:
        return f"recolor({from_c}) impossible: color {from_c} not in grid"


@pruning_rule("remove_color_exists")
def _remove_color_exists(history, grid, ptype, params):
    """Remove: color deve existir no grid."""
    if ptype != "remove":
        return ""
    color = params.get("color", -1)
    colors = _object_colors(grid)
    if color not in colors:
        return f"remove({color}) impossible: color {color} not in grid"


@pruning_rule("move_objects_exist")
def _move_objects_exist(history, grid, ptype, params):
    """Move: grid precisa ter objetos."""
    if ptype != "move":
        return ""
    if not np.any(grid > 0):
        return "move impossible: no objects in grid"


@pruning_rule("add_grid_not_full")
def _add_grid_not_full(history, grid, ptype, params):
    """Add: grid nao pode estar completamente cheio."""
    if ptype != "add":
        return ""
    total = grid.size
    filled = (grid > 0).sum()
    if filled >= total:
        return "add impossible: grid is full"


@pruning_rule("remove_recolor_conflict")
def _remove_recolor_conflict(history, grid, ptype, params):
    """Nao faz sentido recolorir uma cor que acabou de ser removida."""
    if ptype != "recolor":
        return ""
    from_c = params.get("from_color", -1)
    for step in history:
        if step.get("type") == "remove":
            removed_color = step.get("params", {}).get("color", -1)
            if removed_color == from_c:
                return f"recolor({from_c}) after remove({removed_color}): color already removed"


@pruning_rule("recolor_idempotent")
def _recolor_idempotent(history, grid, ptype, params):
    """Recolor so faz sentido se from_color != to_color."""
    if ptype != "recolor":
        return ""
    frm = params.get("from_color", -1)
    to = params.get("to_color", -1)
    if frm == to:
        return f"recolor({frm}->{to}) idempotent: no change"


@pruning_rule("remove_already_removed")
def _remove_already_removed(history, grid, ptype, params):
    """Nao faz sentido remover uma cor que ja foi removida."""
    if ptype != "remove":
        return ""
    color = params.get("color", -1)
    for step in history:
        if step.get("type") == "remove":
            if step.get("params", {}).get("color") == color:
                return f"remove({color}) already done"


@pruning_rule("move_after_move_merges")
def _move_after_move_merges(history, grid, ptype, params):
    """Move apos move: o segundo move herda o grid ja deslocado.
    Isso e permitido estruturalmente, mas raramente util.
    """
    if ptype != "move":
        return ""
    for step in history:
        if step.get("type") == "move":
            return ""  # Apenas warning, nao bloqueio


def get_rule_names() -> List[str]:
    return list(_pruning_rules_names)
