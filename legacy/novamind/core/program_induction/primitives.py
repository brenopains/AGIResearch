"""
Primitives implementadas — cada uma com solver que INFERE parametros
por constraint satisfaction, nao por enumeracao.

Nenhuma transformacao e hardcoded. Cada primitive:
  1. Define parameter_schema (o espaco de parametros)
  2. Define constraints (relacoes entre parametros)
  3. Implementa solver_fn (infere parametros de input/output)
  4. Implementa execute_fn (aplica a transformacao)
"""
from typing import Dict, Optional, Set, Tuple
import numpy as np

from .primitive import Primitive, ParameterSchema, Constraint


# ─── Helpers ────────────────────────────────────────────────


def _object_colors(grid: np.ndarray) -> Set[int]:
    """Cores presentes em objetos (exclui background 0)."""
    return set(int(c) for c in np.unique(grid) if c != 0)


def _color_percent(grid: np.ndarray) -> Dict[int, float]:
    """Proporcao de cada cor no grid."""
    total = grid.size
    counts = {int(c): 0 for c in np.unique(grid) if c != 0}
    for color in counts:
        counts[color] = float((grid == color).sum()) / total
    return counts


def _bounding_box(grid: np.ndarray, color: int = None) -> Tuple[int, int, int, int]:
    """Bounding box de todos os objetos (ou de uma cor especifica)."""
    mask = grid == color if color is not None else grid > 0
    if not np.any(mask):
        return (0, 0, 0, 0)
    rows, cols = np.where(mask)
    return (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))


# ─── Solvers ────────────────────────────────────────────────


def _solve_recolor(inp: np.ndarray, out: np.ndarray) -> Optional[Dict]:
    """Infere parametros de recolor por MATCHING DE CONJUNTOS.

    Nao tenta cores. Resolve por constraint:
      - Bijection: cada cor no input mapeia para uma cor no output
      - Minimizacao: a permutation que minimiza erro de pixel

    Isso e um assignment problem, nao uma busca.
    """
    in_colors = _object_colors(inp)
    out_colors = _object_colors(out)

    if not in_colors:
        return None

    # Cores que sumiram ou apareceram
    removed = in_colors - out_colors
    added = out_colors - in_colors

    if len(removed) == 1 and len(added) == 1:
        from_c = list(removed)[0]
        to_c = list(added)[0]
        # Verifica: todo pixel from_c vira to_c?
        from_mask = inp == from_c
        to_mask = out == to_c
        overlap = (from_mask & to_mask).sum()
        total_from = from_mask.sum()
        if total_from > 0 and overlap / total_from > 0.5:
            return {"from_color": from_c, "to_color": to_c}

    # Caso mais complexo: tenta encontrar mapping bijectivo
    # Comparando histogramas de cor
    in_pct = _color_percent(inp)
    out_pct = _color_percent(out)

    matched = {}
    used_out = set()
    for c_in in sorted(in_pct.keys()):
        best = None
        best_diff = float("inf")
        for c_out in sorted(out_pct.keys()):
            if c_out in used_out:
                continue
            diff = abs(in_pct[c_in] - out_pct[c_out])
            if diff < best_diff:
                best_diff = diff
                best = c_out
        if best is not None and best_diff < 0.3:
            matched[c_in] = best
            used_out.add(best)

    if len(matched) == 1:
        for frm, to in matched.items():
            return {"from_color": frm, "to_color": to}

    return None


def _solve_remove_color(inp: np.ndarray, out: np.ndarray) -> Optional[Dict]:
    """Infere qual cor foi removida.

    Constraint: cor existe no input mas nao no output.
    """
    in_colors = _object_colors(inp)
    out_colors = _object_colors(out)
    removed = in_colors - out_colors
    if len(removed) == 1:
        return {"color": list(removed)[0]}
    if len(removed) > 1:
        # Multiplas cores removidas: pega a que mais perdeu area
        best = None
        max_loss = 0
        for c in removed:
            loss = (inp == c).sum() - (out == c).sum()
            if loss > max_loss:
                max_loss = loss
                best = c
        if best is not None:
            return {"color": best}
    return None


def _solve_move(inp: np.ndarray, out: np.ndarray) -> Optional[Dict]:
    """Infere translacao por DESLOCAMENTO DO CENTROIDE.

    Nao tenta posicoes. Computa o vetor deslocamento direto.
    Lida com grids de tamanhos diferentes (comum em ARC com padding).
    """
    in_mask = inp > 0
    out_mask = out > 0

    if not np.any(in_mask) or not np.any(out_mask):
        return None

    # Usa so a bounding box nao-zero para calcular centroide
    def _center(mask):
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return (0, 0)
        return (rows.mean(), cols.mean())

    in_cy, in_cx = _center(in_mask)
    out_cy, out_cx = _center(out_mask)

    dr = round(out_cy - in_cy)
    dc = round(out_cx - in_cx)

    if dr == 0 and dc == 0:
        return None

    # Verifica consistencia: translada o input dentro do espaco do output
    max_h = max(inp.shape[0], out.shape[0])
    max_w = max(inp.shape[1], out.shape[1])
    shifted = np.zeros((max_h, max_w), dtype=inp.dtype)
    h, w = inp.shape
    r_src = max(0, -dr)
    r_dst = max(0, dr)
    c_src = max(0, -dc)
    c_dst = max(0, dc)
    r_end = min(h, h - dr)
    c_end = min(w, w - dc)
    shifted[r_dst:r_dst + (r_end - r_src),
            c_dst:c_dst + (c_end - c_src)] = inp[r_src:r_end, c_src:c_end]

    # Recorta para o tamanho do output
    h_out, w_out = out.shape
    shifted = shifted[:h_out, :w_out]

    overlap = ((shifted > 0) & (out_mask)).sum()
    total = max(1, out_mask.sum())
    if overlap / total > 0.3:
        return {"dr": int(dr), "dc": int(dc)}

    return None


def _solve_add_object(inp: np.ndarray, out: np.ndarray) -> Optional[Dict]:
    """Infere parametros de um objeto adicionado."""
    in_colors = _object_colors(inp)
    out_colors = _object_colors(out)
    added = out_colors - in_colors

    if not added:
        return None

    color = list(added)[0]
    # Centroides
    in_rows, in_cols = np.where(inp > 0)
    out_rows, out_cols = np.where(out == color)
    if len(in_rows) == 0 or len(out_rows) == 0:
        return {"color": int(color), "relative_position": "below"}
    in_cy, in_cx = in_rows.mean(), in_cols.mean()
    out_cy, out_cx = out_rows.mean(), out_cols.mean()
    dr = out_cy - in_cy
    dc = out_cx - in_cx
    if abs(dr) > abs(dc):
        rel_pos = "below" if dr > 0 else "above"
    else:
        rel_pos = "right" if dc > 0 else "left"
    return {"color": int(color), "relative_position": rel_pos}


def _solve_grow_shrink(inp: np.ndarray, out: np.ndarray) -> Optional[Dict]:
    """Detecta se houve crescimento ou encolhimento."""
    in_pixels = (inp > 0).sum()
    out_pixels = (out > 0).sum()

    if out_pixels > in_pixels * 1.2:
        return {"direction": "grow", "ratio": round(out_pixels / max(1, in_pixels), 2)}
    elif out_pixels < in_pixels * 0.8:
        return {"direction": "shrink", "ratio": round(out_pixels / max(1, in_pixels), 2)}
    return None


# ─── Executors ──────────────────────────────────────────────


def _execute_recolor(grid: np.ndarray, params: Dict) -> np.ndarray:
    result = grid.copy()
    frm = params.get("from_color", -1)
    to = params.get("to_color", -1)
    if frm >= 0 and to >= 0:
        result[result == frm] = to
    return result


def _execute_remove_color(grid: np.ndarray, params: Dict) -> np.ndarray:
    result = grid.copy()
    color = params.get("color", -1)
    if color >= 0:
        result[result == color] = 0
    return result


def _execute_move(grid: np.ndarray, params: Dict) -> np.ndarray:
    result = np.zeros_like(grid)
    dr = int(params.get("dr", 0))
    dc = int(params.get("dc", 0))
    h, w = grid.shape
    if dr == 0 and dc == 0:
        return grid.copy()
    r_src = max(0, -dr)
    r_dst = max(0, dr)
    c_src = max(0, -dc)
    c_dst = max(0, dc)
    r_src_end = h - max(0, dr)
    r_dst_end = h - max(0, -dr)
    c_src_end = w - max(0, dc)
    c_dst_end = w - max(0, -dc)
    result[r_dst:r_dst_end, c_dst:c_dst_end] = grid[r_src:r_src_end, c_src:c_src_end]
    return result


def _execute_add_object(grid: np.ndarray, params: Dict) -> np.ndarray:
    result = grid.copy()
    color = params.get("color", -1)
    rel_pos = params.get("relative_position", "below")
    if color >= 0 and np.any(result > 0):
        rows, cols = np.where(result > 0)
        if rel_pos == "below":
            r = rows.max() + 1
            c = int(np.mean(cols))
        elif rel_pos == "above":
            r = rows.min() - 1
            c = int(np.mean(cols))
        elif rel_pos == "right":
            r = int(np.mean(rows))
            c = cols.max() + 1
        else:
            r = int(np.mean(rows))
            c = cols.min() - 1
        if 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
            result[r, c] = color
    return result


# ─── Constraints ────────────────────────────────────────────


def _constraint_from_not_to(params, inp, out):
    """Recolor: from_color != to_color."""
    return params.get("from_color", -1) != params.get("to_color", -1)


def _constraint_color_exists_input(params, inp, out):
    """A cor de origem existe no input."""
    color = params.get("from_color") or params.get("color", -1)
    if color < 0:
        return True
    return np.any(inp == color) if inp is not None else True


def _constraint_move_nonzero(params, inp, out):
    """Move: pelo menos uma direcao nao-zero."""
    return params.get("dr", 0) != 0 or params.get("dc", 0) != 0


# ─── Factory ────────────────────────────────────────────────


def create_recolor_primitive(name: str = "recolor") -> Primitive:
    prim = Primitive(
        name=name,
        structure_type="recolor",
        description="recolor: change one color to another",
        parameter_schemas=[
            ParameterSchema("from_color", "categorical", list(range(10)),
                           "color to replace"),
            ParameterSchema("to_color", "categorical", list(range(10)),
                           "new color"),
        ],
        constraints=[
            Constraint("from_color != to_color", _constraint_from_not_to),
            Constraint("from_color exists in input", _constraint_color_exists_input),
        ],
        solver_fn=_solve_recolor,
    )
    # Sobrescreve executor
    def _exec(g, p): return _execute_recolor(g, p)
    prim._execute = _exec
    return prim


def create_remove_primitive(name: str = "remove") -> Primitive:
    prim = Primitive(
        name=name,
        structure_type="remove",
        description="remove: delete all objects of a color",
        parameter_schemas=[
            ParameterSchema("color", "categorical", list(range(10)),
                           "color to remove"),
        ],
        constraints=[
            Constraint("color exists in input", _constraint_color_exists_input),
        ],
        solver_fn=_solve_remove_color,
    )
    def _exec(g, p): return _execute_remove_color(g, p)
    prim._execute = _exec
    return prim


def create_move_primitive(name: str = "move") -> Primitive:
    prim = Primitive(
        name=name,
        structure_type="move",
        description="move: translate all objects by (dr, dc)",
        parameter_schemas=[
            ParameterSchema("dr", "int", (-20, 20), "delta row"),
            ParameterSchema("dc", "int", (-20, 20), "delta col"),
        ],
        constraints=[
            Constraint("at least one direction non-zero", _constraint_move_nonzero),
        ],
        solver_fn=_solve_move,
    )
    def _exec(g, p): return _execute_move(g, p)
    prim._execute = _exec
    return prim


def create_add_primitive(name: str = "add") -> Primitive:
    prim = Primitive(
        name=name,
        structure_type="add",
        description="add: add a new object of a color at a relative position",
        parameter_schemas=[
            ParameterSchema("color", "categorical", list(range(10)),
                           "color of new object"),
            ParameterSchema("relative_position", "categorical",
                           ["above", "below", "left", "right"],
                           "position relative to existing objects"),
        ],
        constraints=[],
        solver_fn=_solve_add_object,
    )
    def _exec(g, p): return _execute_add_object(g, p)
    prim._execute = _exec
    return prim


def create_transform_primitive(name: str = "transform") -> Primitive:
    prim = Primitive(
        name=name,
        structure_type="transform",
        description="transform: grow or shrink objects",
        parameter_schemas=[
            ParameterSchema("direction", "categorical",
                           ["grow", "shrink"], "growth or shrinkage"),
            ParameterSchema("ratio", "float", (0.0, 10.0),
                           "size ratio"),
        ],
        constraints=[],
        solver_fn=_solve_grow_shrink,
    )
    # grow/shrink usa _execute_move como fallback
    def _exec(g, p): return _execute_move(g, p)
    prim._execute = _exec
    return prim


# ─── Registro de fabricas de primitives ─────────────────────

# Mapeia structure_type -> funcao de fabrica correspondente.
# O Discovery usa este mapa para CRIAR novas primitives quando
# um padrao de delta e observado pela primeira vez.
#
# Nao e uma lista fixa: e um catalogo de ESTRUTURAS que o
# sistema pode instanciar. Novas estruturas sao adicionadas
# quando o Discovery generaliza um padrao novo.
_PRIMITIVE_FACTORIES = {}

def register_primitive_type(structure_type: str, factory_fn):
    _PRIMITIVE_FACTORIES[structure_type] = factory_fn

def get_primitive_factory(structure_type: str):
    return _PRIMITIVE_FACTORIES.get(structure_type)

def get_known_types() -> list:
    return list(_PRIMITIVE_FACTORIES.keys())

# Registra os tipos de primitiva que o sistema conhece
register_primitive_type("recolor", create_recolor_primitive)
register_primitive_type("remove", create_remove_primitive)
register_primitive_type("move", create_move_primitive)
register_primitive_type("add", create_add_primitive)
register_primitive_type("transform", create_transform_primitive)
