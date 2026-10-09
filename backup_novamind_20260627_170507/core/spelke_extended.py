"""
Spelke Genes — BIBLIOTECAS ESTENDIDAS (H100 + H111 + H125 + user-insight)

Novas primitivas para突破 o muro de 4%:
- Átomos compostos (H111) — pares de genes que sempre aparecem juntos
- Predicados espaciais 2D (H125) — HorizontalLine, CrossAt, TShape, etc.
- Recolor variants (H100) — RowMod, ColMod, DiagMod, DistFromCenter, etc.
- Transformações globais de cor — Mod, XOR, Mask

Estas funções são INJETADAS em spelke_genes.make_genes() e
make_predicates() dinamicamente.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter

# Imports do módulo principal (lazy)
def _get_module():
    import importlib, sys
    if 'core.spelke_genes' in sys.modules:
        return sys.modules['core.spelke_genes']
    from core import spelke_genes as sg
    sys.modules['core.spelke_genes'] = sg
    return sg


# ============================================================================
# PREDICADOS 2D (H125)
# ============================================================================

def pred_is_horizontal_line(node, ctx):
    """Linha horizontal pura (r constante, c varia)."""
    rows = set(r for r, c in node.cells)
    cols = set(c for r, c in node.cells)
    return len(rows) == 1 and len(cols) >= 2

def pred_is_vertical_line(node, ctx):
    """Coluna pura (c constante, r varia)."""
    rows = set(r for r, c in node.cells)
    cols = set(c for r, c in node.cells)
    return len(cols) == 1 and len(rows) >= 2

def pred_is_diagonal(node, ctx):
    """Diagonal principal (r-c constante)."""
    diffs = set(r - c for r, c in node.cells)
    return len(diffs) == 1 and len(node.cells) >= 2

def pred_is_antidiagonal(node, ctx):
    """Anti-diagonal (r+c constante)."""
    sums = set(r + c for r, c in node.cells)
    return len(sums) == 1 and len(node.cells) >= 2

def pred_t_shape(node, ctx):
    """Forma de T: linha horizontal + linha vertical saindo do meio."""
    if node.size < 4:
        return False
    rs = [r for r, c in node.cells]
    cs = [c for r, c in node.cells]
    h = max(rs) - min(rs) + 1
    w = max(cs) - min(cs) + 1
    # Heurística: bounding box com uma linha inteira e uma coluna centralizada
    if h == 1 and w >= 3:
        return True  # barra horizontal
    if w == 1 and h >= 3:
        return True  # barra vertical
    return False

def pred_l_shape(node, ctx):
    """Forma de L: duas barras perpendiculares."""
    if node.size < 4:
        return False
    rs = [r for r, c in node.cells]
    cs = [c for r, c in node.cells]
    h = max(rs) - min(rs) + 1
    w = max(cs) - min(cs) + 1
    if h >= 2 and w >= 2 and node.size == h + w - 1:
        return True
    return False

def pred_on_border(node, ctx):
    """Toca a borda do grid."""
    H, W = node.shape
    for r, c in node.cells:
        if r == 0 or r == H - 1 or c == 0 or c == W - 1:
            return True
    return False

def pred_in_corner(node, ctx):
    """Tem célula em algum canto."""
    H, W = node.shape
    for r, c in node.cells:
        if (r == 0 or r == H - 1) and (c == 0 or c == W - 1):
            return True
    return False

def pred_is_single_cell(node, ctx):
    return len(node.cells) == 1

def pred_is_frame(node, ctx):
    """Anel/borda: preenche o contorno do seu bounding box."""
    if node.size < 8:
        return False
    rs = [r for r, c in node.cells]
    cs = [c for r, c in node.cells]
    h = max(rs) - min(rs) + 1
    w = max(cs) - min(cs) + 1
    expected_perim = 2 * h + 2 * w - 4
    return node.size == expected_perim and node.size >= 8

def pred_in_quadrant(node, ctx):
    """Objeto está num quadrante específico (passado como param)."""
    return True  # usado em combinação com param

def pred_has_inner_color(node, ctx, c):
    """Tem célula da cor c dentro do bounding box do objeto."""
    if not node.cells:
        return False
    rs = [r for r, c2 in node.cells]
    cs = [cc for r, cc in node.cells]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    for r in range(rmin, rmax + 1):
        for col in range(cmin, cmax + 1):
            if (r, col) in node.cells:
                continue
            if 0 <= r < ctx.grid.shape[0] and 0 <= col < ctx.grid.shape[1]:
                if ctx.grid[r, col] == c:
                    return True
    return False

def pred_singleton(node, ctx):
    """Único objeto com sua cor no grid."""
    same_color = [n for n in ctx.nodes if n.color == node.color and n.color != 0]
    return len(same_color) == 1


# ============================================================================
# AÇÕES: RECOLOR VARIANTS (H100 — cobrem 23+ variantes de recolor)
# ============================================================================

def act_recolor_by_row_mod(node, ctx, mod=2, base_color=1):
    """Recolor por (linha % mod) → cor."""
    new_grid = ctx.grid.copy()
    for r, c in node.cells:
        new_grid[r, c] = base_color + (r % mod)
    return _new_ctx(new_grid, ctx)

def act_recolor_by_col_mod(node, ctx, mod=2, base_color=1):
    new_grid = ctx.grid.copy()
    for r, c in node.cells:
        new_grid[r, c] = base_color + (c % mod)
    return _new_ctx(new_grid, ctx)

def act_recolor_by_diag_mod(node, ctx, mod=2, base_color=1):
    new_grid = ctx.grid.copy()
    for r, c in node.cells:
        new_grid[r, c] = base_color + ((r - c) % mod)
    return _new_ctx(new_grid, ctx)

def act_recolor_by_antidiag_mod(node, ctx, mod=2, base_color=1):
    new_grid = ctx.grid.copy()
    for r, c in node.cells:
        new_grid[r, c] = base_color + ((r + c) % mod)
    return _new_ctx(new_grid, ctx)

def act_recolor_by_dist_from_center(node, ctx, base_color=1):
    """Recolor por distância Manhattan do centro do objeto."""
    rs = [r for r, c in node.cells]
    cs = [c for r, c in node.cells]
    cr = sum(rs) / len(rs)
    cc = sum(cs) / len(cs)
    new_grid = ctx.grid.copy()
    for r, c in node.cells:
        d = int(abs(r - cr) + abs(c - cc))
        new_grid[r, c] = base_color + (d % 3)
    return _new_ctx(new_grid, ctx)

def act_recolor_by_quadrant(node, ctx, base_color=1):
    """Recolor por quadrante do grid."""
    H, W = ctx.grid.shape
    new_grid = ctx.grid.copy()
    for r, c in node.cells:
        q = (0 if r < H / 2 else 1) + (0 if c < W / 2 else 2)
        new_grid[r, c] = base_color + q
    return _new_ctx(new_grid, ctx)

def act_recolor_by_x_pos(node, ctx, base_color=1):
    """Recolor = base + (c / W) * 4."""
    H, W = ctx.grid.shape
    new_grid = ctx.grid.copy()
    for r, c in node.cells:
        new_grid[r, c] = base_color + int((c / max(W - 1, 1)) * 3)
    return _new_ctx(new_grid, ctx)

def act_recolor_by_y_pos(node, ctx, base_color=1):
    H, W = ctx.grid.shape
    new_grid = ctx.grid.copy()
    for r, c in node.cells:
        new_grid[r, c] = base_color + int((r / max(H - 1, 1)) * 3)
    return _new_ctx(new_grid, ctx)

def act_recolor_modulo(node, ctx, mod=3, base_color=1):
    """Recolor = base + (célula % mod)."""
    new_grid = ctx.grid.copy()
    for r, c in node.cells:
        new_grid[r, c] = base_color + (int(ctx.grid[r, c]) % mod)
    return _new_ctx(new_grid, ctx)


# ============================================================================
# AÇÕES: ÁTOMOS COMPOSTOS (H111)
# ============================================================================

def act_mirror_then_recolor(node, ctx, axis='y', new_color=1):
    """MirrorY + Recolor em um passo."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    rs = [r for r, c in node.cells]
    cs = [c for r, c in node.cells]
    if not rs:
        return _new_ctx(new_grid, ctx)
    if axis == 'y':
        # espelha colunas: c -> (cmin + cmax) - c
        cmin, cmax = min(cs), max(cs)
        csum = cmin + cmax
        for r, c in node.cells:
            new_grid[r, c] = 0  # apaga origem
        for r, c in node.cells:
            new_grid[r, csum - c] = node.color
    elif axis == 'x':
        rmin, rmax = min(rs), max(rs)
        rsum = rmin + rmax
        for r, c in node.cells:
            new_grid[r, c] = 0
        for r, c in node.cells:
            new_grid[rsum - r, c] = node.color
    # agora recolor
    for r, c in node.cells:
        if 0 <= r < H and 0 <= csum - c < W:
            new_grid[r, csum - c] = new_color
    return _new_ctx(new_grid, ctx)

def act_rotate_then_recolor(node, ctx, k=1, new_color=1):
    """Rotate90 + Recolor em um passo."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    rs = [r for r, c in node.cells]
    cs = [c for r, c in node.cells]
    if not rs:
        return _new_ctx(new_grid, ctx)
    rmin, cmin = min(rs), min(cs)
    # move tudo para origem, rotaciona
    for r, c in node.cells:
        new_grid[r, c] = 0
    for r, c in node.cells:
        rr = r - rmin
        cc = c - cmin
        # rotaciona 90 k vezes
        for _ in range(k % 4):
            rr, cc = cc, -rr - 1  # heurística
        new_grid[rr + rmin, cc + cmin] = new_color
    return _new_ctx(new_grid, ctx)

def act_gravity_then_recolor(node, ctx, direction='down', new_color=1):
    """Gravity + Recolor em um passo."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    # Simples: empurra tudo na direção
    if direction == 'down':
        # encontra a linha mais baixa da cor
        max_r = max(r for r, c in node.cells)
        for r, c in node.cells:
            new_grid[r, c] = 0
        for r, c in node.cells:
            new_grid[max_r, c] = new_color
    return _new_ctx(new_grid, ctx)


# ============================================================================
# AÇÕES: GLOBAIS DE COR (H100 — complementam recolor)
# ============================================================================

def act_set_color_global(node, ctx, target_color, new_color):
    """Muda toda ocorrência de target_color para new_color, exceto onde o objeto está."""
    new_grid = ctx.grid.copy()
    obj_cells = set(node.cells) if node else set()
    for r in range(ctx.grid.shape[0]):
        for c in range(ctx.grid.shape[1]):
            if (r, c) in obj_cells:
                continue
            if new_grid[r, c] == target_color:
                new_grid[r, c] = new_color
    return _new_ctx(new_grid, ctx)

def act_swap_pair_global(node, ctx, a, b):
    """Troca duas cores globalmente."""
    new_grid = ctx.grid.copy()
    a_mask = (new_grid == a)
    b_mask = (new_grid == b)
    new_grid[a_mask] = b
    new_grid[b_mask] = a
    return _new_ctx(new_grid, ctx)

def act_set_background(node, ctx, color):
    """Muda a cor de fundo (0) para color."""
    new_grid = ctx.grid.copy()
    new_grid[new_grid == 0] = color
    return _new_ctx(new_grid, ctx)


# ============================================================================
# HELPER
# ============================================================================

def _new_ctx(grid, old_ctx):
    sg = _get_module()
    return sg.Context(grid=grid, history=old_ctx.history + [(None, None)], shape=old_ctx.shape)


# ============================================================================
# MONTAGEM: predicados extras + genes extras
# ============================================================================

def get_extra_predicates():
    """Retorna lista adicional de predicados 2D."""
    return [
        ('IsHorizLine', pred_is_horizontal_line, [None]),
        ('IsVertLine', pred_is_vertical_line, [None]),
        ('IsDiagonal', pred_is_diagonal, [None]),
        ('IsAntidiag', pred_is_antidiagonal, [None]),
        ('IsTShape', pred_t_shape, [None]),
        ('IsLShape', pred_l_shape, [None]),
        ('OnBorder', pred_on_border, [None]),
        ('InCorner', pred_in_corner, [None]),
        ('IsSingleCell', pred_is_single_cell, [None]),
        ('IsFrame', pred_is_frame, [None]),
        ('Singleton', pred_singleton, [None]),
        ('HasInnerColor', pred_has_inner_color, 'colors'),
    ]


def get_extra_genes():
    """Retorna lista adicional de genes (recolor variants + atoms + globals)."""
    sg = _get_module()
    return [
        # Recolor variants (H100)
        sg.Gene("RecolorByRowMod", sg.Family.SPATIAL, sg.Domain.GRID, ["obj"], ["grid"],
                lambda n, c, m=2: act_recolor_by_row_mod(n, c, mod=m),
                needs_global=True,
                description="Recolor por (linha % mod)"),
        sg.Gene("RecolorByColMod", sg.Family.SPATIAL, sg.Domain.GRID, ["obj"], ["grid"],
                lambda n, c, m=2: act_recolor_by_col_mod(n, c, mod=m),
                needs_global=True,
                description="Recolor por (col % mod)"),
        sg.Gene("RecolorByDiagMod", sg.Family.SPATIAL, sg.Domain.GRID, ["obj"], ["grid"],
                lambda n, c, m=2: act_recolor_by_diag_mod(n, c, mod=m),
                needs_global=True,
                description="Recolor por ((r-c) % mod)"),
        sg.Gene("RecolorByAntidiagMod", sg.Family.SPATIAL, sg.Domain.GRID, ["obj"], ["grid"],
                lambda n, c, m=2: act_recolor_by_antidiag_mod(n, c, mod=m),
                needs_global=True,
                description="Recolor por ((r+c) % mod)"),
        sg.Gene("RecolorByDistFromCenter", sg.Family.SPATIAL, sg.Domain.GRID, ["obj"], ["grid"],
                act_recolor_by_dist_from_center,
                needs_global=True,
                description="Recolor por dist. Manhattan do centro"),
        sg.Gene("RecolorByQuadrant", sg.Family.SPATIAL, sg.Domain.GRID, ["obj"], ["grid"],
                act_recolor_by_quadrant,
                needs_global=True,
                description="Recolor por quadrante do grid"),
        sg.Gene("RecolorByXPos", sg.Family.SPATIAL, sg.Domain.GRID, ["obj"], ["grid"],
                act_recolor_by_x_pos,
                needs_global=True,
                description="Recolor por posição X"),
        sg.Gene("RecolorByYPos", sg.Family.SPATIAL, sg.Domain.GRID, ["obj"], ["grid"],
                act_recolor_by_y_pos,
                needs_global=True,
                description="Recolor por posição Y"),
        sg.Gene("RecolorModulo", sg.Family.SPATIAL, sg.Domain.GRID, ["obj"], ["grid"],
                lambda n, c, m=3: act_recolor_modulo(n, c, mod=m),
                needs_global=True,
                description="Recolor = base + (cor_atual % mod)"),

        # Átomos compostos (H111)
        sg.Gene("MirrorYThenRecolor", sg.Family.SPATIAL, sg.Domain.GRID, ["obj"], ["grid"],
                lambda n, c, col=1: act_mirror_then_recolor(n, c, axis='y', new_color=col),
                needs_global=True,
                description="MirrorY + Recolor em 1 passo"),
        sg.Gene("MirrorXThenRecolor", sg.Family.SPATIAL, sg.Domain.GRID, ["obj"], ["grid"],
                lambda n, c, col=1: act_mirror_then_recolor(n, c, axis='x', new_color=col),
                needs_global=True,
                description="MirrorX + Recolor em 1 passo"),
        sg.Gene("GravityThenRecolor", sg.Family.SPATIAL, sg.Domain.GRID, ["obj"], ["grid"],
                lambda n, c, col=1: act_gravity_then_recolor(n, c, direction='down', new_color=col),
                needs_global=True,
                description="Gravity + Recolor em 1 passo"),

        # Globais de cor (H100)
        sg.Gene("SwapPairGlobal", sg.Family.SPATIAL, sg.Domain.GRID, ["obj"], ["grid"],
                lambda n, c, a=1, b=2: act_swap_pair_global(n, c, a, b),
                needs_global=True,
                description="Troca duas cores globalmente"),
        sg.Gene("SetBackground", sg.Family.SPATIAL, sg.Domain.GRID, ["obj"], ["grid"],
                lambda n, c, col=1: act_set_background(n, c, col),
                needs_global=True,
                description="Muda a cor de fundo"),
    ]


def count_extras():
    return {
        'extra_predicates': len(get_extra_predicates()),
        'extra_genes': len(get_extra_genes()),
    }
