"""Percepcao object-centric: frame (grade de cores) -> lista de objetos persistentes.

Sem HDV, sem CGE, sem hash-de-texto. Componentes conexas de mesma cor via
flood-fill, cada uma vira um objeto com cor, bbox, forma normalizada
(invariante a translacao) e centroide. E o substrato sobre o qual
transition_model.py e skill_memory.py operam.

Ver Fase 1 do plano: novo, nao reusa core/arc_object_centric.py ainda —
essa troca so acontece se um teste isolado mostrar que vale a pena.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Obj:
    color: int
    cells: frozenset  # tuple[int,int] relativos ao bbox (r,c) a partir de (0,0)
    top: int
    left: int
    height: int
    width: int
    size: int
    centroid: tuple

    @property
    def shape_hash(self) -> int:
        return hash((self.color, self.cells))

    @property
    def bbox(self) -> tuple:
        return (self.top, self.left, self.top + self.height - 1, self.left + self.width - 1)


def _flood_fill(grid: list[list[int]], visited, r0: int, c0: int, color: int) -> set:
    h, w = len(grid), len(grid[0]) if grid else 0
    q = deque([(r0, c0)])
    visited[r0][c0] = True
    cells = set()
    while q:
        r, c = q.popleft()
        cells.add((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                visited[nr][nc] = True
                q.append((nr, nc))
    return cells


def extract_objects(layer, background: int = None) -> list[Obj]:
    """Componentes conexas (4-conectividade) de um unico layer HxW de cores.

    `layer` pode ser list[list[int]] ou numpy.ndarray (FrameDataRaw usa
    numpy internamente) — normalizado para list[list[int]] de saida.

    background=None -> heuristica: cor mais frequente da grade (assume que
    o fundo domina em area, valido para a maioria dos jogos ARC-AGI-3).
    """
    if layer is None or len(layer) == 0 or len(layer[0]) == 0:
        return []
    layer = [[int(v) for v in row] for row in layer]
    h, w = len(layer), len(layer[0])
    if background is None:
        counts = Counter(layer[r][c] for r in range(h) for c in range(w))
        background = counts.most_common(1)[0][0]

    visited = [[False] * w for _ in range(h)]
    objects = []
    for r in range(h):
        for c in range(w):
            if visited[r][c] or layer[r][c] == background:
                continue
            color = layer[r][c]
            cells = _flood_fill(layer, visited, r, c, color)
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            top, left = min(rs), min(cs)
            norm_cells = frozenset((p[0] - top, p[1] - left) for p in cells)
            objects.append(Obj(
                color=color,
                cells=norm_cells,
                top=top, left=left,
                height=max(rs) - top + 1, width=max(cs) - left + 1,
                size=len(cells),
                centroid=(sum(rs) / len(rs), sum(cs) / len(cs)),
            ))
    return objects


def extract_scene(frame_layers: list[list[list[int]]]) -> list[list[Obj]]:
    """Uma cena por layer do frame (a maioria dos jogos tem 1 layer)."""
    return [extract_objects(layer) for layer in frame_layers]


def scene_signature(scene: list[list[Obj]], max_objects: int = 12) -> tuple:
    """Assinatura coarse e hashable de uma cena, para indexar o modelo de
    transicao e a memoria de skills. Ordena objetos por (cor, tamanho,
    posicao) para ser estavel entre chamadas equivalentes."""
    flat = []
    for layer_idx, objs in enumerate(scene):
        for o in objs:
            flat.append((layer_idx, o.color, o.size, o.shape_hash,
                         round(o.centroid[0] / 4), round(o.centroid[1] / 4)))
    flat.sort()
    return tuple(flat[:max_objects])
