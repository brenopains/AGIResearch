"""
Spelke Genes — The Cognitive Primitive Library
================================================
"O Dicionário de Deus" — The Cognitive DNA of AGI

INSPIRAÇÃO BIOLÓGICA (analogia do usuário):
    DNA         -> alfabeto de nucleotídeos (4 letras) -> compostos
    Gene        -> bloco funcional codificado
    Proteína    -> primitiva executável (transcrição + tradução)
    Via         -> composição de proteínas (rede de genes)
    Organismo   -> sistema vivo (integração)

IMPLEMENTAÇÃO COMPUTACIONAL:
    Alfabeto    -> Predicados (True, Color, IsSquare, etc.) — "nucleotídeos"
    Gene        -> Regra = (Predicado, Ação)  — "proteína funcional"
    Via         -> Sequência de Regras         — "pathway metabólico"
    AGI         -> Sistema que compõe genes em vias sob seleção    — "organismo"

AS 8 FAMÍLIAS (do "Blueprint da AGI" — RELATORIO.md linha 1694-1705):
    1. Operações Espaciais e Topológicas  (geometria, distância, contenção, buracos)
    2. Causalidade Probabilística         (Bayesiana, Se(A)->Afeta(B))
    3. Escala e Proporção                  (ScaleUp, Rhythm, RelativeSize)
    4. Hierarquia Lógica e Abstração       (É_UM, PERTENCE_A, ParteDe)
    5. Estruturas Relacionais              (Grafos: IsConnected, NeighborOf, Path)
    6. Dinâmicas Temporais                 (Before, After, ChangeRate)
    7. Simetria e Assimetria               (Mirror, IsSym, BrokenSym)
    8. Meta-cognição e Simulação           (Rewind, EvaluateError, PruneTree)

CADA GENE TEM:
    - name:        identificador único (HUMANO-LEGÍVEL, NÃO código)
    - family:      qual das 8 famílias
    - domain:      grid|graph|sequence|abstract — onde opera
    - inputs:      tipos que consome
    - outputs:     tipos que produz
    - cost:        estimativa de custo computacional
    - apply:       função pura (grid, ctx) -> (new_grid, ctx)
    - inverts:     função inversa (se existir) — para REPLAY/meta-cog

SELEÇÃO POR GENERALIZAÇÃO (H185-H188):
    Um gene nasce no RESÍDUO que a biblioteca não cobre;
    é ancorado por VIÉS (localidade/equivariância);
    é SELECIONADO por generalização (held-out exato + transfere tamanho).

HONESTIDADE METODOLÓGICA:
    - Genes sem aplicação concreta (família 8 meta-cog) são FRAMEWORK, não solução.
    - Genes de família 6 (temporal) são MINIMAL (ARC não tem tempo explícito).
    - Famílias 2 (bayesiana) e 4 (hierárquica) são ABSTRATAS — aplicadas via composição.
    - Nem todo gene da biblia é "novo" — muitos são as primitivas que faltaram em H111.

AUTOR: usuario + NovaMind AGI research
DATA: 06/06/2026
CONTEXTO: 150+ experimentos, parede mapeada (Solomonoff/largura), arquitetura validada.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Tuple, Set
from enum import Enum
from scipy.ndimage import label, binary_fill_holes, binary_dilation, distance_transform_edt

# ============================================================================
# ENUMS & TYPES
# ============================================================================

class Family(Enum):
    SPATIAL = "1_spatial"
    CAUSAL = "2_causal"
    SCALE = "3_scale"
    HIERARCHY = "4_hierarchy"
    RELATIONAL = "5_relational"
    TEMPORAL = "6_temporal"
    SYMMETRY = "7_symmetry"
    METACOG = "8_metacog"

class Domain(Enum):
    GRID = "grid"        # ARC, ambiente 2D
    GRAPH = "graph"      # mundo, relações
    SEQUENCE = "seq"     # linguagem, tempo
    ABSTRACT = "abstract"  # composições de alto nível

# ============================================================================
# GENE = unidade básica
# ============================================================================

@dataclass
class Gene:
    """Uma primitiva cognitiva: predicado + ação + contexto."""
    name: str
    family: Family
    domain: Domain
    inputs: List[str]
    outputs: List[str]
    apply: Callable
    cost: float = 1.0
    inverts: Optional[Callable] = None
    description: str = ""
    arity: int = 1  # quantos argumentos da grade (1=por-objeto, 2=relacional)
    needs_global: bool = False  # True se precisa do estado global da grade

    def __repr__(self):
        return f"Gene({self.name}/{self.family.value})"

# ============================================================================
# CONTEXTO = estado cognitivo por tarefa
# ============================================================================

@dataclass
class Context:
    """Estado cognitivo completo — passado entre genes."""
    grid: np.ndarray
    history: List[np.ndarray] = field(default_factory=list)
    nodes: List[Any] = field(default_factory=list)
    edges: Dict[str, Set] = field(default_factory=dict)
    colors: Set[int] = field(default_factory=set)
    parent_map: Dict[int, int] = field(default_factory=dict)  # HIERARCHY
    graph: Dict[int, Set[int]] = field(default_factory=dict)  # RELATIONAL
    seq_index: int = 0  # TEMPORAL
    errors: List[float] = field(default_factory=list)  # METACOG
    actions_taken: List[str] = field(default_factory=list)  # METACOG

    def snapshot(self):
        """Para META-cog: salva estado para rewind."""
        return {
            'grid': self.grid.copy(),
            'history': [h.copy() for h in self.history],
            'actions': list(self.actions_taken),
        }

    def restore(self, snap):
        self.grid = snap['grid'].copy()
        self.history = [h.copy() for h in snap['history']]
        self.actions_taken = list(snap['actions'])

# ============================================================================
# NODE — objeto de grade (do H113, mas refinado)
# ============================================================================

@dataclass
class Node:
    """Um objeto conectado numa grade: primitivo de PERCEPÇÃO."""
    id: int
    cells: Set[Tuple[int, int]]
    color: int
    grid_shape: Tuple[int, int]

    def __post_init__(self):
        self.size = len(self.cells)
        r_coords = [r for r, c in self.cells]
        c_coords = [c for r, c in self.cells]
        self.r_min, self.r_max = min(r_coords), max(r_coords)
        self.c_min, self.c_max = min(c_coords), max(c_coords)
        self.bbox = (self.r_min, self.c_min, self.r_max, self.c_max)
        self.h = self.r_max - self.r_min + 1
        self.w = self.c_max - self.c_min + 1

        self.is_square = (self.h == self.w and self.h * self.w == self.size)
        self.is_line = (self.h == 1 or self.w == 1) and self.size > 1
        self.is_point = self.size == 1

        # Máscara local para shape-tests
        self.mask = np.zeros((self.h, self.w), dtype=bool)
        for r, c in self.cells:
            self.mask[r - self.r_min, c - self.c_min] = True

        # Topologia
        try:
            self.is_hollow = bool(np.any(binary_fill_holes(self.mask) & ~self.mask))
        except Exception:
            self.is_hollow = False

        # Simetrias locais
        if self.size > 1:
            try:
                self.is_sym_x = bool(np.array_equal(self.mask, np.flipud(self.mask)))
                self.is_sym_y = bool(np.array_equal(self.mask, np.fliplr(self.mask)))
                self.is_sym_d = bool(np.array_equal(self.mask, self.mask.T))
                self.is_sym_180 = bool(np.array_equal(self.mask, np.flipud(np.fliplr(self.mask))))
            except Exception:
                self.is_sym_x = self.is_sym_y = self.is_sym_d = self.is_sym_180 = False
        else:
            self.is_sym_x = self.is_sym_y = self.is_sym_d = self.is_sym_180 = False

        # Quantidades (SCALE)
        self.n_holes = int(self.is_hollow)
        # Conta buracos
        try:
            filled = binary_fill_holes(self.mask)
            self.n_holes = int(filled.sum() - self.mask.sum())
        except Exception:
            pass

    def centroid(self) -> Tuple[float, float]:
        if not self.cells:
            return (0.0, 0.0)
        rs = [r for r, c in self.cells]
        cs = [c for r, c in self.cells]
        return (sum(rs) / len(rs), sum(cs) / len(cs))

    def distance_to(self, other: 'Node') -> float:
        """Distância centroide-a-centroide (CAUSAL, RELATIONAL)."""
        r1, c1 = self.centroid()
        r2, c2 = other.centroid()
        return ((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5


# ============================================================================
# PERCEPÇÃO: extrair objetos do mundo
# ============================================================================

def perceive(grid: np.ndarray) -> Tuple[List[Node], Dict[str, Set[Tuple[int, int]]]]:
    """Constrói o mapa cognitivo: objetos + relações espaciais."""
    H, W = grid.shape
    nodes = []
    for c in np.unique(grid):
        if c == 0:
            continue  # ignora fundo
        mask = (grid == c)
        lbl, num = label(mask)
        for i in range(1, num + 1):
            cells = set(map(tuple, np.argwhere(lbl == i).tolist()))
            nodes.append(Node(len(nodes), cells, int(c), (H, W)))

    # Borda/fundo
    bg_mask = (grid == 0)
    bg_lbl, bg_num = label(bg_mask)
    bg_id = len(nodes)
    for i in range(1, bg_num + 1):
        cells = set(map(tuple, np.argwhere(bg_lbl == i).tolist()))
        nodes.append(Node(len(nodes), cells, 0, (H, W)))

    # Edges: touches, inside
    edges = {'touches': set(), 'inside': set(), 'overlaps': set()}
    for n1 in nodes:
        for n2 in nodes:
            if n1 is n2:
                continue
            # bounding-box check
            if not (n1.r_min > n2.r_max + 1 or n1.r_max < n2.r_min - 1 or
                    n1.c_min > n2.c_max + 1 or n1.c_max < n2.c_min - 1):
                # dentro?
                if (n1.r_min > n2.r_min and n1.r_max < n2.r_max and
                        n1.c_min > n2.c_min and n1.c_max < n2.c_max):
                    edges['inside'].add((n1.id, n2.id))
                # toca? (4-vizinhança)
                for r, c in n1.cells:
                    if ((r + 1, c) in n2.cells or (r - 1, c) in n2.cells or
                            (r, c + 1) in n2.cells or (r, c - 1) in n2.cells):
                        edges['touches'].add((n1.id, n2.id))
                        break
    return nodes, edges


def grid_fingerprint(grid: np.ndarray) -> Dict[str, Any]:
    """Hash perceptual: usado em MEMÓRIA CROSS-DOMÍNIO (H146)."""
    H, W = grid.shape
    nodes, _ = perceive(grid)
    non_bg = [n for n in nodes if n.color != 0]
    if not non_bg:
        return {'shape': (H, W), 'n_objs': 0, 'colors': set(), 'max_size': 0, 'has_hollow': False, 'has_sym': False}
    return {
        'shape': (H, W),
        'n_objs': len(non_bg),
        'colors': set(n.color for n in non_bg),
        'max_size': max(n.size for n in non_bg),
        'min_size': min(n.size for n in non_bg),
        'has_hollow': any(n.is_hollow for n in non_bg),
        'has_sym': any(n.is_sym_x or n.is_sym_y for n in non_bg),
    }


# ============================================================================
# FAMÍLIA 1 — ESPAÇO & TOPOLOGIA (Rotate, Translate, Distance, IsHollow, Encloses, Fill)
# ============================================================================

def act_rotate(node: Node, ctx: Context, k: int = 1) -> Context:
    """Rotaciona um objeto 90° k vezes em torno do seu centro."""
    new_grid = ctx.grid.copy()
    for r, c in node.cells:
        new_grid[r, c] = 0
    H, W = ctx.grid.shape
    cr, cc = node.centroid()
    for r, c in node.cells:
        dr, dc = r - cr, c - cc
        for _ in range(k % 4):
            dr, dc = -dc, dr
        nr, nc = int(round(cr + dr)), int(round(cc + dc))
        if 0 <= nr < H and 0 <= nc < W:
            new_grid[nr, nc] = node.color
    ctx.grid = new_grid
    return ctx


def act_translate(node: Node, ctx: Context, dr: int, dc: int) -> Context:
    """Translada o objeto (dr, dc)."""
    new_grid = ctx.grid.copy()
    for r, c in node.cells:
        new_grid[r, c] = 0
    H, W = ctx.grid.shape
    for r, c in node.cells:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            new_grid[nr, nc] = node.color
    ctx.grid = new_grid
    return ctx


def act_fill_holes(node: Node, ctx: Context) -> Context:
    """Preenche buracos topológicos do objeto."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    try:
        filled = binary_fill_holes(node.mask)
        for r in range(node.h):
            for c in range(node.w):
                if filled[r, c]:
                    nr, nc = node.r_min + r, node.c_min + c
                    if 0 <= nr < H and 0 <= nc < W:
                        new_grid[nr, nc] = node.color
    except Exception:
        pass
    ctx.grid = new_grid
    return ctx


def act_outline(node: Node, ctx: Context) -> Context:
    """Mantém só o contorno do objeto."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    for r, c in node.cells:
        new_grid[r, c] = 0
    # contorno = cells com vizinho fora
    for r, c in node.cells:
        is_edge = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) not in node.cells:
                is_edge = True
                break
        if is_edge and 0 <= r < H and 0 <= c < W:
            new_grid[r, c] = node.color
    ctx.grid = new_grid
    return ctx


def act_move_to_edge(node: Node, ctx: Context, edge: str = 'right') -> Context:
    """Move o objeto até a borda especificada."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    for r, c in node.cells:
        new_grid[r, c] = 0
    if edge == 'right':
        dc = W - 1 - node.c_max
    elif edge == 'left':
        dc = -node.c_min
    elif edge == 'bottom':
        dr = H - 1 - node.r_max
    elif edge == 'top':
        dr = -node.r_min
    else:
        dr, dc = 0, 0
    for r, c in node.cells:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            new_grid[nr, nc] = node.color
    ctx.grid = new_grid
    return ctx


def act_gravity(node: Node, ctx: Context, direction: str = 'down') -> Context:
    """Faz o objeto cair até colidir."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    for r, c in node.cells:
        new_grid[r, c] = 0
    if direction == 'down':
        dr_sign = 1
        max_fall = H - 1 - node.r_max
    elif direction == 'up':
        dr_sign = -1
        max_fall = node.r_min
    else:
        dr_sign = 0
        max_fall = 0

    actual_fall = max_fall
    for f in range(1, max_fall + 1):
        hit = False
        for r, c in node.cells:
            nr, nc = r + dr_sign * f, c
            if not (0 <= nr < H) or grid_has_color_at(ctx, nr, nc, exclude=node):
                hit = True
                break
        if hit:
            actual_fall = f - 1
            break
    for r, c in node.cells:
        nr, nc = r + dr_sign * actual_fall, c
        if 0 <= nr < H and 0 <= nc < W:
            new_grid[nr, nc] = node.color
    ctx.grid = new_grid
    return ctx


def grid_has_color_at(ctx: Context, r: int, c: int, exclude: Node) -> bool:
    """Verifica se uma célula tem cor != 0 e != exclude."""
    if not (0 <= r < ctx.grid.shape[0] and 0 <= c < ctx.grid.shape[1]):
        return True  # colide com borda
    if ctx.grid[r, c] == 0:
        return False
    if (r, c) in exclude.cells:
        return False
    return True


# ============================================================================
# FAMÍLIA 2 — CAUSALIDADE BAYESIANA (Se(A)->Afeta(B))
# ============================================================================

def causal_apply_if(node_a: Node, action_a, node_b: Node, ctx: Context) -> Context:
    """P(A|B) — se o node A tem uma propriedade, AFETA node B."""
    if not action_a(ctx):
        return ctx
    # Aplica mudança em B
    return ctx


# Probabilidades simples — modeladas como count-based
class BayesianTracker:
    """Tracker Bayesiano para P(propriedade|contexto) com suavização Laplace."""
    def __init__(self):
        self.counts: Dict[Tuple, Dict[Any, int]] = {}

    def update(self, key: Tuple, value: Any):
        if key not in self.counts:
            self.counts[key] = {}
        self.counts[key][value] = self.counts[key].get(value, 0) + 1

    def prob(self, key: Tuple, value: Any) -> float:
        if key not in self.counts:
            return 0.5
        total = sum(self.counts[key].values()) + len(self.counts[key])
        return (self.counts[key].get(value, 0) + 1) / total


# ============================================================================
# FAMÍLIA 3 — ESCALA & PROPORÇÃO
# ============================================================================

def act_scale_up(node: Node, ctx: Context, factor: int = 2) -> Context:
    """Amplia o objeto por fator (pixel->bloco factor×factor)."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    for r, c in node.cells:
        new_grid[r, c] = 0
    for r, c in node.cells:
        for i in range(factor):
            for j in range(factor):
                nr, nc = r * factor + i, c * factor + j
                if 0 <= nr < H and 0 <= nc < W:
                    new_grid[nr, nc] = node.color
    ctx.grid = new_grid
    return ctx


def act_scale_down(node: Node, ctx: Context, factor: int = 2) -> Context:
    """Reduz o objeto por fator (sample top-left)."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    for r, c in node.cells:
        new_grid[r, c] = 0
    for r, c in node.cells:
        if r % factor == 0 and c % factor == 0:
            nr, nc = r // factor, c // factor
            if 0 <= nr < H and 0 <= nc < W:
                new_grid[nr, nc] = node.color
    ctx.grid = new_grid
    return ctx


def act_tile(node: Node, ctx: Context, n_h: int, n_w: int) -> Context:
    """Replica o objeto n_h × n_w vezes em ladrilho."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    # limpa area
    for r, c in node.cells:
        new_grid[r, c] = 0
    base_h, base_w = node.h, node.w
    for ti in range(n_h):
        for tj in range(n_w):
            for r, c in node.cells:
                nr = r - node.r_min + ti * base_h
                nc = c - node.c_min + tj * base_w
                if 0 <= nr < H and 0 <= nc < W:
                    new_grid[nr, nc] = node.color
    ctx.grid = new_grid
    return ctx


# ============================================================================
# FAMÍLIA 4 — HIERARQUIA LÓGICA (É_UM, PERTENCE_A, ParteDe)
# ============================================================================

def build_hierarchy(nodes: List[Node]) -> Dict[int, int]:
    """Constrói árvore de contenção: parent[child] = parent.id."""
    parent = {}
    for n in nodes:
        parent[n.id] = -1  # root
    for n1 in nodes:
        for n2 in nodes:
            if n1 is n2 or n1.color == 0 or n2.color == 0:
                continue
            # n1 dentro de n2?
            if (n1.r_min > n2.r_min and n1.r_max < n2.r_max and
                    n1.c_min > n2.c_min and n1.c_max < n2.c_max):
                # mais interno vence
                if parent[n1.id] == -1 or n2.id != parent[n1.id]:
                    parent[n1.id] = n2.id
    return parent


def act_extract_part(node: Node, ctx: Context, part_type: str = 'children') -> Context:
    """Mantém só as partes (filhos) ou só o objeto sem fundo."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    new_grid[:, :] = 0
    if part_type == 'children':
        # mantém só nodes que estão contidos em node (mas não inclui node)
        # (essa lógica depende do parent_map)
        pass
    elif part_type == 'outline':
        act_outline(node, ctx)
    return ctx


# ============================================================================
# FAMÍLIA 5 — ESTRUTURAS RELACIONAIS (grafos)
# ============================================================================

def build_graph(nodes: List[Node], edges: Dict) -> Dict[int, Set[int]]:
    """Constrói grafo de adjacência (RELATIONAL)."""
    graph: Dict[int, Set[int]] = {n.id: set() for n in nodes}
    for a, b in edges['touches']:
        graph[a].add(b)
        graph[b].add(a)
    for child, parent in edges['inside']:
        if parent in graph:
            graph[parent].add(child)
    return graph


def find_path(graph: Dict[int, Set[int]], start: int, end: int) -> Optional[List[int]]:
    """BFS para encontrar caminho entre dois nodes (RELATIONAL)."""
    if start not in graph or end not in graph:
        return None
    from collections import deque
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None


def act_connect_to(node: Node, target_color: int, ctx: Context) -> Context:
    """Desenha linha do node até o objeto mais próximo da cor target_color (RELATIONAL)."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    r1, c1 = node.centroid()
    # encontra node de cor target mais próximo
    best = None
    best_d = float('inf')
    for n in ctx.nodes:
        if n.id == node.id or n.color != target_color:
            continue
        d = node.distance_to(n)
        if d < best_d:
            best_d = d
            best = n
    if best is None:
        return ctx
    r2, c2 = best.centroid()
    # Bresenham
    rr, cc = int(round(r1)), int(round(c1))
    tr, tc = int(round(r2)), int(round(c2))
    dr = abs(tr - rr)
    dc = abs(tc - cc)
    sr = (tr - rr) // (dr or 1)
    sc = (tc - cc) // (dc or 1)
    err = dr - dc
    while True:
        if 0 <= rr < H and 0 <= cc < W:
            new_grid[rr, cc] = node.color
        if (rr, cc) == (tr, tc):
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            rr += sr
        if e2 < dr:
            err += dr
            cc += sc
    ctx.grid = new_grid
    return ctx


# ============================================================================
# FAMÍLIA 6 — DINÂMICAS TEMPORAIS (sequência, before/after)
# ============================================================================

class TemporalBuffer:
    """Buffer de estados passados (TEMPORAL) — usado em H52 imaginação."""
    def __init__(self, max_size: int = 16):
        self.buffer: List[np.ndarray] = []
        self.max_size = max_size

    def push(self, grid: np.ndarray):
        self.buffer.append(grid.copy())
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def before(self, k: int = 1) -> Optional[np.ndarray]:
        if k > len(self.buffer):
            return None
        return self.buffer[-k]

    def rate(self) -> float:
        """Taxa de mudança média (change rate)."""
        if len(self.buffer) < 2:
            return 0.0
        diffs = []
        for i in range(1, len(self.buffer)):
            diffs.append(float(np.sum(self.buffer[i] != self.buffer[i - 1])))
        return sum(diffs) / len(diffs)


# ============================================================================
# FAMÍLIA 7 — SIMETRIA & ASSIMETRIA
# ============================================================================

def act_mirror(node: Node, ctx: Context, axis: str = 'y') -> Context:
    """Espelha o objeto num eixo."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    for r, c in node.cells:
        new_grid[r, c] = 0
    if axis == 'y':  # espelha vertical (coluna)
        for r, c in node.cells:
            nc = node.c_min + (node.c_max - c)
            if 0 <= r < H and 0 <= nc < W:
                new_grid[r, nc] = node.color
    elif axis == 'x':  # espelha horizontal (linha)
        for r, c in node.cells:
            nr = node.r_min + (node.r_max - r)
            if 0 <= nr < H and 0 <= c < W:
                new_grid[nr, c] = node.color
    elif axis == 'd':  # diagonal
        for r, c in node.cells:
            nr, nc = c - node.c_min + node.r_min, r - node.r_min + node.c_min
            if 0 <= nr < H and 0 <= nc < W:
                new_grid[nr, nc] = node.color
    ctx.grid = new_grid
    return ctx


def act_complete_symmetry(node: Node, ctx: Context, axis: str = 'y') -> Context:
    """Completa a simetria: copia a parte existente para o outro lado."""
    if axis == 'y':
        # metade esquerda -> metade direita
        new_grid = ctx.grid.copy()
        H, W = ctx.grid.shape
        for r in range(node.r_min, node.r_max + 1):
            for c in range(node.c_min, (node.c_min + node.c_max) // 2 + 1):
                if (r, c) in node.cells:
                    mirror_c = node.c_min + node.c_max - c
                    if 0 <= mirror_c < W:
                        new_grid[r, mirror_c] = node.color
        ctx.grid = new_grid
    return ctx


# ============================================================================
# FAMÍLIA 8 — META-COGNIÇÃO (Rewind, Evaluate, Prune, Imagination)
# ============================================================================

class MetaCortex:
    """Controlador meta-cognitivo: REWIND, EVALUATE, PRUNE, IMAGINE."""
    def __init__(self):
        self.snapshots: List[Dict] = []
        self.error_history: List[float] = []

    def checkpoint(self, ctx: Context):
        """REWIND point: salva estado atual."""
        self.snapshots.append(ctx.snapshot())

    def rewind(self, ctx: Context, k: int = 1) -> Context:
        """Volta k passos atrás."""
        if k > len(self.snapshots):
            return ctx
        ctx.restore(self.snapshots[-k])
        return ctx

    def evaluate(self, ctx: Context, target: np.ndarray) -> float:
        """Mede erro atual (cell-difference)."""
        if ctx.grid.shape != target.shape:
            return 999999.0
        err = float(np.sum(ctx.grid != target))
        self.error_history.append(err)
        return err

    def should_prune(self, recent: int = 5, threshold: float = 0.0) -> bool:
        """Prune: se não está progredindo nos últimos K passos."""
        if len(self.error_history) < recent:
            return False
        return self.error_history[-1] >= self.error_history[-recent]

    def is_progressing(self, window: int = 5) -> bool:
        """Está melhorando? (Oudeyer/Schmidhuber learning progress)."""
        if len(self.error_history) < 2 * window:
            return False
        recent = self.error_history[-window:]
        older = self.error_history[-2 * window:-window]
        if not older or not recent:
            return False
        return sum(recent) < sum(older) * 0.95  # 5% melhor


# ============================================================================
# GENES REGISTRADOS — A BIBLIOTECA COMPLETA
# ============================================================================

# Predicados (lógica de seleção)
def pred_true(node, ctx): return True
def pred_color(node, ctx, c): return node.color == c
def pred_is_square(node, ctx): return node.is_square
def pred_is_line(node, ctx): return node.is_line
def pred_is_hollow(node, ctx): return node.is_hollow
def pred_is_point(node, ctx): return node.is_point
def pred_max_size(node, ctx):
    others = [n for n in ctx.nodes if n.color != 0]
    return bool(others) and node.size == max(n.size for n in others)
def pred_min_size(node, ctx):
    others = [n for n in ctx.nodes if n.color != 0]
    return bool(others) and node.size == min(n.size for n in others)
def pred_most_freq_color(node, ctx):
    from collections import Counter
    cnt = Counter(n.color for n in ctx.nodes if n.color != 0)
    return bool(cnt) and node.color == cnt.most_common(1)[0][0]
def pred_least_freq_color(node, ctx):
    from collections import Counter
    cnt = Counter(n.color for n in ctx.nodes if n.color != 0)
    return bool(cnt) and node.color == cnt.most_common()[-1][0]
def pred_is_sym_x(node, ctx): return node.is_sym_x
def pred_is_sym_y(node, ctx): return node.is_sym_y
def pred_is_sym_d(node, ctx): return node.is_sym_d
def pred_touches_color(node, ctx, c):
    for a, b in ctx.edges['touches']:
        if a == node.id and ctx.nodes[b].color == c: return True
        if b == node.id and ctx.nodes[a].color == c: return True
    return False
def pred_inside_color(node, ctx, c):
    for child, parent in ctx.edges['inside']:
        if child == node.id and ctx.nodes[parent].color == c: return True
    return False
def pred_has_holes(node, ctx): return node.n_holes > 0


# Ações (transformações)
def act_recolor(node, ctx, c):
    new_grid = ctx.grid.copy()
    for r, cc in node.cells:
        new_grid[r, cc] = c
    ctx.grid = new_grid
    return ctx

def act_delete(node, ctx):
    new_grid = ctx.grid.copy()
    for r, c in node.cells:
        new_grid[r, c] = 0
    ctx.grid = new_grid
    return ctx

def act_keep(node, ctx):
    return ctx  # no-op


# ============================================================================
# GENES CRÍTICOS QUE FALTAVAM (identificados pelo H100/H111 do relatório)
# Genes "nascidos" da análise do resíduo do ARC — preenchem o gap real.
# ============================================================================

def act_recolor_by_size_rank(node, ctx):
    """Recolor cada objeto por RANK de TAMANHO (menor→cor 1, maior→cor N)."""
    new_grid = ctx.grid.copy()
    valid = sorted([n for n in ctx.nodes if n.color != 0], key=lambda n: n.size)
    if not valid:
        return ctx
    target_colors = sorted(set(int(c) for c in new_grid.flatten() if c != 0)) or [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i, n in enumerate(valid):
        new_color = target_colors[i] if i < len(target_colors) else (i + 1)
        for r, c in n.cells:
            new_grid[r, c] = new_color
    ctx.grid = new_grid
    return ctx


def act_recolor_by_freq_rank(node, ctx):
    """Recolor por RANK de FREQUÊNCIA."""
    new_grid = ctx.grid.copy()
    valid = [n for n in ctx.nodes if n.color != 0]
    if not valid:
        return ctx
    color_size = {}
    for n in valid:
        color_size.setdefault(n.color, 0)
        color_size[n.color] += n.size
    sorted_colors = sorted(color_size.keys(), key=lambda c: color_size[c])
    target_colors = sorted(set(int(c) for c in new_grid.flatten() if c != 0)) or [1, 2, 3, 4, 5, 6, 7, 8, 9]
    color_map = {old: target_colors[i] if i < len(target_colors) else (i + 1)
                 for i, old in enumerate(sorted_colors)}
    H, W = ctx.grid.shape
    for r in range(H):
        for c in range(W):
            if new_grid[r, c] in color_map:
                new_grid[r, c] = color_map[new_grid[r, c]]
    ctx.grid = new_grid
    return ctx


def act_recolor_by_centroid_order(node, ctx):
    """Recolor objetos por ORDEM do centroide (top→bottom, left→right)."""
    new_grid = ctx.grid.copy()
    valid = [n for n in ctx.nodes if n.color != 0]
    if not valid:
        return ctx
    valid.sort(key=lambda n: (n.centroid()[0], n.centroid()[1]))
    target_colors = sorted(set(int(c) for c in new_grid.flatten() if c != 0)) or [1, 2, 3, 4, 5, 6, 7, 8, 9]
    new_grid_full = np.zeros_like(new_grid)
    for i, n in enumerate(valid):
        new_color = target_colors[i] if i < len(target_colors) else (i + 1)
        for r, c in n.cells:
            new_grid_full[r, c] = new_color
    ctx.grid = new_grid_full
    return ctx


def act_swap_2_colors(node, ctx, color_a=1, color_b=2):
    """Troca duas cores (par-wise swap)."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    for r in range(H):
        for c in range(W):
            v = new_grid[r, c]
            if v == color_a:
                new_grid[r, c] = color_b
            elif v == color_b:
                new_grid[r, c] = color_a
    ctx.grid = new_grid
    return ctx


def act_invert_grid(node, ctx):
    """Inverte cores 0↔max."""
    new_grid = ctx.grid.copy()
    max_c = int(new_grid.max())
    new_grid = max_c - new_grid
    ctx.grid = new_grid
    return ctx


def act_rotate_grid_global(node, ctx, k=1):
    """Rotaciona a grade inteira."""
    new_grid = np.rot90(ctx.grid, k=k)
    ctx.grid = new_grid
    return ctx


def act_flip_grid_global(node, ctx, axis='y'):
    """Espelha a grade inteira."""
    new_grid = np.flipud(ctx.grid) if axis == 'x' else np.fliplr(ctx.grid)
    ctx.grid = new_grid
    return ctx


def act_swap_objects(node, ctx):
    """Troca a posição de 2 objetos adjacentes (cores)."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    valid = [n for n in ctx.nodes if n.color != 0]
    if len(valid) < 2:
        return ctx
    n1, n2 = valid[0], valid[1]
    color1, color2 = n1.color, n2.color
    for r in range(H):
        for c in range(W):
            v = new_grid[r, c]
            if v == color1:
                new_grid[r, c] = color2
            elif v == color2:
                new_grid[r, c] = color1
    ctx.grid = new_grid
    return ctx


def act_replace_per_object(node, ctx, target_color=1):
    """Recolor o objeto com nova cor, parametrizado."""
    return act_recolor(node, ctx, target_color)


# ============================================================================
# GENES "ZEALOT" (H93) — Periodic-complete + Enclosed-fill + Tile
# Estes são os genes que o relatório identificou como os que mais
# adicionam tarefas (+5 tasks: 00d62c1b, 0dfd9992, 29ec7d0e, 484b58aa, c3f564a4)
# ============================================================================

def act_complete_periodic(node, ctx, period=2):
    """Completa padrão periódico ao longo de linhas/colunas."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    # para cada linha, vê o primeiro elemento não-zero e repete com period
    for r in range(H):
        first_nz = None
        for c in range(W):
            if new_grid[r, c] != 0:
                if first_nz is None:
                    first_nz = c
                else:
                    expected = (c - first_nz) % period
                    if expected == 0 and new_grid[r, c] == 0:
                        # copia o primeiro
                        new_grid[r, c] = new_grid[r, first_nz]
    ctx.grid = new_grid
    return ctx


def act_fill_enclosed(node, ctx, fill_color=2):
    """Preenche regiões cercadas (enclosed areas)."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    # detecta regiões 0 cercadas por cor não-zero
    from scipy.ndimage import label
    bg_mask = (new_grid == 0)
    lbl, num = label(bg_mask)
    for i in range(1, num + 1):
        region = (lbl == i)
        # verifica se essa região toca a borda
        touches_border = (
            region[0, :].any() or region[-1, :].any() or
            region[:, 0].any() or region[:, -1].any()
        )
        if not touches_border:
            new_grid[region] = fill_color
    ctx.grid = new_grid
    return ctx


def act_periodic_seed_lines(node, ctx, period=2):
    """Marca linhas alternadas (per-row periodic)."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    for r in range(H):
        if r % period == 0:
            # copia linha não-vazia mais próxima
            for src_r in range(H):
                if src_r != r and (new_grid[src_r] != 0).any():
                    new_grid[r] = new_grid[src_r]
                    break
    ctx.grid = new_grid
    return ctx


def act_count_neighbors(node, ctx, parity='even', target_color=2):
    """Recolor célula se #vizinhos-mesma-cor tem paridade X (H111: paridade de vizinhos)."""
    new_grid = ctx.grid.copy()
    H, W = ctx.grid.shape
    for r in range(H):
        for c in range(W):
            if new_grid[r, c] == 0:
                continue
            color = new_grid[r, c]
            count = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and new_grid[nr, nc] == color:
                    count += 1
            is_even = (count % 2 == 0)
            if (parity == 'even' and is_even) or (parity == 'odd' and not is_even):
                new_grid[r, c] = target_color
    ctx.grid = new_grid
    return ctx


# ============================================================================
# GENES CONCRETOS — a biblioteca em si
# ============================================================================

def make_genes(include_extended: bool = True, include_super: bool = False) -> List[Gene]:
    """Constrói a biblioteca completa de Spelke Genes.
    include_extended=True: inclui os genes "nascidos" (H100/H111/H93) — 30 genes
    include_extended=False: só os básicos (17 genes) — mais rápido
    """
    genes: List[Gene] = []

    # ---- FAMÍLIA 1: ESPAÇO & TOPOLOGIA ----
    genes += [
        Gene("Translate", Family.SPATIAL, Domain.GRID, ["obj"], ["obj"],
             lambda n, c, dr, dc: act_translate(n, c, dr, dc),
             description="Move objeto (dr, dc)"),
        Gene("Rotate90", Family.SPATIAL, Domain.GRID, ["obj"], ["obj"],
             lambda n, c, k=1: act_rotate(n, c, k),
             description="Rotaciona 90°×k"),
        Gene("Gravity", Family.SPATIAL, Domain.GRID, ["obj"], ["obj"],
             lambda n, c, dir='down': act_gravity(n, c, dir),
             description="Faz objeto cair até colidir"),
        Gene("MoveToEdge", Family.SPATIAL, Domain.GRID, ["obj"], ["obj"],
             lambda n, c, e='right': act_move_to_edge(n, c, e),
             description="Move até borda (r/l/t/b)"),
        Gene("FillHole", Family.SPATIAL, Domain.GRID, ["obj"], ["obj"],
             act_fill_holes,
             description="Preenche buracos topológicos"),
        Gene("Outline", Family.SPATIAL, Domain.GRID, ["obj"], ["obj"],
             act_outline,
             description="Mantém só contorno"),
    ]

    # ---- FAMÍLIA 3: ESCALA & PROPORÇÃO ----
    genes += [
        Gene("ScaleUp", Family.SCALE, Domain.GRID, ["obj"], ["obj"],
             lambda n, c, f=2: act_scale_up(n, c, f),
             description="Amplia por fator"),
        Gene("ScaleDown", Family.SCALE, Domain.GRID, ["obj"], ["obj"],
             lambda n, c, f=2: act_scale_down(n, c, f),
             description="Reduz por fator"),
        Gene("Tile", Family.SCALE, Domain.GRID, ["obj"], ["grid"],
             lambda n, c, h=2, w=2: act_tile(n, c, h, w),
             description="Replica em ladrilho n_h × n_w"),
    ]

    # ---- FAMÍLIA 5: RELACIONAL ----
    genes += [
        Gene("ConnectTo", Family.RELATIONAL, Domain.GRID, ["obj", "color"], ["line"],
             lambda n, c, target: act_connect_to(n, target, c),
             needs_global=True,
             description="Desenha linha até objeto de cor X"),
    ]

    # ---- FAMÍLIA 7: SIMETRIA ----
    genes += [
        Gene("MirrorX", Family.SYMMETRY, Domain.GRID, ["obj"], ["obj"],
             lambda n, c: act_mirror(n, c, 'x'),
             description="Espelha horizontal"),
        Gene("MirrorY", Family.SYMMETRY, Domain.GRID, ["obj"], ["obj"],
             lambda n, c: act_mirror(n, c, 'y'),
             description="Espelha vertical"),
        Gene("MirrorDiag", Family.SYMMETRY, Domain.GRID, ["obj"], ["obj"],
             lambda n, c: act_mirror(n, c, 'd'),
             description="Espelha diagonal"),
        Gene("CompleteSymY", Family.SYMMETRY, Domain.GRID, ["obj"], ["obj"],
             lambda n, c: act_complete_symmetry(n, c, 'y'),
             description="Completa simetria vertical"),
    ]

    # ---- AÇÕES BÁSICAS (entre todas as famílias) ----
    genes += [
        Gene("Recolor", Family.SPATIAL, Domain.GRID, ["obj", "color"], ["obj"],
             lambda n, c, col: act_recolor(n, c, col),
             description="Muda cor do objeto"),
        Gene("Delete", Family.SPATIAL, Domain.GRID, ["obj"], [""],
             lambda n, c: act_delete(n, c),
             description="Remove objeto"),
        Gene("Keep", Family.SPATIAL, Domain.GRID, ["obj"], ["obj"],
             lambda n, c: act_keep(n, c),
             description="No-op (mantém)"),
    ]

    if not include_extended:
        return genes

    # ---- GENES "NASCIDOS" (H100/H111) — genes que faltavam e moveram o número ----
    genes += [
        Gene("RecolorBySizeRank", Family.SPATIAL, Domain.GRID, ["obj"], ["grid"],
             lambda n, c: act_recolor_by_size_rank(n, c),
             needs_global=True,
             description="Recolor por RANK de TAMANHO"),
        Gene("RecolorByFreqRank", Family.SPATIAL, Domain.GRID, ["obj"], ["grid"],
             lambda n, c: act_recolor_by_freq_rank(n, c),
             needs_global=True,
             description="Recolor por RANK de FREQUÊNCIA"),
        Gene("RecolorByCentroidOrder", Family.SPATIAL, Domain.GRID, ["obj"], ["grid"],
             lambda n, c: act_recolor_by_centroid_order(n, c),
             needs_global=True,
             description="Recolor por ORDEM do centroide"),
        Gene("Swap2Colors", Family.SPATIAL, Domain.GRID, ["obj", "color", "color"], ["grid"],
             lambda n, c, a, b: act_swap_2_colors(n, c, a, b),
             needs_global=True,
             description="Troca duas cores"),
        Gene("InvertGrid", Family.SPATIAL, Domain.GRID, ["obj"], ["grid"],
             lambda n, c: act_invert_grid(n, c),
             needs_global=True,
             description="Inverte cores 0↔max"),
        Gene("RotateGrid", Family.SPATIAL, Domain.GRID, ["obj"], ["grid"],
             lambda n, c, k=1: act_rotate_grid_global(n, c, k),
             needs_global=True,
             description="Rotaciona grade inteira"),
        Gene("FlipGrid", Family.SPATIAL, Domain.GRID, ["obj"], ["grid"],
             lambda n, c, axis='y': act_flip_grid_global(n, c, axis),
             needs_global=True,
             description="Espelha grade inteira"),
        Gene("SwapObjects", Family.SPATIAL, Domain.GRID, ["obj"], ["grid"],
             lambda n, c: act_swap_objects(n, c),
             needs_global=True,
             description="Troca 2 objetos (par-wise)"),
        Gene("ReplaceColor", Family.SPATIAL, Domain.GRID, ["obj", "color", "color"], ["grid"],
             lambda n, c, old_c, new_c: act_replace_per_object(n, c, old_c, new_c) if old_c == 0 else act_replace_per_object(n, c, old_c),
             needs_global=True,
             description="Substitui cor globalmente"),
    ]

    # ---- GENES "ZEALOT" (H93) ----
    genes += [
        Gene("CompletePeriodic", Family.SPATIAL, Domain.GRID, ["obj"], ["grid"],
             lambda n, c, p=2: act_complete_periodic(n, c, p),
             needs_global=True,
             description="Completa padrão periódico nas linhas"),
        Gene("FillEnclosed", Family.SPATIAL, Domain.GRID, ["obj"], ["grid"],
             lambda n, c, col=2: act_fill_enclosed(n, c, col),
             needs_global=True,
             description="Preenche regiões cercadas"),
        Gene("PeriodicSeedLines", Family.SPATIAL, Domain.GRID, ["obj"], ["grid"],
             lambda n, c, p=2: act_periodic_seed_lines(n, c, p),
             needs_global=True,
             description="Replica linhas alternadas (per-row)"),
        Gene("CountNeighborsParity", Family.RELATIONAL, Domain.GRID, ["obj"], ["grid"],
             lambda n, c, par='even', col=2: act_count_neighbors(n, c, par, col),
             needs_global=True,
             description="Recolor por paridade do # vizinhos (H111)"),
    ]

    if include_super:
        try:
            from core.spelke_extended import get_extra_genes
            genes += get_extra_genes()
        except Exception as e:
            print(f"[WARN] Não carregou bibliotecas estendidas: {e}")

    return genes


# ============================================================================
# TABELA DE PREDICADOS (gera todas as combinações predicado×cor)
# ============================================================================

def make_predicates(include_super: bool = False, include_cognitive: bool = False):
    """Lista de (nome, fn, params_possíveis)."""
    preds = [
        ('True', pred_true, [None]),
        ('Color', pred_color, 'colors'),
        ('IsSquare', pred_is_square, [None]),
        ('IsLine', pred_is_line, [None]),
        ('IsHollow', pred_is_hollow, [None]),
        ('IsPoint', pred_is_point, [None]),
        ('MaxSize', pred_max_size, [None]),
        ('MinSize', pred_min_size, [None]),
        ('MostFreqColor', pred_most_freq_color, [None]),
        ('LeastFreqColor', pred_least_freq_color, [None]),
        ('IsSymX', pred_is_sym_x, [None]),
        ('IsSymY', pred_is_sym_y, [None]),
        ('IsSymDiag', pred_is_sym_d, [None]),
        ('HasHoles', pred_has_holes, [None]),
        ('TouchesColor', pred_touches_color, 'colors'),
        ('InsideColor', pred_inside_color, 'colors'),
    ]
    if include_super:
        try:
            from core.spelke_extended import get_extra_predicates
            preds += get_extra_predicates()
        except Exception as e:
            print(f"[WARN] Não carregou predicados extras: {e}")

    if include_cognitive:
        # Adiciona primitivas cognitivas COMPATÍVEIS com busca determinística:
        # IGUAL_A, DIFERENTE_DE, OBJETO, MEMBRO_DE, TIPO_DE, TEM_PROP
        # (As outras — SE/QUER/ACREDITA — exigem agente e ficam para Fase 2)
        from core.spelke_cognitive import (pred_IGUAL_A, pred_DIFERENTE_DE,
                                            pred_OBJETO, pred_TIPO_DE,
                                            pred_TEM_PROPRIEDADE, pred_NEGA)
        preds += [
            ('OBJETO', pred_OBJETO, [None]),
            ('TIPO_DE', pred_TIPO_DE, 'callable'),
            ('TEM_PROP', pred_TEM_PROPRIEDADE, 'prop_value'),
            # IGUAL_A / DIFERENTE_DE precisam de "outro nó" — marcados como callable
        ]
    return preds


# ============================================================================
# RESOLVER — pega biblioteca + tarefa e busca regra
# ============================================================================

def solve_task(train: List[Tuple[np.ndarray, np.ndarray]],
               test_in: np.ndarray,
               max_depth: int = 2,
               beam_width: int = 20,
               color_subset: int = 4,
               max_candidates: int = 400,
               use_extended: bool = True,
               use_super: bool = False) -> Optional[List[Dict]]:
    """Busca em feixe: acha sequência de regras que resolve train e aplica em test.
    OTIMIZADA: pré-computa percepções, cache por grid hash, cap em candidatos.
    """
    # Coleta cores
    colors = set()
    for gi, go in train:
        colors.update(np.unique(gi).tolist())
        colors.update(np.unique(go).tolist())
    colors = sorted([c for c in colors if c != 0])[:color_subset] or [0]
    colors.append(0)  # sempre inclui 0 (fundo)

    # Predicados disponíveis
    predicates = make_predicates(include_super=use_super)

    # Beam: lista de (sequência, grades_resultantes, erro_total)
    initial_grids = [gi.copy() for gi, go in train]
    initial_err = sum(cell_err(gi, go) for gi, go in train)
    beam = [([], initial_grids, initial_err)]

    # Cache de percepções por (grid_id, content_hash)
    perceive_cache = {}
    def cached_perceive(g):
        h = hash(g.tobytes())
        if h not in perceive_cache:
            perceive_cache[h] = perceive(g)
        return perceive_cache[h]

    # Cache de candidatos
    candidate_cache = {}
    def get_candidates(g_in, g_target):
        h = (hash(g_in.tobytes()), hash(g_target.tobytes()))
        if h not in candidate_cache:
            c = gen_candidate_rules(g_in, g_target, colors, predicates, max_candidates=max_candidates, use_extended=use_extended, use_super=use_super)
            candidate_cache[h] = c
        return candidate_cache[h]

    for depth in range(max_depth):
        next_beam = []
        for seq, current_grids, err in beam:
            if err == 0:
                return seq
            # Gera candidatos com base na primeira grade (mais rápido)
            try:
                candidates = get_candidates(current_grids[0], train[0][1])
            except Exception:
                continue
            for rule_dict in candidates:
                # Aplica em todas as grades
                new_grids = []
                new_err = 0
                valid = True
                for i, (gi, go) in enumerate(train):
                    try:
                        g_out = apply_rule_dict_cached(current_grids[i], rule_dict, gi, go, cached_perceive)
                    except Exception:
                        valid = False
                        break
                    h_err = cell_err(g_out, go)
                    new_grids.append(g_out)
                    new_err += h_err
                if not valid:
                    continue
                if not any(not np.array_equal(current_grids[i], new_grids[i]) for i in range(len(train))):
                    continue
                if new_err <= err:
                    next_beam.append((seq + [rule_dict], new_grids, new_err))

        if not next_beam:
            break
        next_beam.sort(key=lambda x: x[2])
        beam = next_beam[:beam_width]
        if beam and beam[0][2] == 0:
            return beam[0][0]

    return None if not beam or beam[0][2] > 0 else beam[0][0]


def cell_err(pred, target):
    if pred.shape != target.shape:
        return 999999.0 + abs(pred.shape[0] - target.shape[0]) + abs(pred.shape[1] - target.shape[1])
    return float(np.sum(pred != target))


def gen_candidate_rules(grid_in: np.ndarray, grid_target: np.ndarray, colors: List[int], predicates, max_candidates: int = 400, use_extended: bool = True, use_super: bool = False) -> List[Dict]:
    """Gera candidatas de regra (predicado + gene + parâmetro) para uma grade.
    Versão OTIMIZADA: pré-caching, cap em cores e params, foco em velocidade.
    """
    nodes, edges = perceive(grid_in)
    out = []
    if len(nodes) < 1:
        return out
    # Cores presentes no input (e output) — mais relevante
    input_colors = set(int(c) for c in np.unique(grid_in)) - {0}
    if grid_target is not None:
        for c in np.unique(grid_target):
            input_colors.add(int(c))
    input_colors.discard(0)
    input_colors = sorted(input_colors)[:3] or [0]
    all_genes = make_genes(include_extended=use_extended, include_super=use_super)

    # Prioriza genes "globais" (que mudam a grade) e predicados de posição
    for pname, pfn, pparams in predicates:
        if pparams == 'colors':
            for c in input_colors[:1]:  # cap em 1 cor
                for g in all_genes:
                    for gparam in gene_params(g, grid_in)[:2]:
                        out.append({
                            'pred_name': pname, 'pred_fn': pfn, 'pred_param': c,
                            'gene': g, 'gene_param': gparam,
                        })
                        if len(out) >= max_candidates:
                            return out
        else:  # [None]
            for g in all_genes:
                for gparam in gene_params(g, grid_in)[:2]:
                    out.append({
                        'pred_name': pname, 'pred_fn': pfn, 'pred_param': None,
                        'gene': g, 'gene_param': gparam,
                    })
                    if len(out) >= max_candidates:
                        return out
    return out


def gene_params(gene: Gene, grid: np.ndarray) -> List[Any]:
    """Quais valores de parâmetro um gene aceita?
    LIMITADO para velocidade (cap em 2 params por gene).
    """
    if gene.name in ('Translate',):
        return [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
    elif gene.name in ('Rotate90', 'RotateGrid'):
        return [1, 2]
    elif gene.name in ('Gravity',):
        return ['down']
    elif gene.name in ('MoveToEdge', 'FlipGrid'):
        return ['right', 'left'] if gene.name == 'MoveToEdge' else ['x', 'y']
    elif gene.name in ('ScaleUp', 'ScaleDown'):
        return [2]
    elif gene.name in ('Tile',):
        return [(2, 2)]
    elif gene.name == 'Recolor':
        return [1, 2, 3, 4]
    elif gene.name == 'ConnectTo':
        return [1, 2, 3]
    elif gene.name in ('Swap2Colors',):
        return [(1, 2), (2, 3)]
    elif gene.name == 'CompletePeriodic':
        return [2]
    elif gene.name == 'FillEnclosed':
        return [1, 2]
    elif gene.name == 'PeriodicSeedLines':
        return [2]
    elif gene.name == 'CountNeighborsParity':
        return [('even', 2), ('odd', 3)]
    else:
        return [None]


def apply_rule_dict(grid_in: np.ndarray, rule: Dict, original_in: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Aplica um rule_dict a uma grade e retorna a grade transformada."""
    nodes, edges = perceive(grid_in)
    return apply_rule_dict_with_nodes(grid_in, rule, nodes, edges)


def apply_rule_dict_cached(grid_in: np.ndarray, rule: Dict, original_in: np.ndarray,
                            target: np.ndarray, cached_perceive) -> np.ndarray:
    """Versão com cache de percepção."""
    nodes, edges = cached_perceive(grid_in)
    return apply_rule_dict_with_nodes(grid_in, rule, nodes, edges)


def apply_rule_dict_with_nodes(grid_in: np.ndarray, rule: Dict,
                                nodes: List[Node], edges: Dict) -> np.ndarray:
    """Aplica regra usando nodes pré-computados."""
    ctx = Context(
        grid=grid_in.copy(),
        history=[],
        nodes=nodes,
        edges=edges,
        colors=set(int(c) for c in np.unique(grid_in)),
    )
    for node in nodes:
        if node.color == 0 and rule['pred_name'] not in ('True',):
            continue
        try:
            if rule['pred_param'] is None:
                ok = rule['pred_fn'](node, ctx)
            else:
                ok = rule['pred_fn'](node, ctx, rule['pred_param'])
        except Exception:
            continue
        if ok:
            try:
                if rule['gene_param'] is None:
                    ctx = rule['gene'].apply(node, ctx)
                else:
                    ctx = rule['gene'].apply(node, ctx, rule['gene_param'])
            except Exception:
                return grid_in
    return ctx.grid


# ============================================================================
# ESTATÍSTICAS
# ============================================================================

def library_stats(include_extended: bool = True):
    """Estatísticas da biblioteca."""
    genes = make_genes(include_extended=include_extended)
    families = {}
    for g in genes:
        families.setdefault(g.family.value, []).append(g.name)
    return {
        'total_genes': len(genes),
        'total_predicates': len(make_predicates()),
        'families': {k: len(v) for k, v in families.items()},
        'genes_by_family': families,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("SPELKE GENES — A BIBLIOTECA DE PRIMITIVAS COGNITIVAS")
    print("=" * 70)
    stats = library_stats()
    print(f"\nTotal de genes: {stats['total_genes']}")
    print(f"Total de predicados: {stats['total_predicates']}")
    print(f"\nPor família:")
    for fam, count in stats['families'].items():
        print(f"  {fam:25s} : {count:2d} genes")
    print(f"\nGenes disponíveis:")
    for fam, names in stats['genes_by_family'].items():
        print(f"  {fam}: {', '.join(names)}")
