import os, sys, json, glob, time
import numpy as np
from scipy.ndimage import label, binary_fill_holes

DEV = False
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "arc_data", "data", "training")

# =====================================================================
# 1. PERCEPCAO OMNI-CORE: FISICA, TOPOLOGIA E GEOMETRIA
# =====================================================================
class Node:
    def __init__(self, id, cells, color, grid_shape):
        self.id = id
        self.cells = set(cells)
        self.color = color
        self.size = len(cells)
        self.grid_shape = grid_shape
        
        r_coords = [r for r, c in cells]
        c_coords = [c for r, c in cells]
        self.r_min, self.r_max = min(r_coords), max(r_coords)
        self.c_min, self.c_max = min(c_coords), max(c_coords)
        self.bbox = (self.r_min, self.c_min, self.r_max, self.c_max)
        self.h = self.r_max - self.r_min + 1
        self.w = self.c_max - self.c_min + 1
        
        # Geometria Basica
        self.is_square = (self.h == self.w and self.h * self.w == self.size)
        self.is_line = (self.h == 1 or self.w == 1) and self.size > 1
        
        # Simetria e Topologia
        # Construir matriz booleana local do objeto
        self.mask = np.zeros((self.h, self.w), dtype=bool)
        for r, c in self.cells:
            self.mask[r - self.r_min, c - self.c_min] = True
            
        self.is_hollow = np.any(binary_fill_holes(self.mask) & ~self.mask)
        self.is_sym_x = np.array_equal(self.mask, np.flipud(self.mask)) if self.size > 1 else False
        self.is_sym_y = np.array_equal(self.mask, np.fliplr(self.mask)) if self.size > 1 else False

def build_graph(grid):
    H, W = grid.shape
    nodes = []
    # Usar scipy label para encontrar objetos contiguos (4-way)
    for c in np.unique(grid):
        mask = (grid == c)
        lbl, num = label(mask)
        for i in range(1, num + 1):
            cells = tuple(map(tuple, np.argwhere(lbl == i)))
            nodes.append(Node(len(nodes), cells, c, (H, W)))
            
    # Edges OMNI
    edges = {'touches': set(), 'inside': set(), 'aligned_r': set(), 'aligned_c': set()}
    for n1 in nodes:
        for n2 in nodes:
            if n1 is n2: continue
            
            # Touches
            if not (n1.r_min > n2.r_max + 1 or n1.r_max < n2.r_min - 1 or n1.c_min > n2.c_max + 1 or n1.c_max < n2.c_min - 1):
                for r, c in n1.cells:
                    if (r+1, c) in n2.cells or (r-1, c) in n2.cells or (r, c+1) in n2.cells or (r, c-1) in n2.cells:
                        edges['touches'].add((n1.id, n2.id))
                        break
            
            # Inside (Strict Bounding Box Include)
            if n1.r_min > n2.r_min and n1.r_max < n2.r_max and n1.c_min > n2.c_min and n1.c_max < n2.c_max:
                edges['inside'].add((n1.id, n2.id))
                
            # Alignment (Raycasting preparation)
            if n1.r_min == n2.r_min: edges['aligned_r'].add((n1.id, n2.id))
            if n1.c_min == n2.c_min: edges['aligned_c'].add((n1.id, n2.id))
                
    return nodes, edges

# =====================================================================
# 2. O OMNI-DICIONÁRIO (GENES)
# =====================================================================
class Predicate:
    def __init__(self, type_, val=None):
        self.type = type_
        self.val = val
        
    def eval(self, node, nodes, edges):
        # Base
        if self.type == 'True': return True
        if self.type == 'Color': return node.color == self.val
        
        # Arithmetic / Rank
        valid_nodes = [n for n in nodes if n.color != 0]
        if self.type == 'MaxSize': return bool(valid_nodes) and node.color != 0 and node.size == max(n.size for n in valid_nodes)
        if self.type == 'MinSize': return bool(valid_nodes) and node.color != 0 and node.size == min(n.size for n in valid_nodes)
        if self.type == 'MostFreqColor':
            freqs = {c: sum(1 for n in nodes if n.color == c) for c in np.unique([n.color for n in nodes])}
            return bool(freqs) and node.color == max(freqs, key=freqs.get)
        if self.type == 'LeastFreqColor':
            freqs = {c: sum(1 for n in nodes if n.color == c) for c in np.unique([n.color for n in nodes])}
            return bool(freqs) and node.color == min(freqs, key=freqs.get)
        
        # Geometry & Topology
        if self.type == 'IsSquare': return node.is_square
        if self.type == 'IsLine': return node.is_line
        if self.type == 'IsHollow': return node.is_hollow
        if self.type == 'IsSymX': return node.is_sym_x
        if self.type == 'IsSymY': return node.is_sym_y
        
        # Relational Topology
        if self.type == 'TouchesColor':
            return any(nodes[n2].color == self.val for n1, n2 in edges['touches'] if n1 == node.id)
        if self.type == 'InsideColor':
            return any(nodes[n2].color == self.val for n1, n2 in edges['inside'] if n1 == node.id)
            
        return False
        
    def __str__(self): return f"{self.type}({self.val if self.val is not None else ''})"

class Action:
    def __init__(self, type_, val=None):
        self.type = type_
        self.val = val
        
    def apply(self, node, grid_out):
        H, W = grid_out.shape
        
        # Colors
        if self.type == 'Recolor':
            for r, c in node.cells: 
                if 0 <= r < H and 0 <= c < W: grid_out[r, c] = self.val
        elif self.type == 'Delete':
            for r, c in node.cells: 
                if 0 <= r < H and 0 <= c < W: grid_out[r, c] = 0
                
        # Transformations (Symmetry)
        elif self.type == 'MirrorX':
            # Espelha o objeto sobre o seu proprio eixo X local
            for r, c in node.cells: grid_out[r, c] = 0
            for r, c in node.cells:
                new_r = node.r_min + (node.r_max - r)
                if 0 <= new_r < H and 0 <= c < W: grid_out[new_r, c] = node.color
                
        elif self.type == 'MirrorY':
            # Espelha sobre eixo Y local
            for r, c in node.cells: grid_out[r, c] = 0
            for r, c in node.cells:
                new_c = node.c_min + (node.c_max - c)
                if 0 <= r < H and 0 <= new_c < W: grid_out[r, new_c] = node.color
                
        # Topologia
        elif self.type == 'FillHole':
            # Preenche oca com a propria cor
            filled = binary_fill_holes(node.mask)
            for r in range(node.h):
                for c in range(node.w):
                    if filled[r, c]:
                        nr, nc = node.r_min + r, node.c_min + c
                        if 0 <= nr < H and 0 <= nc < W: grid_out[nr, nc] = node.color
                        
        # Movimento Espacial e Gravidade
        elif self.type == 'Translate':
            dr, dc = self.val
            for r, c in node.cells: grid_out[r, c] = 0
            for r, c in node.cells:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W: grid_out[nr, nc] = node.color
                
        elif self.type == 'Gravity':
            # Cai para baixo (maior R) ate tocar em algo q nao eh 0
            dr = self.val # 1 para baixo
            if dr != 0:
                for r, c in node.cells: grid_out[r, c] = 0
                # Calcula o quanto pode cair sem bater no chao H ou em algo nao-zero (ignorando self)
                max_fall = H - 1 - node.r_max
                actual_fall = max_fall
                for f in range(1, max_fall + 1):
                    hit = False
                    for r, c in node.cells:
                        if grid_out[r+f, c] != 0:
                            hit = True
                            break
                    if hit:
                        actual_fall = f - 1
                        break
                for r, c in node.cells:
                    nr = r + actual_fall
                    grid_out[nr, c] = node.color

    def __str__(self): return f"{self.type}({self.val if self.val is not None else ''})"

class Rule:
    def __init__(self, pred, act):
        self.pred = pred
        self.act = act
    def __str__(self): return f"IF {self.pred} THEN {self.act}"
    def execute(self, grid, nodes, edges):
        out = grid.copy()
        for node in nodes:
            if self.pred.eval(node, nodes, edges):
                self.act.apply(node, out)
        return out

def exact_match(pred_grid, target_grid):
    if pred_grid.shape != target_grid.shape: return False
    return np.array_equal(pred_grid, target_grid)

# =====================================================================
# 3. TYPED CROSSOVER & EVOLUTION
# =====================================================================
def get_valid_genes(graphs, colors):
    # Context-Aware Heuristic Primordial Soup
    has_hollow = any(n.is_hollow for gi, go, nodes, edges in graphs for n in nodes)
    has_sym = any(n.is_sym_x or n.is_sym_y for gi, go, nodes, edges in graphs for n in nodes)
    
    ptypes = [
        'True', 'Color', 'MaxSize', 'MinSize', 'MostFreqColor', 'LeastFreqColor',
        'IsSquare', 'IsLine', 'TouchesColor', 'InsideColor'
    ]
    if has_hollow: ptypes.append('IsHollow')
    if has_sym: ptypes.extend(['IsSymX', 'IsSymY'])
        
    atypes = ['Recolor', 'Delete', 'Translate', 'Gravity', 'MirrorX', 'MirrorY']
    if has_hollow: atypes.append('FillHole')
    
    return ptypes, atypes

def generate_random_rule(colors, ptypes, atypes):
    ptype = np.random.choice(ptypes)
    pval = np.random.choice(colors) if ptype in ['Color', 'TouchesColor', 'InsideColor'] and len(colors)>0 else None
    
    atype = np.random.choice(atypes)
    if atype == 'Recolor':
        aval = np.random.choice(colors) if len(colors)>0 else 0
    elif atype == 'Translate':
        aval = (np.random.randint(-3, 4), np.random.randint(-3, 4))
    elif atype == 'Gravity':
        aval = 1 # hardcode down gravity for now
    else:
        aval = None
        
    return Rule(Predicate(ptype, pval), Action(atype, aval))

def crossover(r1, r2):
    return Rule(r1.pred, r2.act) if np.random.rand() > 0.5 else Rule(r2.pred, r1.act)

def mutate(r, colors, ptypes, atypes):
    if np.random.rand() > 0.5:
        return crossover(r, generate_random_rule(colors, ptypes, atypes))
    else:
        return generate_random_rule(colors, ptypes, atypes)

def test_time_training(train_tasks, generations=20, pop_size=400):
    colors = set()
    for gi, go in train_tasks:
        colors.update(np.unique(gi)); colors.update(np.unique(go))
    colors = list(colors)
    
    graphs = []
    for gi, go in train_tasks:
        graphs.append((gi, go, *build_graph(gi)))
        
    ptypes, atypes = get_valid_genes(graphs, colors)
        
    pop = [generate_random_rule(colors, ptypes, atypes) for _ in range(pop_size)]
    
    for gen in range(generations):
        fitness = []
        for r in pop:
            score = 0
            for gi, go, nodes, edges in graphs:
                pred = r.execute(gi, nodes, edges)
                if exact_match(pred, go): score += 1000
                else: score -= np.sum(pred != go)
            fitness.append(score)
            
        best_idx = np.argmax(fitness)
        if fitness[best_idx] == 1000 * len(train_tasks):
            return pop[best_idx]
            
        parents = [pop[i] for i in np.argsort(fitness)[-pop_size//2:]]
        next_pop = parents[:]
        while len(next_pop) < pop_size:
            p1 = parents[np.random.randint(len(parents))]
            p2 = parents[np.random.randint(len(parents))]
            next_pop.append(mutate(crossover(p1, p2), colors, ptypes, atypes))
        pop = next_pop
        
    best_idx = np.argmax(fitness)
    return pop[best_idx] if fitness[best_idx] == 1000 * len(train_tasks) else None

# =====================================================================
# 4. SOLVER MAESTRO (Omni-Run)
# =====================================================================
def solve_all():
    files = sorted(glob.glob(os.path.join(DATA, "*.json")))
    if DEV: files = files[:50]
    
    print(f"Iniciando OMNI-CORE UAC sobre {len(files)} tarefas...")
    t0 = time.time()
    solved = []
    
    for idx, fp in enumerate(files):
        name = os.path.basename(fp)
        t = json.load(open(fp))
        train = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        tests = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
        
        same_shape = all(gi.shape == go.shape for gi, go in train) and all(gi.shape == go.shape for gi, go in tests)
        if not same_shape: continue
            
        rule = test_time_training(train)
        if rule:
            ok = True
            for ti, to in tests:
                pred = rule.execute(ti, *build_graph(ti))
                if not exact_match(pred, to):
                    ok = False
                    break
            if ok:
                solved.append(name)
                print(f"[{idx+1}/{len(files)}] {name} RESOLVIDA! Regra inventada: {rule}")
                
    print("=" * 60)
    print(f"Finalizado em {time.time()-t0:.1f}s")
    print(f"Resolvidas EXATAS (OMNI): {len(solved)}/{len(files)} ({len(solved)/len(files):.1%})")

if __name__ == "__main__":
    solve_all()
