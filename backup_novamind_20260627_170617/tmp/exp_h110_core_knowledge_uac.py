import os, sys, json, glob, time
import numpy as np
from scipy.ndimage import label

DEV = False
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "arc_data", "data", "training")

# =====================================================================
# 1. PERCEPCAO AVANCADA: GRAFOS + SPELKE CORE KNOWLEDGE
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
        
        # Geometry
        self.is_square = (self.h == self.w and self.h * self.w == self.size)
        self.is_line = (self.h == 1 or self.w == 1) and self.size > 1
        
        # Spatial placement
        self.is_top_edge = (self.r_min == 0)
        self.is_bottom_edge = (self.r_max == grid_shape[0] - 1)
        self.is_left_edge = (self.c_min == 0)
        self.is_right_edge = (self.c_max == grid_shape[1] - 1)
        
        # Shape Sig
        norm_cells = tuple(sorted((r - self.r_min, c - self.c_min) for r, c in cells))
        self.shape_sig = hash(norm_cells)

def build_graph(grid):
    H, W = grid.shape
    nodes = []
    # Identify objects (bg = 0 is ignored for objects usually, but in ARC bg isn't always 0.
    # We will treat all contiguous blocks of same color as nodes, including 0).
    for c in np.unique(grid):
        mask = (grid == c)
        lbl, num = label(mask)
        for i in range(1, num + 1):
            cells = tuple(map(tuple, np.argwhere(lbl == i)))
            nodes.append(Node(len(nodes), cells, c, (H, W)))
            
    edges = {'touches': set(), 'inside': set(), 'above': set(), 'left_of': set()}
    for n1 in nodes:
        for n2 in nodes:
            if n1 is n2: continue
            
            # Touches
            if not (n1.r_min > n2.r_max + 1 or n1.r_max < n2.r_min - 1 or n1.c_min > n2.c_max + 1 or n1.c_max < n2.c_min - 1):
                for r, c in n1.cells:
                    if (r+1, c) in n2.cells or (r-1, c) in n2.cells or (r, c+1) in n2.cells or (r, c-1) in n2.cells:
                        edges['touches'].add((n1.id, n2.id))
                        break
            
            # Inside (n1 bounding box strictly inside n2 bounding box, or cells strictly within)
            if n1.r_min > n2.r_min and n1.r_max < n2.r_max and n1.c_min > n2.c_min and n1.c_max < n2.c_max:
                edges['inside'].add((n1.id, n2.id))
                
            # Above (n1 is completely above n2)
            if n1.r_max < n2.r_min:
                edges['above'].add((n1.id, n2.id))
                
            # Left of (n1 is completely left of n2)
            if n1.c_max < n2.c_min:
                edges['left_of'].add((n1.id, n2.id))
                
    return nodes, edges

# =====================================================================
# 2. CORE KNOWLEDGE: GENES LÓGICOS UNIVERSAIS
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
        valid_nodes = [n for n in nodes if n.color != 0] # assume 0 is bg for size ranks
        if self.type == 'MaxSize':
            if not valid_nodes: return False
            return node.color != 0 and node.size == max(n.size for n in valid_nodes)
        if self.type == 'MinSize':
            if not valid_nodes: return False
            return node.color != 0 and node.size == min(n.size for n in valid_nodes)
        if self.type == 'MostFreqColor':
            freqs = {c: sum(1 for n in nodes if n.color == c) for c in np.unique([n.color for n in nodes])}
            if not freqs: return False
            return node.color == max(freqs, key=freqs.get)
        
        # Geometry
        if self.type == 'IsSquare': return node.is_square
        if self.type == 'IsLine': return node.is_line
        
        # Spatial
        if self.type == 'IsTopEdge': return node.is_top_edge
        if self.type == 'IsBottomEdge': return node.is_bottom_edge
        
        # Topology (Relational)
        if self.type == 'TouchesColor':
            for n1, n2 in edges['touches']:
                if n1 == node.id and nodes[n2].color == self.val: return True
            return False
        if self.type == 'InsideColor':
            for n1, n2 in edges['inside']:
                if n1 == node.id and nodes[n2].color == self.val: return True
            return False
        
        return False
        
    def __str__(self): return f"{self.type}({self.val if self.val is not None else ''})"

class Action:
    def __init__(self, type_, val=None):
        self.type = type_
        self.val = val
        
    def apply(self, node, grid_out):
        if self.type == 'Recolor':
            for r, c in node.cells: 
                if 0 <= r < grid_out.shape[0] and 0 <= c < grid_out.shape[1]:
                    grid_out[r, c] = self.val
                    
        elif self.type == 'Delete':
            for r, c in node.cells: 
                if 0 <= r < grid_out.shape[0] and 0 <= c < grid_out.shape[1]:
                    grid_out[r, c] = 0
                    
        elif self.type == 'Translate':
            dr, dc = self.val  # val is a tuple (dr, dc)
            for r, c in node.cells: grid_out[r, c] = 0 # clear old
            for r, c in node.cells:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_out.shape[0] and 0 <= nc < grid_out.shape[1]:
                    grid_out[nr, nc] = node.color
                    
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
# 3. EVOLUÇÃO GENÉTICA AMPLIFICADA
# =====================================================================
def generate_random_rule(colors):
    ptypes = [
        'True', 'Color', 'MaxSize', 'MinSize', 'MostFreqColor',
        'IsSquare', 'IsLine', 'IsTopEdge', 'IsBottomEdge', 
        'TouchesColor', 'InsideColor'
    ]
    ptype = np.random.choice(ptypes)
    pval = np.random.choice(colors) if ptype in ['Color', 'TouchesColor', 'InsideColor'] and len(colors)>0 else None
    
    atypes = ['Recolor', 'Delete', 'Translate']
    atype = np.random.choice(atypes)
    if atype == 'Recolor':
        aval = np.random.choice(colors) if len(colors)>0 else 0
    elif atype == 'Translate':
        # Random small shifts
        aval = (np.random.randint(-2, 3), np.random.randint(-2, 3))
    else:
        aval = None
        
    return Rule(Predicate(ptype, pval), Action(atype, aval))

def crossover(r1, r2):
    return Rule(r1.pred, r2.act) if np.random.rand() > 0.5 else Rule(r2.pred, r1.act)

def mutate(r, colors):
    if np.random.rand() > 0.5:
        return crossover(r, generate_random_rule(colors))
    else:
        return generate_random_rule(colors)

def test_time_training(train_tasks, generations=15, pop_size=300):
    colors = set()
    for gi, go in train_tasks:
        colors.update(np.unique(gi)); colors.update(np.unique(go))
    colors = list(colors)
    
    graphs = []
    for gi, go in train_tasks:
        graphs.append((gi, go, *build_graph(gi)))
        
    pop = [generate_random_rule(colors) for _ in range(pop_size)]
    
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
            next_pop.append(mutate(crossover(p1, p2), colors))
        pop = next_pop
        
    best_idx = np.argmax(fitness)
    return pop[best_idx] if fitness[best_idx] == 1000 * len(train_tasks) else None

# =====================================================================
# 4. SOLVER MAESTRO (Rodando sobre todo o ARC)
# =====================================================================
def solve_all():
    files = sorted(glob.glob(os.path.join(DATA, "*.json")))
    if DEV: files = files[:50]
    
    print(f"Iniciando UAC Core Knowledge sobre {len(files)} tarefas...")
    t0 = time.time()
    solved = []
    
    for idx, fp in enumerate(files):
        name = os.path.basename(fp)
        t = json.load(open(fp))
        train = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        tests = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
        
        # Filtro basico
        same_shape = all(gi.shape == go.shape for gi, go in train) and all(gi.shape == go.shape for gi, go in tests)
        if not same_shape: continue
            
        rule = test_time_training(train)
        if rule:
            # Validar no Teste
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
    print(f"Resolvidas EXATAS (Core Knowledge): {len(solved)}/{len(files)} ({len(solved)/len(files):.1%})")

if __name__ == "__main__":
    solve_all()
