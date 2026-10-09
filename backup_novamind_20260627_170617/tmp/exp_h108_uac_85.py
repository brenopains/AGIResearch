import os, sys, json, glob, time
import numpy as np
from scipy.ndimage import label
from collections import Counter

DEV = True  # Set to False to run all tasks
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "arc_data", "data", "training")

# =====================================================================
# 1. PERCEPCAO: GRAFO UNIVERSAL (Universal Segmenter)
# =====================================================================
class Node:
    def __init__(self, id, cells, color, grid_shape):
        self.id = id
        self.cells = set(cells)
        self.color = color
        self.size = len(cells)
        r_coords = [r for r, c in cells]
        c_coords = [c for r, c in cells]
        self.r_min, self.r_max = min(r_coords), max(r_coords)
        self.c_min, self.c_max = min(c_coords), max(c_coords)
        self.bbox = (self.r_min, self.c_min, self.r_max, self.c_max)
        self.is_square = (self.r_max - self.r_min + 1) * (self.c_max - self.c_min + 1) == self.size
        # Shape signature (invariant to translation and color)
        norm_cells = tuple(sorted((r - self.r_min, c - self.c_min) for r, c in cells))
        self.shape_sig = hash(norm_cells)
        
def build_graph(grid):
    H, W = grid.shape
    nodes = []
    # Background is usually 0, but sometimes not. We assume 0 is background for objects.
    # Group by color
    for c in np.unique(grid):
        mask = (grid == c)
        lbl, num = label(mask)
        for i in range(1, num + 1):
            cells = tuple(map(tuple, np.argwhere(lbl == i)))
            nodes.append(Node(len(nodes), cells, c, (H, W)))
            
    # Edges (touches, inside)
    edges = {'touches': set(), 'inside': set()}
    for n1 in nodes:
        for n2 in nodes:
            if n1 is n2: continue
            # Touches (bbox overlap roughly, then exact check)
            if not (n1.r_min > n2.r_max + 1 or n1.r_max < n2.r_min - 1 or n1.c_min > n2.c_max + 1 or n1.c_max < n2.c_min - 1):
                # Exact adjacent check
                for r, c in n1.cells:
                    if (r+1, c) in n2.cells or (r-1, c) in n2.cells or (r, c+1) in n2.cells or (r, c-1) in n2.cells:
                        edges['touches'].add((n1.id, n2.id))
                        break
            # Inside (n1 is completely inside n2's bbox)
            if n1.r_min > n2.r_min and n1.r_max < n2.r_max and n1.c_min > n2.c_min and n1.c_max < n2.c_max:
                edges['inside'].add((n1.id, n2.id))
                
    return nodes, edges

# =====================================================================
# 2. MOTOR ABDUTIVO GENETICO (ILP)
# =====================================================================
# Regra: Predicate(Node) -> Action(Node)
# Predicates: Color(c), MaxSize(), MinSize(), Touches(Color), HasShape(s)
# Actions: Recolor(c), Delete(), MoveTo(Node)

class Predicate:
    def __init__(self, type_, val=None):
        self.type = type_
        self.val = val
    def eval(self, node, nodes, edges):
        if self.type == 'True': return True
        if self.type == 'Color': return node.color == self.val
        if self.type == 'MaxSize': 
            valid = [n.size for n in nodes if n.color != 0]
            return len(valid) > 0 and node.size == max(valid)
        if self.type == 'MinSize': 
            valid = [n.size for n in nodes if n.color != 0]
            return len(valid) > 0 and node.size == min(valid)
        if self.type == 'TouchesColor':
            for n1, n2 in edges['touches']:
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
            for r, c in node.cells: grid_out[r, c] = self.val
        if self.type == 'Delete':
            for r, c in node.cells: grid_out[r, c] = 0
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

# =====================================================================
# 3. EVOLUCAO
# =====================================================================
def generate_random_rule(colors):
    ptypes = ['True', 'Color', 'MaxSize', 'MinSize', 'TouchesColor']
    ptype = np.random.choice(ptypes)
    pval = np.random.choice(colors) if ptype in ['Color', 'TouchesColor'] and len(colors)>0 else None
    
    atypes = ['Recolor', 'Delete']
    atype = np.random.choice(atypes)
    aval = np.random.choice(colors) if atype == 'Recolor' and len(colors)>0 else None
    
    return Rule(Predicate(ptype, pval), Action(atype, aval))

def crossover(r1, r2):
    return Rule(r1.pred, r2.act) if np.random.rand() > 0.5 else Rule(r2.pred, r1.act)

def mutate(r, colors):
    if np.random.rand() > 0.5:
        return crossover(r, generate_random_rule(colors))
    else:
        return generate_random_rule(colors)

def exact_match(pred_grid, target_grid):
    if pred_grid.shape != target_grid.shape: return False
    return np.array_equal(pred_grid, target_grid)

def test_time_training(train_tasks, pop_size=200, generations=10):
    # Coletar cores validas
    colors = set()
    for gi, go in train_tasks:
        colors.update(np.unique(gi))
        colors.update(np.unique(go))
    colors = list(colors)
    
    # Extrair grafos
    graphs = []
    for gi, go in train_tasks:
        nodes, edges = build_graph(gi)
        graphs.append((gi, go, nodes, edges))
        
    pop = [generate_random_rule(colors) for _ in range(pop_size)]
    
    for gen in range(generations):
        fitness = []
        for r in pop:
            score = 0
            for gi, go, nodes, edges in graphs:
                pred = r.execute(gi, nodes, edges)
                if exact_match(pred, go):
                    score += 100
                else:
                    # Partial match
                    score -= np.sum(pred != go)
            fitness.append(score)
            
        best_idx = np.argmax(fitness)
        best_rule = pop[best_idx]
        if fitness[best_idx] == 100 * len(train_tasks):
            return best_rule
            
        # Select and breed
        parents = [pop[i] for i in np.argsort(fitness)[-pop_size//2:]]
        next_pop = parents[:]
        while len(next_pop) < pop_size:
            p1 = parents[np.random.randint(len(parents))]
            p2 = parents[np.random.randint(len(parents))]
            next_pop.append(mutate(crossover(p1, p2), colors))
        pop = next_pop
        
    # Return best found
    best_idx = np.argmax(fitness)
    return pop[best_idx] if fitness[best_idx] == 100 * len(train_tasks) else None

# =====================================================================
# 4. AVALIACAO GLOBAL (ARC REAL)
# =====================================================================
def solve():
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(DATA, "*.json")))
    if DEV: files = files[:50]
    
    solved = []
    print(f"Iniciando UAC (Unified Abductive Core) sobre {len(files)} tarefas...")
    
    for idx, fp in enumerate(files):
        name = os.path.basename(fp)
        t = json.load(open(fp))
        train = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        tests = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
        
        # Filtro basico: so tentamos grafos onde as dimensoes não mudam para esta versao v1
        same_shape = all(gi.shape == go.shape for gi, go in train) and all(gi.shape == go.shape for gi, go in tests)
        if not same_shape: continue
            
        rule = test_time_training(train)
        if rule:
            # Test generalization
            ok = True
            for ti, to in tests:
                nodes, edges = build_graph(ti)
                pred = rule.execute(ti, nodes, edges)
                if not exact_match(pred, to):
                    ok = False
                    break
            if ok:
                solved.append(name)
                print(f"[{idx+1}/{len(files)}] {name} RESOLVIDA! Regra abduzida: {rule}")
        
    print("=" * 60)
    print(f"UAC (Test-Time Evolution) Finalizado em {time.time()-t0:.1f}s")
    print(f"Resolvidas EXATAS (fora do fecho): {len(solved)}/{len(files)} ({len(solved)/len(files):.1%})")
    
if __name__ == "__main__":
    solve()
