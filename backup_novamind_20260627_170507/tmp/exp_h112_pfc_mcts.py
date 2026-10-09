import os, sys, json, glob, time
import numpy as np
import multiprocessing as mp
from scipy.ndimage import label, binary_fill_holes

DEV = False
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "arc_data", "data", "training")

# =====================================================================
# 1. PERCEPCAO OMNI-CORE (MAPA COGNITIVO)
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
        
        self.is_square = (self.h == self.w and self.h * self.w == self.size)
        self.is_line = (self.h == 1 or self.w == 1) and self.size > 1
        
        self.mask = np.zeros((self.h, self.w), dtype=bool)
        for r, c in self.cells:
            self.mask[r - self.r_min, c - self.c_min] = True
            
        self.is_hollow = np.any(binary_fill_holes(self.mask) & ~self.mask)
        self.is_sym_x = np.array_equal(self.mask, np.flipud(self.mask)) if self.size > 1 else False
        self.is_sym_y = np.array_equal(self.mask, np.fliplr(self.mask)) if self.size > 1 else False

def build_graph(grid):
    H, W = grid.shape
    nodes = []
    for c in np.unique(grid):
        mask = (grid == c)
        lbl, num = label(mask)
        for i in range(1, num + 1):
            cells = tuple(map(tuple, np.argwhere(lbl == i)))
            nodes.append(Node(len(nodes), cells, c, (H, W)))
            
    edges = {'touches': set(), 'inside': set()}
    for n1 in nodes:
        for n2 in nodes:
            if n1 is n2: continue
            if not (n1.r_min > n2.r_max + 1 or n1.r_max < n2.r_min - 1 or n1.c_min > n2.c_max + 1 or n1.c_max < n2.c_min - 1):
                for r, c in n1.cells:
                    if (r+1, c) in n2.cells or (r-1, c) in n2.cells or (r, c+1) in n2.cells or (r, c-1) in n2.cells:
                        edges['touches'].add((n1.id, n2.id))
                        break
            if n1.r_min > n2.r_min and n1.r_max < n2.r_max and n1.c_min > n2.c_min and n1.c_max < n2.c_max:
                edges['inside'].add((n1.id, n2.id))
                
    return nodes, edges

class Predicate:
    def __init__(self, type_, val=None):
        self.type = type_
        self.val = val
        
    def eval(self, node, nodes, edges):
        if self.type == 'True': return True
        if self.type == 'Color': return node.color == self.val
        
        valid_nodes = [n for n in nodes if n.color != 0]
        if self.type == 'MaxSize': return bool(valid_nodes) and node.color != 0 and node.size == max(n.size for n in valid_nodes)
        if self.type == 'MinSize': return bool(valid_nodes) and node.color != 0 and node.size == min(n.size for n in valid_nodes)
        if self.type == 'MostFreqColor':
            freqs = {c: sum(1 for n in nodes if n.color == c) for c in np.unique([n.color for n in nodes])}
            return bool(freqs) and node.color == max(freqs, key=freqs.get)
        if self.type == 'LeastFreqColor':
            freqs = {c: sum(1 for n in nodes if n.color == c) for c in np.unique([n.color for n in nodes])}
            return bool(freqs) and node.color == min(freqs, key=freqs.get)
        
        if self.type == 'IsSquare': return node.is_square
        if self.type == 'IsLine': return node.is_line
        if self.type == 'IsHollow': return node.is_hollow
        if self.type == 'IsSymX': return node.is_sym_x
        if self.type == 'IsSymY': return node.is_sym_y
        
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
        if self.type == 'Recolor':
            for r, c in node.cells: 
                if 0 <= r < H and 0 <= c < W: grid_out[r, c] = self.val
        elif self.type == 'Delete':
            for r, c in node.cells: 
                if 0 <= r < H and 0 <= c < W: grid_out[r, c] = 0
        elif self.type == 'MirrorX':
            for r, c in node.cells: grid_out[r, c] = 0
            for r, c in node.cells:
                new_r = node.r_min + (node.r_max - r)
                if 0 <= new_r < H and 0 <= c < W: grid_out[new_r, c] = node.color
        elif self.type == 'MirrorY':
            for r, c in node.cells: grid_out[r, c] = 0
            for r, c in node.cells:
                new_c = node.c_min + (node.c_max - c)
                if 0 <= r < H and 0 <= new_c < W: grid_out[r, new_c] = node.color
        elif self.type == 'FillHole':
            filled = binary_fill_holes(node.mask)
            for r in range(node.h):
                for c in range(node.w):
                    if filled[r, c]:
                        nr, nc = node.r_min + r, node.c_min + c
                        if 0 <= nr < H and 0 <= nc < W: grid_out[nr, nc] = node.color
        elif self.type == 'Translate':
            dr, dc = self.val
            for r, c in node.cells: grid_out[r, c] = 0
            for r, c in node.cells:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W: grid_out[nr, nc] = node.color
        elif self.type == 'Gravity':
            dr = 1
            for r, c in node.cells: grid_out[r, c] = 0
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

def heuristic_error(pred, target):
    if pred.shape != target.shape: return 999999
    return np.sum(pred != target)

# =====================================================================
# 2. CÓRTEX PRÉ-FRONTAL: BUSCA EM ÁRVORE BEAM SEARCH (MCTS)
# =====================================================================
def get_possible_rules(graphs, colors):
    has_hollow = any(n.is_hollow for gi, go, nodes, edges in graphs for n in nodes)
    has_sym = any(n.is_sym_x or n.is_sym_y for gi, go, nodes, edges in graphs for n in nodes)
    
    ptypes = ['True', 'Color', 'MaxSize', 'MinSize', 'MostFreqColor', 'LeastFreqColor', 'IsSquare', 'IsLine', 'TouchesColor', 'InsideColor']
    if has_hollow: ptypes.append('IsHollow')
    if has_sym: ptypes.extend(['IsSymX', 'IsSymY'])
        
    atypes = ['Recolor', 'Delete', 'Translate', 'Gravity', 'MirrorX', 'MirrorY']
    if has_hollow: atypes.append('FillHole')
    
    rules = []
    for pt in ptypes:
        pvals = [None]
        if pt in ['Color', 'TouchesColor', 'InsideColor']: pvals = colors
        for pv in pvals:
            for at in atypes:
                avals = [None]
                if at == 'Recolor': avals = colors
                elif at == 'Translate': avals = [(1,0), (-1,0), (0,1), (0,-1)]
                for av in avals:
                    rules.append(Rule(Predicate(pt, pv), Action(at, av)))
    return rules

def tree_search_planning(train_tasks, max_depth=2, beam_width=30):
    colors = set()
    for gi, go in train_tasks:
        colors.update(np.unique(gi)); colors.update(np.unique(go))
    colors = list(colors)
    if 0 not in colors: colors.append(0)
    
    graphs = []
    for gi, go in train_tasks:
        graphs.append((gi, go, *build_graph(gi)))
        
    universe_rules = get_possible_rules(graphs, colors)
    
    initial_grids = [gi for gi, go, n, e in graphs]
    initial_error = sum(heuristic_error(gi, go) for gi, go, n, e in graphs)
    
    beam = [([], initial_grids, initial_error)]
    
    for depth in range(max_depth):
        next_beam = []
        for seq, current_grids, err in beam:
            if err == 0:
                return seq
                
            for rule in universe_rules:
                new_grids = []
                new_err = 0
                valid = True
                
                for i in range(len(train_tasks)):
                    g_in = current_grids[i]
                    go = train_tasks[i][1]
                    nodes, edges = build_graph(g_in)
                    
                    try:
                        g_out = rule.execute(g_in, nodes, edges)
                    except Exception:
                        valid = False; break
                        
                    h_err = heuristic_error(g_out, go)
                    new_grids.append(g_out)
                    new_err += h_err
                    
                if not valid: continue
                
                changed_anything = any(not exact_match(current_grids[i], new_grids[i]) for i in range(len(train_tasks)))
                if not changed_anything: continue
                
                if new_err <= err:
                    next_beam.append((seq + [rule], new_grids, new_err))
                    
        if not next_beam: break
        
        next_beam.sort(key=lambda x: x[2])
        beam = next_beam[:beam_width]
        
        if beam and beam[0][2] == 0:
            return beam[0][0]
            
    return None

# =====================================================================
# 3. SOLVER (PARALELIZADO MULTICORE)
# =====================================================================
def process_task(fp):
    name = os.path.basename(fp)
    t = json.load(open(fp))
    train = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
    tests = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
    
    same_shape = all(gi.shape == go.shape for gi, go in train) and all(gi.shape == go.shape for gi, go in tests)
    if not same_shape: return None
        
    rule_seq = tree_search_planning(train, max_depth=2, beam_width=30)
    
    if rule_seq:
        ok = True
        for ti, to in tests:
            pred = ti.copy()
            for rule in rule_seq:
                nodes, edges = build_graph(pred)
                pred = rule.execute(pred, nodes, edges)
            if not exact_match(pred, to):
                ok = False
                break
        if ok:
            seq_str = " -> ".join(str(r) for r in rule_seq)
            return (name, seq_str)
    return None

def solve_all():
    files = sorted(glob.glob(os.path.join(DATA, "*.json")))
    if DEV: files = files[:30]
    
    cpu_cores = mp.cpu_count()
    total_files = len(files)
    print(f"Iniciando PFC MCTS Multi-Passo sobre {total_files} tarefas...", flush=True)
    print(f"Utilizando PARALELISMO TOTAL com {cpu_cores} núcleos de CPU.", flush=True)
    
    t0 = time.time()
    solved = []
    completed = 0
    
    with mp.Pool(processes=cpu_cores) as pool:
        for res in pool.imap_unordered(process_task, files):
            completed += 1
            elapsed = time.time() - t0
            avg_time_per_task = elapsed / completed
            remaining_tasks = total_files - completed
            eta_seconds = remaining_tasks * avg_time_per_task
            
            # Formatar ETA
            horas = int(eta_seconds // 3600)
            minutos = int((eta_seconds % 3600) // 60)
            segundos = int(eta_seconds % 60)
            
            if res is not None:
                name, seq_str = res
                solved.append((name, seq_str))
                print(f"[{completed}/{total_files}] EUREKA! Tarefa: {name} | Sequencia: {seq_str} | ETA: {horas}h {minutos}m {segundos}s", flush=True)
            else:
                print(f"[{completed}/{total_files}] Falha (sem solucao). | ETA: {horas}h {minutos}m {segundos}s", flush=True)
            
    print("=" * 60, flush=True)
    print(f"Finalizado em {time.time()-t0:.1f}s", flush=True)
    print(f"Resolvidas EXATAS (MCTS Paralelo): {len(solved)}/{total_files} ({len(solved)/total_files:.1%})", flush=True)

if __name__ == "__main__":
    solve_all()
