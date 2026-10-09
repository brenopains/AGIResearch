import os, sys, json, glob, time
import numpy as np
from scipy.ndimage import label

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "arc_data", "data", "training")

# =====================================================================
# 1. PERCEPCAO E REGRAS (Core Knowledge)
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
    return nodes, edges

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

def exact_match(pred_grid, target_grid):
    if pred_grid.shape != target_grid.shape: return False
    return np.array_equal(pred_grid, target_grid)

# =====================================================================
# 2. MOTOR EVOLUTIVO (Slow System 2)
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

def test_time_training(train_tasks, generations=10, pop_size=200):
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
                if exact_match(pred, go): score += 100
                else: score -= np.sum(pred != go)
            fitness.append(score)
            
        best_idx = np.argmax(fitness)
        if fitness[best_idx] == 100 * len(train_tasks):
            return pop[best_idx]
            
        parents = [pop[i] for i in np.argsort(fitness)[-pop_size//2:]]
        next_pop = parents[:]
        while len(next_pop) < pop_size:
            p1 = parents[np.random.randint(len(parents))]
            p2 = parents[np.random.randint(len(parents))]
            next_pop.append(mutate(crossover(p1, p2), colors))
        pop = next_pop
        
    best_idx = np.argmax(fitness)
    return pop[best_idx] if fitness[best_idx] == 100 * len(train_tasks) else None

# =====================================================================
# 3. MAESTRO (Fast System 1 + Aprendizado Estrutural)
# =====================================================================
class ArchetypeLibrary:
    def __init__(self):
        # Priors puros iniciais inuteis (para forcar frustracao)
        self.rules = [
            Rule(Predicate('True'), Action('Delete'))  # Apaga tudo (regra ingenua)
        ]

class Maestro:
    def __init__(self):
        self.library = ArchetypeLibrary()
        self.frustration = 0.0
        
    def solve_task(self, name, train_tasks, test_tasks):
        print(f"\n--- Analisando {name} ---")
        
        # 1. Tentar System 1 (Recuperacao Rapida da Memoria)
        t_start = time.time()
        for rule in self.library.rules:
            # Avalia regra no treino
            success = True
            for gi, go in train_tasks:
                nodes, edges = build_graph(gi)
                pred = rule.execute(gi, nodes, edges)
                if not exact_match(pred, go):
                    success = False
                    break
            
            if success:
                print(f"[SYSTEM 1] SUCESSO! Regra reutilizada do Menu: {rule} ({time.time()-t_start:.3f}s)")
                self.frustration = 0.0
                return True
                
        # 2. Falhou o System 1 -> Frustracao sobe
        self.frustration = 1.0
        print(f"[MAESTRO] Frustracao {self.frustration:.1f}. O fecho de primitivas e insuficiente. Acordando ILP Genetico...")
        
        # 3. System 2 Lento (Descoberta fora-do-fecho)
        t_slow = time.time()
        new_rule = test_time_training(train_tasks, generations=15, pop_size=300)
        
        if new_rule:
            print(f"[SYSTEM 2] EUREKA! O Sistema ILP abstraiu uma regra fora do fecho: {new_rule} ({time.time()-t_slow:.1f}s)")
            # Validar no held-out
            ok = True
            for ti, to in test_tasks:
                nodes, edges = build_graph(ti)
                pred = new_rule.execute(ti, nodes, edges)
                if not exact_match(pred, to): ok = False
            
            if ok:
                # 4. CRESCIMENTO ESTRUTURAL
                self.library.rules.append(new_rule)
                print(f"[MAESTRO] A regra {new_rule} foi compilada num Arquetipo e adicionada a memoria inata!")
                self.frustration = 0.0
                return True
        else:
            print(f"[SYSTEM 2] Falha. O motor ILP nao encontrou regra. Tarefa impossivel para o gramario atual.")
        
        return False

# =====================================================================
# 4. FLUXO DE VIDA DO AGENTE (VALIDACAO)
# =====================================================================
def run_stream():
    # Vamos simular um curriculum vitae do Agente.
    # Ele encontrara a mesma regra 2 vezes, pra provar que System 1 a captura instantaneamente apos aprender.
    
    file_b = os.path.join(DATA, "b1948b0a.json")
    file_c = os.path.join(DATA, "c8f0f002.json")
    
    def load_task(fp):
        t = json.load(open(fp))
        train = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        tests = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
        return train, tests
        
    train_b, test_b = load_task(file_b)
    train_c, test_c = load_task(file_c)
    
    # Curriculum:
    # 1. Encontra tarefa B pela primeira vez
    # 2. Encontra tarefa C pela primeira vez
    # 3. Encontra tarefa B novamente (simulando uma tarefa nova porem com mesma lei subjacente)
    # 4. Encontra tarefa C novamente
    stream = [
        ("Episodio 1: Topologia de Vizinhanca (b1948b0a)", train_b, test_b),
        ("Episodio 2: Substituicao de Cor (c8f0f002)", train_c, test_c),
        ("Episodio 3: REPETICAO - Outro cenario de Vizinhanca", train_b, test_b),
        ("Episodio 4: REPETICAO - Outro cenario de Cor", train_c, test_c)
    ]
    
    agent = Maestro()
    for name, tr, te in stream:
        agent.solve_task(name, tr, te)
        
    print("\n[RESULTADO FINAL] O Menu de Arquetipos Expandido:")
    for i, r in enumerate(agent.library.rules):
        print(f"  {i}: {r}")

if __name__ == "__main__":
    run_stream()
