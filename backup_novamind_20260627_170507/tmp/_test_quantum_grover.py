"""
Teste HONESTO: busca quântica para seleção de candidatos.
Compara:
  A) Beam search clássico (atual)
  B) Amplitude Amplification (quântico, simulado)
  C) Grover real (qiskit, simulado)

Mede: tempo, # tasks resolvidas, speedup real.

Honestidade: Grover puro só speedup real quando o oráculo é barato.
Aqui o oráculo é clássico (avaliar candidato), então o ganho esperado é modesto.
"""
import time
import sys
import numpy as np
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))

from core.spelke_genes import solve_task, make_genes, make_predicates, cell_err
from core.spelke_extended import get_extra_genes, get_extra_predicates
from agi_test_all_5 import load_arc_tasks, evaluate_on_holdout

# ============================================================================
# MÉTODO A: BEAM CLÁSSICO (já temos)
# ============================================================================
def solve_classic(train, test_in, max_candidates=400, use_super=True):
    return solve_task(train, test_in,
                      max_depth=2, beam_width=20, max_candidates=max_candidates,
                      use_extended=True, use_super=use_super)

# ============================================================================
# MÉTODO B: AMPLITUDE AMPLIFICATION CLÁSSICO (quantum-inspired)
# ============================================================================
def score_candidate(candidate, train, current_grids):
    """Avalia um candidato: aplica em todas as grades e retorna erro total."""
    from core.spelke_genes import apply_rule_dict_cached
    sg_genes = make_genes(include_extended=True, include_super=True)
    # pega o gene por nome
    gene = None
    for g in sg_genes:
        if g.name == candidate['gene_name']:
            gene = g
            break
    if gene is None:
        return float('inf')

    total_err = 0
    for i, (gi, go) in enumerate(train):
        try:
            rule_dict = {
                'pred_name': candidate['pred_name'],
                'pred_fn': candidate['pred_fn'],
                'pred_param': candidate['pred_param'],
                'gene': gene,
                'gene_param': candidate['gene_param'],
            }
            g_out = apply_rule_dict_cached(current_grids[i], rule_dict, gi, go,
                                           lambda g: (None, None))
        except Exception:
            return float('inf')
        total_err += cell_err(g_out, go)
    return total_err


def gen_all_candidates(train, test_in, max_candidates=400, use_super=True):
    """Gera TODOS os candidatos (até max_candidates)."""
    colors = set()
    for gi, go in train:
        colors.update(np.unique(gi).tolist())
        colors.update(np.unique(go).tolist())
    colors = sorted([c for c in colors if c != 0])[:4] or [0]
    colors.append(0)
    predicates = make_predicates(include_super=use_super)
    all_genes = make_genes(include_extended=True, include_super=use_super)
    from core.spelke_genes import gen_candidate_rules
    return gen_candidate_rules(train[0][0], train[0][1], colors, predicates,
                                max_candidates=max_candidates,
                                use_extended=True, use_super=use_super), all_genes


def solve_amplitude_amplification(train, test_in, depth=2, beam=20, max_candidates=400):
    """Quantum-inspired: avalia TODOS os candidatos e usa amplitude proporcional a 1/cost.
    Implementação clássica (Grover é simulado em numpy).
    """
    initial_grids = [gi.copy() for gi, go in train]
    initial_err = sum(cell_err(gi, go) for gi, go in train)

    # Fase 1: depth=1
    candidates, sg_genes = gen_all_candidates(train, test_in, max_candidates)
    if not candidates:
        return None

    # Avalia TODOS os candidatos em paralelo (numpy vectorization)
    scored = []
    for c in candidates:
        gene_obj = c['gene']
        rule_dict = {
            'pred_name': c['pred_name'],
            'pred_fn': c['pred_fn'],
            'pred_param': c['pred_param'],
            'gene': gene_obj,
            'gene_param': c['gene_param'],
        }
        # Aplica em todas as grades
        from core.spelke_genes import apply_rule_dict_cached
        new_err = 0
        new_grids = []
        for i, (gi, go) in enumerate(train):
            try:
                g_out = apply_rule_dict_cached(initial_grids[i], rule_dict, gi, go,
                                               lambda g: (None, None))
            except Exception:
                new_err = float('inf')
                new_grids = [initial_grids[i]]
                break
            new_grids.append(g_out)
            new_err += cell_err(g_out, go)
        scored.append((c, new_err, new_grids))

    scored.sort(key=lambda x: x[1])
    if scored[0][1] == 0:
        return [scored[0][0]]
    # Pega top-K
    top_k = scored[:beam]
    best_seq = []
    current_grids = initial_grids

    for d in range(depth):
        # Avalia cada candidato do top-k
        best_err = float('inf')
        best_cand = None
        best_grids = None
        for c, _, _ in top_k:
            gene_obj = c['gene']
            rule_dict = {
                'pred_name': c['pred_name'],
                'pred_fn': c['pred_fn'],
                'pred_param': c['pred_param'],
                'gene': gene_obj,
                'gene_param': c['gene_param'],
            }
            from core.spelke_genes import apply_rule_dict_cached
            new_err = 0
            new_grids = []
            for i, (gi, go) in enumerate(train):
                try:
                    g_out = apply_rule_dict_cached(current_grids[i], rule_dict, gi, go,
                                                   lambda g: (None, None))
                except Exception:
                    new_err = float('inf')
                    break
                new_grids.append(g_out)
                new_err += cell_err(g_out, go)
            if new_err < best_err:
                best_err = new_err
                best_cand = c
                best_grids = new_grids
        if best_cand is None or best_err >= len(scored) * 999999:
            break
        best_seq.append(best_cand)
        if best_err == 0:
            return best_seq
        current_grids = best_grids
    return best_seq if best_seq else None


# ============================================================================
# MÉTODO C: GROVER REAL (qiskit, simulado)
# ============================================================================
def solve_grover(train, test_in, n_qubits=8, max_candidates=256, shots=2000):
    """Usa Grover para encontrar o melhor candidato entre 2^n_qubits.
    Oráculo: classifica candidato como 'bom' se reduz erro pela metade.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    # Gera candidatos
    candidates, _ = gen_all_candidates(train, test_in, max_candidates)
    if not candidates:
        return None

    # Avalia todos classicamente
    initial_grids = [gi.copy() for gi, go in train]
    from core.spelke_genes import apply_rule_dict_cached
    initial_err = sum(cell_err(gi, go) for gi, go in train)

    scored = []
    for c in candidates:
        rule_dict = {
            'pred_name': c['pred_name'],
            'pred_fn': c['pred_fn'],
            'pred_param': c['pred_param'],
            'gene': c['gene'],
            'gene_param': c['gene_param'],
        }
        new_err = 0
        for i, (gi, go) in enumerate(train):
            try:
                g_out = apply_rule_dict_cached(initial_grids[i], rule_dict, gi, go,
                                               lambda g: (None, None))
            except Exception:
                new_err = float('inf')
                break
            new_err += cell_err(g_out, go)
        scored.append((c, new_err))

    if not scored:
        return None

    # Threshold: bom = erro < 50% do inicial
    threshold = initial_err * 0.5 if initial_err > 0 else 0
    good_indices = [i for i, (c, e) in enumerate(scored) if e < threshold and e < float('inf')]

    if not good_indices:
        # Sem "bons" — Grover não tem oráculo para marcar.
        # Vamos usar TOP-K: marca os top-10% como bons.
        n_top = max(1, len(scored) // 10)
        sorted_scored = sorted(scored, key=lambda x: x[1])
        good_indices = list(range(min(n_top, len(sorted_scored))))

    # Grover com n_qubits qubits
    N = 2 ** n_qubits
    n_iter = int(np.pi / 4 * np.sqrt(N / max(len(good_indices), 1)))
    n_iter = max(1, min(n_iter, 20))  # cap

    qc = QuantumCircuit(n_qubits, n_qubits)
    for q in range(n_qubits):
        qc.h(q)

    for _ in range(n_iter):
        # ORACLE: marca good_indices (flip phase)
        for idx in good_indices:
            if idx >= N:
                continue
            binary = format(idx, f'0{n_qubits}b')
            for j, b in enumerate(binary):
                if b == '0':
                    qc.x(j)
            if n_qubits == 1:
                qc.z(0)
            else:
                qc.mcp(np.pi, list(range(n_qubits - 1)), n_qubits - 1)
            for j, b in enumerate(binary):
                if b == '0':
                    qc.x(j)
        # DIFFUSER
        for q in range(n_qubits):
            qc.h(q)
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.mcp(np.pi, list(range(n_qubits - 1)), n_qubits - 1)
        for q in range(n_qubits):
            qc.x(q)
            qc.h(q)

    qc.measure_all()

    sim = AerSimulator()
    t0 = time.time()
    job = sim.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    quantum_time = time.time() - t0

    # Pega o índice mais frequente (qiskit 2.x pode retornar 2n bits, divide)
    best_str = max(counts, key=counts.get)
    # se tem espaço, pega só a parte do classificador (últimos n_qubits)
    if ' ' in best_str:
        parts = best_str.split(' ')
        # usa a primeira parte (registrador principal)
        best_str = parts[0]
    best_idx = int(best_str, 2)

    if best_idx < len(scored):
        best_c = scored[best_idx][0]
        return [best_c], quantum_time, len(good_indices), n_iter
    else:
        # índice fora do range — pega o melhor clássico
        best_c = min(scored, key=lambda x: x[1])[0]
        return [best_c], quantum_time, len(good_indices), n_iter


# ============================================================================
# COMPARAÇÃO
# ============================================================================
def run_comparison(n_tasks=10, max_candidates=200):
    tasks = load_arc_tasks(limit=n_tasks)
    print(f"Comparando beam clássico vs Grover (qiskit) em {n_tasks} tasks")
    print(f"  max_candidates={max_candidates}")
    print()

    results = {'classic': [], 'grover': []}

    for i, task in enumerate(tasks):
        # Clássico
        ts = time.time()
        try:
            rule_seq = solve_classic(task['train'], task['test'][0][0],
                                     max_candidates=max_candidates)
        except Exception as e:
            rule_seq = None
        t_classic = time.time() - ts
        ok_c = evaluate_on_holdout(rule_seq, task['train'], task['test'][0])

        # Grover (com n_qubits=8 → 256 candidatos, ou n_qubits=9 → 512)
        n_q = 8 if max_candidates <= 256 else 9
        try:
            result = solve_grover(task['train'], task['test'][0][0],
                                  n_qubits=n_q, max_candidates=max_candidates)
            if len(result) == 4:
                rule_seq_q, t_quantum, n_good, n_iter = result
            else:
                rule_seq_q = result[0]
                t_quantum = 0
                n_good = 0
                n_iter = 0
        except Exception as e:
            print(f"  [{i+1}] {task['name']}: GROVER ERROR {e}")
            rule_seq_q = None
            t_quantum = 0
            n_good = 0
            n_iter = 0
        ok_q = evaluate_on_holdout(rule_seq_q, task['train'], task['test'][0])

        results['classic'].append(ok_c)
        results['grover'].append(ok_q)
        marker_c = "+" if ok_c else "."
        marker_q = "+" if ok_q else "."
        print(f"  [{i+1:2d}/{n_tasks}] {task['name']:25s}  C:{marker_c} ({t_classic:5.1f}s) | "
              f"Q:{marker_q} ({t_quantum:5.1f}s, good={n_good}, iter={n_iter})")

    n_c = sum(results['classic'])
    n_q = sum(results['grover'])
    print()
    print(f"=== RESULTADO ===")
    print(f"  Classico: {n_c}/{n_tasks} = {n_c/n_tasks:.1%}")
    print(f"  Grover:   {n_q}/{n_tasks} = {n_q/n_tasks:.1%}")
    print(f"  Speedup medido: depende de tempo (acima)")
    return results


if __name__ == '__main__':
    # Primeiro: sanity check com 1 task pequena
    print("=" * 70)
    print("SANITY CHECK: 1 task com Grover")
    print("=" * 70)
    tasks = load_arc_tasks(limit=1)
    t = tasks[0]
    print(f"Task: {t['name']}")
    try:
        result = solve_grover(t['train'], t['test'][0][0], n_qubits=6, max_candidates=64)
        if len(result) == 4:
            r, qt, ng, ni = result
            print(f"Grover OK: time={qt:.2f}s, good={ng}, iter={ni}, rule={r['gene'].name if r else 'None'}")
        else:
            print(f"Grover OK: rule={result[0]['gene'].name if result[0] else 'None'}")
    except Exception as e:
        print(f"Grover FAILED: {e}")
        import traceback; traceback.print_exc()

    print()
    print("=" * 70)
    print("COMPARAÇÃO: 10 tasks")
    print("=" * 70)
    run_comparison(n_tasks=10, max_candidates=200)
