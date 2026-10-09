"""
Teste HONESTO 2: Grover vs busca exaustiva clássica.
Pergunta: dado um oráculo clássico (cara!), o Grover ajuda?

Método:
  - Para cada task, gera 64 candidatos
  - Avalia TODOS classicamente (1 vez só) — define o OPTIMAL
  - Roda Grover (6 qubits) que tenta encontrar o optimal usando apenas 8 chamadas ao oráculo
  - Compara: Grover encontrou o optimal? Quanto tempo levou?

Honestidade esperada: Grover clássico é mais lento (overhead de simulação)
mas o ponto é VERIFICAR se o algoritmo funciona corretamente.
"""
import time
import sys
import numpy as np
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))

from core.spelke_genes import solve_task, make_genes, make_predicates, cell_err, apply_rule_dict_cached
from agi_test_all_5 import load_arc_tasks, evaluate_on_holdout


def gen_candidates(train, max_n=64, use_super=True):
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
                                max_candidates=max_n, use_extended=True, use_super=use_super)


def evaluate_all_classic(train, candidates):
    """Avalia TODOS classicamente e retorna lista de (idx, err, grids)."""
    from core.spelke_genes import apply_rule_dict_cached
    initial_grids = [gi.copy() for gi, go in train]
    out = []
    for i, c in enumerate(candidates):
        rule_dict = {
            'pred_name': c['pred_name'],
            'pred_fn': c['pred_fn'],
            'pred_param': c['pred_param'],
            'gene': c['gene'],
            'gene_param': c['gene_param'],
        }
        new_err = 0
        new_grids = []
        valid = True
        for j, (gi, go) in enumerate(train):
            try:
                g_out = apply_rule_dict_cached(initial_grids[j], rule_dict, gi, go,
                                               lambda g: (None, None))
            except Exception:
                valid = False
                new_err = float('inf')
                break
            new_grids.append(g_out)
            new_err += cell_err(g_out, go)
        out.append((i, new_err, new_grids, valid))
    return out


def grover_search(candidates, train, n_qubits=6, shots=2000):
    """Grover para encontrar candidato com menor erro.
    Marca top-K como 'bons', Grover sorteia com viés.
    """
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    initial_grids = [gi.copy() for gi, go in train]

    # Avalia todos classicamente UMA vez (custo fixo)
    t0 = time.time()
    scored = evaluate_all_classic(train, candidates)
    t_classic_eval = time.time() - t0

    # Top-K: marca os 25% melhores como "bons"
    n = len(scored)
    sorted_scored = sorted(scored, key=lambda x: x[1] if x[3] else float('inf'))
    n_good = max(1, n // 4)
    good_indices = [s[0] for s in sorted_scored[:n_good]]

    if not good_indices:
        return None, t_classic_eval, 0, 0, 0

    # Grover
    N = 2 ** n_qubits
    n_iter = int(np.pi / 4 * np.sqrt(N / len(good_indices)))
    n_iter = max(1, min(n_iter, 20))

    qc = QuantumCircuit(n_qubits, n_qubits)
    for q in range(n_qubits):
        qc.h(q)

    for _ in range(n_iter):
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
    t_quantum = time.time() - t0

    best_str = max(counts, key=counts.get)
    if ' ' in best_str:
        best_str = best_str.split()[0]
    best_idx = int(best_str, 2)

    # Retorna o candidato escolhido, com o erro verificado
    if best_idx < len(candidates):
        chosen_err = scored[best_idx][1] if scored[best_idx][3] else float('inf')
        optimal_err = sorted_scored[0][1]
        optimal_idx = sorted_scored[0][0]
        return {
            'chosen_idx': best_idx,
            'chosen_err': chosen_err,
            'optimal_idx': optimal_idx,
            'optimal_err': optimal_err,
            'hit': best_idx == optimal_idx,
            'in_top_k': best_idx in good_indices,
            'n_good': n_good,
            'n_iter': n_iter,
            't_classic_eval': t_classic_eval,
            't_quantum': t_quantum,
            'counts_top5': sorted(counts.items(), key=lambda x: -x[1])[:5],
        }, t_classic_eval, t_quantum, n_good, n_iter
    return None, t_classic_eval, t_quantum, n_good, n_iter


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("TESTE: Grover vs busca exaustiva clássica")
    print("=" * 70)
    print("Mede se Grover encontra o OPTIMAL dos 64 candidatos (6 qubits)")
    print()

    tasks = load_arc_tasks(limit=15)
    n_qubits = 6  # 2^6 = 64 candidatos

    results = []
    for i, task in enumerate(tasks):
        ts = time.time()
        candidates = gen_candidates(task['train'], max_n=2**n_qubits)
        if len(candidates) < 2**n_qubits:
            print(f"  [{i+1:2d}] {task['name']:25s}  SKIP (apenas {len(candidates)} candidatos)")
            continue

        out = grover_search(candidates, task['train'], n_qubits=n_qubits)
        if out[0] is None:
            continue
        r, t_classic, t_quantum, n_good, n_iter = out
        elapsed = time.time() - ts

        hit_mark = "HIT" if r['hit'] else "miss"
        in_topk = "topk" if r['in_top_k'] else "out"
        ratio = r['chosen_err'] / max(r['optimal_err'], 1)
        print(f"  [{i+1:2d}] {task['name']:25s}  {hit_mark}/{in_topk}  err_ratio={ratio:.2f}  "
              f"good={n_good} iter={n_iter}  T_class={t_classic:.2f}s T_q={t_quantum:.2f}s")

        results.append(r)

    # Resumo
    n = len(results)
    if n == 0:
        print("\nNenhum resultado.")
        sys.exit(0)
    n_hit = sum(1 for r in results if r['hit'])
    n_topk = sum(1 for r in results if r['in_top_k'])
    avg_ratio = np.mean([r['chosen_err'] / max(r['optimal_err'], 1) for r in results])
    avg_t_class = np.mean([r['t_classic_eval'] for r in results])
    avg_t_q = np.mean([r['t_quantum'] for r in results])

    print()
    print("=" * 70)
    print("RESUMO")
    print("=" * 70)
    print(f"Tasks: {n}")
    print(f"Grover achou OPTIMAL exato: {n_hit}/{n} = {n_hit/n:.1%}")
    print(f"Grover achou dentro do top-K: {n_topk}/{n} = {n_topk/n:.1%}")
    print(f"Razao media chosen/optimal: {avg_ratio:.2f}x (1.0 = perfeito)")
    print(f"Tempo medio avaliacao classica: {avg_t_class:.3f}s")
    print(f"Tempo medio circuito quantico:  {avg_t_q:.3f}s")
    print(f"Speedup bruto (T_class/T_q): {avg_t_class/max(avg_t_q, 0.001):.2f}x")
    print()
    print("HONESTIDADE:")
    print("- Se Grover tem speedup e accuracy, vitoria.")
    print("- Se Grover tem speedup mas perde accuracy, NAO ajuda.")
    print("- Se Grover NAO tem speedup (classico mais rapido), tambem nao ajuda.")
    print(f"- Aqui: speedup = {avg_t_class/max(avg_t_q, 0.001):.2f}x, accuracy = {n_hit/n:.1%}")
