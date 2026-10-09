"""Test focado nas 5 tasks H93 Zealot +5 com depth=4 + beam=30."""
import time
import sys
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))
from core.spelke_genes import solve_task
from agi_test_all_5 import load_arc_tasks, evaluate_on_holdout

# Carrega todas as 400 tasks
all_tasks = load_arc_tasks(limit=400)
target_names = ['00d62c1b', '0dfd9992', '29ec7d0e', '484b58aa', 'c3f564a4']
target_tasks = [t for t in all_tasks if any(n in t['name'] for n in target_names)]
print(f"Tasks alvo encontradas: {[t['name'] for t in target_tasks]}")
print(f"Total: {len(target_tasks)}")
print()

solved = []
t0 = time.time()
for task in target_tasks:
    ts = time.time()
    print(f"Rodando {task['name']} com depth=4 beam=30 max_cand=500...", end=' ', flush=True)
    try:
        rule_seq = solve_task(task['train'], task['test'][0][0],
                              max_depth=4, beam_width=30, max_candidates=500,
                              use_extended=True, use_super=True)
    except Exception as e:
        print(f"ERROR {e}")
        continue
    elapsed = time.time() - ts
    ok = evaluate_on_holdout(rule_seq, task['train'], task['test'][0])
    if ok:
        solved.append(task['name'])
        print(f"SOLVED em {elapsed:.1f}s!")
    else:
        print(f"falhou em {elapsed:.1f}s")
    if elapsed > 300:  # 5 min cap
        print(f"  TIMEOUT")
        break

elapsed_total = time.time() - t0
print()
print(f"=== H93 Zealot +5: {len(solved)}/5 = {len(solved)/5:.1%} em {elapsed_total:.1f}s ===")
print(f"Tasks resolvidas: {solved}")
