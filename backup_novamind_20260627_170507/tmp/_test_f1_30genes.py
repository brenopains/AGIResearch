import time
import sys
import os
import json
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))
from core.spelke_genes import solve_task, library_stats
from agi_test_all_5 import load_arc_tasks, evaluate_on_holdout

tasks = load_arc_tasks(limit=25)
print(f"Tasks carregadas: {len(tasks)}")
print(f"Biblioteca: {library_stats(include_extended=True)['total_genes']} genes")
print()

# F1: 30 genes
print("=" * 70)
print("F1: BIBLIOTECA (30 genes, 400 cap)")
print("=" * 70)
solved = []
t0 = time.time()
for i, task in enumerate(tasks):
    ts = time.time()
    try:
        rule_seq = solve_task(task['train'], task['test'][0][0],
                              max_depth=2, beam_width=20, max_candidates=400)
    except Exception as e:
        rule_seq = None
    elapsed = time.time() - ts
    ok = evaluate_on_holdout(rule_seq, task['train'], task['test'][0])
    if ok:
        solved.append(task['name'])
        print(f"  [{i+1:2d}] {task['name']:25s}  SOLVED  {elapsed:5.1f}s")
elapsed_f1 = time.time() - t0
print(f"\nF1: {len(solved)}/{len(tasks)} = {len(solved)/len(tasks):.1%} em {elapsed_f1:.1f}s")
print(f"Tasks resolvidas: {solved}")
f1 = {
    'name': 'F1_Biblioteca',
    'solved': len(solved), 'total': len(tasks),
    'pct': len(solved)/len(tasks), 'time': elapsed_f1,
    'solved_names': solved,
}
