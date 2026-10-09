"""Test F1 com biblioteca SUPER (44 genes, 28 predicados)."""
import time
import sys
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))
from core.spelke_genes import solve_task, library_stats, make_genes, make_predicates
from agi_test_all_5 import load_arc_tasks, evaluate_on_holdout

# stats
print("Biblioteca SUPER:")
g = make_genes(include_extended=True, include_super=True)
p = make_predicates(include_super=True)
print(f"  {len(g)} genes, {len(p)} predicados")

tasks = load_arc_tasks(limit=50)
print(f"\nTasks: {len(tasks)}\n")

# Fase 1: SUPER + depth=2 (baseline expandido)
print("=" * 70)
print("FASE 1: SUPER + depth=2")
print("=" * 70)
solved1 = []
t0 = time.time()
for i, task in enumerate(tasks):
    ts = time.time()
    try:
        rule_seq = solve_task(task['train'], task['test'][0][0],
                              max_depth=2, beam_width=20, max_candidates=300,
                              use_extended=True, use_super=True)
    except Exception as e:
        rule_seq = None
    elapsed = time.time() - ts
    ok = evaluate_on_holdout(rule_seq, task['train'], task['test'][0])
    if ok:
        solved1.append(task['name'])
        print(f"  [{i+1:2d}/{len(tasks)}] {task['name']:25s}  SOLVED  {elapsed:5.1f}s")
elapsed_1 = time.time() - t0
print(f"\nF1 SUPER depth=2: {len(solved1)}/{len(tasks)} = {len(solved1)/len(tasks):.1%} em {elapsed_1:.1f}s")
print(f"Tasks: {solved1}")

# Fase 2: SUPER + depth=3 (no mesmo subset, mas só para tasks não resolvidas)
print()
print("=" * 70)
print("FASE 2: SUPER + depth=3 (apenas tasks não resolvidas)")
print("=" * 70)
unsolved = [t for t in tasks if t['name'] not in solved1]
print(f"  Restantes: {len(unsolved)}")
solved2 = []
t0 = time.time()
for i, task in enumerate(unsolved):
    ts = time.time()
    try:
        rule_seq = solve_task(task['train'], task['test'][0][0],
                              max_depth=3, beam_width=15, max_candidates=200,
                              use_extended=True, use_super=True)
    except Exception as e:
        rule_seq = None
    elapsed = time.time() - ts
    ok = evaluate_on_holdout(rule_seq, task['train'], task['test'][0])
    if ok:
        solved2.append(task['name'])
        print(f"  [{i+1:2d}/{len(unsolved)}] {task['name']:25s}  SOLVED  {elapsed:5.1f}s")
    if elapsed > 60:
        print(f"  [{i+1:2d}/{len(unsolved)}] {task['name']:25s}  TIMEOUT")
        break
elapsed_2 = time.time() - t0
print(f"\nF1 SUPER depth=3: +{len(solved2)} tasks em {elapsed_2:.1f}s")
print(f"Tasks extras: {solved2}")
print(f"TOTAL: {len(solved1) + len(solved2)}/{len(tasks)} = {(len(solved1)+len(solved2))/len(tasks):.1%}")
print(f"  Fase1={len(solved1)} Fase2={len(solved2)} Tempo total: {elapsed_1+elapsed_2:.1f}s")
