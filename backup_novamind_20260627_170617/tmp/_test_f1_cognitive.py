"""F1 com biblioteca COGNITIVA integrada (OBJETO, TIPO_DE, TEM_PROP)."""
import time
import sys
import json
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))
from core.spelke_genes import solve_task
from agi_test_all_5 import load_arc_tasks, evaluate_on_holdout

# Modifica make_predicates para incluir cognitive
import core.spelke_genes as sg
orig_make_pred = sg.make_predicates
def make_pred_with_cog(*a, **kw):
    kw['include_cognitive'] = True
    return orig_make_pred(*a, **kw)
sg.make_predicates = make_pred_with_cog

# Patch solve_task também (ele importa make_predicates dentro)
def solve_with_cog(train, test_in, **kw):
    kw['use_super'] = True
    return sg.solve_task(train, test_in, **kw)

tasks = load_arc_tasks(limit=25)
print(f"Tasks: {len(tasks)}")
print(f"Predicados: {len(sg.make_predicates(include_super=True, include_cognitive=True))} (com cognitive)")
print()

solved = []
t0 = time.time()
for i, task in enumerate(tasks):
    ts = time.time()
    try:
        rule_seq = sg.solve_task(task['train'], task['test'][0][0],
                                  max_depth=2, beam_width=20, max_candidates=300,
                                  use_extended=True, use_super=True)
    except Exception as e:
        rule_seq = None
    elapsed = time.time() - ts
    ok = evaluate_on_holdout(rule_seq, task['train'], task['test'][0])
    if ok:
        solved.append(task['name'])
        print(f"  [{i+1:2d}] {task['name']:25s}  SOLVED  {elapsed:5.1f}s")
elapsed_f1 = time.time() - t0
print(f"\nF1 COGNITIVE: {len(solved)}/{len(tasks)} = {len(solved)/len(tasks):.1%} em {elapsed_f1:.1f}s")
print(f"Tasks: {solved}")
