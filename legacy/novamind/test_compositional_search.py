"""Teste do ProgramGraphSearch com beam search composicional."""
import sys, numpy as np
sys.path.insert(0, ".")

from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, ProgramGraphSearch
)
from core.arc_dataset import ARCDataset

print("=" * 70)
print("TESTE: COMPOSITIONAL PROGRAM GRAPH SEARCH")
print("=" * 70)

# Setup primitives
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)

registry = PrimitiveRegistry()
for factory in [create_recolor_primitive, create_remove_primitive,
                create_move_primitive, create_add_primitive]:
    registry.register(factory())

print(f"Primitives: {registry.count}")
for p in registry.get_all():
    print(f"  {p.name}: {p.description}")

# Test 1: Single recolor
print("\n[TEST 1] recolor 2->8")
s = ProgramSynthesizer(registry)
inp = np.array([[0,2,0],[2,2,2],[0,2,0]], dtype=int)
out = np.array([[0,8,0],[8,8,8],[0,8,0]], dtype=int)
progs = s.synthesize(inp, out)
if progs:
    best = progs[0]
    final = best.execute(inp)
    print(f"  {best.name}: score={best.score:.3f} valid={best.validated} exact={np.array_equal(final, out)}")

# Test 2: Two-step (move + recolor)
print("\n[TEST 2] move down + recolor 2->8")
inp2 = np.array([
    [0,0,0,0],
    [0,2,2,0],
    [0,2,2,0],
    [0,0,0,0],
], dtype=int)
out2 = np.array([
    [0,0,0,0],
    [0,0,0,0],
    [0,8,8,0],
    [0,8,8,0],
], dtype=int)
progs2 = s.synthesize(inp2, out2)
print(f"  Programs: {len(progs2)}")
for p in progs2[:5]:
    final = p.execute(inp2)
    exact = np.array_equal(final, out2)
    print(f"    {p.name}: score={p.score:.3f} steps={p.n_steps} exact={exact}")

# Test 3: ARC tasks
print("\n[TEST 3] ARC tasks (30)")
dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)
results = []
for task_idx in range(30):
    name, task = dataset.tasks[task_idx]
    train_pairs = task.get("train", [])
    if not train_pairs:
        continue
    inp = np.array(train_pairs[0]["input"], dtype=int)
    out = np.array(train_pairs[0]["output"], dtype=int)
    progs = s.synthesize(inp, out)
    if progs:
        best = progs[0]
        final = best.execute(inp)
        exact = np.array_equal(final, out) if final.shape == out.shape else False
        results.append({
            "task": name, "program": best.name,
            "score": best.score, "n_steps": best.n_steps, "exact": exact,
        })
        if best.n_steps > 1 or exact or task_idx < 5:
            print(f"  {name}: {best.name} score={best.score:.3f} steps={best.n_steps} exact={exact}")

# Summary
print(f"\n[SUMMARY]")
n_exact = sum(1 for r in results if r["exact"])
n_composed = sum(1 for r in results if r["n_steps"] > 1)
avg_score = np.mean([r["score"] for r in results]) if results else 0
print(f"  Tasks: {len(results)}, Exact: {n_exact}, Composed: {n_composed}, Avg score: {avg_score:.3f}")
print(f"  Programs:")
seen = set()
for r in results:
    if r["program"] not in seen:
        seen.add(r["program"])
        print(f"    {r['program']} (steps={r['n_steps']})")
