"""Teste da sintese por constraint satisfaction."""
import sys, numpy as np
sys.path.insert(0, ".")

from core.program_induction import PrimitiveDiscovery, ProgramSynthesizer, PrimitiveRegistry
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive, create_move_primitive
)
from core.geometry_dataset import GeometryTransformDataset
from core.arc_dataset import ARCDataset

print("=" * 60)
print("TESTE: CONSTRAINT-BASED PARAMETER INFERENCE")
print("=" * 60)

# --- 1. Discovery ---
print("\n[1] Discovery a partir do Geometry Dataset")
registry = PrimitiveRegistry()
discovery = PrimitiveDiscovery(registry=registry)
geo = GeometryTransformDataset(size=500, h=10, w=10)

for idx in range(len(geo)):
    inp, out, label, _ = geo[idx]
    discovery.observe(inp, out)

print(f"  Primitives: {registry.count}")
print(f"  Stats: {registry.get_stats()}")

# --- 2. Constraint Solvers ---
print("\n[2] Teste dos Solvers (inferencia por constraint)")

recolor = create_recolor_primitive()
remove = create_remove_primitive()
move = create_move_primitive()

# Recolor 2 -> 8
inp = np.array([[0,2,0],[2,2,2],[0,2,0]], dtype=int)
out = np.array([[0,8,0],[8,8,8],[0,8,0]], dtype=int)
params = recolor.infer_params(inp, out)
print(f"  recolor 2->8: params={params}")
if params:
    result = recolor.apply(inp, params)
    ok = "OK" if np.array_equal(result, out) else "FAIL"
    print(f"    {ok}: match={np.array_equal(result, out)}")

# Remove color 1
inp = np.array([[1,0,1],[0,2,0],[1,0,1]], dtype=int)
out = np.array([[0,0,0],[0,2,0],[0,0,0]], dtype=int)
params = remove.infer_params(inp, out)
print(f"  remove color 1: params={params}")
if params:
    result = remove.apply(inp, params)
    ok = "OK" if np.array_equal(result, out) else "FAIL"
    print(f"    {ok}: match={np.array_equal(result, out)}")

# Move down
inp = np.array([[0,0,0],[0,2,0],[0,0,0]], dtype=int)
out = np.array([[0,0,0],[0,0,0],[0,2,0]], dtype=int)
params = move.infer_params(inp, out)
print(f"  move down: params={params}")
if params:
    result = move.apply(inp, params)
    ok = "OK" if np.array_equal(result, out) else "FAIL"
    print(f"    Result:\n{result}")
    print(f"    {ok}: match={np.array_equal(result, out)}")

# Move none (should return None)
inp = np.array([[0,0,0],[0,2,0],[0,0,0]], dtype=int)
out = inp.copy()
params = move.infer_params(inp, out)
print(f"  move none: params={params}")  # Should be None

# --- 3. Synthesis on ARC tasks ---
print("\n[3] Synthesis em tasks ARC (com constraint solving)")
dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)
s = ProgramSynthesizer(registry=registry)

results = []
for task_idx in range(20):
    name, task = dataset.tasks[task_idx]
    train_pairs = task["train"]
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
            "task": name,
            "program": best.name,
            "score": best.score,
            "validated": best.validated,
            "exact": exact,
            "n_steps": best.n_steps,
        })

        if task_idx < 5 or best.validated:
            print(f"  {name}: {best.name} score={best.score:.3f} "
                  f"validated={best.validated} exact={exact} steps={best.n_steps}")
            for p in best.primitives:
                print(f"    -> {p.name}")

# --- 4. Summary ---
print("\n[4] Summary")
n_exact = sum(1 for r in results if r["exact"])
n_validated = sum(1 for r in results if r["validated"])
n_composed = sum(1 for r in results if r["n_steps"] > 1)
avg_score = np.mean([r["score"] for r in results]) if results else 0

print(f"  Tasks: {len(results)}")
print(f"  Exact match: {n_exact}")
print(f"  Validated: {n_validated}")
print(f"  Composicoes: {n_composed}")
print(f"  Score medio: {avg_score:.3f}")

print("\n  Programas encontrados:")
for r in results[:10]:
    print(f"    {r['task']}: {r['program']} score={r['score']:.3f} exact={r['exact']}")
