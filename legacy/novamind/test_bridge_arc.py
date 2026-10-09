"""Test bridge on ARC tasks."""
import sys, numpy as np
sys.path.insert(0, ".")

from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph,
    OperatorGroundingBridge,
)
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.arc_dataset import ARCDataset

registry = PrimitiveRegistry()
for fn in [create_recolor_primitive, create_remove_primitive,
           create_move_primitive, create_add_primitive]:
    registry.register(fn())

graph = CompositionGraph()
bridge = OperatorGroundingBridge(registry, graph)
s = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)

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
        uses_spatial = any(
            "flip" in p.name or "rotate" in p.name or "transpose" in p.name
            for p in best.instances
        )
        results.append({
            "task": name, "program": best.name,
            "score": best.score, "n_steps": best.n_steps,
            "exact": exact, "spatial": uses_spatial,
        })

n_exact = sum(1 for r in results if r["exact"])
n_spatial = sum(1 for r in results if r["spatial"])
n_composed = sum(1 for r in results if r["n_steps"] > 1)
avg_score = np.mean([r["score"] for r in results]) if results else 0
print(f"Tasks: {len(results)}, Exact: {n_exact}, Spatial: {n_spatial}, Composed: {n_composed}, Score: {avg_score:.3f}")
for r in results:
    tag = " SPATIAL" if r["spatial"] else ""
    tag2 = " EXACT" if r["exact"] else ""
    print(f"  {r['task']}: {r['program']} score={r['score']:.3f} steps={r['n_steps']}{tag}{tag2}")

print(f"\nRegistry now has {registry.count} primitives:")
for p in registry.get_all():
    print(f"  {p.name}: {p.description[:50]}")
print(f"\nCompositionGraph: {graph.get_stats()}")
