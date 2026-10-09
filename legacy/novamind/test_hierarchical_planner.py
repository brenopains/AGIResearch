"""Teste do HierarchicalPlanner em tasks ARC."""
import sys, numpy as np
sys.path.insert(0, ".")

from core.hierarchical_planner import HierarchicalPlanner, compute_distance_to_solution
from core.goal_driven_engine import GoalDrivenEngine, GoalState, GoalComparator
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph,
    OperatorGroundingBridge,
)
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.arc_dataset import ARCDataset

print("=" * 70)
print("TESTE: HIERARCHICAL PLANNER")
print("=" * 70)

# Setup com todas as 12 primitives
registry = PrimitiveRegistry()
for fn in [create_recolor_primitive, create_remove_primitive,
           create_move_primitive, create_add_primitive]:
    registry.register(fn())

# Adiciona operadores espaciais do dataset
from core.program_induction.operator_proposals import get_all_proposals
from core.program_induction.primitive import Primitive
for op in get_all_proposals():
    fn = op["function"]
    prim = Primitive(name=op["name"], structure_type=op["name"],
                     description=op.get("description", ""))
    def _make_exec(f):
        def _exec(g, p):
            try: return f(g)
            except: return g.copy()
        return _exec
    prim._execute = _make_exec(fn)
    prim.confidence = 0.5
    registry.register(prim)

# Adiciona transform e composite
for name in ["transform", "composite"]:
    prim = Primitive(name=name, structure_type=name,
                     description=f"discovered: {name}")
    registry.register(prim)

print(f"Primitives: {registry.count}")
for p in registry.get_all():
    print(f"  {p.name}: {p.description[:50] if p.description else 'generic'}")

graph = CompositionGraph()
bridge = OperatorGroundingBridge(registry, graph)
pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
cmp = GoalComparator()

# Teste 1: Planejamento simples (recolor)
print("\n[TEST 1] Planner: recolor simple")
planner = HierarchicalPlanner(synthesizer=pie)
inp = np.array([[0,2,0],[2,2,2],[0,2,0]], dtype=int)
target = np.array([[0,8,0],[8,8,8],[0,8,0]], dtype=int)

programs = planner.plan(inp, target)
state = inp.copy()
for p in programs:
    state = p.execute(state)
exact = np.array_equal(state, target)
print(f"  Programs: {len(programs)}")
print(f"  Exact: {exact}")
print(f"  Stats: {planner.get_stats()}")

# Teste 2: ARC com planner
print("\n[TEST 2] ARC with HierarchicalPlanner")
dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)

engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph,
                          use_planner=True)

results = []
for task_idx in range(30):
    name, task = dataset.tasks[task_idx]
    train_pairs = task.get("train", [])
    if not train_pairs:
        continue
    inp = np.array(train_pairs[0]["input"], dtype=int)
    out = np.array(train_pairs[0]["output"], dtype=int)

    engine.set_goal_from_target(out, description=name)
    engine.perceive(inp)
    engine.step()  # Planner tenta uma vez
    final = engine.current_state

    exact = bool(np.array_equal(final, out)) if final.shape == out.shape else False
    acc = 1.0 - int((final != out).sum()) / max(1, int(out.size)) if final.shape == out.shape else 0.0
    dist = compute_distance_to_solution(engine.history, out, final)
    results.append({
        "task": name, "exact": exact, "acc": acc, "dist": dist,
    })

n_exact = sum(1 for r in results if r["exact"])
avg_acc = np.mean([r["acc"] for r in results]) if results else 0
avg_dist = np.mean([r["dist"]["pixel_gap"] for r in results]) if results else 0
avg_coverage = np.mean([r["dist"]["structural_coverage"] for r in results]) if results else 0

print(f"\n[RESULTS]")
print(f"  Exact: {n_exact}/{len(results)}")
print(f"  Avg acc: {avg_acc:.2%}")
print(f"  Avg pixel gap: {avg_dist:.4f}")
print(f"  Avg coverage: {avg_coverage:.2%}")
print(f"  Planner stats: {engine.planner.get_stats() if hasattr(engine, 'planner') else 'N/A'}")
