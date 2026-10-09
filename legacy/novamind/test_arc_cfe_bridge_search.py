"""
ARC-AGI com CFEBridgeSearch — Soft guidance + fallback integrado no GoalDrivenEngine.
Compara: sem CFE, CFE hard filter (velho), CFE soft guidance (novo).
"""
import sys, numpy as np, time
sys.path.insert(0, ".")

from core.goal_driven_engine import GoalDrivenEngine, GoalComparator
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph, OperatorGroundingBridge,
)
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.program_induction.operator_proposals import get_all_proposals
from core.program_induction.primitive import Primitive
from core.arc_dataset import ARCDataset
from core.conceptual_discovery.cfe_bridge import CFEBridge
from core.conceptual_discovery import InvariantDiscovery, ConceptFormation
from core.cfe_bridge_search import CFEBridgeSearch


def setup_registry():
    reg = PrimitiveRegistry()
    for fn in [create_recolor_primitive, create_remove_primitive,
               create_move_primitive, create_add_primitive]:
        reg.register(fn())
    for op in get_all_proposals():
        fn = op["function"]
        prim = Primitive(name=op["name"], structure_type=op["name"],
                         description=op.get("description", ""))
        def _me(f):
            def _e(g, p):
                try: return f(g)
                except: return g.copy()
            return _e
        prim._execute = _me(fn)
        reg.register(prim)
    return reg


def make_engine(registry, cfe_bridge=None, cfe_bridge_search=None):
    graph = CompositionGraph()
    bridge = OperatorGroundingBridge(registry, graph)
    pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
    return GoalDrivenEngine(
        program_induction_engine=pie,
        composition_graph=graph,
        cfe_bridge=cfe_bridge,
        cfe_bridge_search=cfe_bridge_search,
        use_planner=False,
    )


print("=" * 70)
print("ARC-AGI com CFEBridgeSearch — Soft Guidance + Fallback")
print("=" * 70)

dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)
registry = setup_registry()

discovery = InvariantDiscovery()
formation = ConceptFormation()
bridge = CFEBridge(discovery=discovery, formation=formation)

# Engines
engine_base = make_engine(registry)
engine_hard = make_engine(registry, cfe_bridge=bridge)

# Soft guidance: cria CFEBridgeSearch e passa para o engine
soft_graph = CompositionGraph()
soft_bridge = OperatorGroundingBridge(registry, soft_graph)
soft_pie = ProgramSynthesizer(registry, composition_graph=soft_graph, grounding_bridge=soft_bridge)
soft = CFEBridgeSearch(bridge, soft_pie)
engine_soft = GoalDrivenEngine(
    program_induction_engine=soft_pie,
    composition_graph=soft_graph,
    cfe_bridge_search=soft,
    use_planner=False,
)

total = 400
results = {
    "baseline": {"exact": 0, "accs": []},
    "hard_filter": {"exact": 0, "accs": []},
    "soft_guidance": {"exact": 0, "accs": []},
}

start = time.time()
for idx in range(total):
    name, task = dataset.tasks[idx]
    train = task.get("train", [])
    if not train:
        continue
    inp = np.array(train[0]["input"], dtype=int)
    out = np.array(train[0]["output"], dtype=int)

    # --- BASELINE (sem CFE) ---
    engine_base.set_goal_from_target(out)
    engine_base.perceive(inp)
    engine_base.run(max_steps=3)
    final = engine_base.current_state
    if final.shape == out.shape:
        results["baseline"]["exact"] += 1 if np.array_equal(final, out) else 0
        results["baseline"]["accs"].append(
            1.0 - int((final != out).sum()) / max(1, int(out.size))
        )

    # --- HARD FILTER (CFE velho) ---
    engine_hard.set_goal_from_target(out)
    engine_hard.perceive(inp)
    engine_hard.run(max_steps=3)
    final = engine_hard.current_state
    if final.shape == out.shape:
        results["hard_filter"]["exact"] += 1 if np.array_equal(final, out) else 0
        results["hard_filter"]["accs"].append(
            1.0 - int((final != out).sum()) / max(1, int(out.size))
        )

    # --- SOFT GUIDANCE (CFEBridgeSearch novo) ---
    engine_soft.set_goal_from_target(out)
    engine_soft.perceive(inp)
    engine_soft.run(max_steps=3)
    final = engine_soft.current_state
    if final.shape == out.shape:
        results["soft_guidance"]["exact"] += 1 if np.array_equal(final, out) else 0
        results["soft_guidance"]["accs"].append(
            1.0 - int((final != out).sum()) / max(1, int(out.size))
        )

    if (idx + 1) % 20 == 0:
        elapsed = time.time() - start
        e_b = results["baseline"]["exact"]
        e_h = results["hard_filter"]["exact"]
        e_s = results["soft_guidance"]["exact"]
        print(f"  [{idx+1}/{total}] {elapsed:.0f}s | Base:{e_b} Hard:{e_h} Soft:{e_s}")

elapsed = time.time() - start

# Resultados
print(f"\n{'='*70}")
print(f"RESULTADOS ({total} tasks, {elapsed:.0f}s)")
print(f"{'='*70}")
print(f"\n{'Metrica':<30s} {'Baseline':>10s} {'Hard':>10s} {'Soft':>10s}")
print(f"{'-'*65}")
b_exact = results["baseline"]["exact"]
h_exact = results["hard_filter"]["exact"]
s_exact = results["soft_guidance"]["exact"]
b_pa = (np.mean(results["baseline"]["accs"]) * 100) if results["baseline"]["accs"] else 0
h_pa = (np.mean(results["hard_filter"]["accs"]) * 100) if results["hard_filter"]["accs"] else 0
s_pa = (np.mean(results["soft_guidance"]["accs"]) * 100) if results["soft_guidance"]["accs"] else 0
print(f"  {'Exact Match':<30s} {b_exact:>6d}       {h_exact:>6d}       {s_exact:>6d}")
print(f"  {'Pixel Acc %':<30s} {b_pa:>6.2f}%     {h_pa:>6.2f}%     {s_pa:>6.2f}%")

# CFEBridgeSearch stats
soft_stats = soft.get_stats()
print(f"\nCFEBridgeSearch Stats:")
for k, v in soft_stats.items():
    print(f"  {k}: {v}")

# Invariantes
invariants = discovery.get_invariants(min_confidence=0.1)
print(f"\nInvariantes descobertos: {len(invariants)}")
for inv in invariants:
    print(f"  {inv.structure_type}: conf={inv.confidence:.2f}, obs={inv.n_observations}")

# Gravacao
out_path = r"D:\novamind_data\arc_cfe_bridge_search_results.txt"
with open(out_path, "w") as f:
    f.write(f"ARC CFEBridgeSearch Evaluation ({total} tasks, {elapsed:.0f}s)\n")
    f.write(f"Baseline Exact: {b_exact}, Pixel Acc: {b_pa:.2f}%\n")
    f.write(f"Hard Filter Exact: {h_exact}, Pixel Acc: {h_pa:.2f}%\n")
    f.write(f"Soft Guidance Exact: {s_exact}, Pixel Acc: {s_pa:.2f}%\n")
    f.write(f"Soft Stats: {soft_stats}\n")
print(f"\nResultados salvos em: {out_path}")
