"""ARC-AGI com CFE Bridge guiando o ProgramSynthesizer."""
import sys, numpy as np, time
sys.path.insert(0, ".")

from core.goal_driven_engine import GoalDrivenEngine, GoalState, GoalComparator
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

print("=" * 70)
print("ARC-AGI com CFE — Busca guiada por conceitos")
print("=" * 70)

# Setup registry
registry = PrimitiveRegistry()
for fn in [create_recolor_primitive, create_remove_primitive,
           create_move_primitive, create_add_primitive]:
    registry.register(fn())
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
    registry.register(prim)

dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)

# Setup CFE bridge
discovery = InvariantDiscovery()
formation = ConceptFormation()
bridge = CFEBridge(discovery=discovery, formation=formation)

# Setup engine COM CFE
graph1 = CompositionGraph()
bridge1 = OperatorGroundingBridge(registry, graph1)
pie1 = ProgramSynthesizer(registry, composition_graph=graph1, grounding_bridge=bridge1)
engine_cfe = GoalDrivenEngine(program_induction_engine=pie1, composition_graph=graph1,
                                cfe_bridge=bridge, use_planner=False)
cmp = GoalComparator()

# Setup engine SEM CFE (baseline)
graph2 = CompositionGraph()
bridge2 = OperatorGroundingBridge(registry, graph2)
pie2 = ProgramSynthesizer(registry, composition_graph=graph2, grounding_bridge=bridge2)
engine_base = GoalDrivenEngine(program_induction_engine=pie2, composition_graph=graph2,
                                 use_planner=False)

# Test
total = 100
results_cfe = {"exact": 0, "accs": [], "concept_hits": 0}
results_base = {"exact": 0, "accs": []}

for idx in range(total):
    name, task = dataset.tasks[idx]
    train = task.get("train", [])
    if not train:
        continue
    inp = np.array(train[0]["input"], dtype=int)
    out = np.array(train[0]["output"], dtype=int)

    # CFE engine
    engine_cfe.set_goal_from_target(out)
    engine_cfe.perceive(inp)
    engine_cfe.run(max_steps=3)
    final_cfe = engine_cfe.current_state
    if final_cfe.shape == out.shape:
        results_cfe["exact"] += 1 if np.array_equal(final_cfe, out) else 0
        results_cfe["accs"].append(1.0 - int((final_cfe != out).sum()) / max(1, int(out.size)))

    # Baseline engine
    engine_base.set_goal_from_target(out)
    engine_base.perceive(inp)
    engine_base.run(max_steps=3)
    final_base = engine_base.current_state
    if final_base.shape == out.shape:
        results_base["exact"] += 1 if np.array_equal(final_base, out) else 0
        results_base["accs"].append(1.0 - int((final_base != out).sum()) / max(1, int(out.size)))

    # Count concept hits
    if bridge.get_concept_hint(inp, out) is not None:
        results_cfe["concept_hits"] += 1

    if (idx + 1) % 20 == 0:
        print(f"  [{idx+1}/{total}] CFE: {results_cfe['exact']}, Base: {results_base['exact']}")

# Report
avg_cfe = np.mean(results_cfe["accs"]) if results_cfe["accs"] else 0
avg_base = np.mean(results_base["accs"]) if results_base["accs"] else 0

print(f"\n{'='*70}")
print("RESULTADOS")
print(f"{'='*70}")
print(f"\n{'Metrica':<30s} {'Sem CFE':>12s} {'Com CFE':>12s}")
print(f"{'-'*55}")
print(f"  {'Exact Match':<30s} {results_base['exact']:>6d}      {results_cfe['exact']:>6d}")
print(f"  {'Pixel Acc':<30s} {avg_base:.2%}     {avg_cfe:.2%}")
print(f"  {'Concept Hits':<30s} {'N/A':>12s} {results_cfe['concept_hits']:>6d}")

# Invariants discovered
invariants = discovery.get_invariants(min_confidence=0.1)
print(f"\n  Invariantes descobertos: {len(invariants)}")
for inv in invariants:
    print(f"    {inv.structure_type}: conf={inv.confidence:.2f}, obs={inv.n_observations}")

print(f"\n  Mapeamento CFE -> Primitives:")
for concept, prims in bridge.CONCEPT_TO_PRIMITIVES.items():
    has_inv = any(inv.structure_type == concept for inv in invariants)
    mark = "DESCOBERTO" if has_inv else "nao visto"
    print(f"    {concept:25s} -> {len(prims):2d} primitives possiveis [{mark}]")
