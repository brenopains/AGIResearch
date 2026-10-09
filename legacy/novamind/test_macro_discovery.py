"""Teste: MacroDiscovery — descobre macros do CompositionGraph, promove a primitives."""
import sys, numpy as np
sys.path.insert(0, ".")

from core.program_induction.composition_graph import CompositionGraph
from core.macro_discovery import MacroDiscovery
from core.goal_driven_engine import GoalDrivenEngine, GoalState, GoalComparator
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, OperatorGroundingBridge,
)
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.program_induction.operator_proposals import get_all_proposals
from core.program_induction.primitive import Primitive
from core.arc_dataset import ARCDataset

print("=" * 70)
print("MACRO DISCOVERY")
print("=" * 70)

# Setup registry com primitives base + espaciais
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

graph = CompositionGraph()
bridge = OperatorGroundingBridge(registry, graph)
pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph)
dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)

# Popula o grafo rodando ARC
print("\nPopulando CompositionGraph com 60 tasks ARC...")
for task_idx in range(60):
    name, task = dataset.tasks[task_idx]
    train_pairs = task.get("train", [])
    if not train_pairs:
        continue
    inp = np.array(train_pairs[0]["input"], dtype=int)
    out = np.array(train_pairs[0]["output"], dtype=int)
    engine.set_goal_from_target(out, description=name)
    engine.perceive(inp)
    engine.step()
    # O step ja alimenta o composition_graph

print(f"CompositionGraph: {graph.get_stats()}")

# Descobre macros
print("\nEscaneando por macros...")
discovery = MacroDiscovery(composition_graph=graph, registry=registry)
discovery.scan(min_frequency=2, min_success_rate=0.3)
print(f"Macros descobertas: {discovery.get_stats()}")

macros = discovery.get_macros(min_confidence=0.3)
for m in macros:
    print(f"  [{m.confidence:.2f}] {m.name}: {' -> '.join(m.primitive_sequence)} "
          f"(freq={m.frequency}, sr={m.success_rate:.2f})")

# Promove macros com confianca suficiente
print("\nPromovendo macros candidatas...")
promoted = 0
for m in macros:
    if m.confidence > 0.4:
        ok = discovery.promote_to_primitive(m)
        if ok:
            print(f"  PROMOVIDA: {m.name} ({' -> '.join(m.primitive_sequence)})")
            promoted += 1

print(f"\nPrimitives antes: {registry.count - promoted}")
print(f"Primitives depois: {registry.count}")
print(f"Macros promovidas: {promoted}")

print("\nRegistry completo:")
for p in registry.get_all():
    if "macro" in p.name.lower() or p.structure_type == "macro":
        print(f"  >> MACRO << {p.name}: {p.description[:70]}")
    else:
        print(f"     {p.name}: {p.description[:50]}")
