"""Teste da GoalDrivenEngine: goal -> gap -> programa -> acao -> feedback."""
import sys, numpy as np
sys.path.insert(0, ".")

from core.goal_driven_engine import (
    GoalState, GoalComparator, Gap, ActionSelector, Action, GoalDrivenEngine,
)
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph,
)

print("=" * 70)
print("TESTE: GOAL-DRIVEN ENGINE (agencia orientada a objetivos)")
print("=" * 70)

# Setup: ProgramInductionEngine com primitives base
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.program_induction.operator_proposals import get_all_proposals

registry = PrimitiveRegistry()
for fn in [create_recolor_primitive, create_remove_primitive,
           create_move_primitive, create_add_primitive]:
    registry.register(fn())

graph = CompositionGraph()
pie = ProgramSynthesizer(registry, composition_graph=graph)

# ─── TEST 1: GoalComparator com grids ───
print("\n[TEST 1] GoalComparator — gap entre dois grids")
cmp = GoalComparator()

current = np.array([[0, 2, 0], [2, 2, 2], [0, 2, 0]], dtype=int)
goal_grid = np.array([[0, 8, 0], [8, 8, 8], [0, 8, 0]], dtype=int)

goal = GoalState(target=goal_grid, description="recolor 2 to 8", priority=0.8)
gap = cmp.compute(current, goal)
print(f"  Gap: structural={gap.structural_gap:.3f}, semantic={gap.semantic_gap:.3f}")
print(f"  Total: {gap.total:.3f}")
print(f"  Details: {gap.details}")
assert gap.total > 0  # Deve haver gap

# Gap do goal contra ele mesmo (deve ser zero)
gap_zero = cmp.compute(goal_grid, goal)
print(f"  Gap zero (goal vs self): {gap_zero.total:.3f}")
assert gap_zero.is_zero
print("  PASS")

# ─── TEST 2: ActionSelector — escolhe por reducao de gap ───
print("\n[TEST 2] ActionSelector — acao que MAIS reduz gap")
from core.program_induction import Program

progs = pie.synthesize(current, goal_grid)
print(f"  Programas candidatos: {len(progs)}")
for p in progs[:3]:
    final = p.execute(current)
    g = cmp.compute(final, goal)
    print(f"    {p.name}: score={p.score:.3f}, gap_after={g.total:.3f}")

actions = []
for p in progs[:5]:
    final = p.execute(current)
    g = cmp.compute(final, goal)
    actions.append(Action(program=p, estimated_gap_after=g, execution_cost=p.n_steps, confidence=p.score))

selector = ActionSelector()
best = selector.select(actions, goal, gap)
if best:
    print(f"\n  Melhor acao: {best.program.name if hasattr(best.program, 'name') else '?'}")
    print(f"  Gap reduction: {best.gap_reduction:.3f}")
    print(f"  Utility: {best.utility:.3f}")
    print("  PASS")

# ─── TEST 3: GoalDrivenEngine — loop completo ───
print("\n[TEST 3] GoalDrivenEngine — loop goal -> acao -> feedback")
engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph)

# Goal: transformar current em goal_grid
engine.set_goal_from_target(goal_grid, description="recolor 2 to 8", priority=0.8)
engine.perceive(current)

# Step 1
action1 = engine.step()
if action1:
    print(f"  Step 1: {action1.program.name if hasattr(action1.program, 'name') else '?'}")
    gap1 = cmp.compute(engine.current_state, goal)
    print(f"    Gap apos step 1: {gap1.total:.3f}")

# Step 2 (se necessario)
action2 = engine.step()
if action2:
    print(f"  Step 2: {action2.program.name if hasattr(action2.program, 'name') else '?'}")
    gap2 = cmp.compute(engine.current_state, goal)
    print(f"    Gap apos step 2: {gap2.total:.3f}")

# Verifica se goal foi atingido
achieved = engine._is_goal_achieved()
print(f"  Goal atingido: {achieved}")
print(f"  Stats: {engine.get_stats()}")

# ─── TEST 4: GoalComparator para estado CGE ───
print("\n[TEST 4] GoalComparator — gap entre estados CGE")
current_cge = {
    "nodes": {"whale": {}, "mammal": {}, "ocean": {}},
    "edges": [
        {"from": "whale", "to": "mammal"},
        {"from": "whale", "to": "ocean"},
    ],
}
target_cge = {
    "nodes": {"whale": {}, "mammal": {}, "ocean": {}, "breathes": {}},
    "edges": [
        {"from": "whale", "to": "mammal"},
        {"from": "whale", "to": "ocean"},
        {"from": "whale", "to": "breathes"},
    ],
}
cge_gap = cmp.compute(current_cge, GoalState(target=target_cge, description="learn whale breathes"))
print(f"  Nodes missing: diff in target but not current")
print(f"  Gap: structural={cge_gap.structural_gap:.3f}, semantic={cge_gap.semantic_gap:.3f}")
print(f"  Details: {cge_gap.details}")
assert cge_gap.structural_gap > 0
print("  PASS")

# ─── TEST 5: Multi-step goal achievement ───
print("\n[TEST 5] Multi-step — goal em 2 passos")
inp = np.array([[0, 2, 0], [0, 2, 0], [0, 0, 0]], dtype=int)
target = np.array([[0, 0, 0], [0, 8, 0], [0, 8, 0]], dtype=int)  # move down + recolor

engine2 = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph)
engine2.set_goal_from_target(target, description="move down + recolor", priority=0.9)
engine2.perceive(inp)

result = engine2.run(max_steps=5)
print(f"  Goal achieved: {result}")
print(f"  Stats: {engine2.get_stats()}")
print(f"  History:")
for h in engine2.history:
    if h.get("event") in ("goal_set", "action_executed", "goal_achieved"):
        print(f"    {h['event']}: {h.get('action', h.get('goal', '?'))}")

# Final state
final_gap = cmp.compute(engine2.current_state, GoalState(target=target))
print(f"\n  Final gap: {final_gap.total:.3f}")
print(f"  State matches target: {np.array_equal(engine2.current_state, target)}")

print("\n" + "=" * 70)
print("TODOS OS TESTES CONCLUIDOS")
print("=" * 70)
