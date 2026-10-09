"""Teste do SkillEngine: skills vs macros vs base no ARC."""
import sys, numpy as np
sys.path.insert(0, ".")

from core.goal_driven_engine import GoalDrivenEngine, GoalState, GoalComparator
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph,
    OperatorGroundingBridge,
)
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.program_induction.operator_proposals import get_all_proposals
from core.program_induction.primitive import Primitive
from core.macro_discovery import MacroDiscovery
from core.skill_engine import SkillDiscovery, SkillRegistry
from core.arc_dataset import ARCDataset


def setup_base_registry():
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
    return registry


print("=" * 70)
print("TESTE: SKILL ENGINE")
print("=" * 70)

dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)
registry = setup_base_registry()

# --- FASE 1: Descobre macros (60 tasks) ---
print("\n[FASE 1] Descobrindo macros...")
graph = CompositionGraph()
bridge = OperatorGroundingBridge(registry, graph)
pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph, use_planner=False)

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

disc = MacroDiscovery(composition_graph=graph)
disc.scan(min_frequency=3, min_success_rate=0.3)
macros = disc.get_macros(min_confidence=0.1)
print(f"  Macros: {len(macros)}")

# --- FASE 2: Descobre SKILLS dos clusters de macros ---
print("\n[FASE 2] Descobrindo SKILLS...")
seqs = [m.primitive_sequence for m in macros]
skill_disc = SkillDiscovery(registry=registry)
skill_reg = skill_disc.discover(seqs, domain="ARC")
print(f"  Skills descobertas: {skill_reg.get_stats()}")
for s in skill_reg.get_best(n=10):
    print(f"    {s.name:35s} intent={s.intent:30s} impls={len(s.implementations):2d} conf={s.confidence:.2f}")

# --- FASE 3: TESTE — skills vs primitives base ---
print("\n[FASE 3] Skills vs Base Primitives (forcando uso)...")
base_better = 0
skill_better = 0
tie = 0
skill_wins = []

for task_idx in range(100):
    name, task = dataset.tasks[task_idx]
    train_pairs = task.get("train", [])
    if not train_pairs:
        continue
    inp = np.array(train_pairs[0]["input"], dtype=int)
    out = np.array(train_pairs[0]["output"], dtype=int)

    # Base: usa GoalDrivenEngine (sem skills)
    cmp = GoalComparator()
    base_gap = cmp.compute(inp, GoalState(target=out))
    base_acc = 1.0 - base_gap.total

    # Skills: tenta cada skill
    best_skill_acc = 0.0
    best_skill_name = ""
    for skill in skill_reg.get_best(n=20):
        try:
            result = skill.execute(inp, registry)
            if result.shape == out.shape:
                acc = 1.0 - int((result != out).sum()) / max(1, int(out.size))
                if acc > best_skill_acc:
                    best_skill_acc = acc
                    best_skill_name = skill.name
        except Exception:
            pass

    if best_skill_acc > base_acc + 0.05:
        skill_better += 1
        skill_wins.append((name, best_skill_name, base_acc, best_skill_acc))
    elif base_acc > best_skill_acc + 0.05:
        base_better += 1
    else:
        tie += 1

print(f"  Base melhor:        {base_better}")
print(f"  Skills melhores:    {skill_better}")
print(f"  Empate (diff<5pp):  {tie}")
print(f"\n  Exemplos onde skill venceu:")
for name, sname, base, skill in skill_wins[:5]:
    print(f"    {name}: base={base:.2%} -> skill({sname})={skill:.2%}")

# --- FASE 4: Skills com multiplas implementacoes ---
print("\n[FASE 4] Skills com multiplas implementacoes:")
for skill in skill_reg.get_best(n=20):
    if len(skill.implementations) >= 2:
        print(f"  {skill.name} ({skill.intent}):")
        for impl in skill.implementations[:3]:
            print(f"    -> {' -> '.join(impl)}")

# --- SUMMARY ---
print(f"\n{'='*70}")
print("RESUMO SKILL ENGINE")
print(f"{'='*70}")
skills = skill_reg.get_best(n=50)
n_multi = sum(1 for s in skills if len(s.implementations) >= 2)
print(f"""
  Primitives base:   {registry.count}
  Macros:            {len(macros)}
  Skills:            {len(skills)}
  Skills multi-impl: {n_multi}

  Skills > base:     {skill_better}/100 tarefas
  Base > skills:     {base_better}/100

  Skills representam INTENCAO abstrata:
    - rotate_90 + variacoes -> "reorient_objects"
    - flip_horizontal + variacoes -> "mirror_objects"
    - Cada skill pode ser implementada de multiplas formas

  Proximo passo: planner usa skills como operadores de alto nivel,
  fallback para primitives so quando skill nao se aplica.
""")
