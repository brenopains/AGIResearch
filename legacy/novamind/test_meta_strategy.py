"""Teste do MetaPrimitiveRegistry + StrategyMemory em tasks ARC."""
import sys, numpy as np
sys.path.insert(0, ".")

from core.program_induction.meta_registry import MetaPrimitiveRegistry
from core.strategy_memory import StrategyMemory, ProblemSignature
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
from core.arc_dataset import ARCDataset

print("=" * 70)
print("TESTE: MetaPrimitiveRegistry + StrategyMemory")
print("=" * 70)

# Setup
registry = PrimitiveRegistry()
for fn in [create_recolor_primitive, create_remove_primitive,
           create_move_primitive, create_add_primitive]:
    registry.register(fn())
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
    registry.register(prim)

meta = MetaPrimitiveRegistry(registry=registry)
strat = StrategyMemory()
graph = CompositionGraph()
bridge = OperatorGroundingBridge(registry, graph)
pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph)
cmp = GoalComparator()

dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)

print(f"\nPrimitives: {registry.count}")
print(f"Meta inicial: {meta.get_stats()}")

# Roda ARC, registra uso no MetaPrimitiveRegistry e StrategyMemory
for task_idx in range(30):
    name, task = dataset.tasks[task_idx]
    train_pairs = task.get("train", [])
    if not train_pairs:
        continue
    inp = np.array(train_pairs[0]["input"], dtype=int)
    out = np.array(train_pairs[0]["output"], dtype=int)

    engine.set_goal_from_target(out, description=name)
    engine.perceive(inp)
    engine.step()
    final = engine.current_state

    exact = bool(np.array_equal(final, out)) if final.shape == out.shape else False
    acc = 1.0 - int((final != out).sum()) / max(1, int(out.size)) if final.shape == out.shape else 0.0

    # Registra no MetaRegistry
    sig = ProblemSignature.from_grids(inp, out, gap_type="ARC")
    for h_item in engine.history[-3:]:
        if h_item.get("event") == "action_executed":
            meta.record_use(h_item.get("action", "?"), success=exact, domain="ARC")

    # Se resolveu, armazena estrategia
    if exact or acc > 0.8:
        # Extrai plano do engine history
        seq = []
        params = []
        for h_item in engine.history:
            if h_item.get("event") == "action_executed":
                seq.append(h_item.get("action", "?")[:20])
                params.append({})
        if seq:
            strat.store(sig, seq, params, score=acc, task_id=name)

# Resultados
print(f"\n[RESULTADOS MetaPrimitiveRegistry]")
stats = meta.get_stats()
print(f"  Total: {stats['total']}")
print(f"  Arquivadas: {stats['archived']}")
print(f"  Usos totais: {stats['total_uses']}")
print(f"  Utilidade media: {stats['avg_utility']}")
print(f"  Candidatas a fusao: {stats['n_ready_to_merge']}")

# Mostra saude das primitives
print(f"\n  Saude das primitives:")
for name, h in sorted(meta.health.items(), key=lambda x: -x[1].utility)[:10]:
    print(f"    {name:20s} util={h.utility:.2f} freq={h.frequency:3d} "
          f"sr={h.success_rate:.2f} viability={h.viability:.3f}")

# Testa poda
print(f"\n  Poda:")
changes = meta.prune(min_viability=0.01, max_age_hours=0)
print(f"    Arquivadas: {changes['archived']}, Fundidas: {changes['merged']}")
print(f"  Apos poda: {meta.get_stats()}")

# Resultados StrategyMemory
print(f"\n[RESULTADOS StrategyMemory]")
mem_stats = strat.get_stats()
print(f"  Planos armazenados: {mem_stats['total']}")
print(f"  Score medio: {mem_stats['avg_score']}")
print(f"  Tipos de gap unicos: {mem_stats['unique_gap_types']}")
print(f"  Total sucessos: {mem_stats['total_successes']}")

if strat.plans:
    print(f"\n  Planos armazenados:")
    for p in strat.plans[:5]:
        print(f"    {p.task_id}: {' -> '.join(p.primitive_sequence[:4])} "
              f"(score={p.score:.2f}, sr={p.success_rate:.2f})")

# Testa recuperacao
print(f"\n  Recuperacao por similaridade:")
test_sig = ProblemSignature.from_grids(
    np.array([[0,2,0],[2,2,2],[0,2,0]], dtype=int),
    np.array([[0,8,0],[8,8,8],[0,8,0]], dtype=int),
    gap_type="ARC",
)
best = strat.retrieve(test_sig)
if best:
    seq, params = strat.adapt(best, test_sig)
    print(f"    Plano recuperado: {' -> '.join(seq[:4])} (sim={best.signature.similarity(test_sig):.2f})")

print(f"\n[FIM] MetaPrimitiveRegistry + StrategyMemory operacionais")
