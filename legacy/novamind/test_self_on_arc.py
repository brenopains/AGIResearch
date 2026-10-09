"""
Teste do UnifiedSelf + CognitiveExecutive + IntrinsicMotivation no ARC.

Mede:
  - ARC accuracy com self vs sem self
  - Evolucao dos drives (curiosidade, mestria, novidade)
  - Memoria autobiografica (o sistema lembra do que fez?)
  - Neuromodulacao (humor muda com sucesso/falha?)
  - Capacidades (auto-avaliacao de proficiencia)
"""
import sys, numpy as np, time
sys.path.insert(0, ".")

from core.goal_driven_engine import GoalDrivenEngine, GoalState, GoalComparator
from core.cognitive_executive import CognitiveExecutive
from core.unified_self import UnifiedSelf
from core.intrinsic_motivation import IntrinsicMotivationEngine
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

print("=" * 70)
print("ARC-AGI com UnifiedSelf + CognitiveExecutive")
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
    def _me(f):
        def _e(g, p):
            try: return f(g)
            except: return g.copy()
        return _e
    prim._execute = _me(fn)
    registry.register(prim)

dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)

# Self
self_model = UnifiedSelf()
motivation = IntrinsicMotivationEngine(self_model=self_model)
cfe = CFEBridge()

# Engine COM self
graph = CompositionGraph()
bridge = OperatorGroundingBridge(registry, graph)
pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph,
                           cfe_bridge=cfe, use_planner=False)
executive = CognitiveExecutive(engine, cfe_bridge=cfe, self_model=self_model,
                                motivation=motivation)

# Engine SEM self (baseline)
graph2 = CompositionGraph()
bridge2 = OperatorGroundingBridge(registry, graph2)
pie2 = ProgramSynthesizer(registry, composition_graph=graph2, grounding_bridge=bridge2)
engine2 = GoalDrivenEngine(program_induction_engine=pie2, composition_graph=graph2,
                            use_planner=False)
cmp = GoalComparator()

# Test
n_tasks = 100
results_with = {"exact": 0, "accs": []}
results_without = {"exact": 0, "accs": []}

for idx in range(n_tasks):
    name, task = dataset.tasks[idx]
    train = task.get("train", [])
    if not train:
        continue
    inp = np.array(train[0]["input"], dtype=int)
    out = np.array(train[0]["output"], dtype=int)

    # COM self
    cyc = executive.cycle(inp, out)
    results_with["exact"] += 1 if cyc["exact"] else 0
    results_with["accs"].append(cyc["accuracy"])

    # SEM self
    engine2.set_goal_from_target(out)
    engine2.perceive(inp)
    engine2.run(max_steps=3)
    f2 = engine2.current_state
    if f2.shape == out.shape:
        results_without["exact"] += 1 if np.array_equal(f2, out) else 0
        results_without["accs"].append(
            1.0 - int((f2 != out).sum()) / max(1, int(out.size))
        )

    if (idx + 1) % 20 == 0:
        print(f"  [{idx+1}/{n_tasks}] Self={results_with['exact']} Base={results_without['exact']}")

# ─── RELATORIO ───
avg_with = np.mean(results_with["accs"]) if results_with["accs"] else 0
avg_without = np.mean(results_without["accs"]) if results_without["accs"] else 0

print(f"\n{'='*70}")
print("RESULTADOS ARC")
print(f"{'='*70}")
print(f"\n{'Metrica':<35s} {'Sem Self':>12s} {'Com Self':>12s}")
print(f"{'-'*60}")
print(f"  {'Exact Match':<35s} {results_without['exact']:>6d}      {results_with['exact']:>6d}")
print(f"  {'Pixel Acc':<35s} {avg_without:.2%}     {avg_with:.2%}")

print(f"\n{'='*70}")
print("ESTADO DO SELF APOS 100 TASKS")
print(f"{'='*70}")
print(f"\n{executive.summarize_self()}")

print(f"\nDrives finais:")
for name, drive in self_model.drives.items():
    print(f"  {name}: level={drive.level:.2f} last_satisfied={drive.last_satisfied:.1f}")

print(f"\nNeuromodulacao:")
print(f"  DA (dopamina/recompensa): {self_model.dopamine:.2f}")
print(f"  NA (noradrenalina/alerta): {self_model.noradrenaline:.2f}")
print(f"  ACh (acetilcolina/atencao): {self_model.acetylcholine:.2f}")
print(f"  5HT (serotonina/bem-estar): {self_model.serotonin:.2f}")
print(f"  Mood: {self_model.current_mood}")

print(f"\nMemoria Autobiografica:")
print(f"  Total episodios: {len(self_model.memory.episodes)}")
print(f"  Recentes:")
for ep in self_model.memory.get_recent(n=5):
    print(f"    {ep.to_short()}")

print(f"\nCapacidades:")
for skill, prof in sorted(self_model.capabilities.items(), key=lambda x: -x[1])[:5]:
    print(f"  {skill}: {prof:.3f}")

print(f"\nCrencas (top 5):")
for belief, conf in sorted(self_model.beliefs.items(), key=lambda x: -x[1])[:5]:
    print(f"  {belief}: {conf:.3f}")

self_state = self_model.get_state()
print(f"\nSelf state summary:")
print(f"  Mood: {self_state['mood']}")
print(f"  Energy: {self_state['energy']}")
print(f"  Success rate: {self_state['success_rate']:.2%}")
print(f"  Autobiographical memory: {self_state['memory_size']} episodes")
print(f"  Beliefs: {self_state['beliefs']}")
print(f"  Capabilities: {self_state['capabilities']}")
