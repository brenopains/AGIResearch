"""
ARC-AGI com CognitiveExecutive + CognitiveModeManager.
Testa: modo FOCUSED vs EXPLORATION afeta resultados.
"""
import sys, numpy as np, time
sys.path.insert(0, ".")

from core.cognitive_executive import CognitiveExecutive
from core.cognitive_modes import CognitiveModeManager, CognitiveMode
from core.unified_self import UnifiedSelf
from core.intrinsic_motivation import IntrinsicMotivationEngine
from core.goal_driven_engine import GoalDrivenEngine, GoalState, GoalComparator
from core.world_model import WorldModel
from core.hypothesis_engine import HypothesisEngine
from core.conceptual_discovery.constraint_network import ConstraintNetwork
from core.conceptual_discovery.concept_agent import ConceptAgentRegistry
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


def setup_registry():
    reg = PrimitiveRegistry()
    for fn in [create_recolor_primitive, create_remove_primitive,
               create_move_primitive, create_add_primitive]:
        reg.register(fn())
    for op in get_all_proposals():
        fn = op["function"]
        prim = Primitive(name=op["name"], structure_type=op["name"])
        def _me(f):
            def _e(g, p):
                try: return f(g)
                except: return g.copy()
            return _e
        prim._execute = _me(fn)
        reg.register(prim)
    return reg


print("=" * 70)
print("ARC-AGI com CognitiveExecutive + Modos")
print("=" * 70)

dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)
registry = setup_registry()

# Result collectors
results = {
    "focused": {"exact": 0, "accs": [], "mode_changes": 0},
    "exploration": {"exact": 0, "accs": [], "mode_changes": 0},
}

# Setup compartilhado
cg = CompositionGraph()
og = OperatorGroundingBridge(registry, cg)
pie = ProgramSynthesizer(registry, composition_graph=cg, grounding_bridge=og)
engine = GoalDrivenEngine(program_induction_engine=pie, use_planner=False)

for mode_name, force_mode in [("focused", CognitiveMode.FOCUSED),
                               ("exploration", CognitiveMode.EXPLORATION)]:
    self_model = UnifiedSelf()
    motivation = IntrinsicMotivationEngine(self_model=self_model)
    mode_mgr = CognitiveModeManager(initial_mode=force_mode)
    wm = WorldModel()
    he = HypothesisEngine()
    cr = ConceptAgentRegistry()
    cn = ConstraintNetwork()
    exec_ = CognitiveExecutive(
        goal_engine=engine, self_model=self_model, motivation=motivation,
        mode_manager=mode_mgr, world_model=wm, hypothesis_engine=he,
        synthesizer=pie, constraint_network=cn, concept_registry=cr,
    )
    # Forca o modo e aplica config
    mode_mgr._transition_to(force_mode)
    exec_._apply_mode_config()

    print(f"\n--- Modo {mode_name.upper()} ---")
    print(f"  Config: beam={pie._beam_width}, depth={pie._max_depth}, "
          f"temp={mode_mgr.config.temperature}")

    start = time.time()
    for idx in range(400):
        name, task = dataset.tasks[idx]
        train = task.get("train", [])
        if not train:
            continue
        inp = np.array(train[0]["input"], dtype=int)
        out = np.array(train[0]["output"], dtype=int)

        # Usa o ciclo do CognitiveExecutive
        result = exec_.cycle(inp, out)
        r = results[mode_name]

        if result["exact"]:
            r["exact"] += 1
        r["accs"].append(result["accuracy"])
        if result.get("mode") != mode_name:
            r["mode_changes"] += 1

        if (idx + 1) % 100 == 0:
            el = time.time() - start
            print(f"  [{idx+1}/400] {el:.0f}s | exact={r['exact']}")

    elapsed = time.time() - start
    r = results[mode_name]
    pa = (np.mean(r["accs"]) * 100) if r["accs"] else 0
    print(f"  Final: exact={r['exact']}, pixel_acc={pa:.2f}%, "
          f"mode_changes={r['mode_changes']}, {elapsed:.0f}s")

# Summary
print(f"\n{'='*70}")
print(f"RESULTADO FINAL (400 tasks)")
print(f"{'='*70}")
print(f"\n{'Metrica':<35s} {'FOCUSED':>10s} {'EXPLORATION':>13s}")
print(f"{'-'*60}")
for metric in ["exact", "pixel_acc"]:
    vals = []
    for m in ["focused", "exploration"]:
        r = results[m]
        if metric == "exact":
            vals.append(r["exact"])
        else:
            pa = (np.mean(r["accs"]) * 100) if r["accs"] else 0
            vals.append(pa)
    if metric == "pixel_acc":
        print(f"  {'Pixel Acc %':<35s} {vals[0]:>7.2f}%     {vals[1]:>7.2f}%")
    else:
        print(f"  {'Exact Match':<35s} {vals[0]:>6d}         {vals[1]:>6d}")

out_path = r"D:\novamind_data\arc_modes_comparison.txt"
with open(out_path, "w") as f:
    f.write(f"ARC Mode Comparison (400 tasks)\n")
    for m in ["focused", "exploration"]:
        r = results[m]
        pa = (np.mean(r["accs"]) * 100) if r["accs"] else 0
        f.write(f"{m}: exact={r['exact']}, pixel_acc={pa:.2f}%, mode_changes={r['mode_changes']}\n")
print(f"\nResultados: {out_path}")
