"""
ARC-AGI com HypothesisEngine — Arquitetura Concept-Centric.

Fluxo invertido:
  Observacao → HypothesisEngine → conceitos → hipoteses → guidance → ProgramSynthesizer

Compara:
  - Baseline (sem conceitos)
  - Hard Filter (CFE velho)
  - Soft Guidance (CFEBridgeSearch)
  - Concept-Centric (HypothesisEngine)
"""
import sys, numpy as np, time
sys.path.insert(0, ".")

from core.goal_driven_engine import GoalDrivenEngine
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
from core.conceptual_discovery.concept_agent import ConceptAgent
from core.hypothesis_engine import HypothesisEngine
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


def make_engine(registry, **kwargs):
    graph = CompositionGraph()
    bridge = OperatorGroundingBridge(registry, graph)
    pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
    return GoalDrivenEngine(
        program_induction_engine=pie, composition_graph=graph,
        use_planner=False, **kwargs
    )


def setup_hypothesis_engine():
    """Cria HypothesisEngine com ConceptAgents para cada invariante."""
    engine = HypothesisEngine()

    # REFLECTION
    ref = ConceptAgent("reflection", "spatial_transformation")
    ref.add_prediction("orientation_inverted", True)
    ref.add_prediction("area_preserved", True)
    ref.add_prediction("color_preserved", True)
    ref.add_prediction("object_count_preserved", True)
    ref.add_modal_signature("grid", ["flip", "rotate", "mirror"],
                            invariants=["shape_preserved"])
    ref.add_compatibility("symmetry", 0.8)
    ref.add_compatibility("rotation", 0.7)
    ref.add_contradiction("expansion")
    def _ref_hyp(obs):
        if not isinstance(obs, dict): return None
        inp, out = obs.get("grid"), obs.get("target")
        if inp is None or out is None: return None
        if inp.shape == out.shape:
            return {"concept": "reflection", "predictions_confirmed": 1.0}
        return None
    ref.set_hypothesis_fn(_ref_hyp)
    ref.observe("physics"); ref.observe("geometry"); ref.observe("audio")
    engine.register_agent(ref)

    # SYMMETRY
    sym = ConceptAgent("symmetry", "structural_preservation")
    sym.add_prediction("structure_preserved", True)
    sym.add_prediction("count_preserved", True)
    sym.add_prediction("connectivity_preserved", True)
    sym.add_modal_signature("grid", ["bilateral", "rotational"],
                            invariants=["shape_preserved", "count_preserved"])
    sym.add_compatibility("reflection", 0.9)
    sym.add_compatibility("rotation", 0.8)
    sym.add_contradiction("randomness")
    def _sym_hyp(obs):
        if not isinstance(obs, dict): return None
        inp, out = obs.get("grid"), obs.get("target")
        if inp is None or out is None: return None
        if inp.shape == out.shape and obs.get("shape_preserved"):
            return {"concept": "symmetry", "predictions_confirmed": 0.8}
        return None
    sym.set_hypothesis_fn(_sym_hyp)
    sym.observe("geometry"); sym.observe("biology")
    engine.register_agent(sym)

    # EXPANSION
    exp = ConceptAgent("expansion", "scale_increase")
    exp.add_prediction("object_count_increases", True)
    exp.add_prediction("grid_size_increases", True)
    exp.add_prediction("pattern_repeated", True)
    exp.add_modal_signature("grid", ["tile", "scale", "copy"],
                            invariants=["pattern_preserved"])
    exp.add_compatibility("branching", 0.6)
    exp.add_contradiction("contraction")
    def _exp_hyp(obs):
        if not isinstance(obs, dict): return None
        inp, out = obs.get("grid"), obs.get("target")
        if inp is None or out is None: return None
        if out.size > inp.size * 1.5:
            return {"concept": "expansion", "predictions_confirmed": 1.0}
        return None
    exp.set_hypothesis_fn(_exp_hyp)
    exp.observe("geometry"); exp.observe("physics")
    engine.register_agent(exp)

    # CONTRACTION
    con = ConceptAgent("contraction", "scale_decrease")
    con.add_prediction("object_count_decreases", True)
    con.add_prediction("grid_size_decreases", True)
    con.add_modal_signature("grid", ["crop", "compress"],
                            invariants=["content_preserved"])
    con.add_compatibility("filtering", 0.7)
    con.add_contradiction("expansion")
    def _con_hyp(obs):
        if not isinstance(obs, dict): return None
        inp, out = obs.get("grid"), obs.get("target")
        if inp is None or out is None: return None
        if out.size < inp.size * 0.5:
            return {"concept": "contraction", "predictions_confirmed": 1.0}
        return None
    con.set_hypothesis_fn(_con_hyp)
    con.observe("geometry"); con.observe("physics")
    engine.register_agent(con)

    # BRANCHING
    bra = ConceptAgent("branching", "structural_division")
    bra.add_prediction("object_count_increases", True)
    bra.add_prediction("parent_structure_preserved", True)
    bra.add_prediction("recursive_pattern", True)
    bra.add_modal_signature("grid", ["fractal", "recursive_tile"],
                            invariants=["hierarchy_preserved"])
    bra.add_compatibility("expansion", 0.6)
    bra.add_contradiction("contraction")
    def _bra_hyp(obs):
        if not isinstance(obs, dict): return None
        inp, out = obs.get("grid"), obs.get("target")
        if inp is None or out is None: return None
        out_objs = int((out > 0).sum()) if hasattr(out, 'sum') else 0
        in_objs = int((inp > 0).sum()) if hasattr(inp, 'sum') else 0
        if out_objs > in_objs * 2:
            return {"concept": "branching", "predictions_confirmed": 1.0}
        return None
    bra.set_hypothesis_fn(_bra_hyp)
    bra.observe("biology"); bra.observe("geography"); bra.observe("geometry")
    engine.register_agent(bra)

    # CAUSAL_CHAIN
    causal = ConceptAgent("causal_chain", "sequential_transformation")
    causal.add_prediction("order_preserved", True)
    causal.add_prediction("intermediate_steps", True)
    causal.add_modal_signature("grid", ["pipeline", "sequential"],
                               invariants=["order_preserved"])
    causal.add_compatibility("spatial_transformation", 0.5)
    def _causal_hyp(obs):
        if not isinstance(obs, dict): return None
        return {"concept": "causal_chain", "predictions_confirmed": 0.5}
    causal.set_hypothesis_fn(_causal_hyp)
    causal.observe("physics"); causal.observe("programming")
    engine.register_agent(causal)

    return engine


print("=" * 70)
print("ARC-AGI Concept-Centric — HypothesisEngine")
print("=" * 70)

dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)
registry = setup_registry()
hyp_engine = setup_hypothesis_engine()

# Engines
engine_base = make_engine(registry)

# CFE bridge (shared discovery)
discovery = InvariantDiscovery()
bridge = CFEBridge(discovery=discovery)
engine_hard = make_engine(registry, cfe_bridge=bridge)

# CFEBridgeSearch (soft guidance)
sg_pie = ProgramSynthesizer(registry, composition_graph=CompositionGraph(),
                             grounding_bridge=OperatorGroundingBridge(registry, CompositionGraph()))
soft = CFEBridgeSearch(bridge, sg_pie)
engine_soft = make_engine(registry, cfe_bridge_search=soft)

# HypothesisEngine (concept-centric)
he_pie = ProgramSynthesizer(registry, composition_graph=CompositionGraph(),
                             grounding_bridge=OperatorGroundingBridge(registry, CompositionGraph()))
engine_he = make_engine(registry, hypothesis_engine=hyp_engine)

total = 400
results = {
    "baseline": {"exact": 0, "accs": []},
    "hard_filter": {"exact": 0, "accs": []},
    "soft_guidance": {"exact": 0, "accs": []},
    "concept_centric": {"exact": 0, "accs": [], "concepts_used": []},
}

start = time.time()
for idx in range(total):
    name, task = dataset.tasks[idx]
    train = task.get("train", [])
    if not train:
        continue
    inp = np.array(train[0]["input"], dtype=int)
    out = np.array(train[0]["output"], dtype=int)

    # Baseline
    engine_base.set_goal_from_target(out); engine_base.perceive(inp)
    engine_base.run(max_steps=3)
    f = engine_base.current_state
    if f.shape == out.shape:
        results["baseline"]["exact"] += 1 if np.array_equal(f, out) else 0
        results["baseline"]["accs"].append(1.0 - int((f != out).sum()) / max(1, int(out.size)))

    # Hard filter
    engine_hard.set_goal_from_target(out); engine_hard.perceive(inp)
    engine_hard.run(max_steps=3)
    f = engine_hard.current_state
    if f.shape == out.shape:
        results["hard_filter"]["exact"] += 1 if np.array_equal(f, out) else 0
        results["hard_filter"]["accs"].append(1.0 - int((f != out).sum()) / max(1, int(out.size)))

    # Soft guidance
    engine_soft.set_goal_from_target(out); engine_soft.perceive(inp)
    engine_soft.run(max_steps=3)
    f = engine_soft.current_state
    if f.shape == out.shape:
        results["soft_guidance"]["exact"] += 1 if np.array_equal(f, out) else 0
        results["soft_guidance"]["accs"].append(1.0 - int((f != out).sum()) / max(1, int(out.size)))

    # Concept-centric
    engine_he.set_goal_from_target(out); engine_he.perceive(inp)
    engine_he.run(max_steps=3)
    f = engine_he.current_state
    if f.shape == out.shape:
        results["concept_centric"]["exact"] += 1 if np.array_equal(f, out) else 0
        results["concept_centric"]["accs"].append(1.0 - int((f != out).sum()) / max(1, int(out.size)))
    if engine_he.last_concept:
        results["concept_centric"]["concepts_used"].append(engine_he.last_concept)

    if (idx + 1) % 50 == 0:
        el = time.time() - start
        e_b, e_h, e_s, e_he = (results[k]["exact"] for k in ["baseline","hard_filter","soft_guidance","concept_centric"])
        print(f"  [{idx+1}/{total}] {el:.0f}s | Base:{e_b} Hard:{e_h} Soft:{e_s} Concept:{e_he}")

elapsed = time.time() - start

# Results
print(f"\n{'='*70}")
print(f"RESULTADOS ({total} tasks, {elapsed:.0f}s)")
print(f"{'='*70}")
print(f"\n{'Metrica':<35s} {'Base':>8s} {'Hard':>8s} {'Soft':>8s} {'Concept':>8s}")
print(f"{'-'*70}")
for metric in ["exact", "pixel_acc"]:
    vals = []
    for k in ["baseline","hard_filter","soft_guidance","concept_centric"]:
        r = results[k]
        if metric == "exact":
            vals.append(r["exact"])
        else:
            pa = (np.mean(r["accs"]) * 100) if r["accs"] else 0
            vals.append(pa)
    if metric == "pixel_acc":
        print(f"  {'Pixel Acc %':<35s} {vals[0]:>7.2f}% {vals[1]:>7.2f}% {vals[2]:>7.2f}% {vals[3]:>7.2f}%")
    else:
        print(f"  {'Exact Match':<35s} {vals[0]:>6d}    {vals[1]:>6d}    {vals[2]:>6d}    {vals[3]:>6d}")

# Concepts used
concepts = results["concept_centric"]["concepts_used"]
if concepts:
    from collections import Counter
    counts = Counter(concepts)
    print(f"\nConceitos usados ({len(concepts)} deteccoes):")
    for concept, count in counts.most_common():
        print(f"  {concept}: {count} ({count/len(concepts)*100:.1f}%)")

# Results file
out_path = r"D:\novamind_data\arc_concept_centric_results.txt"
with open(out_path, "w") as f:
    f.write(f"ARC Concept-Centric Evaluation ({total} tasks, {elapsed:.0f}s)\n")
    for k in ["baseline","hard_filter","soft_guidance","concept_centric"]:
        r = results[k]
        pa = (np.mean(r["accs"]) * 100) if r["accs"] else 0
        f.write(f"{k}: exact={r['exact']}, pixel_acc={pa:.2f}%\n")
    if concepts:
        f.write(f"Concepts: {Counter(concepts).most_common()}\n")
print(f"\nResultados salvos: {out_path}")
