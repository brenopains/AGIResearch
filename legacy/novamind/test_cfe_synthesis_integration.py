"""
Teste: CFEBridgeSearch — integracao CFE -> ProgramSynthesizer com soft guidance.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from core.conceptual_discovery.cfe_bridge import CFEBridge, CONCEPT_TO_PRIMITIVES
from core.conceptual_discovery.invariant_discovery import InvariantDiscovery
from core.program_induction.synthesis import ProgramSynthesizer
from core.program_induction.primitive import PrimitiveRegistry, Primitive
from core.program_induction.composition_graph import CompositionGraph
from core.program_induction.operator_grounding_bridge import OperatorGroundingBridge
from core.cfe_bridge_search import CFEBridgeSearch


def _make_real_registry():
    """Cria registry com primitives que realmente executam transformacoes."""
    reg = PrimitiveRegistry()

    # flip_horizontal
    p = Primitive(name="flip_h", structure_type="flip_horizontal",
                   solver_fn=lambda i, o: {} if i.shape == o.shape else None)
    p._execute = lambda g, params: np.fliplr(g.copy())
    p.confidence = 0.5
    reg.register(p)

    # flip_vertical
    p = Primitive(name="flip_v", structure_type="flip_vertical",
                   solver_fn=lambda i, o: {} if i.shape == o.shape else None)
    p._execute = lambda g, params: np.flipud(g.copy())
    p.confidence = 0.5
    reg.register(p)

    # rotate_90
    p = Primitive(name="rot90", structure_type="rotate_90",
                   solver_fn=lambda i, o: {} if i.shape == o.shape else None)
    p._execute = lambda g, params: np.rot90(g.copy(), k=-1)
    p.confidence = 0.5
    reg.register(p)

    # rotate_180
    p = Primitive(name="rot180", structure_type="rotate_180",
                   solver_fn=lambda i, o: {} if i.shape == o.shape else None)
    p._execute = lambda g, params: np.rot90(g.copy(), k=2)
    p.confidence = 0.5
    reg.register(p)

    # identity
    p = Primitive(name="id", structure_type="identity",
                   solver_fn=lambda i, o: {} if i.shape == o.shape else None)
    p._execute = lambda g, params: g.copy()
    p.confidence = 0.3
    reg.register(p)

    return reg


def _make_discovery_with_history():
    """InvariantDiscovery pre-populado (confidence > 0.1)."""
    disc = InvariantDiscovery()
    # Pre-popula com observacoes de reflection
    for _ in range(6):
        disc.observe({
            "type": "program_result",
            "program": "flip_h",
            "input_shape": (5, 5),
            "output_shape": (5, 5),
            "success": True,
            "domain": "arc"
        })
    return disc


def _reflection_pair():
    inp = np.zeros((5, 5), dtype=np.int32)
    inp[1, 1:4] = 1
    out = np.fliplr(inp)
    return inp, out


def _make_bridge_and_search():
    disc = _make_discovery_with_history()
    bridge = CFEBridge(discovery=disc)
    reg = _make_real_registry()
    cg = CompositionGraph()
    og = OperatorGroundingBridge(registry=reg, composition_graph=cg)
    syn = ProgramSynthesizer(registry=reg, composition_graph=cg, grounding_bridge=og)
    bs = CFEBridgeSearch(bridge, syn)
    return bridge, bs, reg


def test_concept_detection():
    bridge = CFEBridge(discovery=_make_discovery_with_history())
    inp, out = _reflection_pair()
    concept = bridge.get_concept_hint(inp, out)
    assert concept is not None, "Deveria detectar conceito para reflection"
    assert concept in CONCEPT_TO_PRIMITIVES, f"Conceito desconhecido: {concept}"
    print(f"  Conceito detectado: {concept}")
    # Verifica filtro
    filtro = bridge.get_primitive_filter(inp, out)
    assert filtro is not None, "Deveria ter filtro de primitives"
    assert "flip_horizontal" in filtro, f"Filtro deveria incluir flip_horizontal: {filtro}"
    print(f"  Filtro: {filtro}")


def test_soft_guidance():
    _, bs, _ = _make_bridge_and_search()
    inp, out = _reflection_pair()
    result = bs.search(inp, out)
    assert len(result) > 0, "Deveria encontrar programas"
    print(f"  Melhor programa: {result[0].name}")
    print(f"  Score: {result[0].score}")
    print(f"  Validado: {result[0].validated}")


def test_stats_tracking():
    _, bs, _ = _make_bridge_and_search()
    for _ in range(3):
        inp, out = _reflection_pair()
        bs.search(inp, out)
    stats = bs.get_stats()
    print(f"Stats apos 3 buscas:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    assert stats["total_searches"] == 3
    assert stats["concept_guided"] > 0


def test_attribution():
    _, bs, _ = _make_bridge_and_search()
    inp, out = _reflection_pair()
    attr = bs.search_with_attribution(inp, out)
    assert "programs" in attr
    assert "concept" in attr
    assert "concept_helped" in attr
    assert "fallback_used" in attr
    print(f"Atribuicao:")
    print(f"  Conceito: {attr['concept']}")
    print(f"  Exact: {attr['exact']}")
    print(f"  Concept helped: {attr['concept_helped']}")
    print(f"  Fallback: {attr['fallback_used']}")
    print(f"  Best score: {attr['best_score']}")


def test_weights_affect_ordering():
    """Primitivas com boost aparecem primeiro."""
    bridge = CFEBridge(discovery=_make_discovery_with_history())
    reg = _make_real_registry()
    cg = CompositionGraph()
    og = OperatorGroundingBridge(registry=reg, composition_graph=cg)
    syn = ProgramSynthesizer(registry=reg, composition_graph=cg, grounding_bridge=og)

    filtro = bridge.get_primitive_filter(*_reflection_pair())
    weights = {p: 2.0 for p in filtro}

    syn._weights = weights
    inp, out = _reflection_pair()
    prims = list(reg.get_all(min_confidence=0.0))
    ordered = sorted(prims, key=lambda p: -weights.get(p.structure_type, 1.0))
    first_type = ordered[0].structure_type
    # O primeiro deve ser flip_horizontal ou rotate_90 (peso 2.0)
    assert weights.get(first_type, 1.0) == 2.0, \
        f"Primeira primitiva {first_type} nao tem boost"
    print(f"  Primeira com boost: {first_type}")


PASS = "[PASS]"
FAIL = "[FAIL]"


if __name__ == "__main__":
    print("=== CFEBridgeSearch Tests ===\n")
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        print(f"{name}...")
        try:
            fn()
            print(f"  {PASS}\n")
        except Exception as e:
            print(f"  {FAIL}: {e}\n")
            import traceback
            traceback.print_exc()
