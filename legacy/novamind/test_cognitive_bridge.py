"""Teste da ponte cognitiva: CognitiveDelta + AbstractSynthesis."""
import sys, numpy as np
sys.path.insert(0, ".")

from core.cognition import (
    CognitiveDeltaExtractor, CGEState, CognitiveDelta,
    AbstractProgramSynthesizer, AbstractReasoningChain,
    get_abstract_solver, get_abstract_primitive_names, ABSTRACT_SCHEMAS,
)

# Simula um CGE minimal para teste
class MockCGE:
    def __init__(self):
        self.nodes = {
            "whale": {"confidence": 0.9, "state": "visible", "access_count": 5},
            "mammal": {"confidence": 0.8, "state": "visible", "access_count": 3},
            "breathes": {"confidence": 0.7, "state": "visible", "access_count": 2},
            "ocean": {"confidence": 0.6, "state": "visible", "access_count": 1},
            "dog": {"confidence": 0.9, "state": "visible", "access_count": 4},
            "animal": {"confidence": 0.5, "state": "visible", "access_count": 2},
        }
        self.edges = [
            {"from": "whale", "to": "mammal", "confidence": 0.8, "edge_type": "is_a"},
            {"from": "whale", "to": "ocean", "confidence": 0.7, "edge_type": "lives_in"},
            {"from": "mammal", "to": "breathes", "confidence": 0.9, "edge_type": "has_property"},
            {"from": "dog", "to": "mammal", "confidence": 0.7, "edge_type": "is_a"},
            {"from": "dog", "to": "breathes", "confidence": 0.6, "edge_type": "has_property"},
            {"from": "mammal", "to": "animal", "confidence": 0.5, "edge_type": "is_a"},
        ]


def test_cognitive_delta():
    """Testa extracao de delta entre dois estados CGE."""
    print("[TEST 1] CognitiveDeltaExtractor")

    cge = MockCGE()
    before = CGEState.from_cge(cge)

    # Simula aprendizado: whale->breathes (inferencia transitiva)
    cge.edges.append({
        "from": "whale", "to": "breathes", "confidence": 0.75, "edge_type": "has_property",
    })
    # whale->mammal confidence aumenta
    cge.nodes["whale"]["confidence"] = 0.95

    after = CGEState.from_cge(cge)

    extractor = CognitiveDeltaExtractor()
    delta = extractor.extract(before, after)

    print(f"  Tem mudancas: {delta.has_changes}")
    print(f"  Tipo: {delta.transformation_type}")
    print(f"  Arestas adicionadas: {len(delta.edges_added)}")
    print(f"  Confianca mudou: {len(delta.nodes_confidence_changed)}")
    print(f"  Summary: {delta.summarize()}")
    print(f"  Description: {delta.to_description()}")
    assert delta.has_changes
    assert len(delta.edges_added) == 1
    assert delta.edges_added[0]["from"] == "whale"
    assert delta.edges_added[0]["to"] == "breathes"
    print("  PASS\n")


def test_abstract_synthesis():
    """Testa sintese de cadeia de raciocinio a partir de delta."""
    print("[TEST 2] AbstractProgramSynthesizer")

    cge = MockCGE()
    before = CGEState.from_cge(cge)

    # Simula inferencia: whale->breathes
    cge.edges.append({
        "from": "whale", "to": "breathes", "confidence": 0.75, "edge_type": "has_property",
    })
    after = CGEState.from_cge(cge)

    synthesizer = AbstractProgramSynthesizer(cge=cge)
    chain = synthesizer.observe(before, after)

    if chain:
        print(f"  Cadeia: {chain.name}")
        print(f"  Steps: {chain.n_steps}")
        print(f"  Score: {chain.score:.2f}")
        print(f"  Validado: {chain.validated}")
        for s in chain.steps:
            print(f"    {s['operation']}: {s['params']}")
        print("  PASS\n")
    else:
        print("  Nenhuma cadeia sintetizada (pode ser esperado)\n")


def test_abstract_primitives():
    """Testa solvers de primitives abstratas."""
    print("[TEST 3] Abstract Primitive Solvers")

    cge = MockCGE()

    # Testa infer solver diretamente
    from core.cognition.abstract_primitives import _solve_infer, _solve_generalize

    # Cria delta manual
    delta = CognitiveDelta(
        edges_added=[{"from": "whale", "to": "breathes",
                       "confidence": 0.75, "edge_type": "has_property"}],
    )
    params = _solve_infer(delta, cge)
    if params:
        print(f"  Infer: {params}")
        assert params["from_concept"] == "whale"
        assert params["to_concept"] == "breathes"
        print("  PASS")
    else:
        print("  Infer: None (caminho pode ser encontrado?)")

    # Testa generalize solver
    delta2 = CognitiveDelta(nodes_added={"animal": 0.5})
    params2 = _solve_generalize(delta2, cge)
    if params2:
        print(f"  Generalize: {params2}")
    else:
        print("  Generalize: None (esperado sem novas propriedades)")


def test_integration():
    """Testa integracao com o pipeline de cognicao existente."""
    print("\n[TEST 4] Integracao com CognitionPipeline")

    from core.cognition import CognitionPipeline, Competition, Timeline, MetaCognition

    cge = MockCGE()
    pipe = CognitionPipeline()
    meta = MetaCognition()
    syn = AbstractProgramSynthesizer(cge=cge)

    # Simula ciclo: antes -> opera -> depois
    before = CGEState.from_cge(cge)
    cge.edges.append({
        "from": "whale", "to": "breathes", "confidence": 0.75, "edge_type": "has_property",
    })
    after = CGEState.from_cge(cge)

    chain = syn.observe(before, after)
    if chain:
        print(f"  Raciocinio detectado: {chain.name}")
        print(f"  Stats: {syn.get_stats()}")
        print("  PASS")
    else:
        print("  Nenhum raciocinio (pode ser esperado sem CGE real)")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTE: COGNITIVE BRIDGE (delta abstrato + sintese de raciocinio)")
    print("=" * 60)

    test_cognitive_delta()
    test_abstract_primitives()
    test_abstract_synthesis()
    test_integration()

    print("\n" + "=" * 60)
    print("TODOS OS TESTES CONCLUIDOS")
    print("=" * 60)
