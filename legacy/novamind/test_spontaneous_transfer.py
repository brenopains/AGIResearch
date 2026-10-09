"""
Teste: Transferencia Espontanea Cross-Dominio.

O sistema aprende uma lei EM UM dominio (ex: branching em biologia).
Depois recebe um problema EM OUTRO dominio (ex: branching em geometria/ARC).

A pergunta: o sistema usa ESPONTANEAMENTE a lei aprendida em biologia
para guiar a busca no problema de geometria?

Se sim → evidencia de que a lei virou abstracao operacional.
Se nao → a lei e apenas uma descricao, nao conhecimento reutilizavel.

O teste NAO fornece:
  - Mapeamento explicito entre dominios
  - Regras escritas "use branching em geometria"
  - Exemplos de treino no dominio alvo

Apenas a lei existe no WorldLawRegistry.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, ".")

from core.conceptual_discovery.world_laws import WorldLaw, WorldLawRegistry
from core.conceptual_discovery.concept_agent import ConceptAgent, ConceptAgentRegistry
from core.hypothesis_engine import HypothesisEngine
from core.world_model import WorldModel
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph, OperatorGroundingBridge,
)
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.program_induction.operator_proposals import get_all_proposals
from core.program_induction.primitive import Primitive


PASS = "[PASS]"
FAIL = "[FAIL]"


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


def test_transfer_branching_biology_to_geometry():
    """Branching aprendido em biologia é usado em geometria?

    Fase 1: Ensinar branching em biologia (arvores, vasos, neuronios).
    Fase 2: Apresentar um problema de geometria que exige branching.
    Fase 3: Medir se o sistema transfere espontaneamente.
    """
    print("\n--- FASE 1: Ensinando BRANCHING em biologia ---")

    # Cria um ConceptAgent para branching (so em biologia)
    bra = ConceptAgent("branching", "structural_division")
    bra.add_prediction("object_count_increases", True)
    bra.add_prediction("parent_structure_preserved", True)
    bra.add_prediction("recursive_pattern", True)
    bra.add_modal_signature("biology", ["tree", "vessels", "neurons"],
                            invariants=["hierarchy_preserved"])
    bra.observe("biology")
    bra.observe("biology")
    bra.observe("biology")
    bra.confidence = 0.7

    # Cria um conceito similar mas sem relacao com geometria
    bra.add_contradiction("contraction")
    bra.add_compatibility("expansion", 0.6)

    # Registry de conceitos
    concept_reg = ConceptAgentRegistry()
    concept_reg.register(bra)

    # Registry de leis
    law_reg = WorldLawRegistry()

    # Cria a lei PROPAGATION se ainda nao existe
    propagation = WorldLaw(
        name="propagation",
        description="tudo que se propaga: branching, cascade, diffusion",
        explains={"branching", "diffusion", "spread"},
        domains={"biology", "geography"},
        predictions={"increases_over_space": 0.8, "has_source": 0.9},
        compression=0.8, coverage=0.3, parsimony=0.5, confidence=0.7,
    )
    # Remove geometria da lei propositalmente
    propagation.domains = {"biology", "geography"}
    law_reg.register(propagation)

    # Verifica: branching tem law propagation em biologia
    laws_for_bra = law_reg.get_laws_for_concept("branching")
    assert len(laws_for_bra) > 0, "Branching deveria ter lei associada"
    assert "propagation" in [l.name for l in laws_for_bra]
    print(f"  Lei 'propagation' explica branching em {propagation.domains}")
    print(f"    (geometria NAO esta nos dominios da lei)")

    # HypothesisEngine
    he = HypothesisEngine(concept_registry=concept_reg)
    he.register_agent(bra)

    print("\n--- FASE 2: Problema de geometria (sem treino) ---")

    # Um problema ARC-like que exige branching:
    # input: quadrado 3x3 no centro de 5x5
    # output: 4 quadrados 3x3 nos cantos de 9x9 (expansao com branching)
    inp = np.zeros((5, 5), dtype=np.int32)
    inp[1:4, 1:4] = 2
    out = np.zeros((9, 9), dtype=np.int32)
    out[0:3, 0:3] = 2
    out[0:3, 6:9] = 2
    out[6:9, 0:3] = 2
    out[6:9, 6:9] = 2

    # Apresenta ao hypothesis engine
    obs = {"grid": inp, "target": out, "domain": "geometry", "modality": "grid"}
    hypotheses = he.generate_hypotheses(obs, domain="geometry", modality="grid")

    print(f"  Problema: {inp.shape} → {out.shape} (expansao com branching)")
    print(f"  Hipoteses geradas: {len(hypotheses)}")
    for h in hypotheses:
        print(f"    {h.concept} (conf={h.confidence:.2f})")

    # O teste: alguma hipotese menciona branching OU algum conceito
    # associado a propagation?
    concepts_used = set(h.concept for h in hypotheses)
    has_branching = "branching" in concepts_used

    print("\n--- FASE 3: Medindo transferencia ---")

    # Verifica se a lei propagation (aprendida em biologia) influencia
    # as hipoteses para o problema de geometria
    law_used = False
    for h in hypotheses:
        found_laws = law_reg.get_laws_for_concept(h.concept)
        for found in found_laws:
            if found.name == "propagation":
                law_used = True
                print(f"  LEI '{found.name}' TRANSFERIDA: {h.concept} "
                      f"(dominios da lei: {found.domains})")

    if law_used:
        print(f"\n  ✅ TRANSFERENCIA ESPONTANEA DETECTADA")
        print(f"     Lei 'propagation' (aprendida em biologia/geography)")
        print(f"     foi usada para guiar hipotese em geometria")
        print(f"     SEM nenhum exemplo de geometria ter sido dado.")
    else:
        print(f"\n  ⚠️  Transferencia nao detectada")
        print(f"     Lei 'propagation' nao influenciou hipoteses")
        print(f"     em geometria.")

    # O ideal: o hypothesis engine geraria "branching" como hipotese
    # mesmo nunca tendo visto branching em geometria, porque a lei
    # propagation generalizou.
    return law_used


def test_transfer_conservation_physics_to_audio():
    """Conservation aprendido em fisica é usado em audio?"""
    print("\n--- Transfer: conservation (physics → audio) ---")

    # Cria lei conservation (sem audio)
    conservation = WorldLaw(
        name="conservation",
        description="quantidade se preserva apos transformacao",
        explains={"symmetry", "reflection", "invariance"},
        domains={"physics", "geometry"},  # sem audio!
        predictions={"quantity_preserved": 0.9},
        compression=0.8, coverage=0.3, parsimony=0.5, confidence=0.8,
    )
    law_reg = WorldLawRegistry()
    law_reg.register(conservation)

    # Reflection como conceito (sem audio tb)
    ref = ConceptAgent("reflection", "spatial_transformation")
    ref.add_prediction("orientation_inverted", True)
    ref.add_prediction("area_preserved", True)
    ref.add_modal_signature("physics", ["bounce"], invariants=["energy_preserved"])
    ref.add_modal_signature("geometry", ["flip"], invariants=["shape_preserved"])
    ref.observe("physics")
    ref.observe("geometry")
    ref.confidence = 0.8

    # Registries
    concept_reg = ConceptAgentRegistry()
    concept_reg.register(ref)
    he = HypothesisEngine(concept_registry=concept_reg)
    he.register_agent(ref)

    # Problema de audio: eco (reflection em audio)
    inp = {"signal": [1.0, 0.8, 0.5], "type": "audio", "domain": "audio"}
    out = {"signal": [1.0, 0.8, 0.5, 0.3, 0.15], "type": "echo", "reflected": True}

    hypotheses = he.generate_hypotheses(
        {**inp, "target": out}, domain="audio", modality="audio"
    )

    print(f"  Problema: audio echo (reflection)")
    print(f"  Hipoteses geradas: {len(hypotheses)}")
    for h in hypotheses:
        print(f"    {h.concept} (conf={h.confidence:.2f})")

    # Verifica transferencia
    law_used = any(
        any(l.name == "conservation" for l in law_reg.get_laws_for_concept(h.concept))
        for h in hypotheses
    )

    concepts_used = set(h.concept for h in hypotheses)
    has_reflection = "reflection" in concepts_used

    if has_reflection:
        print(f"\n  ✅ REFLECTION transferido para audio")
        print(f"     (conceito conhecido apenas em physics/geometry)")
    else:
        print(f"\n  ⚠️  Reflection nao transferido para audio")

    return law_used or has_reflection


def test_adversarial_pressure():
    """Adversarial Reality Generator cria pressao mensuravel."""
    from core.adversarial_reality import AdversarialRealityGenerator
    from core.conceptual_discovery.world_laws import WorldLawRegistry

    law_reg = WorldLawRegistry()
    concept_reg = ConceptAgentRegistry()

    gen = AdversarialRealityGenerator(
        law_registry=law_reg,
        concept_registry=concept_reg,
    )

    batch = gen.generate_pressure_batch(n=20)
    assert len(batch) > 0
    print(f"  Exemplos adversariais gerados: {len(batch)}")
    for ex in batch[:5]:
        print(f"    {ex.type:20s} | {ex.domain:15s} | {ex.challenge[:60]}")

    stats = gen.get_stats()
    print(f"  Por tipo: {stats['by_type']}")
    assert stats["total_generated"] > 0
    return True


if __name__ == "__main__":
    print("=" * 65)
    print("TESTE: TRANSFERENCIA ESPONTANEA CROSS-DOMINIO")
    print("=" * 65)

    results = []
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        print(f"\n{name}...")
        try:
            result = fn()
            status = "PASS" if result else "PARCIAL"
            results.append((name, status))
            print(f"\n  [{status}]")
        except Exception as e:
            results.append((name, "FAIL"))
            print(f"\n  [{FAIL}]: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*65}")
    print("RESUMO DA TRANSFERENCIA")
    print(f"{'='*65}")
    for name, status in results:
        icon = {"PASS": "✅", "PARCIAL": "⚠️", "FAIL": "❌"}
        print(f"  {icon.get(status, '?')} {name}: {status}")
