"""
Teste: Arquitetura Concept-Centric — a inversao.

Prova que:
  1. Conceitos sao agentes com predicoes, nao rotulos
  2. HypothesisEngine gera hipoteses conceituais ANTES de programas
  3. TheoryBuilder encontra teorias unificadoras com economia
  4. O fluxo invertido: Observacao → Conceito → Hipotese → Programa funciona
  5. O sistema pensa em termos de explicacoes, nao de primitives
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, ".")

import numpy as np
from core.conceptual_discovery.concept_agent import ConceptAgent, ConceptAgentRegistry
from core.hypothesis_engine import HypothesisEngine, Hypothesis
from core.conceptual_discovery.theory_builder import TheoryBuilder
from core.program_induction.synthesis import ProgramSynthesizer, _compute_score
from core.program_induction.primitive import PrimitiveRegistry, Primitive
from core.program_induction.composition_graph import CompositionGraph
from core.program_induction.operator_grounding_bridge import OperatorGroundingBridge


PASS = "[PASS]"
FAIL = "[FAIL]"


# ============================================================
# CONFIG: Criamos ConceptAgents com comportamento real
# ============================================================

def _make_reflection_agent() -> ConceptAgent:
    """Reflection como agente cognitivo."""
    a = ConceptAgent("reflection", "spatial_transformation",
                     "trajetoria ou estado inverte apos interacao")

    # Predicoes testaveis
    a.add_prediction("orientation_inverted", True)
    a.add_prediction("area_preserved", True)
    a.add_prediction("color_preserved", True)
    a.add_prediction("object_count_preserved", True)

    # Assinaturas modais
    a.add_modal_signature("geometry", ["flip", "rotate", "mirror"],
                          invariants=["area_preserved", "connectivity_preserved"])
    a.add_modal_signature("audio", ["echo", "reverb"],
                          invariants=["amplitude_decays", "frequency_preserved"])
    a.add_modal_signature("physics", ["bounce", "reflect"],
                          invariants=["energy_preserved", "angle_preserved"])

    # Relacoes
    a.add_compatibility("rotation", 0.7)
    a.add_compatibility("symmetry", 0.8)
    a.add_contradiction("expansion")
    a.add_compatibility("spatial_transformation", 0.9)

    # Hipotese personalizada
    def _hypothesis_fn(obs):
        if not isinstance(obs, dict):
            return None
        inp = obs.get("grid")
        out = obs.get("target")
        if inp is None or out is None:
            return {"concept": "reflection", "reason": "no_grid_data",
                    "predictions": {}, "predictions_confirmed": 0.0}
        if inp.shape == out.shape:
            return {
                "concept": "reflection",
                "predictions": {"orientation_inverted": True,
                                "area_preserved": True,
                                "color_preserved": True},
                "predictions_confirmed": 1.0,
                "explanation": "shape preserved, reflection candidate",
            }
        return None

    a.set_hypothesis_fn(_hypothesis_fn)
    a.observe("geometry")
    a.observe("physics")
    a.observe("audio")
    return a


def _make_branching_agent() -> ConceptAgent:
    """Branching como agente cognitivo."""
    a = ConceptAgent("branching", "structural_division",
                     "estrutura se divide em multiplas ramificacoes")

    a.add_prediction("object_count_increases", True)
    a.add_prediction("parent_structure_preserved", True)
    a.add_prediction("recursive_pattern", True)

    a.add_modal_signature("biology", ["tree", "blood_vessels", "neurons"],
                          invariants=["hierarchy_preserved"])
    a.add_modal_signature("geometry", ["fractal", "recursive_tile"],
                          invariants=["self_similarity"])
    a.add_modal_signature("geography", ["river_delta", "tributaries"],
                          invariants=["flow_preserved"])

    a.add_compatibility("expansion", 0.6)
    a.add_compatibility("recursion", 0.9)
    a.add_contradiction("contraction")

    a.observe("biology")
    a.observe("geometry")
    return a


def _make_symmetry_agent() -> ConceptAgent:
    """Symmetry como agente cognitivo."""
    a = ConceptAgent("symmetry", "structural_preservation",
                     "estrutura preservada apos transformacao")

    a.add_prediction("structure_preserved", True)
    a.add_prediction("count_preserved", True)
    a.add_prediction("connectivity_preserved", True)

    a.add_modal_signature("geometry", ["bilateral", "rotational", "translational"],
                          invariants=["shape_preserved", "proportions_preserved"])
    a.add_modal_signature("biology", ["body_plan", "crystal"],
                          invariants=["organization_preserved"])

    a.add_compatibility("reflection", 0.9)
    a.add_compatibility("rotation", 0.8)
    a.add_contradiction("randomness")

    a.observe("geometry")
    a.observe("biology")
    return a


def _make_expansion_agent() -> ConceptAgent:
    """Expansion como agente cognitivo."""
    a = ConceptAgent("expansion", "scale_increase",
                     "algo aumenta de escala mantendo estrutura")

    a.add_prediction("object_count_increases", True)
    a.add_prediction("pattern_repeated", True)
    a.add_prediction("proportions_preserved", True)

    a.add_modal_signature("geometry", ["tile", "scale", "fractal"],
                          invariants=["pattern_preserved"])
    a.add_modal_signature("physics", ["thermal_expansion"],
                          invariants=["proportions_preserved"])

    a.add_compatibility("branching", 0.6)
    a.add_contradiction("contraction")

    a.observe("geometry")
    a.observe("physics")
    return a


# ============================================================
# TESTES
# ============================================================

def test_concept_agent_is_cognitive():
    """ConceptAgent tem comportamento preditivo, nao e rotulo."""
    agent = _make_reflection_agent()
    assert len(agent.predictions) > 0, "Deve ter predicoes"
    assert len(agent.modal_signatures) > 0, "Deve ter assinaturas modais"
    assert len(agent.compatible_with) > 0, "Deve ter relacoes"
    assert len(agent.contradicts) > 0, "Deve ter contradicoes"
    assert agent.is_viable, "Deve ser viavel"
    print(f"  ConceptAgent: {agent.name}")
    print(f"    Predicoes: {len(agent.predictions)}")
    print(f"    Modalidades: {list(agent.modal_signatures.keys())}")
    print(f"    Compatível com: {list(agent.compatible_with.keys())}")
    print(f"    Contradiz: {agent.contradicts}")


def test_concept_generates_hypothesis():
    """ConceptAgent gera hipotese conceitual a partir de observacao."""
    agent = _make_reflection_agent()
    inp = np.zeros((5, 5), dtype=np.int32)
    inp[1, 1:4] = 1
    out = np.fliplr(inp)

    obs = {"grid": inp, "target": out, "domain": "geometry"}
    hypothesis = agent.generate_hypothesis(obs)

    assert hypothesis is not None, "Deveria gerar hipotese"
    assert hypothesis["concept"] == "reflection"
    assert "predictions" in hypothesis
    print(f"  Hipotese gerada: {hypothesis['concept']}")
    print(f"    Predicoes: {len(hypothesis.get('predictions', {}))}")


def test_hypothesis_engine_generates_multiple():
    """HypothesisEngine gera hipoteses concorrentes."""
    engine = HypothesisEngine()
    for maker in [_make_reflection_agent, _make_branching_agent,
                  _make_symmetry_agent, _make_expansion_agent]:
        engine.register_agent(maker())

    inp = np.zeros((5, 5), dtype=np.int32)
    inp[1, 1:4] = 1
    out = np.fliplr(inp)

    hypotheses = engine.generate_hypotheses({"grid": inp, "target": out})
    assert len(hypotheses) > 1, "Deveria gerar multiplas hipoteses"
    assert any(h.concept == "reflection" for h in hypotheses)

    for h in hypotheses:
        print(f"  Hipotese: {h.concept} (conf={h.confidence:.2f}, "
              f"falsif={h.falsifiability:.2f}, prims={h.expected_primitives[:2]})")


def test_hypothesis_selection():
    """HypothesisEngine seleciona melhor hipotese."""
    engine = HypothesisEngine()
    for maker in [_make_reflection_agent, _make_symmetry_agent,
                  _make_expansion_agent]:
        engine.register_agent(maker())

    inp = np.zeros((5, 5), dtype=np.int32)
    inp[1, 1:4] = 1
    out = np.fliplr(inp)

    hypotheses = engine.generate_hypotheses({"grid": inp, "target": out})
    best = engine.select_best(hypotheses)

    assert best is not None, "Deveria selecionar uma hipotese"
    print(f"  Melhor hipotese: {best.concept} (score composto)")
    print(f"    Confianca: {best.confidence:.2f}")
    print(f"    Falsificabilidade: {best.falsifiability:.2f}")


def test_hypothesis_to_guidance():
    """Hipotese vira guia para o ProgramSynthesizer."""
    engine = HypothesisEngine()
    engine.register_agent(_make_reflection_agent())

    inp = np.zeros((5, 5), dtype=np.int32)
    inp[1, 1:4] = 1
    out = np.fliplr(inp)

    hypotheses = engine.generate_hypotheses({"grid": inp, "target": out})
    hyp = hypotheses[0]
    guidance = engine.hypothesis_to_guidance(hyp)

    assert "concept" in guidance
    assert "primitive_weights" in guidance
    assert "invariant_type" in guidance
    assert guidance["concept"] == "reflection"
    print(f"  Guidance gerada:")
    for k, v in guidance.items():
        print(f"    {k}: {v}")


def test_theory_builder_economy():
    """TheoryBuilder encontra teorias com economia de teorias."""
    tb = TheoryBuilder(economy_threshold=0.2)

    # Cada chamada de observe_concept incrementa o contador de co-ocorrencia
    # entre o conceito novo e todos os existentes. Precisamos de 3+ co-ocorrencias
    # para formar uma teoria. Entao chamamos observe_concept em pares repetidos.
    for _ in range(4):
        tb.observe_concept("reflection", {"geometry", "physics"})
        tb.observe_concept("symmetry", {"geometry", "biology"})

    theories = tb.build_theories()
    assert len(theories) > 0, "Deveria criar teorias"

    best = tb.select_best_theory(theories)
    assert best is not None
    assert best.economy > 0

    print(f"  Teorias criadas: {len(theories)}")
    for t in theories:
        print(f"    {t.name}: economia={t.economy:.3f}, "
              f"conceitos={len(t.concepts)}, "
              f"dominios={len(t.domains)}")
    print(f"  Melhor teoria: {best.name} (economia={best.economy:.3f})")


def test_full_cycle():
    """Ciclo completo invertido: Observacao → Conceito → Hipotese → Guidance."""
    engine = HypothesisEngine()
    for maker in [_make_reflection_agent, _make_symmetry_agent,
                  _make_branching_agent, _make_expansion_agent]:
        engine.register_agent(maker())

    inp = np.zeros((5, 5), dtype=np.int32)
    inp[1, 1:4] = 1
    out = np.fliplr(inp)

    # 1. Observacao → Hipoteses
    hypotheses = engine.generate_hypotheses({"grid": inp, "target": out})
    assert len(hypotheses) > 0

    # 2. Selecao
    best = engine.select_best(hypotheses)
    assert best is not None

    # 3. Hipotese → Guidance
    guidance = engine.hypothesis_to_guidance(best)
    assert "primitive_weights" in guidance

    # 4. Guidance → Synthesizer
    reg = PrimitiveRegistry()
    for name, stype in [
        ("flip_h", "flip_horizontal"), ("flip_v", "flip_vertical"),
        ("rot90", "rotate_90"), ("tile", "tile"), ("id", "identity"),
    ]:
        p = Primitive(name=name, structure_type=stype,
                       solver_fn=lambda i, o: {} if i.shape == o.shape else None)
        if stype == "flip_horizontal":
            p._execute = lambda g, params: np.fliplr(g.copy())
        elif stype == "flip_vertical":
            p._execute = lambda g, params: np.flipud(g.copy())
        elif stype == "rotate_90":
            p._execute = lambda g, params: np.rot90(g.copy(), k=-1)
        elif stype == "identity":
            p._execute = lambda g, params: g.copy()
        p.confidence = 0.5
        reg.register(p)

    cg = CompositionGraph()
    og = OperatorGroundingBridge(reg, cg)
    syn = ProgramSynthesizer(reg, composition_graph=cg, grounding_bridge=og)

    # Guidance weights
    weights = guidance["primitive_weights"]
    results = syn.synthesize(inp, out, primitive_weights=weights)

    assert len(results) > 0, "Deveria encontrar programas"
    assert any(p.validated for p in results), "Deveria ter solucao"

    print(f"  Ciclo completo: {best.concept} → programa {results[0].name}")
    print(f"    Score: {results[0].score}, Validado: {results[0].validated}")


def test_concept_is_independent_of_modality():
    """Conceito reflection funciona em geometria E audo."""
    ref = _make_reflection_agent()
    sym = _make_symmetry_agent()

    # Reflection tem assinaturas em 3 modalidades
    assert ref.has_modality("geometry")
    assert ref.has_modality("audio")
    assert ref.has_modality("physics")

    geo_sig = ref.get_modality_signature("geometry")
    audio_sig = ref.get_modality_signature("audio")
    assert geo_sig.modality == "geometry"
    assert audio_sig.modality == "audio"
    assert geo_sig.observable_features != audio_sig.observable_features

    print(f"  Reflection em geometry: {geo_sig.observable_features}")
    print(f"  Reflection em audio: {audio_sig.observable_features}")
    print(f"  Conceito e independente da modalidade!")


if __name__ == "__main__":
    print("=== Arquitetura Concept-Centric ===\n")
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        print(f"{name}...")
        try:
            fn()
            print(f"  {PASS}\n")
        except Exception as e:
            print(f"  {FAIL}: {e}\n")
            import traceback
            traceback.print_exc()
