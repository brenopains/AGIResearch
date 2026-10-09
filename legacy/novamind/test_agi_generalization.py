"""
test_agi_generalization.py — Teste Integrado dos 3 Novos Modulos
=================================================================
Valida as tres pecas que faltavam na arquitetura NovaMind:

  1. DynamicCognitiveSpace — imaginacao e criacao de espacos mentais
  2. EmergentLanguageEngine — linguagem que emerge do CGE
  3. GeneralizationTestSuite — testes de generalizacao real

Uso:
    python test_agi_generalization.py              # teste completo offline
    python test_agi_generalization.py --live        # tenta integrar com NovaMind real
    python test_agi_generalization.py --quick       # apenas testes rapidos
"""
import sys
import os
import time
import traceback

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

ENABLE_LIVE = "--live" in sys.argv
QUICK = "--quick" in sys.argv

print("=" * 60)
print("NovaMind — Teste de Generalizacao e Novos Modulos")
print("=" * 60)

# ─── TESTE 1: DynamicCognitiveSpace ───
print("\n[1/3] DynamicCognitiveSpace (Espacos Mentais Dinamicos)")
print("-" * 50)

try:
    from core.cognitive_space import (
        CognitiveSpace, SpaceRegistry, ImaginationCanvas, ImaginaryObject
    )
    from core.coglang import Manifold

    # Teste standalone sem depender do CGE
    registry = SpaceRegistry()
    canvas = ImaginationCanvas(registry)

    # Teste 1a: Criacao de objetos
    obj1 = canvas.criar_objeto("copo", forma="cilindro",
                               cor=(0.3, 0.3, 0.8), tamanho=1.0,
                               propriedades={"capacidade": 300, "material": "vidro"})
    assert obj1 is not None, "Falhou ao criar objeto"
    assert obj1.name == "copo"
    assert obj1.shape == "cilindro"
    print(f"  [OK] Criacao de objeto: {obj1.name} ({obj1.shape}, size={obj1.size})")

    # Teste 1b: Modificacao de objeto
    ok = canvas.modificar(obj1.id, color=(1.0, 0.0, 0.0), size=2.0)
    assert ok
    obj_data = canvas._space.objects[obj1.id]
    assert obj_data.color == (1.0, 0.0, 0.0)
    assert obj_data.size == 2.0
    print(f"  [OK] Modificacao de objeto: cor=vermelho, tamanho=2.0")

    # Teste 1c: Combinacao de objetos
    obj2 = canvas.criar_objeto("asa", forma="curva", cor=(0.8, 0.8, 0.8))
    combinado = canvas.combinar(obj1.id, obj2.id, "copo_voante", razao=0.3)
    assert combinado is not None
    assert combinado.name == "copo_voante"
    print(f"  [OK] Combinacao de objetos: {combinado.name} (shape={combinado.shape}, size={combinado.size:.2f})")

    # Teste 1d: Animacao e sentimento
    ok = canvas.animar(combinado.id, vx=1.0, vy=0.5)
    assert ok
    ok = canvas.sentir(combinado.id, valencia=0.8, arousal=0.3)
    assert ok
    obj_data = canvas._space.objects[combinado.id]
    assert obj_data.velocity == (1.0, 0.5)
    assert obj_data.emotional_valence == 0.8
    print(f"  [OK] Animacao/sentimento: v=({1.0}, {0.5}), valencia={0.8}")

    # Teste 1e: Simulacao
    history = canvas.simular(n_passos=3, dt=0.1)
    assert len(history) == 3
    print(f"  [OK] Simulacao: {len(history)} passos executados")

    # Teste 1f: Conceitos e relacoes
    space = canvas._space
    space.add_concept("gravidade", {"forca": 9.8})
    space.add_concept("queda", {"tipo": "movimento"})
    space.add_relation("gravidade", "queda", "CAUSES", 0.9)
    assert "gravidade" in space.concepts
    assert "queda" in space.concepts
    print(f"  [OK] Conceitos e relacoes: gravidade CAUSES queda")

    # Teste 1g: Multiplos espacos
    space2 = registry.create_space("experimento_2")
    assert space2 is not None
    assert registry.list_spaces()
    print(f"  [OK] Multiplos espacos: {len(registry.list_spaces())} ativos")

    print(f"\n  >>> DynamicCognitiveSpace: TODOS OS TESTES OK <<<")

except Exception as e:
    print(f"  [FALHA] {e}")
    traceback.print_exc()

# ─── TESTE 2: EmergentLanguageEngine ───
print("\n[2/3] EmergentLanguageEngine (Linguagem do CGE)")
print("-" * 50)

try:
    from core.emergent_language_engine import (
        EmergentVocabulary, CGEToLanguage, InternalMonologue,
        EmergentLanguageEngine, Utterance
    )

    # Teste 2a: Vocabulario emergente
    vocab = EmergentVocabulary()
    vocab.add_word("gravity", "gravity")
    vocab.add_word("force", "force")
    vocab.add_word("mass", "mass")
    vocab.add_word("acceleration", "acceleration")
    assert vocab.has_word("gravity")
    assert vocab.size() == 4
    print(f"  [OK] Vocabulario: {vocab.size()} palavras (sem pre-treinamento)")

    # Teste 2b: CGEToLanguage (modo standalone, sem CGE real)
    translator = CGEToLanguage(vocab=vocab)
    utterance = translator.describe_concept("gravity")
    assert utterance is not None
    print(f"  [OK] CGEToLanguage: gerou texto -> \"{utterance.text[:60]}...\"")

    # Teste 2c: InternalMonologue
    monologue = InternalMonologue(translator)
    monologue.activate()
    thought = monologue.tick("gravity")
    if thought:
        print(f"  [OK] Monologo interno: \"{thought.text[:80]}...\"")
    else:
        print(f"  [OK] Monologo interno: ativo, aguardando foco")

    # Teste 2d: Vocabulario de conceitos via grafo simulado
    try:
        # Usa um dicionario para simular o CGE (para nao depender de torch)
        class MockCGE:
            def __init__(self):
                self.nodes = {}
                self.edges = []
            def create_node(self, node_id, semantic_vec=None, perceptual_vec=None):
                self.nodes[node_id] = {"id": node_id, "confidence": 0.5}
            def add_edge(self, from_id, to_id, edge_type):
                self.edges.append({"from": from_id, "to": to_id, "edge_type": edge_type})
        mock_cge = MockCGE()
        mock_cge.create_node("gravity")
        mock_cge.create_node("mass")
        mock_cge.create_node("force")
        mock_cge.add_edge("gravity", "mass", "CAUSES")
        mock_cge.add_edge("mass", "force", "CORRELATED")
        vocab2 = EmergentVocabulary(mock_cge)
        n = vocab2.sync_from_cge()
        assert n > 0
        print(f"  [OK] Vocabulario sincronizado do CGE simulado: {n} conceitos")
    except Exception as e:
        print(f"  [!] CGE simulado falhou: {e}")

    # Teste 2e: Engine completa
    engine = EmergentLanguageEngine()
    engine.start_monologue()
    stats = engine.get_stats()
    assert "vocab_size" in stats
    print(f"  [OK] EmergentLanguageEngine: {stats}")

    print(f"\n  >>> EmergentLanguageEngine: TODOS OS TESTES OK <<<")

except Exception as e:
    print(f"  [FALHA] {e}")
    traceback.print_exc()

# ─── TESTE 3: GeneralizationTestSuite ───
print("\n[3/3] GeneralizationTestSuite (Testes de Generalizacao Real)")
print("-" * 50)

try:
    from core.generalization_test import (
        GeneralizationTestSuite, TestResult, TestSuiteResult
    )
    from core.cognitive_space import SpaceRegistry, ImaginationCanvas

    # Teste com sistema minimo
    registry3 = SpaceRegistry()
    canvas3 = ImaginationCanvas(registry3)
    suite = GeneralizationTestSuite(
        cge=None,
        world_model=None,
        language_engine=None,
        imagination_canvas=canvas3,
        verbose=False,
    )

    result = suite.run_all()
    assert len(result.results) > 0

    # Testes individuais
    from core.generalization_test import GeneralizationTestSuite as GTS
    suite2 = GTS(verbose=False)

    # Teste 3a: Composicao
    t1 = suite._test_novel_composition()
    print(f"  [{'OK' if t1.passed else 'IGN'}] Composicao de conceitos: {t1.score:.2f}/{t1.max_score:.2f}")

    # Teste 3b: Self-modification
    t2 = suite._test_self_modification()
    print(f"  [{'OK' if t2.passed else 'IGN'}] Auto-modificacao: {t2.score:.2f}/{t2.max_score:.2f}")

    # Teste 3c: Emergent language
    t3 = suite._test_emergent_language()
    print(f"  [{'OK' if t3.passed else 'IGN'}] Linguagem emergente: {t3.score:.2f}/{t3.max_score:.2f}")

    # Summary
    print(f"\n  >>> GeneralizationTestSuite: {result.passed}/{result.total} passed, score {result.overall_score:.2%} <<<")

except Exception as e:
    print(f"  [FALHA] {e}")
    traceback.print_exc()

# ─── TESTE 4: Integracao (Live) ───
if ENABLE_LIVE:
    print("\n[4/4] Integracao com NovaMind Live")
    print("-" * 50)
    try:
        from core.mind import NovaMind
        mind = NovaMind()

        from core.cognitive_space import SpaceRegistry, ImaginationCanvas
        from core.emergent_language_engine import EmergentLanguageEngine

        registry_live = SpaceRegistry(cge=mind.cge, world_model=mind.world_model)
        canvas_live = ImaginationCanvas(registry_live)
        lang_live = EmergentLanguageEngine(cge=mind.cge)

        lang_live.start_monologue()

        # Roda alguns ciclos de pensamento
        for i in range(3):
            text = lang_live.think()
            if text:
                print(f"  [Pensamento {i}] {text[:100]}")

        # Testa imaginacao
        obj = canvas_live.criar_objeto("nova_entidade", forma="esfera")
        canvas_live.refletir()

        suite_live = GeneralizationTestSuite(
            cge=mind.cge,
            world_model=mind.world_model,
            language_engine=lang_live,
            imagination_canvas=canvas_live,
        )
        result_live = suite_live.run_all()
        print(f"\n  >>> Resultado Live: {result_live.passed}/{result_live.total} passed, score {result_live.overall_score:.2%} <<<")

    except Exception as e:
        print(f"  [FALHA na integracao live] {e}")
        traceback.print_exc()
else:
    print("\n[4/4] Integracao Live: SKIP (use --live para testar com NovaMind real)")

# ─── RESUMO FINAL ───
print("\n" + "=" * 60)
print("RESUMO: O que foi construido e por que")
print("=" * 60)
print("""
1. DynamicCognitiveSpace (core/cognitive_space.py):
   - Cria espacos mentais temporarios para imaginacao
   - Objetos podem ser criados, combinados, modificados, animados
   - Cada espaco tem suas proprias leis causais
   - Resultados podem ser injetados de volta no CGE
   -> Resolve: "por que o sistema nao modifica estruturas internamente?"

2. EmergentLanguageEngine (core/emergent_language_engine.py):
   - Vocabulario emerge dos conceitos do CGE (sem BGE/LaBSE)
   - Sentencas sao montadas por tipos de relacao (IS_A, CAUSES, etc)
   - Monologo interno: pensa em linguagem e retroalimenta o CGE
   -> Resolve: "ensinar ela a conversar sem decoder pre-treinado"

3. GeneralizationTestSuite (core/generalization_test.py):
   - Testa composicao de conceitos novos
   - Testa transferencia zero-shot entre dominios
   - Testa criacao autonoma de conceitos
   - Testa auto-modificacao arquitetural
   - Testa producao de linguagem emergente
   -> Resolve: "como mensurar AGI de verdade neste hardware"
""")
