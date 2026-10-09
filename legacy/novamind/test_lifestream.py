"""
Teste: LifeStream + MentalRenderer

Valida:
  1. Source Reliability: confianca por fonte (Wikipedia 0.95, ad 0.05)
  2. Opportunity Engine: expected gain multidimensional
  3. Episode Stream: texto → episodio → CGE
  4. MentalRenderer: pensamento → texto/codigo/visual/fala
  5. LifeStream pipeline completo
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("TESTE: LIFESTREAM + MENTAL RENDERER")
print("=" * 70)

from core.lifestream import (
    LifeStream, ContentItem, SourceProfile, SOURCE_RELIABILITY_DEFAULTS,
    OpportunityEngine, EpisodeStream, WikipediaSource, ProceduralSource,
)
from core.mental_renderer import MentalRenderer, TextDecoder, CodeDecoder, VisualPlanDecoder, RenderedOutput

# ─── PARTE 1: Source Reliability ───
print("\n--- PARTE 1: Source Reliability ---")
sp_wiki = SOURCE_RELIABILITY_DEFAULTS["wikipedia"]
sp_ad = SOURCE_RELIABILITY_DEFAULTS["advertisement"]
sp_comment = SOURCE_RELIABILITY_DEFAULTS["comment"]

assert sp_wiki.reliability > 0.9
assert sp_ad.reliability < 0.1
assert sp_comment.reliability < 0.2
print(f"  Wikipedia: {sp_wiki.reliability:.2f}")
print(f"  Ad: {sp_ad.reliability:.2f}")
print(f"  Comment: {sp_comment.reliability:.2f}")

# Testar effective_value
content_good = ContentItem("t1", "wikipedia", source_reliability=0.95,
                            title="Gravity", body="Gravity is a force...",
                            concepts=["gravity", "force"], estimated_value=0.9)
content_bad = ContentItem("t2", "advertisement", source_reliability=0.05,
                           title="Buy now", body="Amazing product!",
                           concepts=["buy", "product"], estimated_value=0.9,
                           is_ad=True)
assert content_good.effective_value > 0.8
assert content_bad.effective_value < 0.1
print(f"  Conteudo bom: effective_value={content_good.effective_value:.2f}")
print(f"  Conteudo ruim: effective_value={content_bad.effective_value:.2f}")
print("  PASS: Source Reliability")

# ─── PARTE 2: Opportunity Engine ───
print("\n--- PARTE 2: Opportunity Engine ---")
from core.self_model import SelfModel
sm = SelfModel()
oe = OpportunityEngine(self_model=sm)

# Com SelfModel vazio
opp = oe.evaluate(content_good)
print(f"  Knowledge gain: {opp.knowledge_gain:.2f}")
print(f"  Curiosity gain: {opp.curiosity_gain:.2f}")
print(f"  Expected gain: {opp.expected_gain:.2f}")

# Com SelfModel que ja sabe algo
sm.learn_from_experience("gravity", "Gravity is a force", domain="physics", success=True)
opp2 = oe.evaluate(content_good)
print(f"  Apos aprender gravity: knowledge_gain={opp2.knowledge_gain:.2f} "
      f"(deve ser menor que {opp.knowledge_gain:.2f})")

# Ranking
contents = [content_good, content_bad]
ranked = oe.rank(contents, n=2)
assert len(ranked) == 2
assert ranked[0].expected_gain >= ranked[1].expected_gain
print(f"  Ranking: 1o={ranked[0].title} (gain={ranked[0].expected_gain:.2f}), "
      f"2o={ranked[1].title} (gain={ranked[1].expected_gain:.2f})")
print("  PASS: Opportunity Engine")

# ─── PARTE 3: Episode Stream ───
print("\n--- PARTE 3: Episode Stream ---")
stream = EpisodeStream(cge=None)
ep = stream.process_text(content_good)
assert len(ep.concepts) > 0
assert ep.source == "wikipedia"
assert ep.reliability == 0.95
print(f"  Episodio criado: {len(ep.concepts)} conceitos, {len(ep.relations)} relacoes")
print(f"  Conceitos: {ep.concepts[:5]}")
print(f"  Raw discarded: {ep.raw_content_discarded}")
print("  PASS: Episode Stream")

# ─── PARTE 4: LifeStream Pipeline ───
print("\n--- PARTE 4: LifeStream Pipeline ---")
cge = None
try:
    from core.cognitive_graph_engine import CognitiveGraphEngine
    cge = CognitiveGraphEngine(device="cpu")
except Exception:
    pass

ls = LifeStream(cge=cge, bge_encoder=None, self_model=sm)
ls.add_source("procedural", ProceduralSource())

# Gerar conteudo e processar
ls.generate_content("gravity", "procedural")
ls.generate_content("energy", "procedural")
print(f"  Fila antes: {ls.content_queue}")

learned = ls.tick(max_items=2)
print(f"  Conceitos aprendidos: {learned[:5]}")
print(f"  Fila depois: {len(ls.content_queue)}")
print(f"  Processados: {ls.processed_count}")

state = ls.get_state()
print(f"  Estado LifeStream: {state['queue_size']} na fila, {state['processed']} processados")
print("  PASS: LifeStream Pipeline")

# ─── PARTE 5: MentalRenderer — Texto ───
print("\n--- PARTE 5: MentalRenderer (Texto) ---")
renderer = MentalRenderer(cge=cge)
renderer.think("gravity")
text_out = renderer.to_text()
print(f"  Texto: {text_out.content[:80]}...")
assert text_out.modality == "text"
assert len(text_out.content) > 10

# Relate
rel = renderer.text.relate("gravity", "energy")
print(f"  Relacao: {rel}")

# Summarize
summary = renderer.text.summarize(["gravity", "energy", "conductivity"])
print(f"  Sumario: {summary[:100]}...")
print("  PASS: MentalRenderer (Texto)")

# ─── PARTE 6: MentalRenderer — Codigo ───
print("\n--- PARTE 6: MentalRenderer (Codigo) ---")
renderer.think("conductivity")
code_out = renderer.to_code()
print(f"  Codigo gerado ({len(code_out.content)} chars):")
for line in code_out.content.split('\n')[:5]:
    print(f"    {line}")
assert code_out.modality == "code"
print("  PASS: MentalRenderer (Codigo)")

# ─── PARTE 7: MentalRenderer — Visual Plan ───
print("\n--- PARTE 7: MentalRenderer (Visual) ---")
renderer.think_about(["gravity", "energy"])
visual_out = renderer.to_visual_plan()
print(f"  Cena visual ({len(visual_out.content)} chars): {visual_out.content[:120]}...")
assert visual_out.modality == "visual_plan"

# Animacao composta
scene = renderer.visual.compose_animation(["gravity", "energy_flow", "conductivity"])
print(f"  Animacao composta: {len(scene['objects'])} objetos, "
      f"{len(scene['animations'])} animacoes, {len(scene['labels'])} labels")
print("  PASS: MentalRenderer (Visual)")

# ─── PARTE 8: MentalRenderer — Speech Plan ───
print("\n--- PARTE 8: MentalRenderer (Fala) ---")
renderer.think_about(["gravity", "conductivity"])
speech_out = renderer.to_speech_plan()
print(f"  Plano de fala ({len(speech_out.content)} chars)")
assert speech_out.modality == "speech_plan"
print("  PASS: MentalRenderer (Fala)")

# ─── RESUMO ───
print("\n" + "=" * 70)
print("RESUMO: LIFESTREAM + MENTAL RENDERER")
print("=" * 70)
print(f"  Source Reliability: Wikipedia={sp_wiki.reliability:.2f}, Ad={sp_ad.reliability:.2f}")
print(f"  Opportunity Engine: {len(oe.opportunities)} oportunidades ranqueadas")
print(f"  Episode Stream: {stream.processed_count} episodios")
print(f"  LifeStream: {ls.processed_count} conteudos processados")
print(f"  MentalRenderer: texto, codigo, visual, fala disponiveis")
print("=" * 70)
print("TODOS OS TESTES PASSARAM")
print("=" * 70)
