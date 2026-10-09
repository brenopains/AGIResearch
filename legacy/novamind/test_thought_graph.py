"""Testa o ThoughtGraph com pergunta real sobre cognicao humana."""
import sys
sys.path.insert(0, ".")

from core.cognitive_graph_engine import CognitiveGraphEngine
from core.multilingual_encoder import MultilingualEncoder
from core.cognition import MetaCognition
from core.thought_graph import ThoughtTracer, ThoughtGraph
from core.knowledge_data import KNOWLEDGE, CODE_CURRICULUM

print("=" * 70)
print("THOUGHT GRAPH: MApeamento completo do fluxo de pensamento")
print("=" * 70)

# Carrega ou prepara CGE com conhecimento
cge = CognitiveGraphEngine(hdc_memory=None, device="cuda")
enc = MultilingualEncoder()
meta = MetaCognition()

# Se CGE estiver vazio, injeta conhecimento novamente
if len(cge.nodes) < 50:
    print("[ThoughtGraph] CGE vazio, injetando conhecimento...")
    pass  # KNOWLEDGE ja importado de core.knowledge_data
    for module_name, text in {**KNOWLEDGE, **{k: v for k, v in CODE_CURRICULUM.items()}}.items():
        cge.write_text_concepts(text)
    print(f"  CGE apos injecao: {len(cge.nodes)} nodes")

tracer = ThoughtTracer(cge=cge, encoder=enc, meta=meta)

# ===========================================================
# PERGUNTA: COMO HUMANOS PENSAM?
# ===========================================================
graph = tracer.trace("como humanos pensam?")

print("\n" + graph.render())
print("\n" + graph.render_flow())

print("\n" + "=" * 70)
print("ESTATISTICAS DO GRAFO")
print("=" * 70)
stats = graph.get_stats()
print(f"  Nos: {stats['n_nodes']}")
print(f"  Arestas: {stats['n_edges']}")
print(f"  Fluxo total: {stats['total_flow']}")
print(f"  Tipos de no:")
for t, c in stats["node_types"].items():
    print(f"    {t}: {c}")

# Salva grafo
graph.save("D:/novamind_data/output/thought_graph_como_humanos_pensam.json")
print(f"\n[SAVE] Grafo salvo em D:/novamind_data/output/")

# ===========================================================
# PERGUNTA: O QUE E CONSCIENCIA?
# ===========================================================
print("\n" + "=" * 70)
print("SEGUNDA PERGUNTA: O QUE E CONSCIENCIA?")
print("=" * 70)
graph2 = tracer.trace("o que e consciencia?")
print(graph2.render_flow())

# ===========================================================
# PERGUNTA: O QUE SAO NEUROTRANSMISSORES?
# ===========================================================
print("\n" + "=" * 70)
print("TERCEIRA PERGUNTA: O QUE SAO NEUROTRANSMISSORES?")
print("=" * 70)
graph3 = tracer.trace("o que sao neurotransmissores")
print(graph3.render_flow())
