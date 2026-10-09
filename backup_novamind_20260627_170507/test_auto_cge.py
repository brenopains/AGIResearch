"""
Teste: Pipeline Completo CGE
- Escrita automatica no CGE (write_text_concepts)
- Inibicao lateral
- Tag emocional
- Recall por emocao
"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from core.cognitive_graph_engine import CognitiveGraphEngine

print("=" * 60)
print("Teste: Pipeline Completo CGE")
print("=" * 60)

cge = CognitiveGraphEngine(device="cuda" if torch.cuda.is_available() else "cpu")

# Simular 3 pensamentos (como se viessem do Maestro)
sentences = [
    "The cat sits on the mat near the warm fireplace",
    "A dog runs through the green forest chasing a squirrel",
    "The cat and dog play together in the sunny garden",
]

for i, sent in enumerate(sentences):
    valence = [0.3, 0.7, 0.9][i]  # cada vez mais positivo
    arousal = [0.2, 0.6, 0.4][i]
    created = cge.write_text_concepts(sent, valence, arousal)
    print(f"\nSentenca {i+1}: '{sent}'")
    print(f"  Nos criados: {created}")

print(f"\n--- Nos no CGE: {len(cge.nodes)} ---")
for nid, ndata in sorted(cge.nodes.items()):
    print(f"  {nid}: conf={ndata['confidence']:.2f}, "
          f"val={ndata['emotional_valence']:.1f}, "
          f"aro={ndata['emotional_arousal']:.1f}")

print(f"\n--- Arestas: {len(cge.edges)} ---")
for e in cge.edges[:15]:
    print(f"  {e['from']} --causes--> {e['to']}")

# Testar recall emocional
print("\n--- Recall por emocao (valencia > 0.5) ---")
recall = cge.recall_by_emotion(target_valence=0.8, top_k=5)
for r in recall:
    print(f"  {r['id']}: conf={r['confidence']}, val={r['valence']}")

# Testar inibicao lateral
print("\n--- Inibicao Lateral ---")
# Dog e cat compartilham vizinhos (forest, garden, play, run, squirrel)
# A inibicao vai reduzir a confianca do mais fraco
cge.lateral_inhibition(strength=0.5, sim_threshold=0.30)
print("Apos inibicao:")
for nid, ndata in sorted(cge.nodes.items()):
    if ndata["confidence"] < 0.95:
        print(f"  {nid}: conf={ndata['confidence']:.2f} (INIBIDO)")

# Testar spreading activation
print("\n--- Spreading Activation ---")
act = cge.spread_activation(["cat"], decay=0.5, max_hops=2)
for nid, val in sorted(act.items(), key=lambda x: -x[1]):
    if val > 0:
        print(f"  {nid}: {val:.2f}")

print("\nTudo OK!")
