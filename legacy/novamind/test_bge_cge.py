"""
Teste: CGE com BGE 768D (vs hash fallback)

Mostra que com BGE, cat e dog tem vizinhos reais e inibicao lateral funciona.
"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.bge_encoder import BGEEncoder

bge = BGEEncoder()
cge = CognitiveGraphEngine(device="cuda", bge_encoder=bge)

sentences = [
    "The cat sits on the mat near the warm fireplace",
    "A dog runs through the green forest chasing a squirrel",
    "The cat and dog play together in the sunny garden",
]
for s in sentences:
    created = cge.write_text_concepts(s, 0.5, 0.3)
    print(f"Criados {created} nos para: {s}")

print(f"\nNos no CGE (BGE-based): {len(cge.nodes)}")
for nid, nd in sorted(cge.nodes.items()):
    print(f"  {nid}: conf={nd['confidence']:.2f}")

# Similaridade cat vs dog NO ESPACO BGE
print("\n--- Similaridade (BGE 768D -> 10000D bipolar) ---")
cat_vec = cge.nodes["cat"]["vector"].cpu()
dog_vec = cge.nodes["dog"]["vector"].cpu()
cat_dog = float(torch.cosine_similarity(cat_vec.unsqueeze(0), dog_vec.unsqueeze(0)))
print(f"Cos(cat, dog) = {cat_dog:.4f}")

# Comparar palavras relacionadas vs nao relacionadas
pairs = [("cat", "dog"), ("cat", "squirrel"), ("cat", "fireplace"),
         ("runs", "chasing"), ("forest", "garden"), ("sunny", "warm"),
         ("cat", "forest")]  # menos relacionado
for a, b in pairs:
    if a in cge.nodes and b in cge.nodes:
        va = cge.nodes[a]["vector"].cpu()
        vb = cge.nodes[b]["vector"].cpu()
        sim = float(torch.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)))
        print(f"  {a:12s} vs {b:12s}: {sim:.4f}")

# Inibicao lateral (agora com vizinhanca REAL)
print("\n--- Inibicao Lateral ---")
cge.lateral_inhibition(strength=0.5, sim_threshold=0.30)
inhibited = [(n, nd["confidence"]) for n, nd in cge.nodes.items() if nd["confidence"] < 0.95]
if inhibited:
    for nid, conf in sorted(inhibited, key=lambda x: x[1]):
        print(f"  {nid}: conf={conf:.2f} (INIBIDO)")
else:
    print("  Nenhum no inibido (proxemia insuficiente)")

# Spreading activation
print("\n--- Spreading Activation ---")
act = cge.spread_activation(["cat"], decay=0.5, max_hops=2)
for nid, val in sorted(act.items(), key=lambda x: -x[1]):
    if val > 0:
        print(f"  {nid}: act={val:.2f}")

# Teste de pergunta simples (como BGE melhora)
print("\n--- Teste: encaixe semantico ---")
# "What is gravity?" antes com spaCy dava lixo
# Com BGE, a pergunta e o texto sobre gravity tem similaridade real
grav = bge.similarity("gravity", "gravitational force")
print(f"BGE sim('gravity', 'gravitational force') = {grav:.4f}")
print("(Antes com spaCy: ~0.01)")
