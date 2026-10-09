"""
Teste: Forja Consolidation

Objetivo: Verificar se a Forja descobre padroes transitivos
e cria arestas diretas (consolidacao estilo hipocampo).

Cenario:
  FLORB → lightweight, strong, flexible
  lightweight → structure_role
  strong → structure_role

A Forja deve consolidar: FLORB → structure_role (direta)
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.cognitive_graph_engine import CognitiveGraphEngine

DIM = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Teste: Forja Consolidation")
print("=" * 60)

# 1. Criar CGE e Answerer
cge = CognitiveGraphEngine(device=DEVICE)


# 2. Criar HDVs aleatorios para cada conceito
torch.manual_seed(42)
florb = torch.sign(torch.randn(DIM, device=DEVICE))
trek = torch.sign(torch.randn(DIM, device=DEVICE))
submarine = torch.sign(torch.randn(DIM, device=DEVICE))

lightweight = torch.sign(torch.randn(DIM, device=DEVICE))
strong = torch.sign(torch.randn(DIM, device=DEVICE))
flexible = torch.sign(torch.randn(DIM, device=DEVICE))
heat_absorbing = torch.sign(torch.randn(DIM, device=DEVICE))
insulating = torch.sign(torch.randn(DIM, device=DEVICE))
dark = torch.sign(torch.randn(DIM, device=DEVICE))
structure_role = torch.sign(torch.randn(DIM, device=DEVICE))
thermal_role = torch.sign(torch.randn(DIM, device=DEVICE))

# 3. Adicionar nos ao CGE
concepts = {
    "FLORB": florb, "TREK": trek, "submarine": submarine,
    "lightweight": lightweight, "strong": strong, "flexible": flexible,
    "heat_absorbing": heat_absorbing, "insulating": insulating, "dark": dark,
    "structure_role": structure_role, "thermal_role": thermal_role,
}
for name, vec in concepts.items():
    cge.create_node(name, vec)

# 4. Adicionar arestas CAUSES (FLORB → properties → roles)
causes_edge = cge.EDGE_CAUSES

# FLORB → suas propriedades
cge.add_edge("FLORB", "lightweight", causes_edge)
cge.add_edge("FLORB", "strong", causes_edge)
cge.add_edge("FLORB", "flexible", causes_edge)

# Propriedades → structure_role (leve, forte, flexivel sao para estrutura)
cge.add_edge("lightweight", "structure_role", causes_edge)
cge.add_edge("strong", "structure_role", causes_edge)

# TREK → suas propriedades
cge.add_edge("TREK", "heat_absorbing", causes_edge)
cge.add_edge("TREK", "insulating", causes_edge)
cge.add_edge("TREK", "dark", causes_edge)

# Propriedades → thermal_role (absorve calor, isola = funcao termica)
cge.add_edge("heat_absorbing", "thermal_role", causes_edge)
cge.add_edge("insulating", "thermal_role", causes_edge)

print("\n--- Grafo antes da consolidacao ---")
print(cge.get_scene_text())

# 5. Testar se as arestas diretas JA EXISTEM (nao deveriam)
def has_direct_edge(cge, from_id, to_id):
    return any(e["from"] == from_id and e["to"] == to_id for e in cge.edges)

print("\n--- Teste: arestas diretas antes da consolidacao ---")
print(f"FLORB -> structure_role direta? {has_direct_edge(cge, 'FLORB', 'structure_role')}")
print(f"TREK -> thermal_role direta? {has_direct_edge(cge, 'TREK', 'thermal_role')}")

# 6. Testar similaridade da cadeia antes da consolidacao
print("\n--- Similaridade da cadeia transitiva ---")
vec_florb = florb
vec_structure = structure_role
direct_vec = torch.sign(vec_florb * causes_edge.to(DEVICE) * vec_structure)
holo = cge.compress_to_hologram()
chain_sim = float(torch.cosine_similarity(holo.unsqueeze(0), direct_vec.unsqueeze(0)))
print(f"Sim(holo, FLORB*causes*structure_role) = {chain_sim:.4f}")

# 7. RODAR FORJA CONSOLIDATION
print("\n--- Rodando Forja Consolidation ---")
created = cge.forge_consolidate()
print(f"Arestas consolidadas: {created}")

# 8. Testar se as arestas diretas AGORA EXISTEM
def get_edge_confidence(cge, from_id, to_id):
    for e in cge.edges:
        if e["from"] == from_id and e["to"] == to_id:
            return e.get("confidence", 1.0)
    return None

print("\n--- Teste: arestas diretas DEPOIS da consolidacao ---")
florb_conf = get_edge_confidence(cge, "FLORB", "structure_role")
trek_conf = get_edge_confidence(cge, "TREK", "thermal_role")
print(f"FLORB -> structure_role direta? {florb_conf is not None}")
if florb_conf is not None:
    print(f"  Confianca: {florb_conf:.4f}")
print(f"TREK -> thermal_role direta? {trek_conf is not None}")
if trek_conf is not None:
    print(f"  Confianca: {trek_conf:.4f}")

# 9. Testar generalizacao: submarino
print("\n--- Generalizacao: submarino ---")
# Adicionar que submarino usa FLORB para estrutura
cge.create_node("submarine", submarine)
cge.add_edge("submarine", "FLORB", causes_edge)
cge.add_edge("submarine", "TREK", causes_edge)

# Rodar consolidacao de novo
created2 = cge.forge_consolidate()
print(f"Arestas consolidadas na 2a passada: {created2}")

# Verificar se submarino → structure_role existe
print(f"submarine -> structure_role direta? {get_edge_confidence(cge, 'submarine', 'structure_role') is not None}")
print(f"submarine -> thermal_role direta? {get_edge_confidence(cge, 'submarine', 'thermal_role') is not None}")
if get_edge_confidence(cge, 'submarine', 'structure_role'):
    print(f"  structure conf: {get_edge_confidence(cge, 'submarine', 'structure_role'):.4f}")
if get_edge_confidence(cge, 'submarine', 'thermal_role'):
    print(f"  thermal conf: {get_edge_confidence(cge, 'submarine', 'thermal_role'):.4f}")

print("\n--- Grafo final ---")
print(cge.get_scene_text())
