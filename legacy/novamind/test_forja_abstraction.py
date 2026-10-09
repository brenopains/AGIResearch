"""
Teste: Forja Abstraction + Consolidation Cycle

Cenario: 3 instancias de "cadeira" com propriedades compartilhadas
A Forja deve abstrair o conceito "cadeira" via bundle.

Instancia1: tem assento, tem encosto, tem 4 pernas
Instancia2: tem assento, tem encosto, tem 4 pernas, e macia
Instancia3: tem assento, tem encosto, tem 4 pernas, e giratoria

A Forja deve:
1. Criar no abstrato 'cadeira' do bundle das 3
2. Criar arestas 'part_of' das instancias -> cadeira
3. Detectar propriedades comuns (assento, encosto, 4_pernas) e criar
   arestas CAUSES do conceito -> propriedades
"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from core.cognitive_graph_engine import CognitiveGraphEngine

DIM = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

print("=" * 60)
print("Teste: Forja Abstraction")
print("=" * 60)

cge = CognitiveGraphEngine(device=DEVICE)

# Propriedades (HDVs aleatorios)
assento = torch.sign(torch.randn(DIM, device=DEVICE))
encosto = torch.sign(torch.randn(DIM, device=DEVICE))
pernas = torch.sign(torch.randn(DIM, device=DEVICE))
macia = torch.sign(torch.randn(DIM, device=DEVICE))
giratoria = torch.sign(torch.randn(DIM, device=DEVICE))

# Instancias (HDVs aleatorios diferentes)
inst1 = torch.sign(torch.randn(DIM, device=DEVICE))
inst2 = torch.sign(torch.randn(DIM, device=DEVICE))
inst3 = torch.sign(torch.randn(DIM, device=DEVICE))

cge.create_node("instancia1", inst1)
cge.create_node("instancia2", inst2)
cge.create_node("instancia3", inst3)
cge.create_node("assento", assento)
cge.create_node("encosto", encosto)
cge.create_node("4_pernas", pernas)
cge.create_node("macia", macia)
cge.create_node("giratoria", giratoria)

causes = cge.EDGE_CAUSES

# instancia1: assento, encosto, 4 pernas
cge.add_edge("instancia1", "assento", causes)
cge.add_edge("instancia1", "encosto", causes)
cge.add_edge("instancia1", "4_pernas", causes)

# instancia2: assento, encosto, 4 pernas, macia
cge.add_edge("instancia2", "assento", causes)
cge.add_edge("instancia2", "encosto", causes)
cge.add_edge("instancia2", "4_pernas", causes)
cge.add_edge("instancia2", "macia", causes)

# instancia3: assento, encosto, 4 pernas, giratoria
cge.add_edge("instancia3", "assento", causes)
cge.add_edge("instancia3", "encosto", causes)
cge.add_edge("instancia3", "4_pernas", causes)
cge.add_edge("instancia3", "giratoria", causes)

print("\n--- Grafo antes da abstracao ---")
print(cge.get_scene_text())

# Rodar abstracao manualmente
print("\n--- Forja Abstracao: instancia1+instancia2+instancia3 -> cadeira ---")
result = cge.forge_abstract(
    ["instancia1", "instancia2", "instancia3"],
    "cadeira",
    min_sim=0.05
)

if result:
    print(f"Conceito 'cadeira' criado! Confianca: {result['confidence']:.2f}")
else:
    print("FALHA: nao criou o conceito")

# Verificar arestas 'part_of'
part_of_edges = [e for e in cge.edges if e["edge_type"] is cge.EDGE_PART_OF]
print(f"\nArestas 'part_of' criadas: {len(part_of_edges)}")
for e in part_of_edges:
    print(f"  {e['from']} --part_of--> {e['to']}")

# Verificar propriedades comuns herdadas
cadeira_props = [e["to"] for e in cge.edges
                 if e["from"] == "cadeira" and e["edge_type"] is causes]
print(f"\nPropriedades herdadas por 'cadeira': {len(cadeira_props)}")
for p in cadeira_props:
    print(f"  cadeira --causes--> {p}")

# Deveria herdar: assento, encosto, 4_pernas (compartilhados por 2+)
# Nao deveria herdar: macia, giratoria (so 1 instancia cada)
expected_shared = {"assento", "encosto", "4_pernas"}
got_shared = set(cadeira_props)
print(f"\nPropriedades compartilhadas (esperado): {expected_shared}")
print(f"Propriedades herdadas (obtido): {got_shared}")
print(f"Heranca correta? {got_shared == expected_shared}")

# Testar Spreading Activation
print("\n--- Spreading Activation a partir de 'cadeira' ---")
activation = cge.spread_activation(["cadeira"], decay=0.5, max_hops=2)
activated = {k: round(v, 3) for k, v in activation.items() if v > 0}
for nid, act in sorted(activated.items(), key=lambda x: -x[1]):
    print(f"  {nid}: {act}")

# Testar Consolidation Cycle
print("\n--- Consolidation Cycle ---")
stats = cge.consolidation_cycle()
print(f"Arestas consolidadas: {stats['edges_created']}")
print(f"Conceitos abstratos criados: {stats['concepts_created']}")
print(f"Nos podados: {stats['pruned_nodes']}")

print("\n--- Grafo final ---")
print(cge.get_scene_text())
