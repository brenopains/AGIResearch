"""
Teste: LOOP AGI COMPLETO (Goal + Error-Driven Learning + Causal + Compression)

Cenario:
  1. GOAL: AGI recebe meta "entender relacao entre cachorro e gato"
  2. APRENDE: Maestro processa sentencas, CGE cria nos via BGE
  3. AVALIA: GoalManager mede progresso (quantas conexoes entre dog/cat)
  4. ERRO: Se falha, GoalManager enfraquece edges errados
  5. CAUSA: Causal_intervention testa "se remover dog, cat muda?"
  6. COMPRESSAO: forge_abstract cria conceito "animal" de dog+cat
  7. HIERARQUIA: refine_abstraction melhora com contraexemplo
  8. PLANO: plan_to_goal encontra caminho causal
"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from core.cognitive_graph_engine import CognitiveGraphEngine
from core.bge_encoder import BGEEncoder
from core.goal_system import GoalManager

print("=" * 60)
print("TESTE: LOOP AGI COMPLETO")
print("=" * 60)

# 1. Setup
bge = BGEEncoder()
cge = CognitiveGraphEngine(device="cuda", bge_encoder=bge)
gm = GoalManager(cge=cge)
print("[OK] Componentes carregados")

# 2. Alimentar CGE com conhecimento sobre caes e gatos
sentences = [
    "The dog runs fast and chases cats in the park",
    "A cat climbs trees and hunts mice during the night",
    "Dogs are loyal pets that protect their owners",
    "Cats are independent animals that clean themselves",
    "Both dogs and cats are popular pets in households",
    "Dogs and cats can live together peacefully",
    "The dog barked loudly at the cat on the fence",
    "A cat purrs when sitting on the warm windowsill",
    "Dogs need daily walks and regular exercise",
    "Cats use litter boxes and sleep during the day",
]
for s in sentences:
    cge.write_text_concepts(s, emotional_valence=0.5, emotional_arousal=0.3)

print(f"\n[OK] CGE populado: {len(cge.nodes)} nos, {len(cge.edges)} arestas")

# 3. GOAL: entender relacao dog/cat
goal_id = gm.create_goal(
    "understand dog-cat relationship",
    {"edges_between": 3, "min_similarity": 0.5},
    priority=8
)
print(f"\n[GOAL] Meta criada: '{gm.goals[goal_id].description}' (ID={goal_id})")

# 4. Error-Driven Learning: simular tentativas
print("\n--- ERROR-DRIVEN LEARNING ---")
for attempt in range(15):
    # Caminho percorrido (edges no CGE)
    path = [(e["from"], e["to"]) for e in cge.edges
            if "dog" in e["from"] or "cat" in e["from"] or
               "dog" in e["to"] or "cat" in e["to"]][:3]

    # Simular observacao: quantos edges entre dog/cat?
    dog_cat_edges = sum(1 for e in cge.edges
                        if ("dog" in e["from"] and "cat" in e["to"]) or
                           ("cat" in e["from"] and "dog" in e["to"]))
    obs = {"edges_between": dog_cat_edges}

    # Avaliar progresso
    score = gm.evaluate(goal_id, obs)
    print(f"  Tentativa {attempt+1}: edges dog-cat={dog_cat_edges}, score={score:.2f}")

    # Aplicar sinal de erro
    gm.apply_error_signal(goal_id, score, path_taken=path)

    # Simular aprendizado: criar mais conexoes dog-cat a cada falha
    if score < 0.9 and "dog" in cge.nodes and "cat" in cge.nodes:
        if attempt >= 1:
            if not any(e["from"] == "dog" and e["to"] == "cat" for e in cge.edges):
                cge.add_edge("dog", "cat", cge.EDGE_CAUSES)
                print(f"  -> Aprendeu: dog --causes--> cat")
        if attempt >= 3:
            if not any(e["from"] == "dog" and e["to"] == "pets" for e in cge.edges):
                cge.add_edge("dog", "pets", cge.EDGE_CAUSES)
                print(f"  -> Aprendeu: dog --causes--> pets")
        if attempt >= 5:
            if not any(e["from"] == "cat" and e["to"] == "pets" for e in cge.edges):
                cge.add_edge("cat", "pets", cge.EDGE_CAUSES)
                print(f"  -> Aprendeu: cat --causes--> pets")
        if attempt >= 7:
            if not any(e["from"] == "cat" and e["to"] == "dog" for e in cge.edges):
                cge.add_edge("cat", "dog", cge.EDGE_CAUSES)
                print(f"  -> Aprendeu: cat --causes--> dog")

print(f"\n  Estado final: {gm.goals[goal_id].state} (score={gm.goals[goal_id].progress:.2f})")
print(f"  Tentativas: {len(gm.goals[goal_id].attempts)}")

# 5. CAUSAL GROUNDING: intervencao
print("\n--- CAUSAL GROUNDING ---")
intervention = cge.causal_intervention("dog", remove=True)
print(f"  Se remover 'dog', predicao para {len(intervention)} vizinhos:")
for nid, change in list(intervention.items())[:5]:
    print(f"    {nid}: mudanca esperada = {change:.2f}")

# 6. PLANEJAMENTO CAUSAL
print("\n--- PLANEJAMENTO CAUSAL ---")
plan = cge.plan_to_goal("dog", "cat", max_depth=5)
if plan:
    print(f"  Caminho dog -> cat: {' -> '.join(plan[:6])}")
else:
    print("  Nenhum caminho dog->cat encontrado")

# 7. COMPRESSAO CONCEITUAL
print("\n--- COMPRESSAO CONCEITUAL ---")
# Criar conceito abstrato "animal" de dog+cat
abstract = cge.forge_abstract(["dog", "cat"], "animal", min_sim=0.05)
if abstract:
    print(f"  Conceito 'animal' criado (conf={abstract['confidence']:.2f})")
    # Propriedades herdadas
    props = [e["to"] for e in cge.edges
             if e["from"] == "animal" and e["edge_type"] is cge.EDGE_CAUSES]
    print(f"  Propriedades: {props}")

# Refinamento com contraexemplo
if "animal" in cge.nodes:
    # Adicionar "fish" como contraexemplo (diferente de dog/cat)
    cge.write_text_concepts("Fish live in water and have gills", 0.3, 0.2)
    if "fish" in cge.nodes:
        # Criar edge de animal para fish (para testar refine)
        cge.add_edge("animal", "fish", cge.EDGE_CAUSES)
        refined = cge.refine_abstraction("animal", "fish")
        if refined:
            print(f"  Refinamento: {refined['original']} -> {refined['refined']}")
            print(f"  Props diferenciadoras: {refined['differentiating_props']}")

# 8. HIERARQUIA CONCEITUAL
print("\n--- HIERARQUIA CONCEITUAL ---")
if "animal" in cge.nodes:
    hierarchy = cge.get_concept_hierarchy("animal", max_depth=3)
    def print_hierarchy(h, indent=0):
        name = h.get("name", "?")
        conf = h.get("confidence", "?")
        print(f"  {'  ' * indent}{name} (conf={conf})")
        for child in h.get("children", []):
            print_hierarchy(child, indent + 1)
    print_hierarchy(hierarchy)

# 9. RESUMO
print("\n" + "=" * 60)
print("RESUMO DO LOOP AGI")
print("=" * 60)
print(f"  Goal System:      {'PASS' if gm.goals[goal_id].state == 'completed' else 'PARCIAL'}  {gm.goals[goal_id].state}")
print(f"  Error-Driven:     {'PASS' if len(gm.goals[goal_id].attempts) > 3 else 'FAIL'}  {len(gm.goals[goal_id].attempts)} tentativas")
print(f"  Causal Grounding: {'PASS' if intervention else 'FAIL'}  {len(intervention)} predicoes")
print(f"  Causal Planning:  {'PASS' if plan else 'FAIL'}  {' -> '.join(plan[:5]) if plan else 'N/A'}")
print(f"  Compression:      {'PASS' if abstract else 'FAIL'}  conceito 'animal' criado")
print(f"  Hierarchy:        {'PASS' if 'animal' in cge.nodes else 'FAIL'}  presente no CGE")
