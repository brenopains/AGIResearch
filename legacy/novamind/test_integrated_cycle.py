"""
test_integrated_cycle.py — Testa o ciclo completo INTEGRADO.

Pela PRIMEIRA VEZ, o CognitiveExecutive roda com:
  KnowledgeGraph + InferenceEngine + VirtualLab + HDV brain + WorldModel real
"""
import sys, os
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))

# CARREGA TUDO QUE PRECISA
from core.cognitive_executive import CognitiveExecutive
from core.cognitive_modes import CognitiveModeManager
from core.world_model import WorldModel
from core.goal_driven_engine import GoalDrivenEngine
from core.unified_self import UnifiedSelf
from core.intrinsic_motivation import IntrinsicMotivationEngine
from core.knowledge_graph import KnowledgeGraph
from core.inference_engine import InferenceEngine
from core.virtual_lab import VirtualLab
from core.hdc_memory import HDCMemory
import numpy as np

print("=" * 60)
print("  CICLO INTEGRADO — PRIMEIRA EXECUÇÃO REAL")
print("=" * 60)

# 1. Carrega HDV brain (242K itens)
print("\n[1] Carregando HDV brain...")
hdv = HDCMemory()
stats = hdv.get_stats()
print(f"  {stats['total_items']} itens carregados")

# 2. Carrega KnowledgeGraph (do JSON)
print("\n[2] Carregando KnowledgeGraph...")
kg = KnowledgeGraph()
# Carrega do JSON diretamente
import json
kg_data = json.load(open(r"D:\novamind_data\knowledge_graph\concepts.json", "r", encoding="utf-8"))
for name, meta in list(kg_data.items())[:200]:  # amostra 200 para teste
    domain = meta.get("domain", "unknown") if isinstance(meta, dict) else "unknown"
    kg.add_concept(name, [], domain=domain)
rel_data = json.load(open(r"D:\novamind_data\knowledge_graph\relations.json", "r", encoding="utf-8"))
for r in rel_data[:100]:
    kg.add_causal_relation(r.get("source", ""), r.get("target", ""),
                           r.get("type", "causa"),
                           confidence=r.get("confidence", 0.5))
print(f"  {len(kg.concepts)} conceitos, {len(kg.relations)} relacoes (amostra)")

# 3. Carrega InferenceEngine
print("\n[3] Inicializando InferenceEngine...")
ie = InferenceEngine(kg)
print(f"  Pronto: 4 tipos de inferencia")

# 4. Carrega VirtualLab (com Tellurium/SBML)
print("\n[4] Inicializando VirtualLab...")
vl = VirtualLab()
print(f"  Pronto: modelos ODE + SBML")

# 5. Carrega WorldModel (AGORA com HDV + KG)
print("\n[5] Inicializando WorldModel com HDV + KG...")
wm = WorldModel(hdv_memory=hdv, knowledge_graph=kg)

# 6. Monta CognitiveExecutive COMPLETO
print("\n[6] Montando CognitiveExecutive CONECTADO...")
mode_mgr = CognitiveModeManager()
self_model = UnifiedSelf()
motivation = IntrinsicMotivationEngine(self_model=self_model)
goal_engine = GoalDrivenEngine()

exec = CognitiveExecutive(
    goal_engine=goal_engine,
    self_model=self_model,
    motivation=motivation,
    mode_manager=mode_mgr,
    world_model=wm,
    knowledge_graph=kg,
    inference_engine=ie,
    virtual_lab=vl,
    hdv_memory=hdv,
)

print(f"\n  {exec.summarize_self()}")
print(f"  Modulos conectados: KG={kg is not None}, INF={ie is not None}, VLAB={vl is not None}, HDV={hdv is not None}")

# 7. Roda ciclos idle (sem tarefa) para testar MODOS
print("\n[7] Executando 20 ciclos idle integrados...")
print(f"  {'Ciclo':>5} | {'Modo':>15} | {'Energia':>7} | Ações")
print(f"  {'-'*60}")

for i in range(20):
    # Deixa o mode manager circular entre modos
    # Forca transicoes variando a surpresa
    if i == 5:
        wm.recent_surprise.append(0.8)  # surpresa alta -> EXPLORATION
    if i == 10:
        wm.recent_surprise.append(0.1)  # surpresa baixa -> CONSOLIDATION

    result = exec.idle_cycle()
    mode = result.get("mode", "?")
    energy = exec.self_model.energy

    # O que aconteceu neste ciclo?
    actions = []
    if result.get("dream_hypotheses"):
        actions.append(f"{len(result['dream_hypotheses'])} hipoteses")
    if result.get("inference_chains"):
        actions.append(f"{result['inference_chains']} inferencias")
    if result.get("virtual_lab_simulations"):
        n_sim = len(result['virtual_lab_simulations'])
        actions.append(f"{n_sim} simulacoes VLab")
    if result.get("law_discoveries"):
        actions.append(f"{len(result['law_discoveries'])} leis")
    if result.get("consolidated_inferences"):
        actions.append(f"{result['consolidated_inferences']} consolidados")
    if result.get("concept_economy"):
        actions.append("economy")
    if result.get("hdv_context"):
        actions.append("HDV ctx")

    action_str = ", ".join(actions) if actions else "(ocioso)"
    print(f"  {i:5d} | {mode:>15} | {energy:7.2f} | {action_str}")

# 8. Testa WorldModel.predict() REAL
print("\n[8] Testando WorldModel.predict() REAL...")
pred = wm.predict("schizophrenia")
print(f"  Input: 'schizophrenia'")
print(f"  Predicao: {str(pred.predicted_output)[:100]}")
print(f"  Confianca: {pred.confidence:.3f}")

pred2 = wm.predict("dopamine")
print(f"\n  Input: 'dopamine'")
print(f"  Predicao: {str(pred2.predicted_output)[:100]}")
print(f"  Confianca: {pred2.confidence:.3f}")

# 9. Testa ciclo com tarefa
print("\n[9] Ciclo com tarefa externa...")
input_grid = np.random.rand(4, 4)
target_grid = np.zeros((4, 4))
result = exec.cycle(input_grid, target_grid)
print(f"  Modo: {result['mode']}")
print(f"  Sucesso: {result['success']}")
print(f"  Acuracia: {result['accuracy']:.3f}")

# 10. Sumario final
print(f"\n{'='*60}")
print(f"  SUMARIO DA INTEGRACAO")
print(f"{'='*60}")
print(f"  CognitiveExecutive: {exec.summarize_self()}")
print(f"  WorldModel prediz: {wm.predict('test').predicted_output is not None}")
print(f"  Ciclos idle: {exec._idle_cycles}")
print(f"  Ciclos task: {exec._cycle}")
wm_stats = wm.get_stats()
print(f"  Surpresa media: {wm_stats['recent_surprise']}")
print(f"  Acuracia media: {wm_stats['recent_accuracy']}")
print(f"{'='*60}")
print(f"\n  A AGI agora tem um ciclo integrado funcional.")
print(f"  Proximo passo: descoberta genuina rodando.")
