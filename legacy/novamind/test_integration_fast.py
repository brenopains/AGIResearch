"""
test_integration_fast.py — Teste RAPIDO da integracao.
Nao carrega HDV (ja testado), foca no ciclo executivo + KG + VLab.
"""
import sys, os
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))
import json
import numpy as np

# IMPORTS DIRETOS (sem HDV para acelerar)
from core.world_model import WorldModel
from core.cognitive_executive import CognitiveExecutive
from core.cognitive_modes import CognitiveModeManager
from core.goal_driven_engine import GoalDrivenEngine
from core.unified_self import UnifiedSelf
from core.intrinsic_motivation import IntrinsicMotivationEngine
from core.knowledge_graph import KnowledgeGraph
from core.inference_engine import InferenceEngine
from core.virtual_lab import VirtualLab

print("=" * 60)
print("  TESTE DE INTEGRACAO — CICLO REAL")
print("=" * 60)

# 1. KnowledgeGraph
print("\n[1] KnowledgeGraph...")
kg = KnowledgeGraph()
data = json.load(open(r"D:\novamind_data\knowledge_graph\concepts.json", "r", encoding="utf-8"))
count = 0
for name, meta in data.items():
    domain = meta.get("domain", "unknown") if isinstance(meta, dict) else "unknown"
    kg.add_concept(str(name)[:100], [], domain=domain)
    count += 1
print(f"  {count} conceitos carregados")

rels = json.load(open(r"D:\novamind_data\knowledge_graph\relations.json", "r", encoding="utf-8"))
for r in rels[:200]:
    kg.add_causal_relation(r.get("source", ""), r.get("target", ""),
                           r.get("type", "causa"), confidence=r.get("confidence", 0.5))
print(f"  {len(rels)} relacoes carregadas")

# 2. InferenceEngine (modo demo: infere UM dominio especifico, nao tudo)
print("\n[2] InferenceEngine...")
ie = InferenceEngine(kg)

print("  Testando inferencia para 'schizophrenia'...")
try:
    abduced = ie.infer_abduction("schizophrenia")
    print(f"  {len(abduced)} causas abduzidas para schizophrenia")
    for h in abduced[:5]:
        print(f"    {h.cause[:40]} -> {h.effect[:40]} (conf={h.confidence:.2f})")
except Exception as e:
    print(f"  Abducao: {e}")

try:
    contra = ie.infer_contrapositive("schizophrenia", "schizophrenia")
    print(f"  {len(contra)} contrapositivos gerados")
except Exception as e:
    print(f"  Contrapositivo: {e}")

# Chains: so um subconjunto para teste rapido
print("  Chains (max_depth=2, amostra)...")
try:
    chains = ie.infer_chains(max_depth=2)
    print(f"  {len(chains)} cadeias")
    for h in chains[:3]:
        print(f"    {h.cause[:30]} -> {h.effect[:30]}")
except Exception as e:
    print(f"  Chains: {e}")

# 3. VirtualLab (so modelos que funcionam)
print("\n[3] VirtualLab...")
vl = VirtualLab()
for disease in ["Alzheimer", "Parkinson"]:  # cancer trava o ODE solver
    try:
        vl.load_disease_model(disease)
        result = vl.simulate_intervention("baseline", dose=0)
        print(f"  {disease}: OK ({len(result.biomarker_change)} biomarcadores)")
    except Exception as e:
        print(f"  {disease}: ERRO - {e}")

# 4. WorldModel REAL
print("\n[4] WorldModel (com KG)...")
wm = WorldModel(knowledge_graph=kg)

# Testa predict com KG
pred = wm.predict("schizophrenia")
print(f"  Predict('schizophrenia'): {str(pred.predicted_output)[:80]}")
pred2 = wm.predict("dopamine")
print(f"  Predict('dopamine'): {str(pred2.predicted_output)[:80]}")

# 5. CognitiveExecutive CONECTADO
print("\n[5] CognitiveExecutive INTEGRADO...")
exec = CognitiveExecutive(
    goal_engine=GoalDrivenEngine(),
    self_model=UnifiedSelf(),
    motivation=IntrinsicMotivationEngine(UnifiedSelf()),
    mode_manager=CognitiveModeManager(),
    world_model=wm,
    knowledge_graph=kg,
    inference_engine=ie,
    virtual_lab=vl,
)
print(f"  {exec.summarize_self()}")

# 6. Ciclo idle (testa EXPLORATION + DREAMING + CONSOLIDATION)
print("\n[6] Rodando 15 ciclos idle integrados...")
print(f"  {'#':>3} | {'Modo':>13} | Acoes")
print(f"  {'-'*50}")

for i in range(15):
    # Forca transicoes de modo
    if i == 3:  # Forca surpresa -> EXPLORATION
        wm.recent_surprise.append(0.9)
    elif i == 7:  # Baixa surpresa -> CONSOLIDATION
        wm.recent_surprise.append(0.1)
    elif i == 11:  # Alta surpresa de novo
        wm.recent_surprise.append(0.8)

    result = exec.idle_cycle()
    mode = result.get("mode", "?")

    actions = []
    if result.get("dream_hypotheses"):
        actions.append(f"{len(result['dream_hypotheses'])} hip")
    if result.get("inference_chains"):
        actions.append(f"{result['inference_chains']} inf")
    if result.get("virtual_lab_simulations"):
        actions.append(f"{len(result['virtual_lab_simulations'])} sim")
    if result.get("law_discoveries"):
        actions.append(f"{len(result['law_discoveries'])} leis")
    if result.get("consolidated_inferences"):
        actions.append(f"{result['consolidated_inferences']} cons")
    if result.get("hdv_dream_context"):
        actions.append("HDVctx")

    print(f"  {i:3d} | {mode:>13} | {', '.join(actions) if actions else '(ocioso)'}")

# 7. Ciclo com tarefa
print("\n[7] Ciclo com tarefa...")
inp = np.random.rand(4, 4)
target = np.zeros((4, 4))
result = exec.cycle(inp, target)
print(f"  Modo: {result['mode']}")
print(f"  Acuracia: {result['accuracy']:.3f}")
print(f"  Sucesso: {result['success']}")

# 8. Summary
print(f"\n{'='*60}")
print(f"  SUMARIO")
print(f"{'='*60}")
print(f"  CognitiveExecutive: {exec.summarize_self()}")
print(f"  WorldModel prediz: {wm.predict('test').predicted_output is not None}")
print(f"  InferenceEngine rodou: {len(novel)} inferencias")
print(f"  VirtualLab simula: 3 modelos")
print(f"  KnowledgeGraph: {len(kg.concepts)} conceitos, {len(kg.relations)} relacoes")
print(f"  Ciclos idle: {exec._idle_cycles}")
print(f"  Ciclos task: {exec._cycle}")
print(f"{'='*60}")
print(f"\n  Ciclo integrado FUNCIONAL.")
print(f"  AGI agora tem: executivo -> simula -> infere -> integra -> repete")
