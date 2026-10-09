"""
test_core_integration.py — Testa a integracao BASE da AGI.
Sem inference engine pesado. Foco: ciclo executivo + KG + WorldModel real.
"""
import sys, os, json, numpy as np
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))

from core.world_model import WorldModel
from core.cognitive_executive import CognitiveExecutive
from core.cognitive_modes import CognitiveModeManager
from core.goal_driven_engine import GoalDrivenEngine
from core.unified_self import UnifiedSelf
from core.intrinsic_motivation import IntrinsicMotivationEngine
from core.knowledge_graph import KnowledgeGraph
from core.virtual_lab import VirtualLab

print("=" * 60)
print("  INTEGRACAO BASE - CICLO REAL")
print("=" * 60)

# 1. KnowledgeGraph
print("\n[1] KG: 13822 conceitos...")
kg = KnowledgeGraph()
data = json.load(open(r"D:\novamind_data\knowledge_graph\concepts.json", "r", encoding="utf-8"))
for name, meta in list(data.items()):
    d = meta.get("domain", "unknown") if isinstance(meta, dict) else "unknown"
    kg.add_concept(str(name)[:100], [], domain=d)
rels = json.load(open(r"D:\novamind_data\knowledge_graph\relations.json", "r", encoding="utf-8"))
for r in rels:
    kg.add_causal_relation(r["source"], r["target"], r.get("type", "causa"),
                           confidence=r.get("confidence", 0.5))
print(f"  {len(kg.concepts)} conceitos, {len(kg.relations)} relacoes")

# 2. WorldModel REAL
print("\n[2] WorldModel com KG...")
wm = WorldModel(knowledge_graph=kg)

# Testa predict
for query in ["schizophrenia", "dopamine", "Alzheimer"]:
    pred = wm.predict(query)
    out = str(pred.predicted_output)[:80] if pred.predicted_output else "None"
    print(f"  predict('{query}'): {out}")

# 3. VirtualLab
print("\n[3] VirtualLab...")
vl = VirtualLab()
for d in ["Alzheimer", "Parkinson"]:
    vl.load_disease_model(d)
    r = vl.simulate_intervention("baseline", dose=0)
    print(f"  {d}: {len(r.biomarker_change)} biomarkers")

# 4. CognitiveExecutive
print("\n[4] Executive integrado...")
exec = CognitiveExecutive(
    goal_engine=GoalDrivenEngine(),
    self_model=UnifiedSelf(),
    motivation=IntrinsicMotivationEngine(UnifiedSelf()),
    mode_manager=CognitiveModeManager(),
    world_model=wm,
    knowledge_graph=kg,
    virtual_lab=vl,
)
print(f"  {exec.summarize_self()}")

# 5. Ciclos idle
print("\n[5] 10 ciclos idle...")
print(f"  {'#':>3} | {'Modo':>13} | Surpresa | Acoes")
print(f"  {'-'*50}")
for i in range(10):
    if i == 3:
        wm.recent_surprise.append(0.9)
    elif i == 7:
        wm.recent_surprise.append(0.1)
    result = exec.idle_cycle()
    mode = result.get("mode", "?")
    surprise = wm.get_surprise()
    n_sim = len(result.get("virtual_lab_simulations", []))
    n_laws = len(result.get("law_discoveries", []))
    actions = []
    if n_sim: actions.append(f"{n_sim} sim")
    if n_laws: actions.append(f"{n_laws} leis")
    print(f"  {i:3d} | {mode:>13} | {surprise:.2f}    | {', '.join(actions) if actions else '(ocioso)'}")

# 6. Ciclo com tarefa
print("\n[6] Ciclo com tarefa...")
r = exec.cycle(np.random.rand(4,4), np.zeros((4,4)))
print(f"  Modo: {r['mode']}, acuracia: {r['accuracy']:.3f}")

# 7. Teste WorldModel update
print("\n[7] WorldModel update...")
pred = wm.predict("test")
err = wm.update("test", "real_value", pred)
print(f"  Erro: {err.error_magnitude:.3f}, surpresa: {err.surprise:.3f}")

# SUMMARY
print(f"\n{'='*60}")
print(f"  STATUS DA INTEGRACAO")
print(f"{'='*60}")
print(f"  Executive: {exec.summarize_self()}")
print(f"  WorldModel prediz: {wm.predict('x').predicted_output is not None}")
print(f"  KG: {len(kg.concepts)} conceitos, {len(kg.relations)} relacoes")
print(f"  VLab: Alzheimer + Parkinson OK")
print(f"  Ciclos: {exec._idle_cycles} idle, {exec._cycle} task")
print(f"{'='*60}")
print("  INTEGRACAO BASE FUNCIONAL.")
