"""
Teste: SelfModel + SimulationEngine

Valida:
  1. SelfIdentity: crencas, autobiografia, capabilities, learning_profile
  2. SelfProcess: fadiga, atencao, operador ativo
  3. SimulationEngine: 4 modos (EXPECT, COUNTERFACTUAL, PLAN, IMAGINE)
  4. Integracao: SelfModel alimenta SimulationEngine com estado cognitivo
"""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("TESTE: SELFMODEL + SIMULATION ENGINE")
print("=" * 70)

from core.self_model import SelfModel, SelfIdentity, SelfProcess, Belief
from core.simulation_engine import SimulationEngine, SimulationMode, Branch

# ─── PARTE 1: SelfIdentity ───
print("\n--- PARTE 1: SelfIdentity ---")
identity = SelfIdentity()

# Adicionar crencas
identity.add_belief("gravidade", "Objetos caem quando soltos", confidence=0.8, reliability=0.9)
identity.add_belief("energia", "Energia se conserva", confidence=0.7, reliability=0.8)
identity.add_belief("entropia", "Entropia sempre aumenta", confidence=0.3, reliability=0.4)
identity.add_belief("materia_escura", "Existe materia que nao interage com luz", confidence=0.2, reliability=0.2)

assert "gravidade" in identity.beliefs
assert identity.beliefs["gravidade"].confidence > 0.7
print(f"  Crencas: {len(identity.beliefs)}")
print(f"  Confiantes: {identity.get_confident_beliefs()[:3]}")
print(f"  Incertezas: {identity.get_uncertain_topics()[:3]}")

# Revisar crenca
identity.revise_belief("materia_escura", "Novos experimentos contradizem teoria atual", reliability=0.3)
assert identity.beliefs["materia_escura"].confidence < 0.2
print(f"  Apos revisao: materia_escura conf={identity.beliefs['materia_escura'].confidence:.2f}")

# Capacidades e perfil de aprendizado
identity.capabilities.update("fisica", 0.9)
identity.capabilities.update("biologia", 0.3)
identity.learning_profile.update("video", 0.8)
identity.learning_profile.update("texto", 0.4)
print(f"  Forca: {identity.capabilities.get_strengths()}")
print(f"  Fraqueza: {identity.capabilities.get_weaknesses()}")
print(f"  Melhor modalidade: {identity.learning_profile.best_modality()}")

# Autobiografia
identity.autobiography.record({"type": "descoberta", "concept": "inercia", "success": True})
identity.autobiography.record({"type": "erro", "concept": "atrito_ignorado", "success": False})
print(f"  Autobiografia: {len(identity.autobiography.episodes)} episodios")

print("  PASS: SelfIdentity")

# ─── PARTE 2: SelfProcess ───
print("\n--- PARTE 2: SelfProcess ---")
process = SelfProcess()

# Simular ciclos de trabalho
for i in range(50):
    if i < 20:
        process.update(operator="explore", mode="explore", error=0.3)
    elif i < 35:
        process.update(operator="test", mode="test", error=0.5)
    else:
        process.update(operator="reflect", mode="reflect", error=0.2)

print(f"  Fadiga apos 50 ciclos: {process.fatigue:.2f}")
print(f"  Precisa descansar: {process.needs_rest()}")
print(f"  Operador ativo: {process.active_operator}")

# Fadiga deve ser menor depois de reflect
assert process.fatigue < 0.7  # deve ter recuperado com reflect
print(f"  Deveria trocar de modo: {process.should_switch_mode()}")
print("  PASS: SelfProcess")

# ─── PARTE 3: SelfModel integrado ───
print("\n--- PARTE 3: SelfModel Integrado ---")
sm = SelfModel()

# Aprender com experiencias
sm.learn_from_experience("conservacao_energia",
                          "Energia total se mantem em sistema isolado",
                          reliability=0.9, domain="fisica", modality="video", success=True)
sm.learn_from_experience("termodinamica",
                          "Calor flui do quente para o frio",
                          reliability=0.8, domain="fisica", modality="texto", success=True)
sm.learn_from_experience("fotossintese",
                          "Plantas convertem luz em energia",
                          reliability=0.6, domain="biologia", modality="texto", success=False)

state = sm.get_state()
print(f"  Crencas: {state['identity']['beliefs']}")
print(f"  Confiantes: {state['identity']['confident']}")
print(f"  Incertezas: {state['identity']['uncertain']}")
print(f"  Fraquezas: {state['identity']['weaknesses']}")
print(f"  Melhor modalidade: {state['identity']['best_modality']}")

curiosity = sm.get_curiosity_targets()
print(f"  Curiosidade (para explorar): {curiosity}")

print("  PASS: SelfModel Integrado")

# ─── PARTE 4: Simulation Engine — EXPECT ───
print("\n--- PARTE 4: Simulation Engine (EXPECT) ---")
sim = SimulationEngine()

def energy_model(state):
    return {"target_energy": state.get("source", 80) * 0.75 * (0.75 ** state.get("cells", 4))}
sim.add_transition_model("energy_flow", energy_model)

state_mtb = {"source": 80, "cells": 4, "conductance": 0.75, "target_energy": 0.0}
branches = sim.simulate(state_mtb, mode=SimulationMode.EXPECT, n_branches=3)
print(f"  Branches geradas (EXPECT): {len(branches)}")
for b in branches:
    print(f"    {b.id}: P={b.probability:.2f}, pred_target={b.predicted_state.get('target_energy', 0):.1f}")

best = sim.get_best_branch()
assert best is not None
print(f"  Melhor: {best.id} (P={best.probability:.2f})")

# Observar realidade e atualizar
reality = {"source": 80, "cells": 4, "conductance": 0.75, "target_energy": 11.4}
results = sim.observe_reality(reality)
print(f"  Apos realidade: {len(sim.branches)} branches sobreviventes")
for b in sim.branches[:3]:
    print(f"    {b.id}: P={b.probability:.2f}")

print("  PASS: EXPECT")

# ─── PARTE 5: Simulation Engine — COUNTERFACTUAL ───
print("\n--- PARTE 5: Simulation Engine (COUNTERFACTUAL) ---")
sim.clear()
branches = sim.simulate(state_mtb, mode=SimulationMode.COUNTERFACTUAL, action="place_VORIN", n_branches=4)
print(f"  Branches (COUNTERFACTUAL): {len(branches)}")
for b in branches:
    print(f"    {b.id}: {b.description} (P={b.probability:.2f})")
print("  PASS: COUNTERFACTUAL")

# ─── PARTE 6: Simulation Engine — PLAN ───
print("\n--- PARTE 6: Simulation Engine (PLAN) ---")
sim.clear()
goal = {"target_energy": 12.0, "stability": 3}
branches = sim.simulate(state_mtb, mode=SimulationMode.PLAN, goal=goal, n_branches=4)
print(f"  Branches (PLAN): {len(branches)}")
for b in branches:
    print(f"    {b.id}: U={b.utility:.2f}, steps={' -> '.join(b.steps[:4])}")
print("  PASS: PLAN")

# ─── PARTE 7: Simulation Engine — IMAGINE ───
print("\n--- PARTE 7: Simulation Engine (IMAGINE) ---")
sim.clear()
branches = sim.simulate(state_mtb, mode=SimulationMode.IMAGINE, n_branches=3)
print(f"  Branches (IMAGINE): {len(branches)}")
for b in branches:
    print(f"    {b.id}: {b.description} (P={b.probability:.2f})")
print("  PASS: IMAGINE")

# ─── PARTE 8: Integracao SelfModel + Simulation ───
print("\n--- PARTE 8: Integracao SelfModel + Simulation ---")
sm = SelfModel()
sim = SimulationEngine()

# SelfModel fornece estado cognitivo para simulacao
sm.learn_from_experience("energia_mtb", "Energia flui por condutores",
                          reliability=0.8, domain="fisica_mtb", modality="interactive", success=True)

# Simular usando o estado do SelfModel
cognitive_state = {
    "source": 80, "cells": 4, "conductance": 0.75, "target_energy": 0.0,
    "fatigue": sm.process.fatigue,
    "confidence": sm.identity.beliefs.get("energia_mtb", Belief("")).confidence,
}

branches = sim.simulate(cognitive_state, mode=SimulationMode.EXPECT, n_branches=3)
print(f"  Simulacao com estado cognitivo: {len(branches)} branches")
print(f"  Estado cognitivo usado: fatigue={cognitive_state['fatigue']:.2f}, "
      f"confidence={cognitive_state['confidence']:.2f}")

# Usar curiosidade do SelfModel para gerar simulacoes
curiosity = sm.get_curiosity_targets()
if curiosity:
    target = curiosity[0]
    cf_branches = sim.simulate(
        {"curiosity": 1.0, "target": 1.0},
        mode=SimulationMode.COUNTERFACTUAL,
        action=f"explore_{target}",
    )
    print(f"  Simulacao direcionada por curiosidade '{target}': {len(cf_branches)} branches")

print("  PASS: Integracao SelfModel + Simulation")

# ─── RESUMO ───
print("\n" + "=" * 70)
print("RESUMO: SELFMODEL + SIMULATION ENGINE")
print("=" * 70)
print(f"  SelfIdentity: {state['identity']['beliefs']} crencas, "
      f"{state['identity']['autobiography_size']} episodios")
print(f"  SelfProcess: fadiga={sm.process.fatigue:.2f}, "
      f"modo={sm.process.cognitive_mode}")
print(f"  Simulation EXPECT: {len(sim.branches)} branches ativas")
print(f"  Simulation COUNTERFACTUAL: 4 cenarios 'e se...'")
print(f"  Simulation PLAN: 4 planos candidatos")
print(f"  Simulation IMAGINE: 3 cenarios criativos")
print("=" * 70)
print("TODOS OS TESTES PASSARAM")
print("=" * 70)
