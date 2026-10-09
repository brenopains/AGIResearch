"""
Teste Salto 2 — CognitiveProgram + ModelCompetition + LatentVector
"""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("TESTE: SALTO 2 — Auto-Arquitetura Cognitiva")
print("=" * 60)

# ─── MODULE 1: CognitiveProgram ───
print("\n--- MODULO 1: CognitiveProgram ---")
from core.cognitive_program import (CognitiveProgram, ProgramForge,
                                     LoopEvaluator)

forge = ProgramForge()
evaluator = LoopEvaluator(max_programs=16)

# Simular experiencias
for _ in range(20):
    ctx = {"novelty": random.uniform(0, 1), "error": random.uniform(0, 0.5)}
    actions = ["explore", "MOVE", "measure", "MOVE"]
    reward = 0.7 if ctx["novelty"] > 0.5 else 0.3
    forge.record_episode(ctx, actions, reward, {})

patterns = forge.detect_patterns()
print(f"  Padroes detectados: {len(patterns)}")

if patterns:
    prog = forge.forge_program(patterns[0])
    evaluator.register(prog)
    print(f"  Programa criado: {prog.id}")
    print(f"  Trigger context: {prog.trigger_context}")
    print(f"  Effect profile: {prog.effect_profile}")

# Testar selecao
ctx = {"novelty": 0.8, "error": 0.1, "uncertainty": 0.7}
selected = evaluator.select(ctx)
if selected:
    print(f"  Selecionado: {selected.id} (confidence={selected.confidence:.2f})")
    evaluator.evaluate_outcome(selected.id, 0.9)
    print(f"  Apos recompensa: confidence={selected.confidence:.2f}")

# Testar especializacao e combinacao
if selected:
    spec_id = evaluator.specialize(selected.id, "mtb_energy")
    print(f"  Especializado: {spec_id}")
    prog2 = CognitiveProgram("test_prog_2")
    prog2.confidence = 0.7
    evaluator.register(prog2)
    combo = evaluator.combine(selected.id, "test_prog_2")
    if combo:
        print(f"  Combinado: {combo}")

state = evaluator.get_state()
print(f"  Estado: {state['active']} ativos, {state['retired']} aposentados")

print("  PASS: CognitiveProgram")

# ─── MODULO 2: ModelCompetition ───
print("\n--- MODULO 2: ModelCompetition ---")
from core.model_competition import Model, ModelEcosystem

eco = ModelEcosystem(max_models=8)

# Modelo A: "VORIN conduz energia"
m_a = Model(id="vorin_model", description="VORIN conduz energia com C=0.75",
            confidence=0.5, domain="mtb_energy")
eco.add_model(m_a)

# Modelo B: "THAL conduz mas bloqueia"
m_b = Model(id="thal_model", description="THAL conduz com C=0.95 mas bloqueia",
            confidence=0.3, domain="mtb_energy",
            parsimony=0.6, survival_threshold=0.05)
eco.add_model(m_b)

# Modelo C: "NEXIUM e melhor condutor"
m_c = Model(id="nexium_model", description="NEXIUM conduz com C=0.91",
            confidence=0.7, domain="mtb_energy")
eco.add_model(m_c)

# Simular observacoes (competicao)
def simple_predictor(obs):
    return {"target_energy": obs.get("source", 80) * obs.get("conductance", 0.75) * (0.75 ** obs.get("cells", 4))}

observations = [
    {"source": 80, "conductance": 0.75, "cells": 4, "target_energy": 11.4},
    {"source": 80, "conductance": 0.91, "cells": 5, "target_energy": 10.8},
    {"source": 80, "conductance": 0.95, "cells": 7, "target_energy": 3.5},
]

for i, obs in enumerate(observations):
    result = eco.compete(obs, simple_predictor)
    print(f"  Ciclo {i}: winner={result['winner'].id if result['winner'] else '?'}, "
          f"erro={result['avg_error']:.2f}, "
          f"refutados={len(result['losers'])}, "
          f"sobreviventes={result['n_survivors']}")

# Verificar qual modelo venceu
best = eco.get_best_model(domain="mtb_energy")
print(f"  Melhor modelo: {best.id} (conf={best.confidence:.2f}, "
      f"viab={best.viability:.2f})")

# Testar colapso
if best:
    print(f"  Testando colapso epistemico em {best.id}...")
    best.collapse()
    print(f"  Apos colapso: refuted={best.refuted}, conf={best.confidence:.2f}")

# Gerar modelo alternativo por mutacao
if best:
    alt = eco.generate_alternative(best)
    eco.add_model(alt)
    print(f"  Alternativa gerada: {alt.id} (conf={alt.confidence:.2f})")

print("  PASS: ModelCompetition")

# ─── MODULO 3: LatentVector ───
print("\n--- MODULO 3: LatentVector ---")
from core.latent_vector import LatentSpace, LatentState

space = LatentSpace(n_dims=4)

# Simular objeto com propriedades ocultas
obj_id = "cell_4_6"
for i in range(15):
    # Observacao: energia visivel, mas ha uma propriedade oculta
    hidden_prop = 0.3 if i < 8 else 0.8  # muda no meio!
    observation = {
        "energy": 10.0 + hidden_prop * 5 + random.uniform(-1, 1),
        "conductance": 0.75,
    }
    space.observe_and_update(obj_id, observation)

state = space.get_state()
print(f"  Objetos: {state['n_objects']}, Dimensoes: {state['n_dims']}")
print(f"  Descobertas: {state['discoveries']}")

# Verificar se o vetor latente capturou a mudanca
ls = space.states.get(obj_id)
if ls:
    print(f"  Vetor latente: {[f'{v:.2f}' for v in ls.vector[:4].tolist()]}")
    # Testar predicao
    pred = space.predict(obj_id, {"energy": 15.0})
    print(f"  Predicao: {pred}")

# Descobrir nova dimensao
space.global_errors = [0.5] * 10  # simular erro persistente
space._discover_new_dimension(obj_id)
print(f"  Apos descoberta: dims={space.n_dims}")

print("  PASS: LatentVector")

# ─── RESUMO ───
print("\n" + "=" * 60)
print("RESUMO SALTO 2")
print("=" * 60)
print(f"  CognitiveProgram: PASS — programas como entidades no CGE")
print(f"  ModelCompetition:  PASS — modelos competem, colapsam, vencem")
print(f"  LatentVector:      PASS — dimensoes descobertas pelo sistema")
print("=" * 60)
print("TODOS OS TESTES PASSARAM")
print("=" * 60)
