"""
test_dmn_cycle.py — Ciclo DMN completo: falha -> analogia -> invento -> novo modelo -> teste.

Quando a AGI esta presa (nao melhora), ela:
  1. Ativa DMN: busca analogias em dominios DISTANTES
  2. Inventa conceitos NOVOS que podem estar faltando
  3. Gera modelos inspirados nas analogias
  4. Testa contra a realidade (ADNI)
  5. Se melhorou -> novo modelo base
  6. Se nao -> volta para DMN com erro maior

Isso e CIENCIA REAL: quando o paradigma falha, busca-se inspiracao em outro lugar.
"""
import sys, json, time
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))

import numpy as np
from scipy.integrate import solve_ivp
from core.cross_domain_analogy import CrossDomainAnalogyEngine
from core.concept_inventor import ConceptInventor
from core.parameter_calibrator import ParameterCalibrator
from core.model_library import ModelLibrary

REALITY = {"cognitive_function": 0.770, "neurodegeneration": 0.249}

def compute_error(model):
    try:
        sol = solve_ivp(model.ode_fn, (0, 365), model.initial_state,
                       method="RK45", max_step=50, rtol=1e-3, atol=1e-6)
        if sol.y.shape[1] == 0: return 1.0
        errors = []
        for var, obs in REALITY.items():
            if var in model.variables:
                idx = model.variables.index(var)
                pred = float(sol.y[idx, -1]) if idx < sol.y.shape[0] else 0.5
                pred = 1/(1+abs(pred)) if abs(pred) > 1 else max(0, min(1, pred))
                if abs(obs) > 0.01:
                    errors.append(abs(pred - obs) / obs)
        return float(np.mean(errors)) if errors else 1.0
    except:
        return 1.0


print("="*70)
print("CICLO DMN: FALHA -> ANALOGIA -> INVENCAO -> NOVO MODELO")
print("="*70)

lib = ModelLibrary()
models = lib.get_models("Alzheimer")
analogy_engine = CrossDomainAnalogyEngine()
concept_inventor = ConceptInventor()
calibrator = ParameterCalibrator(REALITY)

pool = list(models)
error_cache = {}

def get_error(m):
    if m.name not in error_cache:
        error_cache[m.name] = compute_error(m)
    return error_cache[m.name]

best_error = min(get_error(m) for m in pool)
best_model = min(pool, key=lambda m: get_error(m))

print(f"\nErro inicial: {best_error:.4f} ({best_model.name})")
print(f"\n--- CICLO DMN ---\n")

dmn_cycles = 0
analogies_applied = 0
concepts_invented = 0
models_from_dmn = 0

for cycle in range(30):
    # Atualiza melhor
    best_error = min(get_error(m) for m in pool)
    best_model = min(pool, key=lambda m: get_error(m))
    
    # Se erro e baixo -> calibracao fina
    if best_error < 0.5:
        result, calibrated = calibrator.calibrate(best_model, n_iterations=50)
        cal_err = get_error(calibrated)
        if cal_err < best_error:
            error_cache[calibrated.name] = cal_err
            pool.append(calibrated)
            best_error = cal_err
            print(f"  Ciclo {cycle}: Calibracao melhorou para {best_error:.4f}")
        continue

    # DMN: FALHA -> buscar analogias em dominios DISTANTES
    dmn_cycles += 1
    print(f"\n  DMN Cycle {dmn_cycles}: Erro={best_error:.4f} > 0.5. Buscando analogias...")

    # 1. Busca analogias
    analogies = analogy_engine.search("Alzheimer", n_analogies=3)
    for a in analogies:
        new_model = analogy_engine.apply_analogy(best_model, a)
        err = get_error(new_model)
        analogies_applied += 1
        if err < best_error:
            pool.append(new_model)
            best_error = err
            models_from_dmn += 1
            print(f"    [OK] Analogia de {a.source_domain}: erro caiu para {err:.4f}")
            print(f"         {a.description[:60]}...")
        else:
            print(f"    [--] Analogia de {a.source_domain}: erro={err:.4f} (nao melhorou)")

    # 2. Inventa conceitos
    concepts = concept_inventor.invent("Alzheimer", error=best_error, n_concepts=2)
    for c in concepts:
        new_model = concept_inventor.apply_to_model(best_model, c)
        err = get_error(new_model)
        concepts_invented += 1
        if err < best_error:
            pool.append(new_model)
            best_error = err
            models_from_dmn += 1
            print(f"    [OK] Conceito inventado '{c.name}': erro caiu para {err:.4f}")
        else:
            print(f"    [--] Conceito '{c.name}': erro={err:.4f} (nao melhorou)")

    # Poda
    if len(pool) > 10:
        pool.sort(key=lambda m: get_error(m))
        pool = pool[:10]
    
    # Se melhorou, proximo ciclo
    if best_error < 0.5:
        print(f"  Erro caiu abaixo de 0.5! Mudando para modo calibracao.\n")
        continue

    # Se nao melhorou em 3 ciclos DMN, para
    if dmn_cycles >= 3:
        print(f"\n  DMN exauriu apos {dmn_cycles} ciclos. Melhor erro: {best_error:.4f}")
        break


print(f"\n{'='*70}")
print(f"RESULTADO DO CICLO DMN")
print(f"{'='*70}")
print(f"  Erro inicial: {min(get_error(m) for m in models):.4f}")
print(f"  Erro final:   {best_error:.4f}")
print(f"  Melhoria:     {(1-best_error/max(0.001, min(get_error(m) for m in models)))*100:.1f}%")
print(f"\n  Ciclos DMN ativados: {dmn_cycles}")
print(f"  Analogias testadas:   {analogies_applied}")
print(f"  Conceitos inventados: {concepts_invented}")
print(f"  Modelos da DMN:       {models_from_dmn}")
print(f"  Dominios explorados:  {len(set(a.source_domain for a in analogy_engine.history))}")

print(f"\n  Analogias utilizadas:")
for a in analogy_engine.history[-5:]:
    print(f"    {a.source_domain:20s} | {a.principle[:50]}")

print(f"\n  Conceitos inventados:")
for c in concept_inventor.inventions[-5:]:
    print(f"    {c.name:25s} | {c.rationale[:50]}")

print(f"\n  Melhor modelo final: {min(pool, key=lambda m: get_error(m)).name}")
print(f"  Erro final: {best_error:.4f}")

report = {
    "error_initial": float(min(get_error(m) for m in models)),
    "error_final": float(best_error),
    "dmn_cycles": dmn_cycles,
    "analogies_tested": analogies_applied,
    "concepts_invented": concepts_invented,
    "models_from_dmn": models_from_dmn,
    "domains_explored": len(set(a.source_domain for a in analogy_engine.history)),
}
report_path = "D:/novamind_data/dmn_cycle_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nRelatorio: {report_path}")
