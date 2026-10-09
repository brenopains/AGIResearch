"""
test_agi_scientific_creativity.py — Mede a CRIATIVIDADE CIENTIFICA da AGI.

Metricas:
  - Modelos criados (novas estruturas causais)
  - Modelos descartados (nao superaram o pai)
  - Modelos que superaram o pai (evoluiram)
  - Queda de erro por INVENCAO ESTRUTURAL (nao por ajuste de parametro)
  - Taxa de sucesso das mutacoes

O objetivo NAO e minimizar erro de simulacao.
O objetivo e MAXIMIZAR a taxa de descoberta de novas estruturas.

A AGI prova que e AGI porque INVENTA novas hipoteses causais,
nao apenas ajusta parametros de modelos existentes.
"""
import sys, json, time
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))

import numpy as np
import random

from core.model_library import ModelLibrary
from core.parameter_calibrator import ParameterCalibrator
from core.deep_structure_mutator import DeepStructureMutator

# ─── REALIDADE: ADNI ───
REALITY = {
    "cognitive_function": 0.770,
    "neurodegeneration": 0.249,
}

def compute_error(model):
    """Erro do modelo contra dados ADNI."""
    from scipy.integrate import solve_ivp
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


# ─── MAIN ───

print("="*70)
print("TESTE DE CRIATIVIDADE CIENTIFICA DA AGI")
print("="*70)

# Carrega modelos
lib = ModelLibrary()
models = lib.get_models("Alzheimer")
calibrator = ParameterCalibrator(REALITY)
mutator = DeepStructureMutator()

pool = list(models)
error_cache = {}

def get_error(m):
    if m.name not in error_cache:
        error_cache[m.name] = compute_error(m)
    return error_cache[m.name]

# Inicializa
print(f"\nModelos iniciais: {len(pool)}")
for m in pool:
    e = get_error(m)
    print(f"  {m.name:25s} erro={e:.4f}")

# Metricas de inventividade
metrics = {
    "models_created": 0,
    "models_discarded": 0,
    "models_improved": 0,
    "errors_by_generation": [],
    "structural_mutations_tried": 0,
    "structural_mutations_successful": 0,
    "calibrations_performed": 0,
    "calibrations_improved": 0,
}

best_error = min(get_error(m) for m in pool)
last_improvement_gen = 0

print(f"\n--- CICLO DE INVENCAO CIENTIFICA ---")
print(f"{'Etapa':20s} {'Modelos':>8s} {'Melhor erro':>12s} {'Acao':>35s}")
print(f"{'-'*75}")

for cycle in range(100):
    # Fase 1: Tenta CALIBRAR o melhor modelo (ajuste de parametros)
    best_model = min(pool, key=lambda m: get_error(m))
    best_err = get_error(best_model)
    error_cache = {}  # limpa cache para reavaliar

    cal_result, calibrated = calibrator.calibrate(best_model, n_iterations=100)
    metrics["calibrations_performed"] += 1
    cal_err = compute_error(calibrated)

    if cal_err < best_err - 0.001:
        calibrated.name = f"{best_model.name}_cal{cycle}"
        pool.append(calibrated)
        best_err = cal_err
        metrics["calibrations_improved"] += 1
        error_cache[calibrated.name] = cal_err
        print(f"{'Calibrate':20s} {len(pool):8d} {best_err:12.4f} {f'Calibration improved {cal_result.improvement:.1%}':35s}")
        last_improvement_gen = cycle
        continue

    # Fase 2: Tenta MUTAR ESTRUTURALMENTE (inventar nova hipotese)
    children = mutator.mutate(best_model, n_mutations=2)
    metrics["structural_mutations_tried"] += len(children)

    best_child_err = 999.0
    best_child = None

    for child in children:
        metrics["models_created"] += 1
        child_err = get_error(child)

        if child_err < best_err:
            metrics["structural_mutations_successful"] += 1
            if child_err < best_child_err:
                best_child_err = child_err
                best_child = child

    if best_child is not None and best_child_err < best_err - 0.001:
        pool.append(best_child)
        best_err = best_child_err
        metrics["models_improved"] += 1
        last_improvement_gen = cycle
        print(f"{'StructuralMut':20s} {len(pool):8d} {best_err:12.4f} {f'NEW: {best_child.mutation_description[:35]}':35s}")
    else:
        metrics["models_discarded"] += len(children)
        if cycle % 10 == 0:
            print(f"{'Stagnant':20s} {len(pool):8d} {best_err:12.4f} {'No improvement this cycle':35s}")

    # Poda: mantem no maximo 15 modelos
    if len(pool) > 15:
        pool.sort(key=lambda m: get_error(m))
        pool = pool[:15]

    metrics["errors_by_generation"].append(best_err)

    # Criterio de parada: 15 ciclos sem melhoria
    if cycle - last_improvement_gen > 15:
        print(f"\nConverged after {cycle} cycles ({15} cycles without improvement)")
        break

# ─── RELATORIO FINAL ───
print(f"\n{'='*70}")
print(f"RELATORIO DE CRIATIVIDADE CIENTIFICA")
print(f"{'='*70}")

best_final = min(pool, key=lambda m: get_error(m))

print(f"\n  Erro inicial: {min(get_error(m) for m in models):.4f}")
print(f"  Erro final:   {get_error(best_final):.4f}")
print(f"  Melhoria:     {(1 - get_error(best_final)/max(0.001, min(get_error(m) for m in models)))*100:.1f}%")
print(f"\n  Modelos criados (novas hipoteses):     {metrics['models_created']}")
print(f"  Modelos descartados (refutados):      {metrics['models_discarded']}")
print(f"  Modelos que superaram o melhor:       {metrics['models_improved']}")
print(f"  Mutacoes estruturais tentadas:        {metrics['structural_mutations_tried']}")
print(f"  Mutacoes estruturais bem-sucedidas:   {metrics['structural_mutations_successful']}")
print(f"  Calibracoes de parametro realizadas:  {metrics['calibrations_performed']}")
print(f"  Calibracoes que melhoraram:           {metrics['calibrations_improved']}")

if metrics["structural_mutations_successful"] > 0:
    taxa = metrics["structural_mutations_successful"] / max(1, metrics["structural_mutations_tried"])
    print(f"\n  Taxa de sucesso de mutacoes: {taxa:.1%}")
    print(f"  A AGI inventou {metrics['models_improved']} novas hipoteses causais que superaram as anteriores.")

print(f"\n  Melhor modelo: {best_final.name}")
print(f"  Hipotese: {best_final.hypothesis[:80] if hasattr(best_final, 'hypothesis') else 'Modelo evoluido'}")

# Mutacoes realizadas
print(f"\n  Mutacoes realizadas (ultimas 10):")
for m in mutator.history[-10:]:
    print(f"    {m.operation:25s} | {m.description[:60]}")

# Salva
report = {
    "models_created": metrics["models_created"],
    "models_discarded": metrics["models_discarded"],
    "models_improved": metrics["models_improved"],
    "structural_mutations": metrics["structural_mutations_successful"],
    "error_initial": min(get_error(m) for m in models),
    "error_final": get_error(best_final),
    "winner": best_final.name,
    "mutations": [m.to_dict() for m in mutator.history],
}
report_path = "D:/novamind_data/creativity_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nRelatorio: {report_path}")
