"""
test_model_comparison_adni.py — Compara modelos rivais com dados REAIS do ADNI.
Pela primeira vez, a AGI pode discriminar qual modelo explica melhor os dados observados.
"""
import sys, json
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))

from core.model_library import ModelLibrary
from core.model_comparator import ModelComparator
from scipy.integrate import solve_ivp
import numpy as np

# Carrega ADNI observations
with open("D:/novamind_data/quantitative_data/adni_observations.json") as f:
    adni_obs = json.load(f)

print("="*65)
print("COMPARACAO BAYESIANA COM DADOS REAIS DO ADNI")
print("="*65)
print(f"\nDados ADNI:")
for o in adni_obs:
    print(f"  {o['group']:3s} | {o['biomarker']:25s} | MMSE/ND={o['value']:.2f} | change={o['change']:+.3f} | n={o['n']}")

# Prepara observacoes no formato que o comparator espera
# Agora com grupos (CN, AD) para comparacao de DIFERENCAS
observations = {
    "cognitive_function_CN": 29.07 / 30.0,
    "cognitive_function_AD": 23.09 / 30.0,
    "neurodegeneration_CN": 0.04 / 18.0,
    "neurodegeneration_AD": 4.48 / 18.0,
}

print(f"\nObservacoes para o comparador:")
for k, v in observations.items():
    print(f"  {k}: {v:.3f}")

# Carrega modelos rivais
lib = ModelLibrary()
models = lib.get_models("Alzheimer")
print(f"\nModelos rivais: {len(models)}")
for m in models:
    print(f"  {m.name:30s} | {m.hypothesis[:50]}")

# Compara TODOS os modelos com dados REAIS
comparator = ModelComparator()
comparisons = comparator.compare(models, observations)

print(f"\n{'='*65}")
print(f"RESULTADO - MODELO VENCEDOR")
print(f"{'='*65}")
print(f"\n{'Modelo':30s} {'Posterior':>10s} {'Bayes':>8s} {'Vars':>5s} {'Erro':>10s} {'Best':>6s}")
print(f"{'-'*70}")
for c in comparisons:
    best_mark = "<<<" if c.is_best else ""
    print(f"{c.model_name:30s} {c.posterior:>10.1f} {c.bayes_factor_vs_best:>8.3f} "
          f"{c.n_vars:>5d} {c.prediction_error:>10.4f} {best_mark:>6s}")

best = comparisons[0]

print(f"\n\nMODELO VENCEDOR: {best.model_name}")
print(f"Hipotese: {[m.hypothesis for m in models if m.name==best.model_name][0][:80]}")
print(f"Posterior: {best.posterior:.1f}")
print(f"Bayes factor: {best.bayes_factor_vs_best:.3f}")
print(f"Erro de predicao: {best.prediction_error:.4f}")

# Mostra a diferenca entre o melhor e o segundo
if len(comparisons) > 1:
    delta = best.posterior - comparisons[1].posterior
    print(f"\nDiferenca para o 2o lugar: {delta:.1f}")

print(f"\nPela PRIMEIRA VEZ, a AGI pode discriminar entre modelos rivais")
print(f"usando dados REAIS de pacientes (ADNI, n=1.857).")
