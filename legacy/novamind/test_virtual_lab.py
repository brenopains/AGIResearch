"""
test_virtual_lab.py — Testa o VirtualLab com a AGI.
Ciclo completo: AGI propoe cura → VirtualLab simula → AGI aprende com resultado.
"""
import sys, json
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))

from core.virtual_lab import VirtualLab
import numpy as np

print("=== VIRTUAL LAB TEST ===\n")

lab = VirtualLab()
print(f"Doencas disponiveis: {lab.list_diseases()}")

# 1. Teste Alzheimer
print("\n--- Alzheimer ---")
lab.load_disease_model("Alzheimer")

# Controle (sem intervencao)
ctrl = lab.simulate_intervention("none", dose=0.0, duration_days=730)
print(f"  Controle (2 anos):")
for k, v in sorted(ctrl.biomarker_change.items()):
    print(f"    {k}: {v:+.1%}")

# Intervencao: anti-amyloid
result = lab.simulate_intervention("BACE1_inhibition", dose=0.4, duration_days=730)
print(f"\n  BACE1_inhibition (dose=0.4, 2 anos):")
for k, v in sorted(result.biomarker_change.items()):
    print(f"    {k}: {v:+.1%}")
if result.adverse_events:
    print(f"  Eventos adversos: {result.adverse_events}")

# Intervencao: anti-inflammatory
result2 = lab.simulate_intervention("anti-inflammatory", dose=0.6, duration_days=730)
print(f"\n  anti-inflammatory (dose=0.6, 2 anos):")
for k, v in sorted(result2.biomarker_change.items()):
    print(f"    {k}: {v:+.1%}")

# 2. Teste Parkinson
print("\n--- Parkinson ---")
lab.load_disease_model("Parkinson")
ctrl = lab.simulate_intervention("none", 0, 730)
print(f"  Controle (2 anos):")
for k, v in sorted(ctrl.biomarker_change.items()):
    print(f"    {k}: {v:+.1%}")

result3 = lab.simulate_intervention("dopamine_replacement", dose=0.5, duration_days=730)
print(f"\n  dopamine_replacement (dose=0.5, 2 anos):")
for k, v in sorted(result3.biomarker_change.items()):
    print(f"    {k}: {v:+.1%}")

# 3. Teste Cancer
print("\n--- Cancer ---")
lab.load_disease_model("cancer")
result4 = lab.simulate_intervention("immunotherapy", dose=0.7, duration_days=365)
print(f"  immunotherapy (dose=0.7, 1 ano):")
for k, v in sorted(result4.biomarker_change.items()):
    print(f"    {k}: {v:+.1%}")

result5 = lab.simulate_intervention("chemotherapy", dose=0.5, duration_days=365)
print(f"\n  chemotherapy (dose=0.5, 1 ano):")
for k, v in sorted(result5.biomarker_change.items()):
    print(f"    {k}: {v:+.1%}")

# 4. Performance
print(f"\n--- Performance ---")
import time
times = []
for _ in range(20):
    start = time.time()
    lab.simulate_intervention("BACE1_inhibition", 0.4, 365)
    times.append(time.time() - start)
print(f"  20 simulacoes: media={np.mean(times)*1000:.1f}ms, "
      f"min={np.min(times)*1000:.1f}ms, max={np.max(times)*1000:.1f}ms")

print(f"\n✅ VirtualLab operacional")
