"""
test_agi_virtual_lab_fast.py — AGI + VirtualLab integration test.
"""
import sys, time
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
from core.virtual_lab import VirtualLab

lab = VirtualLab()

CANDIDATES = {
    "Alzheimer": [("BACE1_inhibition", 0.35), ("BACE1_inhibition", 0.5),
                  ("anti-inflammatory", 0.5), ("BDNF_therapy", 0.4),
                  ("immunotherapy_lecanemab", 0.6)],
    "Parkinson": [("dopamine_replacement", 0.4), ("dopamine_replacement", 0.6),
                  ("mitochondrial_enhancement", 0.5), ("anti-inflammatory", 0.5)],
    "cancer": [("chemotherapy", 0.4), ("chemotherapy", 0.6),
               ("immunotherapy", 0.5), ("immunotherapy", 0.7)],
}

print("="*60)
print("AGI -> VIRTUAL LAB (CICLO DE TESTE)")
print("="*60)

for disease, candidates in CANDIDATES.items():
    if not lab.load_disease_model(disease):
        continue

    print(f"\n--- {disease} ---")
    best_eff = -999
    best_interv = ""
    best_dose = 0

    for interv, dose in candidates:
        result = lab.simulate_intervention(interv, dose, duration_days=365)
        benefit = sum(v for k, v in result.biomarker_change.items()
                     if v > 0 and any(g in k for g in ["cognitive","synaptic",
                                                       "motor","healthy"]))
        benefit -= sum(v for k, v in result.biomarker_change.items()
                      if v < 0 and any(b in k for b in ["tumor","inflammatory",
                                                         "loss","degeneration"]))
        efficacy = benefit
        ms = result.computation_time * 1000
        print(f"    {interv:30s} dose={dose:.1f} | eff={efficacy:+.3f} | {ms:.0f}ms")

        if efficacy > best_eff:
            best_eff = efficacy
            best_interv = interv
            best_dose = dose

    print(f"\n  MELHOR: {best_interv} (dose={best_dose}, eff={best_eff:+.3f})")
    print(f"  AGI learned: {best_interv} at dose {best_dose} optimal for {disease}")

# Performance
print(f"\n{'='*60}")
print("PERFORMANCE")
print(f"{'='*60}")
lab.load_disease_model("Alzheimer")
start = time.time()
for _ in range(50):
    lab.simulate_intervention("BACE1_inhibition", 0.4, 365)
elapsed = time.time() - start
print(f"  50 simulations: {elapsed:.2f}s ({(elapsed/50)*1000:.1f}ms each)")
print(f"  Total interventions tested: {sum(len(v) for v in CANDIDATES.values())}")
print(f"\nOK: AGI + VirtualLab cycle complete")
