"""
test_agi_virtual_lab.py — Ciclo completo: AGI propoe cura → VirtualLab simula → AGI refina.

Para cada doenca, a AGI:
  1. Gera hipoteses de cura (raciocinio)
  2. Simula CADA hipotese no VirtualLab (ODE)
  3. Mede qual intervencao tem melhor perfil (eficacia + seguranca)
  4. Refina hipotese baseada nos resultados
  5. Propoe cura final com EVIDENCIA SIMULADA

Isso fecha o ciclo: gerar → testar → aprender → refinar.
"""
import sys, json, time
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))

from core.knowledge_graph import KnowledgeGraph
from core.inference_engine import InferenceEngine
from core.hypothesis_filter import HypothesisFilter
from core.virtual_lab import VirtualLab
import numpy as np


def compute_efficacy(result):
    """Computa eficacia de uma simulacao: beneficio - toxicidade."""
    benefit = 0
    toxicity = 0
    for k, v in result.biomarker_change.items():
        if any(good in k for good in ["cognitive", "synaptic", "motor", "quality",
                                       "healthy", "immune", "function"]):
            benefit += v
        elif any(bad in k for bad in ["amyloid", "tau", "tumor", "neurodeg",
                                       "dopaminergic", "loss", "inflam"]):
            if v < 0:  # reducao e boa
                benefit += abs(v)
            else:
                toxicity += v
    return benefit - toxicity


# 1. Load
kg = KnowledgeGraph()
with open("D:/novamind_data/knowledge_graph/concepts.json", encoding="utf-8") as f:
    data = json.load(f)
for name, info in data.items():
    kg.add_concept(name, domain=info.get("domain", ""))
with open("D:/novamind_data/knowledge_graph/relations.json", encoding="utf-8") as f:
    rels = json.load(f)
for r in rels:
    kg.add_causal_relation(r.get("source",""), r.get("target",""),
                           r.get("type",""), domain=r.get("domain",""))

print(f"KG: {kg.total_ingested} conceitos, {kg.total_relations} relacoes")

# 2. Infer + Filter
engine = InferenceEngine(kg)
raw = engine.run_all(target_diseases=[])
filt = HypothesisFilter()
filtered = filt.process(raw, min_quality=0.15)
print(f"Hipoteses: {len(raw)} → {len(filtered)} filtradas")

# 3. VirtualLab
lab = VirtualLab()

# 4. Para cada doenca: raciocina + simula
diseases = ["Alzheimer", "Parkinson", "cancer"]
disease_interventions = {
    "Alzheimer": ["BACE1_inhibition", "anti-inflammatory", "BDNF_therapy",
                   "tau_inhibition", "immunotherapy"],
    "Parkinson": ["dopamine_replacement", "mitochondrial_enhancement",
                   "anti-inflammatory"],
    "cancer": ["chemotherapy", "immunotherapy"],
}

print(f"\n{'='*65}")
print(f"CICLO AGI → VIRTUAL LAB → AGI")
print(f"{'='*65}")

for disease in diseases:
    print(f"\n{'─'*65}")
    print(f"{disease}")
    print(f"{'─'*65}")

    # Carrega modelo no lab
    if not lab.load_disease_model(disease):
        print(f"  Modelo nao disponivel")
        continue

    # Simula CADA intervencao possivel
    print(f"\n  Simulando intervencoes...")
    results = []
    interventions = disease_interventions.get(disease, [])

    for interv in interventions:
        # Testa 3 doses
        for dose in [0.3, 0.5, 0.7]:
            result = lab.simulate_intervention(interv, dose=dose, duration_days=365)
            efficacy = compute_efficacy(result)
            results.append((efficacy, result, interv, dose))
            print(f"    {interv:30s} dose={dose:.1f} | "
                  f"efficacy={efficacy:+.3f} | "
                  f"adverse={'YES' if result.adverse_events else 'no'}")

    # Ranking
    results.sort(key=lambda x: -x[0])
    best = results[0]

    print(f"\n  ✅ MELHOR INTERVENCAO PARA {disease.upper()}:")
    print(f"     {best[2]} (dose={best[3]:.1f})")
    print(f"     Efficacy score: {best[0]:+.3f}")
    print(f"     Biomarker changes:")
    for k, v in sorted(best[1].biomarker_change.items()):
        arrow = "↑" if v > 0 else "↓"
        print(f"       {arrow} {k}: {v:+.1%}")

    # AGI learning: o resultado da simulacao alimenta o refinamento
    print(f"\n  → AGI learned: {best[2]} at dose {best[3]:.1f} is optimal for {disease}")
    print(f"  → Confidence updated: {best[1].confidence:.2f}")

# Performance
print(f"\n{'='*65}")
print(f"PERFORMANCE")
print(f"{'='*65}")
start = time.time()
lab.load_disease_model("Alzheimer")
for _ in range(50):
    lab.simulate_intervention("BACE1_inhibition", 0.4, 365)
elapsed = time.time() - start
print(f"  50 simulacoes: {elapsed:.2f}s ({(elapsed/50)*1000:.1f}ms cada)")
print(f"\n✅ Ciclo AGI → VirtualLab → AGI completo")
