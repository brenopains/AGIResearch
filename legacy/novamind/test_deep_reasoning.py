"""
test_deep_reasoning.py — Testa o pipeline completo:
  200k inferencias → filtro → raciocinio convergente → cura proposta
  Com rastreabilidade completa de cada etapa.
"""
import sys, json, time
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))

from core.knowledge_graph import KnowledgeGraph
from core.inference_engine import InferenceEngine
from core.hypothesis_filter import HypothesisFilter
from core.cure_reasoner import CureReasoner

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
print(f"Base: {kg.total_ingested} conceitos, {kg.total_relations} relacoes")

# 2. Infer (divergent)
engine = InferenceEngine(kg)
start = time.time()
raw = engine.run_all(target_diseases=[])
elapsed = time.time() - start
print(f"\n--- FASE 1: DIVERGENTE ({elapsed:.1f}s) ---")
print(f"Inferencias brutas: {len(raw):,}")

# 3. Filter
filt = HypothesisFilter()
start = time.time()
filtered = filt.process(raw, min_quality=0.15)
elapsed = time.time() - start
print(f"\n--- FASE 2: FILTRAGEM ({elapsed:.1f}s) ---")
print(f"Apos filtro: {len(filtered):,} ({len(filtered)/max(1,len(raw))*100:.1f}% do total)")
print(f"Candidatas terapeuticas: {len(filt.get_therapeutic_candidates())}")
print(f"Plausibilidade biologica: {sum(1 for h in filtered if h.is_biologically_plausible)}")

# 4. Reason (convergent) for multiple diseases
reasoner = CureReasoner()
diseases = ["Alzheimer", "Parkinson", "depression", "schizophrenia",
            "cancer", "diabetes", "stroke", "ALS", "Huntington", "autism"]

print(f"\n--- FASE 3: RACIOCINIO CONVERGENTE ---")
for disease in diseases:
    start = time.time()
    cure = reasoner.reason_deep(disease, filtered, kg, max_rounds=8)
    elapsed = time.time() - start
    print(f"\n  {disease}: {cure.intervention[:60]}")
    print(f"    Mecanismo: {cure.mechanism}")
    print(f"    Confianca: {cure.confidence:.2f}")
    print(f"    Experimento: {cure.suggested_experiment[:80]}...")
    print(f"    ({elapsed:.2f}s)")

# 5. Traceability report
trace = reasoner.get_traceability_report("Alzheimer")
trace_path = "D:/novamind_data/agi_alzheimer_cure_trace.md"
with open(trace_path, "w", encoding="utf-8") as f:
    f.write(trace)

print(f"\n\n=== RELATORIO DE RASTREABILIDADE ===")
print(f"Alzheimer: {trace_path}")
print(trace[:1500])
print(f"\n[... full trace saved]")

# Stats
print(f"\n\n=== RESUMO ===")
print(f"{'Etapa':30s} {'Quantidade':>12s} {'Tx reducao':>12s}")
print(f"{'-'*55}")
print(f"{'Inferencias brutas':30s} {len(raw):>12,d} {'100%':>12s}")
print(f"{'Apos filtro':30s} {len(filtered):>12,d} {len(filtered)/max(1,len(raw))*100:>11.1f}%")
ther = len(filt.get_therapeutic_candidates())
print(f"{'Candidatas terapeuticas':30s} {ther:>12,d} {ther/max(1,len(filtered))*100:>11.1f}%")
print(f"{'Doencas com cura proposta':30s} {len(reasoner.proposals):>12,d} {'':>12s}")
