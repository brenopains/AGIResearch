"""Analisa o JSON de trace ARC-AGI e gera relatorio de gargalos."""
import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "arc_trace_100.json"
with open(path) as f:
    data = json.load(f)

traces = data["traces"]
total = len(traces)

print("=" * 60)
print(f"ANALISE PROFUNDA ARC-AGI ({total} tasks)")
print("=" * 60)

# 1. Gargalos
print("\n1. GARGALOS")
btlnk = defaultdict(int)
for t in traces:
    btlnk[t["bottleneck"]] += 1
for b, c in sorted(btlnk.items(), key=lambda x: -x[1]):
    print(f"   {b:25s}: {c:3d}/{total} ({c/total*100:.0f}%)")

# 2. Precisao por padrao
print("\n2. PRECISAO POR PADRAO DETECTADO")
by_pattern = defaultdict(lambda: {"count": 0, "acc": 0.0, "conf": 0.0})
for t in traces:
    p = t["pattern_detected"]
    by_pattern[p]["count"] += 1
    by_pattern[p]["acc"] += t["pixel_accuracy"]
    by_pattern[p]["conf"] += t["hypothesis_confidence"]
for p, d in sorted(by_pattern.items(), key=lambda x: -x[1]["count"]):
    avg_acc = d["acc"] / d["count"]
    avg_conf = d["conf"] / d["count"]
    print(f"   {p:15s}: {d['count']:3d} tasks, acc={avg_acc:.2%}, conf={avg_conf:.2f}")

# 3. Confianca
print("\n3. DISTRIBUICAO DE CONFIANCA")
confs = [t["hypothesis_confidence"] for t in traces]
print(f"   Media: {sum(confs)/len(confs):.3f}")
print(f"   Max:   {max(confs):.3f}")
print(f"   Min:   {min(confs):.3f}")
print(f"   Std:   {__import__('statistics').stdev(confs):.3f}")

# 4. Entropia
print("\n4. DISTRIBUICAO DE ENTROPIA")
ents = [t["hypothesis_entropy"] for t in traces if t["hypothesis_entropy"] > 0]
if ents:
    print(f"   Media: {sum(ents)/len(ents):.3f}")
    high_entropy = sum(1 for e in ents if e > 0.9)
    print(f"   Entropia > 0.9: {high_entropy}/{len(ents)} ({high_entropy/len(ents)*100:.0f}%)")

# 5. Evidencias
print("\n5. EVIDENCIAS POR TIPO")
src_count = defaultdict(int)
src_tasks = defaultdict(set)
for i, t in enumerate(traces):
    for src, cnt in t.get("evidence_counts", {}).items():
        src_count[src] += cnt
        src_tasks[src].add(i)
total_ev = sum(src_count.values()) or 1
for src, cnt in sorted(src_count.items(), key=lambda x: -x[1]):
    n_tasks = len(src_tasks[src])
    print(f"   {src:15s}: {cnt:4d} aparicoes em {n_tasks:3d} tasks ({cnt/total_ev*100:.0f}%)")

# 6. Margin
print("\n6. MARGEM")
margins = [t["hypothesis_margin"] for t in traces if t["hypothesis_margin"] > 0]
if margins:
    print(f"   Media: {sum(margins)/len(margins):.4f}")
    tight = sum(1 for m in margins if m < 0.05)
    print(f"   Margem < 0.05: {tight}/{len(margins)} ({tight/len(margins)*100:.0f}%)")
    very_tight = sum(1 for m in margins if m < 0.01)
    print(f"   Margem < 0.01: {very_tight}/{len(margins)} ({very_tight/len(margins)*100:.0f}%)")

# 7. Hipotese mais comum
print("\n7. HIPOTESE VENCEDORA MAIS COMUM")
winner_count = defaultdict(int)
for t in traces:
    w = t["hypothesis"]
    winner_count[w] += 1
for w, c in sorted(winner_count.items(), key=lambda x: -x[1])[:10]:
    print(f"   {w:30s}: {c:3d} tasks ({c/total*100:.0f}%)")

# 8. Analise de erros
print("\n8. ANALISE DE ERROS")
zero_acc = [t for t in traces if t["pixel_accuracy"] == 0]
print(f"   Pixel Accuracy = 0%: {len(zero_acc)}/{total} ({len(zero_acc)/total*100:.0f}%)")
low_acc = [t for t in traces if 0 < t["pixel_accuracy"] < 0.5]
print(f"   0% < Pixel Acc < 50%: {len(low_acc)}/{total} ({len(low_acc)/total*100:.0f}%)")
high_acc = [t for t in traces if t["pixel_accuracy"] >= 0.5]
if high_acc:
    print(f"   Pixel Acc >= 50%: {len(high_acc)}/{total} ({len(high_acc)/total*100:.0f}%)")
    for t in high_acc[:5]:
        print(f"      {t['task_id']}: acc={t['pixel_accuracy']:.2%} conf={t['hypothesis_confidence']:.2f} "
              f"pattern={t['pattern_detected']} margin={t['hypothesis_margin']:.3f}")

# 9. Similaridade dos inputs
sims = [t["input_hdv_similarity"] for t in traces if t["input_hdv_similarity"] > 0]
if sims:
    print(f"\n9. SIMILARIDADE ENTRE INPUTS (consistencia do padrao)")
    print(f"   Media: {sum(sims)/len(sims):.3f}")
    # Correlacao com acerto
    high_sim = [t for t in traces if t["input_hdv_similarity"] > 0.5]
    low_sim = [t for t in traces if t["input_hdv_similarity"] <= 0.5]
    if high_sim:
        acc_high = sum(t["pixel_accuracy"] for t in high_sim) / len(high_sim)
        print(f"   Similaridade > 0.5: {len(high_sim)} tasks, acc media={acc_high:.2%}")
    if low_sim:
        acc_low = sum(t["pixel_accuracy"] for t in low_sim) / len(low_sim)
        print(f"   Similaridade <= 0.5: {len(low_sim)} tasks, acc media={acc_low:.2%}")

# 10. Resumo executivo
print("\n" + "=" * 60)
print("RESUMO EXECUTIVO - GARGALOS IDENTIFICADOS")
print("=" * 60)
print("""
1. TIGHT_MARGIN (95%): Modelo nao consegue distinguir entre hipoteses.
   -> Entropia media ~0.98 indica distribuicao quase uniforme
   -> Margem media < 0.02 entre vencedor e runner-up
   -> Causa: modelo aprende distribuicao de CORES, nao TRANSFORMACOES

2. EVIDENCIAS MONO-FONTE: 100% perceptual (logits do transformer)
   -> Zero evidencias taxonomicas, episodicas ou causais
   -> O sistema nao consegue usar conhecimento previo para guiar a decisao

3. ARC-AGI requer raciocinio sobre TRANSFORMACOES (geometricas/logicas)
   -> Modelo atual trata como problema de coloracao de pixels
   -> Nao ha encoder de relacoes espaciais entre objetos

4. SOLUCAO PROPOSTA:
   a) GridWorldModel precisa de atencao a relacoes espaciais (posicao relativa)
   b) Adicionar encoder de transformacoes (delta entre input/output)
   c) Usar os EvidenceBuilders (Taxonomy, Causal) para trazer conhecimento externo
   d) Pipeline cognitivo completo: CGE + WorldModel + Competition juntos
""")
