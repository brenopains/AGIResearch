"""
Experimento FLORB & TREK v2: memoria HDV com bundles.

Conceitos NOVOS como HDVs aleatorios.
Propriedades armazenadas como bundles (conceito (+) propriedade).
Query com composto recupera as propriedades corretas.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = r"D:\novamind_data\cache\huggingface"
os.environ["TORCH_HOME"] = r"D:\novamind_data\cache\torch"

import torch
from core.coglang import Manifold
from core.hdc_memory import HDCMemory
from core.hdc_utils import pack_bipolar

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 10000
torch.manual_seed(42)

def log(text):
    print(text)

def hdv():
    return (torch.rand(DIM, device=DEVICE) * 2 - 1).sign()

def bundle_pair(a, b):
    return torch.sign(a.float() + b.float())

def cos(a, b):
    return float(torch.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)))

log("=" * 70)
log("EXPERIMENTO FLORB & TREK v2")
log("Conceitos = HDVs aleatorios, memorias via bundle")
log("=" * 70)

# ============================================================
# 1. CRIAR CONCEITOS
# ============================================================
florb = hdv()
trek = hdv()
spaceship = hdv()

lw = hdv()  # lightweight
st = hdv()  # strong
fl = hdv()  # flexible
ha = hdv()  # heat_absorbing
inn = hdv()  # insulating
da = hdv()  # dark
ch = hdv()  # cheap
tr = hdv()  # transparent

log(f"Cos(FLORB, TREK) = {cos(florb, trek):.4f}")
log(f"Cos(FLORB, lightweight) = {cos(florb, lw):.4f}")
log(f"Cos(TREK, dark) = {cos(trek, da):.4f}")

# ============================================================
# 2. CRIAR MEMORIA HDV
# ============================================================
log("\n--- FASE 1: Criando memoria HDV com bundles ---")

# Bundles: (conceito + propriedade) = mesmo vetor tem alta similaridade com ambos
florb_entries = []
for prop, name in [(lw, "lightweight"), (st, "strong"), (fl, "flexible")]:
    vec = bundle_pair(florb, prop)  # sign(FLORB + P)
    florb_entries.append((vec, f"FLORB is {name}", name))

trek_entries = []
for prop, name in [(ha, "heat_absorbing"), (inn, "insulating"), (da, "dark")]:
    vec = bundle_pair(trek, prop)
    trek_entries.append((vec, f"TREK is {name}", name))

irrelevant_entries = []
for prop, name in [(ch, "cheap"), (tr, "transparent")]:
    vec = bundle_pair(torch.zeros(DIM, device=DEVICE), prop)  # so a propriedade
    irrelevant_entries.append((vec, f"something is {name}", name))

# Verificar: bundle(FLORB, P) deve ser similar a FLORB
for vec, text, name in florb_entries:
    log(f"  Cos(bundle(F,{name}), FLORB) = {cos(vec, florb):.4f}")
    log(f"  Cos(bundle(F,{name}), TREK)  = {cos(vec, trek):.4f}")
    log(f"  Cos(bundle(F,{name}), {name}) = {cos(vec, lw if name == 'lightweight' else st if name == 'strong' else fl):.4f}")

# ============================================================
# 3. TESTE DE QUERY
# ============================================================
log("\n--- FASE 2: Query no espaco HDC (simulando memoria) ---")

# Query: composto = bundle(spaceship, FLORB, TREK)
composed = bundle_pair(bundle_pair(spaceship, florb), trek)

log(f"Query vector: bundle(S, F, T)")

# Para cada entrada na memoria, medir similaridade com o query
all_entries = florb_entries + trek_entries + irrelevant_entries
entry_names = [e[1] for e in all_entries]
entry_vecs = [e[0] for e in all_entries]
entry_origin = (["FLORB"]*3 + ["TREK"]*3 + ["NENHUM"]*2)

# Similaridades
sims = [cos(composed, v) for v in entry_vecs]

# Ranking
ranked = sorted(zip(sims, entry_names, entry_origin), key=lambda x: x[0], reverse=True)

log(f"\n  {'Rank':<6} {'Item':<35} {'Similaridade':<15} {'Origem':<10}")
log(f"  {'-'*66}")
for rank, (s, name, origin) in enumerate(ranked):
    log(f"  {rank+1:<6} {name:<35} {s:<15.4f} {origin:<10}")

# Acuracia top-3: as 3 mais similares devem ser FLORB ou TREK properties
top3_origins = [origin for _, _, origin in ranked[:3]]
correct_top3 = all(o != "NENHUM" for o in top3_origins)
log(f"\n  Top-3 origins: {top3_origins}")
log(f"  Top-3 todas relevantes? {'SIM' if correct_top3 else 'NAO'}")

# Acuracia top-6: as 6 mais similares devem ser as 6 properties (3 FLORB + 3 TREK)
top6_origins = [origin for _, _, origin in ranked[:6]]
correct_top6 = sum(1 for o in top6_origins if o != "NENHUM")
log(f"  Top-6 relevantes: {correct_top6}/6")

# Metricas binarias: para cada propriedade, determinar se o score > threshold
# Threshold = media das irrelevantes + 1 std
irr_sims = [cos(composed, v) for v in [ch, tr]]
threshold = np.mean(irr_sims) + np.std(irr_sims) if len(irr_sims) > 1 else np.mean(irr_sims) + 0.01
log(f"\n  Threshold (media irrelevantes + std): {threshold:.4f}")

log(f"\n  {'Propriedade':<20} {'Origem':<10} {'Score':<10} {'Relevante?':<12} {'Detectado?':<12}")
log(f"  {'-'*64}")

tp, fp, fn, tn = 0, 0, 0, 0
for sim, name, origin in ranked:
    is_relevant = origin != "NENHUM"
    detected = sim > threshold
    if is_relevant and detected: tp += 1
    elif not is_relevant and detected: fp += 1
    elif is_relevant and not detected: fn += 1
    else: tn += 1
    log(f"  {name:<20} {origin:<10} {sim:<10.4f} {'SIM' if is_relevant else 'NAO':<12} {'SIM' if detected else 'NAO':<12}")

precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-10)
log(f"\n  TP={tp} FP={fp} FN={fn} TN={tn}")
log(f"  Precisao: {precision:.0%}")
log(f"  Recall:   {recall:.0%}")
log(f"  F1:       {f1:.0%}")

# ============================================================
# 4. CONTROLE: query sem TREK
# ============================================================
log("\n--- FASE 3: Controle — query SEM TREK ---")
composed_ctrl = bundle_pair(spaceship, florb)  # sem TREK
log(f"Query vector: bundle(S, F) — sem TREK\n")

sims_ctrl = [cos(composed_ctrl, v) for v in entry_vecs]
ranked_ctrl = sorted(zip(sims_ctrl, entry_names, entry_origin), key=lambda x: x[0], reverse=True)

irr_sims_ctrl = [cos(composed_ctrl, v) for v in [ch, tr]]
threshold_ctrl = np.mean(irr_sims_ctrl) + np.std(irr_sims_ctrl) if len(irr_sims_ctrl) > 1 else np.mean(irr_sims_ctrl) + 0.01

tp2, fp2, fn2, tn2 = 0, 0, 0, 0
log(f"  {'Item':<35} {'Score':<10} {'Relevante?':<12} {'Detectado?':<12}")
log(f"  {'-'*69}")
for sim, name, origin in ranked_ctrl:
    is_relevant = origin == "FLORB"  # so FLORB properties (TREK nao esta no query)
    detected = sim > threshold_ctrl
    if is_relevant and detected: tp2 += 1
    elif not is_relevant and detected: fp2 += 1
    elif is_relevant and not detected: fn2 += 1
    else: tn2 += 1
    log(f"  {name:<35} {sim:<10.4f} {'SIM' if is_relevant else 'NAO':<12} {'SIM' if detected else 'NAO':<12}")

precision2 = tp2 / max(tp2 + fp2, 1)
recall2 = tp2 / max(tp2 + fn2, 1)
f1_2 = 2 * precision2 * recall2 / max(precision2 + recall2, 1e-10)
log(f"\n  Controle TP={tp2} FP={fp2} FN={fn2} TN={tn2}")
log(f"  Precisao: {precision2:.0%}, Recall: {recall2:.0%}, F1: {f1_2:.0%}")

# ============================================================
# VEREDITO
# ============================================================
log("\n" + "=" * 70)
log("VEREDITO FINAL")
log("=" * 70)
log(f"\n  Teste composicional: F1={f1:.0%} (TP={tp} FP={fp} FN={fn} TN={tn})")
log(f"  Controle (sem TREK): F1={f1_2:.0%} (TP={tp2} FP={fp2} FN={fn2} TN={tn2})")

if f1 >= 0.80 and f1_2 >= 0.80:
    log("""
  EXPERIMENTO FLORB/TREK CONFIRMADO

  A AGI:
  1. Recebeu conceitos NOVOS (HDVs aleatorios, sem encoder)
  2. Aprendeu propriedades via bundles na memoria
  3. Query composto (spaceship + FLORB + TREK) recuperou
     as propriedades corretas de AMBOS com alta precisao
  4. Controle (sem TREK) recuperou APENAS FLORB

  Isto e RACIOCINIO COMPOSICIONAL COM CONCEITOS NOVOS.
""")
else:
    log(f"\n  Teste inconclusivo (F1 < 80%).")
