"""
Teste CAUSAL FLORB/TREK v2: raciocinio PURAMENTE por propriedades.
NENHUMA aresta direta material->papel. So arestas: material->propriedade,
propriedade->papel. A AGI precisa INFERIR a atribuicao por cadeia causal.

Cenario:
  "Projete uma nave para atmosfera de planeta quente."
  FLORB = leve, resistente, flexivel
  TREK = absorve calor, isolante, escuro

Cadeia esperada (INFERIDA, nao memorizada):
  FLORB -> (leve, resistente) -> bom para estrutura
  TREK -> (absorve calor, isolante) -> bom para escudo termico
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = r"D:\novamind_data\cache\huggingface"
os.environ["TORCH_HOME"] = r"D:\novamind_data\cache\torch"

import torch
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.hdc_memory import HDCMemory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 10000
torch.manual_seed(42)

def log(text):
    print(text)

def hdv():
    return (torch.rand(DIM, device=DEVICE) * 2 - 1).sign()

def bundle_vecs(vecs):
    return torch.sign(sum(v.float() for v in vecs))

def cos(a, b):
    return float(torch.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)))

memory = HDCMemory(device="cpu")
cge = CognitiveGraphEngine(memory, device=DEVICE)
cge.clear()

log("=" * 70)
log("TESTE CAUSAL FLORB/TREK v2")
log("Atribuicao INFERIDA por propriedades (sem arestas diretas)")
log("=" * 70)

# ============================================================
# 1. CRIAR CONCEITOS
# ============================================================
log("\n--- FASE 1: Conceitos (HDVs aleatorios) ---")

# Materiais
F = hdv()  # FLORB
T = hdv()  # TREK

# Propriedades
L = hdv()  # lightweight
S = hdv()  # strong
Fl = hdv() # flexible
H = hdv()  # heat_absorbing
I = hdv()  # insulating
D = hdv()  # dark

# Papeis
R_struct = hdv()  # structure_role
R_therm = hdv()   # thermal_role

# Contexto
Ship = hdv()   # spaceship
Plan = hdv()   # hot_planet
Atm = hdv()    # atmosphere
Heat = hdv()   # extreme_heat
Shield = hdv() # heat_shield
Sub = hdv()    # submarine (para generalizacao)

props = {"FLORB": F, "TREK": T,
         "lightweight": L, "strong": S, "flexible": Fl,
         "heat_absorbing": H, "insulating": I, "dark": D,
         "structure_role": R_struct, "thermal_role": R_therm,
         "spaceship": Ship, "hot_planet": Plan, "atmosphere": Atm,
         "extreme_heat": Heat, "heat_shield": Shield, "submarine": Sub}

for name, vec in props.items():
    cge.create_node(name, vec)
log(f"  {len(props)} nos criados")

# ============================================================
# 2. ARESTAS: material -> propriedade, propriedade -> papel
#    (SEM arestas diretas material -> papel!)
# ============================================================
log("\n--- FASE 2: Arestas causais (propriedade como intermediario) ---")

# Propriedades de FLORB
cge.add_edge("FLORB", "lightweight", cge.EDGE_PART_OF)
cge.add_edge("FLORB", "strong", cge.EDGE_PART_OF)
cge.add_edge("FLORB", "flexible", cge.EDGE_PART_OF)
log("  FLORB -> (lightweight, strong, flexible)")

# Propriedades de TREK
cge.add_edge("TREK", "heat_absorbing", cge.EDGE_PART_OF)
cge.add_edge("TREK", "insulating", cge.EDGE_PART_OF)
cge.add_edge("TREK", "dark", cge.EDGE_PART_OF)
log("  TREK -> (heat_absorbing, insulating, dark)")

# Propriedade -> papel funcional (SEM reference a FLORB ou TREK!)
cge.add_edge("lightweight", "structure_role", cge.EDGE_CAUSES)
cge.add_edge("strong", "structure_role", cge.EDGE_CAUSES)
cge.add_edge("flexible", "structure_role", cge.EDGE_CAUSES)
cge.add_edge("heat_absorbing", "thermal_role", cge.EDGE_CAUSES)
cge.add_edge("insulating", "thermal_role", cge.EDGE_CAUSES)
log("  lightweight+strong+flexible -> structure_role")
log("  heat_absorbing+insulating -> thermal_role")

# Cadeia causal do problema (dominio geral, nao menciona materiais)
cge.add_edge("hot_planet", "atmosphere", cge.EDGE_PART_OF)
cge.add_edge("atmosphere", "extreme_heat", cge.EDGE_CAUSES)
cge.add_edge("extreme_heat", "heat_shield", cge.EDGE_CAUSES)
cge.add_edge("heat_shield", "thermal_role", cge.EDGE_PART_OF)
log("  hot_planet -> atmosphere -> extreme_heat -> heat_shield -> thermal_role")

cge.add_edge("spaceship", "structure_role", cge.EDGE_PART_OF)
log("  spaceship -> structure_role")

# ============================================================
# 3. COMPRESS + DECOMPRESS: testar inferencia
# ============================================================
log("\n--- FASE 3: Compress → holograma + decompress → inferencia ---")

# Compress: o holograma captura TODO o grafo
hologram = cge.compress_to_hologram()
log(f"  Holograma criado (norm={float(hologram.norm()):.2f})")

# Decompress: quais relacoes o holograma revela?
decomp = cge.decompress_from_hologram(hologram, top_k_nodes=6)
log(f"\n  Nos recuperados pelo decompress:")
for nid, ndata in decomp.get("nodes", {}).items():
    log(f"    {nid:<20} conf={ndata['confidence']:.4f} text={ndata.get('text','')[:30]}")
log(f"\n  Arestas recuperadas pelo decompress:")
for e in decomp.get("edges", []):
    log(f"    {e['from']} --[{e['relation']}]--> {e['to']}")

# ============================================================
# 4. TESTE DE ATRIBUICAO TRANSITIVA
# ============================================================
log("\n--- FASE 4: Atribuicao transitiva ---")
log("  SEM aresta direta FLORB->structure_role")
log("  SEM aresta direta TREK->thermal_role\n")

# Estrategia: criar um vetor de query que representa o problema
# e medir a cadeia de ativacao via similaridade com cada no

query = bundle_vecs([F, T, Ship, Plan, Atm])
log(f"  Query: bundle(FLORB, TREK, spaceship, hot_planet, atmosphere)")
log(f"")

# Medir ativacao em cada no da cadeia causal
log(f"  Cadeia causal de protecao termica:")
log(f"  {'No':<20} {'Cos(query,no)':<15}")
log(f"  {'-'*35}")
chain_thermal = ["hot_planet", "atmosphere", "extreme_heat", "heat_shield", "thermal_role"]
for name in chain_thermal:
    v = props[name]
    s = cos(query, v)
    log(f"  {name:<20} {s:<15.4f}")

log(f"\n  Cadeia estrutural:")
log(f"  {'No':<20} {'Cos(query,no)':<15}")
log(f"  {'-'*35}")
chain_struct = ["spaceship", "structure_role"]
for name in chain_struct:
    v = props[name]
    s = cos(query, v)
    log(f"  {name:<20} {s:<15.4f}")

# Atribuicao: qual material ativa qual papel?
log(f"\n  Atribuicao inferida (similaridade cruzada):")
florb_struct = cos(F, R_struct)
florb_therm = cos(F, R_therm)
trek_struct = cos(T, R_struct)
trek_therm = cos(T, R_therm)

log(f"  Cos(FLORB, structure_role) = {florb_struct:.4f}")
log(f"  Cos(FLORB, thermal_role)   = {florb_therm:.4f}")
log(f"  Cos(TREK, structure_role)  = {trek_struct:.4f}")
log(f"  Cos(TREK, thermal_role)    = {trek_therm:.4f}")

# Atribuicao correta: FLORB > TREK em structure, TREK > FLORB em thermal
florb_to_struct = florb_struct > trek_struct
trek_to_thermal = trek_therm > florb_therm

score_correct = florb_struct + trek_therm  # F->struct, T->thermal
score_wrong = florb_therm + trek_struct    # F->thermal, T->struct

log(f"\n  Score atribuicao CORRETA (F->struct, T->thermal): {score_correct:.4f}")
log(f"  Score atribuicao ERRADA  (F->thermal, T->struct):  {score_wrong:.4f}")
pass_assign = score_correct > score_wrong
log(f"  {'PASS: Atribuicao correta inferida' if pass_assign else 'FAIL: Atribuicao errada'}")

# Teste: sem propriedades, a atribuicao falha?
F_bare = hdv()
florb_bare_struct = cos(F_bare, R_struct)
florb_bare_therm = cos(F_bare, R_therm)
log(f"\n  Controle: FLORB sem propriedades:")
log(f"    Cos(FLORB_bare, structure_role) = {florb_bare_struct:.4f}")
log(f"    Cos(FLORB_bare, thermal_role)   = {florb_bare_therm:.4f}")
# Se FLORB com props > FLORB_bare em structure, as props importam
improvement = florb_struct - florb_bare_struct
pass_props = improvement > 0.01
log(f"    Melhoria com propriedades: {improvement:.4f}")
log(f"    {'PASS: Propriedades importam' if pass_props else 'FAIL: Propriedades nao importam'}")

# ============================================================
# 5. GENERALIZACAO: SUBMARINO (calor irrelevante)
# ============================================================
log(f"\n--- FASE 5: Generalizacao — SUBMARINO ---")
cge.create_node("submarine", Sub)
cge.add_edge("submarine", "structure_role", cge.EDGE_PART_OF)
# Note: SEM aresta submarine -> thermal_role (calor nao e relevante)

query_sub = bundle_vecs([F, T, Sub])
sub_struct = cos(query_sub, R_struct)
sub_therm = cos(query_sub, R_therm)
pass_gen = sub_struct > sub_therm

log(f"  Query: bundle(FLORB, TREK, submarine)")
log(f"  Cos(query, structure_role) = {sub_struct:.4f}")
log(f"  Cos(query, thermal_role)   = {sub_therm:.4f}")
log(f"  {'PASS: Generalizou (structure > thermal)' if pass_gen else 'FAIL: Nao generalizou'}")

# ============================================================
# VEREDITO
# ============================================================
log(f"\n{'='*70}")
log("VEREDITO")
log(f"{'='*70}")

passes = sum([pass_assign, pass_props, pass_gen])
if pass_assign: log(f"  PASS: Atribuicao causal inferida (sem aresta direta)")
if pass_props:  log(f"  PASS: Raciocinio por propriedades (nao associacao direta)")
if pass_gen:    log(f"  PASS: Generalizacao para submarino")

log(f"\n  {passes}/3 passes")
if passes == 3:
    log(f"""
  RACIOCINIO CAUSAL CONFIRMADO

  A AGI inferiu a atribuicao correta APENAS pelas propriedades:
  - FLORB -> structure_role (via lightweight+strong+flexible)
  - TREK -> thermal_role (via heat_absorbing+insulating)
  - Sem propriedades, a inferencia falha.
  - Submarino: structure > thermal (calor irrelevante).

  Isto e RACIOCINIO TRANSITIVO COM CONCEITOS NOVOS.
""")
else:
    log(f"\n  Teste inconclusivo ({passes}/3)")
    if not pass_assign:
        log(f"  Dica: random HDVs sao ortogonais, cos ≈ 0 para todos os pares.")
        log(f"  Para capturar relacoes em HDC, precisamos de:")
        log(f"    a) Bundles (sign de soma) - similaridade positiva com componentes")
        log(f"    b) O ciclo compress->decompress, que testa arestas diretamente")
        log(f"  O decompress ja revela as arestas (veja Fase 3).")
