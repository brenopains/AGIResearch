"""
Demonstracao: Sistema Integrado fazendo o que LLM nao faz
==========================================================
LLMs sao modelos de linguagem. Eles:
  - NAO tem grafo causal com confianca computada
  - NAO tem tracking visual de objetos com relacoes espaciais tipadas
  - NAO tem Betti numbers mostrando o que sabem e o que nao sabem
  - NAO tem cadeias causais transitivas com confianca via FEP

Este script demonstra o sistema completo integrado:
  1. OpenIE extrai triplas tipadas de texto (CAUSES, IS_A, etc)
  2. SpatialTemporalMemory trackeia objetos em video sintetico
  3. TypedCognitiveGraph integra conhecimento textual + visual
  4. SymbolicEngine faz inferencia (ex: se A CAUSES B e B CAUSES C -> A CAUSES C)
  5. Topologia mostra Betti numbers (gaps de conhecimento)
"""
import sys, os, time, numpy as np
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); os.chdir(BASE)

print("=" * 70)
print("SISTEMA INTEGRADO: Raciocinio que LLM nao faz")
print("=" * 70)

# ==============================================================
# 1. OPENIE: Extrair conhecimento textual tipado
# ==============================================================
from core.openie_extractor import OpenIEKnowledgeExtractor
ext = OpenIEKnowledgeExtractor()

text_corpus = [
    "Gravity causes objects to fall toward Earth.",
    "Smoking causes lung cancer in humans.",
    "Vaccines prevent disease by building immunity.",
    "Exercise improves cardiovascular health.",
    "Stress causes elevated cortisol levels.",
    "Force causes acceleration according to Newton.",
]
all_triples = []
for sent in text_corpus:
    all_triples.extend(ext.extract_from_text(sent))

print(f"\n[1] OpenIE textual: {len(all_triples)} triplas tipadas")
rel_counts = {}
for t in all_triples:
    r = t['relation']
    rel_counts[r] = rel_counts.get(r, 0) + 1
for r, c in sorted(rel_counts.items()):
    print(f"    {r}: {c}")

# ==============================================================
# 2. STM: Tracking visual de objetos
# ==============================================================
from core.spatial_temporal_memory import SpatialTemporalMemory
from core.typed_graph import TypedCognitiveGraph

tcg = TypedCognitiveGraph(dim=30, curvature=1.0)
stm = SpatialTemporalMemory(tcg=tcg)

H, W = 240, 320
frames = []
for t in range(60):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    # Objeto A (verde) move-se horizontalmente
    x1 = max(5, min(W-5, int(20 + t * 3)))
    frame[100:120, x1-4:x1+4] = [0, 255, 0]
    # Objeto B (vermelho) move-se verticalmente
    y2 = max(5, min(H-5, int(30 + t * 2)))
    frame[y2-4:y2+4, 150:170] = [255, 0, 0]
    frames.append(frame)

for i, f in enumerate(frames):
    stm.process_frame(f, timestamp=i*0.033)

print(f"\n[2] STM visual: {len(stm.objects)} objetos rastreados, "
      f"{len(stm._spatial_relations)} relacoes espaciais")

# ==============================================================
# 3. ADICIONAR CONHECIMENTO AO TCG
# ==============================================================
for t in all_triples:
    subj = t['subject'].lower().replace(' ', '_')
    obj = t['object'].lower().replace(' ', '_')
    conf = t['confidence']
    rel = t['relation']
    rtype = 'CAUSES' if rel in ('CAUSES','ENABLES','PREVENTS','AFFECTS') else 'CORRELATED'
    tcg.add_relation(subj, obj, rtype, confidence=conf, observation_evidence=conf)

print(f"\n[3] TCG: {len(tcg.nodes)} conceitos, {len(tcg.edges)} edges")

# ==============================================================
# 4. CADEIAS CAUSAIS via SymbolicEngine (com CGE compativel)
# ==============================================================
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.bge_encoder import BGEEncoder
from core.cognition.symbolic_engine import SymbolicEngine

bge = BGEEncoder()
cge_sym = CognitiveGraphEngine(bge_encoder=bge)

# Adicionar nos e arestas do TCG ao CGE para o SymbolicEngine
for node_name in list(tcg.nodes.keys()):
    pos = tcg.nodes[node_name]['position']
    # Criar vetor HDV a partir da posicao hiperbolica
    vec = bge.encode_bipolar([node_name.replace('_', ' ')])[0].to(cge_sym.device)
    if node_name not in cge_sym.nodes:
        cge_sym.create_node(node_name, vec)

# Adicionar arestas CAUSES do TCG como edges no CGE
for key, edge in tcg.edges.items():
    frm, to, rel = key
    if rel == 'CAUSES' and frm in cge_sym.nodes and to in cge_sym.nodes:
        cge_sym.add_edge(frm, to, cge_sym.EDGE_CAUSES)

se = SymbolicEngine(cge_sym, bge)
se.register_semantic_roles(cge_sym.EDGE_PART_OF, cge_sym.EDGE_CAUSES, None, None, None, None)

# Registrar regras universais das edges CAUSES
for key, edge in tcg.edges.items():
    frm, to, rel = key
    if rel == 'CAUSES':
        se.register_universal(frm, to, edge['confidence'])

# Registrar IS_A (apenas se existir no CGE)
for key, edge in tcg.edges.items():
    frm, to, rel = key
    if rel == 'IS_A' and frm in cge_sym.nodes and to in cge_sym.nodes:
        se.register_universal(frm, to, edge['confidence'])

# Also register CAUSES (já adicionamos arestas, também registrar como regras)
for key, edge in tcg.edges.items():
    frm, to, rel = key
    if rel == 'CAUSES' and frm in cge_sym.nodes and to in cge_sym.nodes:
        se.register_universal(frm, to, edge['confidence'])

# Inferir cadeias
print(f"\n[4] Cadeias causais via SymbolicEngine:")
chain_tests = [
    ('smoking', 'lung_cancer'),
    ('gravity', 'objects'),
    ('exercise', 'cardiovascular_health'),
]
is_a_tests = [
    ('force', 'acceleration'),
    ('lung_cancer', 'humans'),
]

for cause, effect in chain_tests:
    result = se.infer(cause, effect)
    if result:
        print(f"    {cause} -> {effect}: {result.rule} (conf={result.confidence:.3f})")
    else:
        print(f"    {cause} -> {effect}: (inferencia falhou)")

# ==============================================================
# 5. TOPOLOGIA: Betti numbers
# ==============================================================
print(f"\n[5] Topologia do grafo de conhecimento:")
topo = tcg.build_topology(max_concepts=min(100, len(tcg.nodes)))
if 'betti_numbers' in topo:
    b = topo['betti_numbers']
    print(f"    beta_0 (clusters): {b.get('beta_0', '?')} "
          f"[quantos grupos isolados - AGI ideal = 1]")
    print(f"    beta_1 (ciclos): {b.get('beta_1', '?')} "
          f"[raciocinio circular - ideal = 0]")
    print(f"    beta_2 (vazios): {b.get('beta_2', '?')} "
          f"[lacunas inferenciais - conhecimento incompleto]")
    print(f"    Entropia de persistencia: {topo.get('persistence_entropy', 0):.4f}")

# ==============================================================
# 6. O QUE LLM NAO FAZ
# ==============================================================
print(f"\n{'='*70}")
print("O QUE LLM NAO FAZ (e este sistema faz)")
print("=" * 70)

print(f"""
  1. GRAFO CAUSAL TIPADO COM CONFIRMACAO COMPUTADA:
     LLM: texto fluente, zero verificacao. 
     Este: {len([k for k in tcg.edges if k[2]=='CAUSES'])} edges CAUSES
           com confianca FEP (media {np.mean([e['confidence'] for e in tcg.edges.values()]):.2f}).
           Se A CAUSES B e B CAUSES C, o SymbolicEngine infere A CAUSES C 
           com confianca = min(conf_AB, conf_BC) * 0.85 -- verificavel linha a linha.

  2. TRACKING VISUAL COM RELACOES ESPACIAIS TIPADAS:
     LLM: vê texto, nao movimento.
     Este: STM rastreou {sum(1 for o in stm.objects.values() if o.speed()>0)} objetos 
           em movimento, computou {len(stm._spatial_relations)} relacoes espaciais
           (LEFT_OF, RIGHT_OF, ABOVE, BELOW, MOVING_TOWARD).

  3. METACOGNICAO TOPOLOGICA (Betti numbers):
     LLM: nao sabe o que nao sabe.
     Este: beta_0={b.get('beta_0', '?')} significa que o conhecimento esta em 
           {b.get('beta_0', '?')} grupos separados. AGI integrada precisaria de 1.
           beta_2={b.get('beta_2', '?')} significa que ha {b.get('beta_2', '?')} 
           lacunas de conhecimento (A->B e B->C sem A->C).

  4. RAZAO PARA ARC-AGI NAO ESTAR EM 80%:
     ARC e programa sintese VISUAL em grids. As melhorias desta sessaoforam 
     SEMANTICAS (typed graph, FEP, OpenIE, hiperbolico). A pipeline ARC
     (object_centric + program_induction) existe separada e precisa de 
     expansao de busca composicional -- nao de conhecimento semantico.
""")
