"""
Pipeline completo: OpenIE -> TypedCognitiveGraph -> Causal Chains
Tempo estimado: < 5 minutos para todo o processo

Passos:
  1. Processa corpus textual com OpenIE (parser dependencias + extracao tipada)
  2. Adiciona triplas ao TypedCognitiveGraph (hiperbolico + Clifford + FEP)
  3. Extrai cadeias causais (CAUSES transitivo)
  4. Compara com BGE-only (baseline)
  5. Relatorio final
"""
import sys, os, json, time
from collections import defaultdict
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

print("=" * 70)
print("PIPELINE: OpenIE -> TypedCognitiveGraph -> Causal Chains")
print("=" * 70)

# ==============================================================
# 1. CARREGAR E PROCESSAR CORPUS
# ==============================================================
t_start = time.time()

# Corpus cientifico (dominios variados)
corpus = [
    # Fisica
    "Gravity causes objects to fall toward Earth.",
    "Gravity is a fundamental force of nature.",
    "Force causes acceleration according to Newton.",
    "Friction prevents motion between surfaces.",
    "Friction produces heat during contact.",
    "Energy enables work to be performed on objects.",
    "Heat flows from hot objects to cold objects.",
    "Light enables vision in biological organisms.",
    "Mass is the amount of matter in an object.",
    "Mass creates gravitational fields.",
    
    # Biologia
    "Smoking causes lung cancer in humans.",
    "Cancer causes pain and inflammation.",
    "Viruses cause infectious diseases.",
    "Vaccines prevent disease by building immunity.",
    "Antibiotics kill bacteria that cause infections.",
    "Exercise improves cardiovascular health.",
    "Diet affects metabolism and energy levels.",
    "Sleep enables memory consolidation.",
    "Stress causes elevated cortisol levels.",
    "Meditation reduces anxiety and depression.",
    
    # Quimica
    "Heat increases the rate of chemical reactions.",
    "Catalysts speed up chemical reactions.",
    "Acids corrode metals over time.",
    "Water dissolves many ionic compounds.",
    "Oxygen enables combustion in fire.",
    "Carbon dioxide causes global warming.",
    "Pollution affects air quality and health.",
    "pH level determines the acidity of solutions.",
    
    # Medicina
    "Insulin regulates blood sugar levels.",
    "High blood pressure causes heart disease.",
    "Cholesterol contributes to artery blockage.",
    "Exercise strengthens the cardiovascular system.",
    "Obesity increases the risk of diabetes.",
    "Early detection improves treatment outcomes.",
    "Chemotherapy kills rapidly dividing cells.",
    "Inflammation is a response to tissue damage.",
    
    # Tecnologia
    "Electricity enables computers to function.",
    "Algorithms process data to produce information.",
    "Machine learning models identify patterns in data.",
    "Neural networks approximate complex functions.",
    "Encryption protects sensitive information.",
    "Software bugs cause system failures.",
    "Redundancy prevents data loss in storage.",
    "Networks enable communication between devices.",
    
    # Economia
    "Inflation reduces purchasing power over time.",
    "Interest rates affect borrowing and spending.",
    "Supply and demand determine market prices.",
    "Investment generates returns over time.",
    "Taxation funds public services and infrastructure.",
    "Trade enables specialization between countries.",
    "Innovation drives economic growth.",
]

# Adicionar sentencas de Alice in Wonderland para diversidade
alice_sentences = [
    "Alice followed the rabbit down the hole.",
    "The rabbit caused Alice to be curious.",
    "Curiosity leads to adventure in Wonderland.",
    "The Queen orders executions for minor offenses.",
    "The Cheshire Cat appears and disappears mysteriously.",
    "The Mad Hatter hosts a tea party every day.",
    "Time enables the Hatter's perpetual tea party.",
    "Growth potions change Alice's size dramatically.",
    "Size changes affect how Alice perceives the world.",
    "The Caterpillar gives Alice important advice.",
    "Advice helps Alice navigate Wonderland.",
    "Wonderland challenges logic and reason.",
    "Nonsense reveals the limitations of language.",
]

corpus += alice_sentences
corpus += [
    "The caterpillar transforms into a butterfly.",
    "Butterflies emerge from cocoons after metamorphosis.",
    "Metamorphosis involves complete reorganization of body tissues.",
    "Enzymes break down larval tissues during metamorphosis.",
    "Hormones trigger the transformation process.",
    "Puberty causes physical and hormonal changes in adolescents.",
    "Testosterone causes development of male secondary characteristics.",
    "Estrogen regulates the female reproductive system.",
    "Growth hormone stimulates cell reproduction and regeneration.",
    "Melatonin regulates sleep-wake cycles in humans.",
    "Serotonin affects mood, appetite, and sleep.",
    "Dopamine is a neurotransmitter that controls reward and pleasure.",
    "Low dopamine levels cause symptoms of Parkinson's disease.",
    "Dopamine replacement therapy improves motor function in Parkinson's.",
    "Deep brain stimulation reduces tremors in Parkinson's patients.",
    "Stem cells differentiate into specialized cell types.",
    "Gene therapy corrects genetic disorders by replacing defective genes.",
    "CRISPR enables precise editing of DNA sequences.",
    "Epigenetic modifications alter gene expression without changing DNA.",
    "Methylation of DNA silences gene expression.",
    "Histone acetylation increases gene transcription activity.",
    "Telomeres shorten with each cell division.",
    "Telomere shortening causes cellular aging.",
]

print(f"Corpus: {len(corpus)} sentences")

# ==============================================================
# 2. EXTRAIR TRIPLAS VIA OPENIE
# ==============================================================
from core.openie_extractor import OpenIEKnowledgeExtractor
ext = OpenIEKnowledgeExtractor()

t0 = time.time()
all_triples = []
for sent in corpus:
    triples = ext.extract_from_text(sent)
    all_triples.extend(triples)
t_extract = time.time() - t0

rel_types = defaultdict(int)
for t in all_triples:
    rel_types[t['relation']] += 1

print(f"\n[OpenIE] {len(all_triples)} triples in {t_extract:.1f}s")
for rel in sorted(rel_types):
    print(f"  {rel:15s}: {rel_types[rel]}")

# ==============================================================
# 3. BUILD TYPED COGNITIVE GRAPH
# ==============================================================
from core.typed_graph import TypedCognitiveGraph
tcg = TypedCognitiveGraph(dim=50, curvature=1.0)

for t in all_triples:
    subj = t['subject'].lower().replace(' ', '_')
    obj = t['object'].lower().replace(' ', '_')
    conf = t['confidence']
    rel = t['relation']
    
    if rel == 'IS_A':
        rtype = 'IS_A'
    elif rel in ('CAUSES', 'ENABLES', 'PREVENTS', 'AFFECTS', 'INFLUENCES', 'FOLLOWS'):
        rtype = 'CAUSES'
    elif rel == 'HAS_PROPERTY':
        rtype = 'PART_OF'
    else:
        rtype = 'CORRELATED'
    
    tcg.add_relation(subj, obj, rtype, confidence=conf, observation_evidence=conf)

t_tcg = time.time() - t_start
print(f"\n[TCG] {len(tcg.nodes)} concepts, {len(tcg.edges)} edges in {t_tcg:.1f}s")

# ==============================================================
# 4. EXTRACT CAUSAL CHAINS
# ==============================================================
print(f"\n{'='*70}")
print("CADEIAS CAUSAIS (transitividade sobre edges CAUSES)")
print("=" * 70)

# Find all CAUSES edges
causes = [(k, e) for k, e in tcg.edges.items() if k[2] == 'CAUSES']

# Build adjacency: cause -> effects
adj = defaultdict(list)
for key, edge in causes:
    frm, to, _ = key
    adj[frm].append((to, edge['confidence']))

# Walk chains of length >= 2
chains = []
for start in list(adj.keys()):
    for mid, conf1 in adj[start]:
        for end, conf2 in adj[mid]:
            chain_conf = conf1 * conf2 * 0.9  # chaining discount
            chains.append((start, mid, end, chain_conf, conf1, conf2))

chains.sort(key=lambda x: -x[3])

print(f"\n  {len(chains)} causal chains found (length >= 2)")
print(f"\n  Top 20 causal chains by confidence:")
for i, (start, mid, end, chain_conf, conf1, conf2) in enumerate(chains[:20]):
    print(f"    {i+1:2d}. {start:20s} -> {mid:20s} (c={conf1:.2f})")
    print(f"        -> {end:20s} (c={conf2:.2f})  chain_conf={chain_conf:.3f}")

# ==============================================================
# 5. COMPARISON WITH BGE-ONLY
# ==============================================================
print(f"\n{'='*70}")
print("COMPARACAO: Causal Chain (typed graph) vs BGE cosine similarity")
print("=" * 70)

from core.bge_encoder import BGEEncoder
import torch
bge = BGEEncoder()

# For each causal pair, compare with BGE similarity
test_pairs = [
    ('smoking', 'lung_cancer'),
    ('exercise', 'cardiovascular_health'), 
    ('stress', 'cortisol_levels'),
    ('vaccines', 'disease'),
    ('inflation', 'purchasing_power'),
    ('gravity', 'objects'),
    ('force', 'motion'),
]

print(f"\n  {'Causal Pair':35s} {'BGE sim':10s} {'TCG conf':10s} {'CGE edge?':10s}")
for cause_effect in test_pairs:
    cause, effect = cause_effect
    # BGE similarity
    bge_sim = bge.similarity(cause.replace('_', ' '), effect.replace('_', ' '))
    
    # TCG confidence
    key = (cause, effect, 'CAUSES')
    tcg_conf = tcg.edges.get(key, {}).get('confidence', 0)
    
    # CGE direct edge (co-ocorrencia check via BGE CGE)
    # (can't really check in < 10 min, skip)
    
    sim_str = f"{bge_sim:.4f}"
    tcg_str = f"{tcg_conf:.4f}" if tcg_conf > 0 else "NO_EDGE"
    print(f"  {cause:20s} -> {effect:20s} {sim_str:10s} {tcg_str:10s}")

# ==============================================================
# 6. QUERY TEST: "What does smoking cause?"
# ==============================================================
print(f"\n{'='*70}")
print("QUERY: What does smoking cause?")
print("=" * 70)
smoking_results = tcg.query('smoking', 'CAUSES', min_confidence=0.1)
if smoking_results:
    for target, conf in smoking_results:
        print(f"  smoking -> {target:20s} (conf={conf:.3f})")
else:
    print(f"  (No CAUSES edges from 'smoking' in TCG)")
    # Fallback: BGE query
    print(f"  BGE fallback: similar concepts to 'smoking':")
    test_concepts = ['lung_cancer', 'cancer', 'disease', 'health', 'exercise', 'pollution']
    smoke_vec = bge.encode_bipolar(['smoking'])[0]
    for c in test_concepts:
        c_vec = bge.encode_bipolar([c])[0]
        sim = float(torch.cosine_similarity(smoke_vec.unsqueeze(0), c_vec.unsqueeze(0)))
        print(f"    sim(smoking, {c:20s}) = {sim:.4f}")

# ==============================================================
# 7. SAVE
# ==============================================================
results = {
    'pipeline': {
        'corpus_size': len(corpus),
        'triples_extracted': len(all_triples),
        'extraction_time': t_extract,
    },
    'typed_graph': {
        'concepts': len(tcg.nodes),
        'edges': len(tcg.edges),
        'relation_distribution': dict(rel_types),
    },
    'causal_chains': {
        'n_chains': len(chains),
        'max_chain_confidence': max([c[3] for c in chains]) if chains else 0,
        'top_chains': [{'from': c[0], 'mid': c[1], 'to': c[2], 'confidence': c[3]} 
                       for c in chains[:20]],
    },
}

out = r'evals/results/pipeline_complete.json'
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    json.dump(results, f, indent=2, default=str)

t_total = time.time() - t_start
print(f"\nTempo total: {t_total:.1f}s")
print(f"Resultados salvos em {out}")
