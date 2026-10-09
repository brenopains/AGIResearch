"""
Experimento: OpenIE Knowledge Extractor + TypedCognitiveGraph
=============================================================
Pipeline completo:
  1. Carrega corpus de texto cientifico
  2. Extrai triplas tipadas via OpenIE (CAUSES, ENABLES, IS_A, PREVENTS, etc.)
  3. Adiciona conhecimento ao TypedCognitiveGraph
  4. Computa topologia (Betti numbers, gaps)
  5. Compara com o grafo de co-ocorrencia original
"""
import sys, os, json, time
from collections import defaultdict
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

print("=" * 70)
print("EXPERIMENTO: OpenIE -> TypedCognitiveGraph")
print("=" * 70)

# 1. Build corpus from existing knowledge
# Use the first 30 concepts as seed to generate simple KD-tests
physics_corpus = [
    "Gravity causes objects to fall toward Earth.",
    "Gravity is a fundamental force of nature.",
    "Gravity attracts all objects with mass.",
    "Gravity follows the inverse square law.",
    "Mass causes gravitational attraction between objects.",
    "Force causes acceleration according to Newton's second law.",
    "Force is a vector quantity that changes motion.",
    "Force has both magnitude and direction.",
    "Friction prevents motion between surfaces that touch.",
    "Friction produces heat during contact.",
    "Friction opposes the relative motion of objects.",
    "Inertia resists changes in the motion of objects.",
    "Inertia is a property of all massive objects.",
    "Motion follows force according to Newton's laws.",
    "Motion requires energy to change direction.",
    "Energy enables work to be performed on objects.",
    "Energy is the capacity to do work.",
    "Energy follows the law of conservation.",
    "Energy transforms between different forms.",
    "Kinetic energy is the energy of motion.",
    "Potential energy is stored energy.",
    "Heat is a form of energy transfer.",
    "Heat increases the temperature of materials.",
    "Heat flows from hot objects to cold objects.",
    "Light is electromagnetic radiation.",
    "Light travels at a constant speed in vacuum.",
    "Light enables vision in biological organisms.",
    "Light carries energy through space.",
    "Mass is the amount of matter in an object.",
    "Mass creates gravitational fields.",
    "Mass resists acceleration through inertia.",
    "Time is a dimension of the physical universe.",
    "Time proceeds forward according to thermodynamics.",
    "Time enables causal relationships between events.",
    "Velocity is speed with direction.",
    "Velocity changes when force is applied.",
    "Acceleration is the rate of change of velocity.",
    "Acceleration follows force according to Newton.",
    "Speed is the magnitude of velocity.",
    "Distance measures the separation between objects.",
    "Electricity enables modern technology.",
    "Electricity flows through conductive materials.",
    "Electricity produces magnetic fields.",
    "Magnetism is a fundamental force.",
    "Magnetism affects the motion of charged particles.",
    "Cells are the basic unit of life.",
    "Cells contain genetic information in DNA.",
    "Cells produce energy through metabolism.",
    "DNA contains the genetic code for organisms.",
    "DNA enables inheritance of traits.",
    "Proteins perform most functions in cells.",
    "Enzymes speed up chemical reactions in cells.",
    "Evolution changes species over generations.",
    "Evolution follows natural selection.",
    "Natural selection preserves beneficial traits.",
    "Genetic mutations cause evolutionary changes.",
    "Species adapt to their environment through evolution.",
    "Ecosystems maintain balance through feedback loops.",
]

print(f"Corpus: {len(physics_corpus)} sentences")

# 2. Extract triples via OpenIE
print(f"\n[OpenIE] Extracting triples...")
from core.openie_extractor import OpenIEKnowledgeExtractor
ext = OpenIEKnowledgeExtractor()
t0 = time.time()

all_triples = []
for sent in physics_corpus:
    triples = ext.extract_from_text(sent)
    all_triples.extend(triples)

elapsed = time.time() - t0
print(f"[OpenIE] Extracted {len(all_triples)} triples in {elapsed:.1f}s")

# Show relation type distribution
rel_types = defaultdict(int)
for t in all_triples:
    rel_types[t['relation']] += 1
print(f"\nRelation type distribution:")
total = sum(rel_types.values()) or 1
for rel in sorted(rel_types):
    print(f"  {rel:15s}: {rel_types[rel]:3d} ({rel_types[rel]/total*100:.1f}%)")

# 3. Build TypedCognitiveGraph from triples
print(f"\n[TCG] Building graph from typed triples...")
from core.typed_graph import TypedCognitiveGraph
tcg = TypedCognitiveGraph(dim=50, curvature=1.0)

for t in all_triples:
    subj = t['subject'].lower().replace(' ', '_')
    obj = t['object'].lower().replace(' ', '_')
    conf = t['confidence']
    rel = t['relation']
    
    # Map to known relation types
    if rel == 'IS_A':
        rtype = 'IS_A'
    elif rel in ('CAUSES', 'ENABLES', 'PREVENTS', 'AFFECTS', 'INFLUENCES', 'FOLLOWS'):
        rtype = 'CAUSES'
    elif rel == 'HAS_PROPERTY':
        rtype = 'PART_OF'
    else:
        rtype = 'CORRELATED'
    
    tcg.add_relation(subj, obj, rtype, confidence=conf, observation_evidence=conf)

print(f"[TCG] Concepts: {len(tcg.nodes)}, Edges: {len(tcg.edges)}")

# 4. Topology analysis
print(f"\n[TOPOLOGY] Computing Betti numbers...")
t_start = time.time()
topo = tcg.build_topology(max_concepts=len(tcg.nodes))
topo_time = time.time() - t_start

if 'betti_numbers' in topo:
    print(f"  Time: {topo_time:.1f}s")
    print(f"  Betti numbers: {topo['betti_numbers']}")
    print(f"  Persistence entropy: {topo.get('persistence_entropy', 0):.4f}")
    print(f"  Knowledge gaps (H2): {topo.get('knowledge_gaps', 'N/A')}")
    print(f"  Circular loops (H1): {topo.get('circular_loops', 'N/A')}")

# 5. Compare with old co-occurrence graph
print(f"\n{'='*70}")
print("COMPARISON: Co-ocorrencia vs OpenIE Typed")
print("=" * 70)

old_system = json.load(open(r'evals/results/experiment_comparison.json'))

print(f"""
  METRICA                    CO-OCORRENCIA      OPENIE TYPED       DIFERENCA
  -------------------------  -----------------  -----------------  ----------------
  Relacoes CAUSES/ENABLES    0 (inexistente)    {rel_types.get('CAUSES',0)+rel_types.get('ENABLES',0)+rel_types.get('PREVENTS',0):3d}             NOVAS - causalidade real
  Relacoes IS_A              heuristicas        {rel_types.get('IS_A',0):3d}             de copula gramatical
  Extracao                   janela bigrama     parser dependencias base gramatical
  Confianca                  constante 1.0      {np.mean([t['confidence'] for t in all_triples]):.2f} media     computada por negacao/modal
  Betti beta_0               {old_system['new_system']['topology'].get('betti_numbers',{}).get('beta_0', '?'):>5s}               {topo.get('betti_numbers',{}).get('beta_0', '?'):>5s}               clusters de conhecimento
  Betti beta_1               {old_system['new_system']['topology'].get('betti_numbers',{}).get('beta_1', '?'):>5s}               {topo.get('betti_numbers',{}).get('beta_1', '?'):>5s}               ciclos causais
  Betti beta_2               {old_system['new_system']['topology'].get('betti_numbers',{}).get('beta_2', '?'):>5s}               {topo.get('betti_numbers',{}).get('beta_2', '?'):>5s}               lacunas de conhecimento
""")

# 6. Show sample causal chains found
print(f"\n  Sample CAUSES chains found:")
causes_edges = [(k, e) for k, e in tcg.edges.items() if k[2] == 'CAUSES']
for key, edge in sorted(causes_edges, key=lambda x: -x[1]['confidence'])[:10]:
    frm, to, rel = key
    print(f"    {frm:20s} -> {to:20s} (conf={edge['confidence']:.3f})")

# 7. Show IS_A hierarchy
print(f"\n  Sample IS_A hierarchy:")
is_a_edges = [(k, e) for k, e in tcg.edges.items() if k[2] == 'IS_A']
for key, edge in sorted(is_a_edges, key=lambda x: -x[1]['confidence'])[:10]:
    frm, to, rel = key
    d_frm = tcg.nodes.get(frm, {}).get('depth', 0)
    d_to = tcg.nodes.get(to, {}).get('depth', 0)
    print(f"    {frm:25s} -> IS_A -> {to:25s} (d_src={d_frm:.2f}, d_tgt={d_to:.2f}, deeper_correct={d_frm > d_to})")

# Save results
results = {
    'experiment': 'openie_vs_cooccurrence',
    'corpus': {'sentences': len(physics_corpus)},
    'openie': {
        'triples': len(all_triples),
        'relation_types': dict(rel_types),
        'extraction_time': elapsed,
    },
    'typed_graph': {
        'concepts': len(tcg.nodes),
        'edges': len(tcg.edges),
        'topology': topo,
    },
}

out = r'evals/results/openie_experiment.json'
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {out}")
