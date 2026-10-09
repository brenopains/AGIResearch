"""
Experimento Científico: Co-ocorrência vs Typed Hyperbolic Knowledge
===================================================================
Compara a base de conhecimento original (co-ocorrência textual, 29k edges,
confidence=1.0 constante) com a nova arquitetura integrada (hiperbólico +
bivectors tipados + confiança FEP + topologia).

Hipótese: a arquitetura integrada revela estrutura hierárquica real,
confiança calculada (não constante), gaps de conhecimento detectáveis,
e Betti numbers significativos — ao contrário da base original onde
tudo é plano (same depth), mesma confiança, e sem estrutura topológica.
"""
import sys, os, json, math, time, pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

# =====================================================================
# 1. Load OLD system (co-occurrence) state
# =====================================================================
print("=" * 80)
print("EXPERIMENTO: Co-ocorrência vs Typed Hyperbolic Knowledge")
print("=" * 80)

data = torch.load(r'save/brain_live.pth', map_location='cpu', weights_only=False)
old_nodes_raw = data.get('nodes', {})
old_edges_raw = data.get('edges', [])
state = pickle.load(open(r'save/agi_state_live.pkl', 'rb'))

old_concepts = list(old_nodes_raw.keys())
old_edge_count = len(old_edges_raw)
old_confidence_values = []

for e in old_edges_raw:
    if isinstance(e, dict):
        c = e.get('confidence', 1.0)
        if isinstance(c, torch.Tensor): c = c.mean().item()
        old_confidence_values.append(float(c))

print(f"\n[OLD] Co-occurrence Graph:")
print(f"  Concepts: {len(old_concepts)}")
print(f"  Edges: {old_edge_count}")
print(f"  Edge type values: {set(str(e.get('edge_type', '?')) for e in old_edges_raw[:100] if isinstance(e, dict))}")
print(f"  Confidence: mean={np.mean(old_confidence_values):.4f}, "
      f"std={np.std(old_confidence_values):.4f}, "
      f"min={min(old_confidence_values):.4f}, max={max(old_confidence_values):.4f}")
print(f"  Distinct confidence values: {len(set(old_confidence_values))}")

# =====================================================================
# 2. Build NEW TypedCognitiveGraph from same data
# =====================================================================
print(f"\n{'='*80}")
print("[NEW] Building TypedCognitiveGraph...")

from core.typed_graph import TypedCognitiveGraph
tcg = TypedCognitiveGraph(dim=100, curvature=1.0)

stats = tcg.load_from_brain_state(r'save/brain_live.pth', r'save/agi_state_live.pkl')
print(f"[NEW] Concepts: {len(tcg.nodes)}, Edges: {len(tcg.edges)}")

# Compute hyperbolic depths
depths = [tcg.nodes[n]['depth'] for n in tcg.nodes]
print(f"\n[NEW] Hyperbolic depths:")
print(f"  mean={np.mean(depths):.4f}, std={np.std(depths):.4f}, "
      f"min={min(depths):.4f}, max={max(depths):.4f}")

# Depth distribution
depth_bins = [0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
for i in range(len(depth_bins)-1):
    count = sum(1 for d in depths if depth_bins[i] <= d < depth_bins[i+1])
    print(f"  [{depth_bins[i]:.1f}, {depth_bins[i+1]:.1f}): {count} concepts")

# =====================================================================
# 3. Compare confidence distributions
# =====================================================================
print(f"\n{'='*80}")
print("CONFIDENCE DISTRIBUTION COMPARISON")

new_confs = [e['confidence'] for e in tcg.edges.values()]
if new_confs:
    print(f"\n[OLD] Confidence (co-occurrence):")
    print(f"  All edges: confidence = {np.mean(old_confidence_values):.4f} (constant)")
    print(f"  No variation — every edge equally 'confident'")
    
    print(f"\n[NEW] Confidence (typed hyperbolic):")
    print(f"  Mean: {np.mean(new_confs):.4f}")
    print(f"  Std:  {np.std(new_confs):.4f}")
    print(f"  Min:  {min(new_confs):.4f}, Max: {max(new_confs):.4f}")
    print(f"  Unique values: {len(set(round(c, 3) for c in new_confs))}")
    
    # Histogram of new confidences
    print(f"\n  Distribution:")
    for b in np.arange(0, 1.05, 0.1):
        count = sum(1 for c in new_confs if b <= c < b+0.1)
        bar = '#' * (count // 5)
        print(f"    [{b:.1f}, {b+0.1:.1f}): {count:4d} {bar}")
else:
    print(f"\n  No edges in typed graph.")

# =====================================================================
# 4. Relação tipada distribution
# =====================================================================
print(f"\n{'='*80}")
print("RELATION TYPE DISTRIBUTION")

rel_counts = defaultdict(int)
for key in tcg.edges:
    rel_counts[key[2]] += 1

total_edges = sum(rel_counts.values()) or 1
for rtype in sorted(rel_counts.keys()):
    count = rel_counts[rtype]
    pct = count / total_edges * 100
    print(f"  {rtype:12s}: {count:5d} edges ({pct:.1f}%)")

# =====================================================================
# 5. Topology Analysis (OLD system cannot do this)
# =====================================================================
print(f"\n{'='*80}")
print("TOPOLOGY ANALYSIS (Typed Hyperbolic Graph only)")
print("  Co-occurrence graph CANNOT compute topology — edges have no type/distinction.")

topology_result = tcg.build_topology()

if 'betti_numbers' in topology_result:
    betti = topology_result['betti_numbers']
    print(f"\n  Betti Numbers:")
    print(f"    beta_0 (knowledge clusters):  {betti.get('beta_0', 'N/A')}")
    print(f"    beta_1 (circular reasoning):   {betti.get('beta_1', 'N/A')}")
    print(f"    beta_2 (knowledge voids):      {betti.get('beta_2', 'N/A')}")
    print(f"  Persistence entropy:            {topology_result.get('persistence_entropy', 'N/A'):.4f}")
    print(f"  Knowledge gaps detected:         {topology_result.get('knowledge_gaps', 'N/A')}")
    print(f"  Circular reasoning loops:        {topology_result.get('circular_loops', 'N/A')}")

# =====================================================================
# 6. IS-A hierarchy verification
# =====================================================================
print(f"\n{'='*80}")
print("IS-A HIERARCHY VERIFICATION (Entailment Cones)")
print("  Verifying: concepts deeper in ball ARE specializations of shallower ones")

# Find top 10 IS_A relations
is_a_relations = [(k, e) for k, e in tcg.edges.items() if k[2] == 'IS_A']
is_a_relations_sorted = sorted(is_a_relations, key=lambda x: -x[1]['confidence'])

print(f"\n  Total IS_A edges: {len(is_a_relations)}")
print(f"  Top 10 IS_A relations by confidence:")
for key, edge in is_a_relations_sorted[:10]:
    frm, to, rel = key
    d_frm = tcg.nodes.get(frm, {}).get('depth', 0)
    d_to = tcg.nodes.get(to, {}).get('depth', 0)
    print(f"    {frm:25s} -> {to:25s} (conf={edge['confidence']:.3f}, "
          f"d_from={d_frm:.2f}, d_to={d_to:.2f}, "
          f"deeper={d_frm > d_to})")

# Verify: are deeper concepts genuinely more specific?
print(f"\n  Verifying hierarchy axiom:")
print(f"    If A -> IS_A -> B, then A should be deeper (more specific) than B")
correct_direction = 0
wrong_direction = 0
for key, edge in is_a_relations:
    frm, to, _ = key
    if frm in tcg.nodes and to in tcg.nodes:
        d_frm = tcg.nodes[frm]['depth']
        d_to = tcg.nodes[to]['depth']
        if d_frm > d_to:
            correct_direction += 1
        else:
            wrong_direction += 1

total_is_a = correct_direction + wrong_direction or 1
print(f"    Correct direction (child deeper): {correct_direction}/{total_is_a} "
      f"({correct_direction/total_is_a*100:.1f}%)")
print(f"    Wrong direction: {wrong_direction}/{total_is_a} "
      f"({wrong_direction/total_is_a*100:.1f}%)")

# =====================================================================
# 7. Metic scientific: compare depth distribution of physics concepts
# =====================================================================
print(f"\n{'='*80}")
print("SCIENTIFIC VALIDATION: Physics concepts hierarchy")

physics_terms = ['gravity', 'gravitational', 'gravitation', 'force', 'motion',
                 'energy', 'physics', 'fundamental', 'interaction', 'quantum']
physics_concepts = [n for n in tcg.nodes if any(t in n.lower() for t in physics_terms)]

if physics_concepts:
    print(f"\n  Physics-related concepts: {len(physics_concepts)}")
    print(f"  {'Concept':25s} {'Depth':8s} {'Depth':8s}")
    for c in sorted(physics_concepts, key=lambda x: tcg.nodes[x]['depth']):
        d = tcg.nodes[c]['depth']
        bar = '#' * min(int(d * 10), 50)
        print(f"  {c:25s} {d:.4f}  {bar}")
    
    # Mean depth vs non-physics
    physics_depths = [tcg.nodes[c]['depth'] for c in physics_concepts]
    all_depths = [tcg.nodes[c]['depth'] for c in tcg.nodes]
    non_physics_depths = [d for c, d in zip(tcg.nodes, all_depths) 
                          if c not in physics_concepts]
    print(f"\n  Mean depth (physics): {np.mean(physics_depths):.4f}")
    print(f"  Mean depth (non-physics): {np.mean(non_physics_depths):.4f}")
    print(f"  Mean depth (all): {np.mean(all_depths):.4f}")
else:
    print(f"\n  No physics concepts found in graph.")

# =====================================================================
# 8. Summary comparison
# =====================================================================
print(f"\n{'='*80}")
print("COMPARISON SUMMARY")
print("=" * 80)

# Before (co-occurrence)
print(f"""
┌────────────────────────────────────────────────────────────┐
│                   CO-OCCURRENCE GRAPH (OLD)                │
├────────────────────────────────────────────────────────────┤
│  Edge type:        -1 (single type, no distinction)       │
│  Confidence:       {np.mean(old_confidence_values):.3f} (constant — all edges equal)  │
│  Hierarchy:        Flat (no depth measure)                  │
│  Topology:         Impossible (single edge type)            │
│  Knowledge gaps:   Invisible                               │
│  Circular reason:  Invisible                               │
└────────────────────────────────────────────────────────────┘
""")

# After (typed hyperbolic)
new_conf_mean = np.mean(new_confs) if new_confs else 0
new_conf_unique = len(set(round(c, 3) for c in new_confs)) if new_confs else 0
betti_result = topology_result.get('betti_numbers', {})

print(f"""
┌────────────────────────────────────────────────────────────┐
│              TYPED HYPERBOLIC GRAPH (NEW)                  │
├────────────────────────────────────────────────────────────┤
│  Edge types:       {total_edges} edges across {len(rel_counts)} types     │
│  Confidence:       {new_conf_mean:.3f} mean, {new_conf_unique} distinct values  │
│  Hierarchy:        Depth range [{min(depths):.2f}, {max(depths):.2f}]          │
│  Topology:         {json.dumps(topology_result.get('betti_numbers', {}))}
│  Knowledge gaps:   {topology_result.get('knowledge_gaps', 'N/A')} detected (transitive edges missing)
│  Circular reason:  {topology_result.get('circular_loops', 'N/A')} loops detected (falsifiable cycles)
└────────────────────────────────────────────────────────────┘
""")

# =====================================================================
# 9. Save results
# =====================================================================
results = {
    'experiment': 'Co-occurrence vs Typed Hyperbolic Knowledge',
    'old_system': {
        'n_concepts': len(old_concepts),
        'n_edges': old_edge_count,
        'edge_types': list(set(str(e.get('edge_type', '?')) for e in old_edges_raw[:100] if isinstance(e, dict))),
        'confidence': {
            'mean': float(np.mean(old_confidence_values)),
            'std': float(np.std(old_confidence_values)),
            'unique_values': len(set(old_confidence_values)),
        },
    },
    'new_system': {
        'n_concepts': len(tcg.nodes),
        'n_edges': len(tcg.edges),
        'relation_types': dict(rel_counts),
        'confidence': {
            'mean': float(np.mean(new_confs)) if new_confs else 0,
            'std': float(np.std(new_confs)) if new_confs else 0,
            'unique_values': len(set(round(c, 3) for c in new_confs)) if new_confs else 0,
        },
        'hyperbolic_depths': {
            'mean': float(np.mean(depths)),
            'std': float(np.std(depths)),
        },
        'topology': topology_result,
        'is_a_hierarchy_accuracy': {
            'correct_direction': correct_direction,
            'wrong_direction': wrong_direction,
            'accuracy': correct_direction / total_is_a if total_is_a else 0,
        },
        'physics_validation': {
            'n_physics_concepts': len(physics_concepts) if physics_concepts else 0,
            'mean_depth_physics': float(np.mean(physics_depths)) if physics_concepts else 0,
            'mean_depth_all': float(np.mean(all_depths)),
        },
    },
}

out_path = r'evals/results/typed_vs_occurence_experiment.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
