"""
Experimento Otimizado: Typed Hyperbolic vs Co-ocorrência
=========================================================
Focado nos resultados científicos essenciais, sem overflow de saída.
Topologia computada em subamostra para evitar timeout.
"""
import sys, os, json, math, time, pickle
from collections import defaultdict
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

print("=" * 70)
print("EXPERIMENTO: Co-ocorrencia vs Typed Hyperbolic")
print("=" * 70)

# 1. Load OLD system stats
data = torch.load(r'save/brain_live.pth', map_location='cpu', weights_only=False)
old_nodes = list(data.get('nodes', {}).keys())
old_edges = data.get('edges', [])
old_edge_types = set()
old_confs = []
for e in old_edges:
    if isinstance(e, dict):
        old_edge_types.add(str(e.get('edge_type', '?')))
        c = e.get('confidence', 1.0)
        if isinstance(c, torch.Tensor): c = c.mean().item()
        old_confs.append(float(c))

print(f"\n[OLD - Co-ocorrencia]")
print(f"  Conceitos: {len(old_nodes)}")
print(f"  Arestas: {len(old_edges)}")
print(f"  Tipos de aresta: {old_edge_types}")
print(f"  Confianca: media={np.mean(old_confs):.4f}, desvio={np.std(old_confs):.4f}")
print(f"  Valores unicos de confianca: {len(set(round(c,4) for c in old_confs))}")

# 2. Build NEW system
print(f"\n[NEW - Typed Hyperbolic]")
from core.typed_graph import TypedCognitiveGraph
tcg = TypedCognitiveGraph(dim=100, curvature=1.0)
t_start = time.time()
stats = tcg.load_from_brain_state(r'save/brain_live.pth', r'save/agi_state_live.pkl')
build_t = time.time() - t_start

deps = [tcg.nodes[n]['depth'] for n in tcg.nodes]
new_confs = [e['confidence'] for e in tcg.edges.values()]

rel_counts = defaultdict(int)
for key in tcg.edges: rel_counts[key[2]] += 1
total_edges = sum(rel_counts.values()) or 1

print(f"  Construcao: {build_t:.1f}s")
print(f"  Conceitos: {len(tcg.nodes)}")
print(f"  Arestas: {len(tcg.edges)}")
print(f"  Tipos de relacao: {list(rel_counts.keys())}")
print(f"  Profundidade hiperbolica: media={np.mean(deps):.4f}, desvio={np.std(deps):.4f}")
for lo, hi, label in [(0, 0.5, 'abstrato'), (0.5, 2, 'medio'), (2, 5, 'especifico'), (5, 10, 'muito especifico')]:
    c = sum(1 for d in deps if lo <= d < hi)
    if c: print(f"    {lo}-{hi}: {c} conceitos ({label})")
print(f"  Confianca: media={np.mean(new_confs):.4f}, desvio={np.std(new_confs):.4f}")
print(f"    min={min(new_confs):.4f}, max={max(new_confs):.4f}")
print(f"    valores unicos: {len(set(round(c,3) for c in new_confs))}")

# 3. IS-A hierarchy check
is_a_edges = [(k, e) for k, e in tcg.edges.items() if k[2] == 'IS_A']
correct_is_a = 0
wrong_is_a = 0
for key, edge in is_a_edges:
    frm, to, _ = key
    if frm in tcg.nodes and to in tcg.nodes:
        if tcg.nodes[frm]['depth'] > tcg.nodes[to]['depth'] and abs(tcg.nodes[frm]['depth'] - tcg.nodes[to]['depth']) > 0.5:
            correct_is_a += 1
        else:
            wrong_is_a += 1
total_is_a = correct_is_a + wrong_is_a
is_a_acc = correct_is_a / total_is_a if total_is_a else 0
print(f"\n  Hierarquia IS-A: {correct_is_a}/{total_is_a} corretos ({is_a_acc*100:.1f}%)")

# 4. Topology (subsampled for speed)
print(f"\n  Topologia (subamostra de 80 conceitos)...")
t_start = time.time()
result = tcg.build_topology()
topo_t = time.time() - t_start
if 'betti_numbers' in result:
    print(f"  Tempo: {topo_t:.1f}s")
    print(f"  Betti numbers: {result['betti_numbers']}")
    print(f"  Entropia de persistencia: {result.get('persistence_entropy', 0):.4f}")
    print(f"  Gaps de conhecimento (H2): {result.get('knowledge_gaps', 'N/A')}")
    print(f"  Loops circulares (H1): {result.get('circular_loops', 'N/A')}")
else:
    print(f"  Erro: {result}")

# 5. Relacao tipada distribution
print(f"\n  Distribuicao por tipo de relacao:")
for rtype in sorted(rel_counts):
    print(f"    {rtype}: {rel_counts[rtype]}")

# 6. Summary
print(f"\n{'='*70}")
print("SUMARIO COMPARATIVO")
print("=" * 70)
print(f"""
  METRICA                    CO-OCORRENCIA      TYPED HYPERBOLIC   DIFERENCA
  -------------------------  -----------------  -----------------  ---------------
  Conceitos                  {len(old_nodes):5d}              {len(tcg.nodes):5d}              +{len(tcg.nodes)-len(old_nodes)}
  Arestas                    {len(old_edges):5d}              {len(tcg.edges):5d}              +{len(tcg.edges)-len(old_edges)}
  Tipos de relacao           {len(old_edge_types):5d}              {len(rel_counts):5d}              +{len(rel_counts)-len(old_edge_types)}
  Confianca (media)          {np.mean(old_confs):.4f}           {np.mean(new_confs):.4f}           FEP agora calcula
  Confianca (desvio)         {np.std(old_confs):.4f}           {np.std(new_confs):.4f}           variacao real
  Valores unicos confianca   {len(set(round(c,4) for c in old_confs)):4d}              {len(set(round(c,3) for c in new_confs)):4d}              +{len(set(round(c,3) for c in new_confs))-len(set(round(c,4) for c in old_confs))}
  Hierarquia (IS-A correto)  N/A                 {is_a_acc*100:.1f}%            axioma verificavel
  Betti beta_0 (clusters)    N/A                 {result.get('betti_numbers', {}).get('beta_0', '?'):>5s}            so possivel com tipos
  Betti beta_1 (ciclos)      N/A                 {result.get('betti_numbers', {}).get('beta_1', '?'):>5s}            so possivel com tipos
  Betti beta_2 (vazios)      N/A                 {result.get('betti_numbers', {}).get('beta_2', '?'):>5s}            so possivel com tipos
""")

# Save
results = {
    'experiment': 'co-occurrence_vs_typed_hyperbolic',
    'old_system': {
        'concepts': len(old_nodes),
        'edges': len(old_edges),
        'edge_types': list(old_edge_types),
        'confidence': {'mean': float(np.mean(old_confs)), 'std': float(np.std(old_confs)), 'unique': len(set(round(c,4) for c in old_confs))},
    },
    'new_system': {
        'concepts': len(tcg.nodes),
        'edges': len(tcg.edges),
        'relation_types': dict(rel_counts),
        'depths': {'mean': float(np.mean(deps)), 'std': float(np.std(deps))},
        'confidence': {'mean': float(np.mean(new_confs)), 'std': float(np.std(new_confs)), 'unique': len(set(round(c,3) for c in new_confs))},
        'is_a_hierarchy_accuracy': is_a_acc,
        'topology': result,
        'build_time_s': build_t,
    }
}
out = r'evals/results/experiment_comparison.json'
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"Resultados salvos em {out}")
