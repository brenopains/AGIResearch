"""Test decode + Caixa de Pandora with 50 triples"""
import sys, os, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(sys.path[0])
import torch
from core.bge_encoder import BGEEncoder
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.cognition.hdc_decoder import HDCDecoder

print("=" * 70)
print("F3 - HDCDecoder + Caixa de Pandora")
print("=" * 70)

state = pickle.load(open(r'save/agi_state_live.pkl', 'rb'))
print(f"[STATE] CGE nodes: {len(state.get('cge_nodes', {}))}")

bge = BGEEncoder()
cge = CognitiveGraphEngine(bge_encoder=bge)
decoder = HDCDecoder(bge, cge)

for node_id, node_data in state.get('cge_nodes', {}).items():
    if isinstance(node_id, str) and isinstance(node_data, dict) and 'vector' in node_data:
        vec = node_data['vector']
        if isinstance(vec, torch.Tensor):
            cge.nodes[node_id] = node_data

decoder.build_vocabulary()
print(f"[DECODER] Vocab: {len(decoder.vocab_names)} concepts")

pool = json.load(open(r'evals/pool/pandora_pool.json'))

results = []
total_recovery = 0
for idx, item in enumerate(pool[:50]):
    A, B, C = item['A'], item['B'], item['C']
    inv = decoder.create_invention([A, B, C])
    if inv is None:
        continue
    report = decoder.component_recovery(inv, [A, B, C], top_n_candidates=50)
    total_recovery += report['recovery_rate']
    results.append({
        'id': item['id'],
        'components': [A, B, C],
        'recovery_rate': report['recovery_rate'],
        'found': report['n_components_found'],
        'decoded': report['top_decoded'][:8],
        'semantic': report['semantic_relations'][:6],
    })

avg_recovery = total_recovery / len(results) if results else 0
print(f"\n=== T1 - Caixa de Pandora (n={len(results)}) ===")
print(f"Component recovery rate: {avg_recovery*100:.1f}%")
from collections import Counter
recovery_dist = Counter()
for r in results:
    recovery_dist[r['found']] += 1
for k in sorted(recovery_dist):
    pct = recovery_dist[k]/len(results)*100
    print(f"  {k}/3 components: {recovery_dist[k]} triples ({pct:.0f}%)")

print("\n=== Examples ===")
for r in results[:5]:
    comps = ', '.join(r['components'])
    decoded = ' | '.join([f"{d[0]}({d[1]:.3f})" for d in r['decoded'][:5]])
    print(f"  [{r['id']}] {comps}")
    print(f"     -> {decoded}")
    if r['recovery_rate'] > 0:
        matched = [d[0] for d in r['decoded'][:20] 
                   if d[0].lower() in [c.lower() for c in r['components']]]
        if matched:
            print(f"     -> Direct matches: {', '.join(matched)}")
    print()

top1_sims = [r['decoded'][0][1] for r in results if r['decoded']]
if top1_sims:
    print(f"Top-1 avg similarity: {sum(top1_sims)/len(top1_sims):.4f}")

total_sem = 0
n_sem = 0
for r in results[:20]:
    for comp, dname, sim in r.get('semantic', []):
        total_sem += sim
        n_sem += 1
if n_sem:
    print(f"Avg semantic sim decoded<->component: {total_sem/n_sem:.4f}")

out_path = r'evals/results/t1_pandora_results.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump({
        'n_triples': len(results),
        'avg_component_recovery': avg_recovery,
        'results': results,
    }, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
