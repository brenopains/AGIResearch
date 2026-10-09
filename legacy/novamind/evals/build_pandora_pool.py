"""
Build the Caixa de Pandora concept pool from the saved NovaMind state.
Selects 200 triples of orthogonal concepts (cos < 0.3 between all pairs).
"""
import pickle, sys, random, json, torch
sys.path.insert(0, ('C:\\Users\\usuario\\Documents\\AGI 2.0\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))
random.seed(20260712)

# Load saved state
print("[POOL] Loading saved state...")
state = pickle.load(open(('C:\\Users\\usuario\\Documents\\AGI 2.0\\novamind\\save\\agi_state_live.pkl').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')), 'rb'))
cge_nodes = state['cge_nodes']
node_ids = list(cge_nodes.keys())
print(f"[POOL] CGE nodes: {len(node_ids)}")

# Try loading BGE
print("[POOL] Loading BGE encoder...")
from core.bge_encoder import BGEEncoder
bge = BGEEncoder()

# Filter valid concept names
valid = [n for n in node_ids if isinstance(n, str) and len(n) > 2 and len(n) < 40 
         and not n.startswith('__') and n != '']
valid_sorted = sorted(valid)[:500]
print(f"[POOL] Valid concepts: {len(valid)} (sampling top 500)")

# Compute BGE embeddings
concepts = []
for i, name in enumerate(valid_sorted):
    try:
        vec = bge.encode_bipolar([name])[0]
        concepts.append({'name': name, 'vector': vec.cpu()})
        if (i+1) % 100 == 0:
            print(f"[POOL]  Embedded {i+1}/{len(valid_sorted)}")
    except Exception as e:
        pass
print(f"[POOL] Total embedded: {len(concepts)}")

# Compute similarity matrix
n = len(concepts)
sim_matrix = torch.zeros(n, n)
for i in range(n):
    for j in range(i+1, n):
        s = torch.cosine_similarity(
            concepts[i]['vector'].unsqueeze(0), 
            concepts[j]['vector'].unsqueeze(0)
        ).item()
        sim_matrix[i,j] = s
        sim_matrix[j,i] = s

# Find orthogonal triples
triples = []
found = set()
for i in range(n):
    for j in range(i+1, n):
        if sim_matrix[i,j] >= 0.3:
            continue
        for k in range(j+1, n):
            if sim_matrix[i,k] >= 0.3 or sim_matrix[j,k] >= 0.3:
                continue
            key = tuple(sorted([concepts[i]['name'], concepts[j]['name'], concepts[k]['name']]))
            if key in found:
                continue
            found.add(key)
            triples.append((concepts[i]['name'], concepts[j]['name'], concepts[k]['name']))
            if len(triples) >= 200:
                break
        if len(triples) >= 200:
            break
    if len(triples) >= 200:
        break

# Save only names (vectors can be re-computed with same seed)
pool = [{'A': t[0], 'B': t[1], 'C': t[2], 'id': f'PANDORA-{i:03d}'} 
        for i, t in enumerate(triples)]

out_path = ('C:\\Users\\usuario\\Documents\\AGI 2.0\\novamind\\evals\\pool\\pandora_pool.json').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~'))
with open(out_path, 'w') as f:
    json.dump(pool, f, indent=2)

print(f"\n[POOL] Saved {len(pool)} triples to {out_path}")
print("Sample:")
for t in triples[:5]:
    print(f"  {t[0]} ++ {t[1]} ++ {t[2]}")
