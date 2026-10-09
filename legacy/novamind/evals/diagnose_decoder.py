"""Diagnóstico: similaridade par-a-par entre HDVs de invenção + debug abstract_"""
import sys, os, json, pickle, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(sys.path[0])

from core.bge_encoder import BGEEncoder
from core.cognition.hdc_decoder import HDCDecoder

state = pickle.load(open(r'save/agi_state_live.pkl', 'rb'))
bge = BGEEncoder()

pool = json.load(open(r'evals/pool/pandora_pool.json'))

# Compute all 20 invention vectors
inventions = []
comps_list = []
for item in pool[:20]:
    A, B, C = item['A'], item['B'], item['C']
    vec_A = bge.encode_bipolar([A])[0]
    vec_B = bge.encode_bipolar([B])[0]
    vec_C = bge.encode_bipolar([C])[0]
    inv = torch.sign(vec_A * vec_B * vec_C)
    inventions.append(inv.cpu())
    comps_list.append((A, B, C))

# Pairwise similarities
n = len(inventions)
print(f"=== Pairwise Invention HDV Similarities (n={n}) ===")
sims = torch.zeros(n, n)
for i in range(n):
    for j in range(i, n):
        s = torch.cosine_similarity(inventions[i].unsqueeze(0), inventions[j].unsqueeze(0)).item()
        sims[i,j] = s
        sims[j,i] = s

# Stats
triu = torch.triu(sims, diagonal=1)
vals = triu[triu > -2]
print(f"\nMean pairwise sim: {vals.mean():.4f} +- {vals.std():.4f}")
print(f"Min: {vals.min():.4f}, Max: {vals.max():.4f}")
print(f"Median: {vals.median():.4f}")

# Check: are they correlated by shared components?
# Many share 'ability' and 'assinou'
print(f"\n=== By component overlap ===")
for i in range(min(10, n)):
    for j in range(i+1, min(10, n)):
        ci = set(comps_list[i])
        cj = set(comps_list[j])
        overlap = ci & cj
        s = sims[i,j].item()
        stars = '*' * len(overlap) if overlap else ''
        print(f"  sim({comps_list[i]}, {comps_list[j]}) = {s:.4f} {stars} overlapped={overlap}")

# Are all inventions significantly different from each other?
below_03 = (vals < 0.3).sum().item()
above_05 = (vals > 0.5).sum().item()
print(f"\nPairs with sim < 0.3: {below_03}/{len(vals)} ({below_03/len(vals)*100:.1f}%)")
print(f"Pairs with sim > 0.5: {above_05}/{len(vals)} ({above_05/len(vals)*100:.1f}%)")

# Now debug: what does the vocabulary look like?
print(f"\n=== Vocabulary Structure ===")
# Check which concepts have 'abstract_' prefix
prefixes = {}
for nid in list(state.get('cge_nodes', {}).keys()):
    if isinstance(nid, str):
        prefix = nid.split('_')[0] if '_' in nid else nid[:5]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
top = sorted(prefixes.items(), key=lambda x: -x[1])[:10]
print(f"Top prefixes in CGE nodes:")
for p, c in top:
    print(f"  '{p}': {c} occurrences")

# How many of the nearest concepts come from CGE vs extra words?
from core.cognitive_graph_engine import CognitiveGraphEngine
cge = CognitiveGraphEngine(bge_encoder=bge)
for nid, nd in state.get('cge_nodes', {}).items():
    if isinstance(nid, str) and isinstance(nd, dict) and 'vector' in nd:
        if isinstance(nd['vector'], torch.Tensor):
            cge.nodes[nid] = nd

decoder = HDCDecoder(bge, cge)
extra_words = []
for nid in list(state.get('cge_nodes', {}).keys()):
    if isinstance(nid, str):
        for part in nid.replace('_', ' ').replace('-', ' ').split():
            if len(part) > 2 and part.isalpha():
                extra_words.append(part.lower())
        cleaned = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in nid.replace('_', ' '))
        if cleaned.strip() and len(cleaned) > 2:
            extra_words.append(cleaned.strip())
extra_words = list(set(extra_words))[:2000]
decoder.build_vocabulary(extra_words=extra_words, max_vocab=3000)

# Count CGE vs extra
cge_count = sum(1 for n in decoder.vocab_names if n in state.get('cge_nodes', {}))
extra_count = len(decoder.vocab_names) - cge_count
print(f"\nVocab: CGE={cge_count}, Extras={extra_count}, Total={len(decoder.vocab_names)}")

# Check for 'abstract_' in vocab
abs_count = sum(1 for n in decoder.vocab_names if n.startswith('abstract'))
print(f"Concepts starting with 'abstract_': {abs_count}")

# Test: for a specific invention, what are the top matches WITHOUT CGE abstract_ concepts?
inv0 = inventions[0]
# All vocab
if decoder.vocab_vectors is not None:
    voc = decoder.vocab_vectors
    sims_all = torch.cosine_similarity(inv0.unsqueeze(0), voc)
    top5_all = sims_all.topk(5)
    print(f"\nTop-5 ALL vocab for inv0 ({comps_list[0]}):")
    for s, idx in zip(top5_all.values.tolist(), top5_all.indices.tolist()):
        print(f"  {decoder.vocab_names[idx]}: {s:.4f}")
    
    # Filter out abstract_ concepts
    non_abstract = [i for i, n in enumerate(decoder.vocab_names) if not n.startswith('abstract')]
    if non_abstract:
        sims_na = sims_all[non_abstract]
        top5_na = sims_na.topk(5)
        print(f"\nTop-5 WITHOUT 'abstract_' for inv0:")
        for s, idx_na in zip(top5_na.values.tolist(), top5_na.indices.tolist()):
            idx = non_abstract[idx_na]
            print(f"  {decoder.vocab_names[idx]}: {s:.4f}")
