"""Decoder fix: build vocabulary from CLEAN English words, not CGE abstract_*"""
import sys, os, json, pickle, torch, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(sys.path[0])

from core.bge_encoder import BGEEncoder
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.cognition.hdc_decoder import HDCDecoder

state = pickle.load(open(r'save/agi_state_live.pkl', 'rb'))
bge = BGEEncoder()
cge = CognitiveGraphEngine(bge_encoder=bge)

# Load CGE nodes
for nid, nd in state.get('cge_nodes', {}).items():
    if isinstance(nid, str) and isinstance(nd, dict) and 'vector' in nd:
        if isinstance(nd['vector'], torch.Tensor):
            cge.nodes[nid] = nd

decoder = HDCDecoder(bge, cge)

# Build CLEAN vocabulary: extract individual English words from CGE concepts
# and add them WITHOUT the "abstract_" prefix
clean_words = set()

for nid in list(state.get('cge_nodes', {}).keys()):
    if not isinstance(nid, str): continue
    # Split by underscore and space
    parts = re.split(r'[_\s\-]+', nid)
    for p in parts:
        p_clean = re.sub(r'[^a-zA-Z]', '', p).lower()
        # Accept only: length 3-15, no "abstract", no portuguese-only words
        if len(p_clean) >= 3 and len(p_clean) <= 15:
            if p_clean != 'abstract' and not p_clean.startswith('abstract'):
                # Basic English-like filter: has a vowel
                if any(v in p_clean for v in 'aeiou'):
                    clean_words.add(p_clean)

# Also add the full concept names (cleaned)
for nid in list(state.get('cge_nodes', {}).keys()):
    if not isinstance(nid, str): continue
    cleaned = re.sub(r'abstract_', '', nid)  # Remove abstract_ prefix
    cleaned = re.sub(r'_', ' ', cleaned)
    if len(cleaned) >= 3 and len(cleaned) <= 30:
        if not cleaned.startswith('abstract'):
            clean_words.add(cleaned)

# Add some basic English nouns for diversity
english_base = ['water', 'fire', 'earth', 'air', 'metal', 'wood', 'stone', 'light',
    'sound', 'force', 'energy', 'motion', 'time', 'space', 'number', 'shape',
    'machine', 'tool', 'wheel', 'lever', 'gear', 'pulley', 'screw', 'wedge',
    'engine', 'motor', 'spring', 'chain', 'belt', 'axle', 'bearing', 'valve',
    'switch', 'circuit', 'wire', 'battery', 'magnet', 'coil', 'lens', 'mirror',
    'pipe', 'tank', 'pump', 'filter', 'turbine', 'generator', 'transformer',
    'animal', 'plant', 'cell', 'nucleus', 'membrane', 'protein', 'enzyme',
    'brain', 'nerve', 'muscle', 'bone', 'blood', 'heart', 'lung', 'eye', 'ear',
    'atom', 'molecule', 'electron', 'proton', 'neutron', 'photon', 'wave',
    'solid', 'liquid', 'gas', 'plasma', 'crystal', 'glass', 'rubber', 'paper',
    'code', 'data', 'signal', 'network', 'server', 'memory', 'clock', 'logic',
]

for w in english_base:
    if w not in clean_words:
        clean_words.add(w)

print(f"Clean vocabulary size: {len(clean_words)}")

# Build decoder vocabulary
decoder.build_vocabulary(extra_words=list(clean_words), max_vocab=5000)

# Verify: how many 'abstract_' remain?
abs_left = sum(1 for n in decoder.vocab_names if 'abstract' in n.lower())
print(f"Concepts with 'abstract' in name: {abs_left} / {len(decoder.vocab_names)}")

# Compute anisotropy baseline
decoder.vocab_vectors = decoder.vocab_vectors.to('cpu') if decoder.vocab_vectors is not None else None
all_vecs = decoder.vocab_vectors

if all_vecs is not None and len(all_vecs) > 10:
    n = min(300, len(all_vecs))
    vecs = all_vecs[:n]
    sim_samples = []
    for i in range(min(150, n)):
        for j in range(i+1, min(150, n)):
            sim = torch.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)).item()
            sim_samples.append(sim)
    mean_sim = sum(sim_samples)/len(sim_samples)
    std_sim = (sum((s-mean_sim)**2 for s in sim_samples)/len(sim_samples))**0.5
    print(f"\nClean vocab anisotropy: mean={mean_sim:.4f} +- {std_sim:.4f} (n={len(sim_samples)} pairs)")

# Now generate blind packets with clean vocabulary
pool = json.load(open(r'evals/pool/pandora_pool.json'))
import random
random.seed(20260712)

n_trials = 20
all_vocab_names = list(decoder.vocab_names)

# First compute all inventions
inventions = []
for item in pool[:n_trials]:
    A, B, C = item['A'], item['B'], item['C']
    inv = decoder.create_invention([A, B, C])
    if inv is None: continue
    inventions.append(inv.cpu())

# Compute pairwise similarities
print(f"\n=== Pairwise invention similarities (clean vocab) ===")
n_inv = len(inventions)
for i in range(n_inv):
    for j in range(i+1, n_inv):
        s = torch.cosine_similarity(inventions[i].unsqueeze(0), inventions[j].unsqueeze(0)).item()
        ci = [pool[i]['A'], pool[i]['B'], pool[i]['C']]
        cj = [pool[j]['A'], pool[j]['B'], pool[j]['C']]
        overlap = len(set(ci) & set(cj))
        if overlap >= 2:
            print(f"  sim({ci}, {cj}) = {s:.4f} (overlap={overlap})")

# Generate clean blind packets
results = []
for idx, inv in enumerate(inventions):
    item = pool[idx]
    A, B, C = item['A'], item['B'], item['C']
    
    # Decode with z-score normalization
    decoded = decoder.decode(inv, exclude=[A, B, C], top_k=15)
    z_scored = [(name, (sim - mean_sim) / std_sim) for name, sim in decoded]
    
    # Filter: remove items that contain any component as substring
    decoded_features = []
    for name, z in z_scored:
        name_lower = name.lower()
        # Skip if ANY component is a substring of this name
        if any(c.lower() in name_lower for c in [A, B, C]):
            continue
        decoded_features.append((name, z))
    
    # Take top 5 after filtering
    top5_clean = decoded_features[:5]
    
    # Generate 6 distractors (ensure no duplicates and no component matches)
    distractors = []
    while len(distractors) < 6:
        candidate = random.choice(all_vocab_names)
        if candidate in [A, B, C] or candidate in distractors:
            continue
        # Skip if candidate contains component substring or vice versa
        cand_lower = candidate.lower()
        if any(c.lower() in cand_lower or cand_lower in c.lower() for c in [A, B, C]):
            continue
        distractors.append(candidate)
    
    # 9 candidates: 3 components + 6 distractors
    candidates = [A, B, C] + distractors
    random.shuffle(candidates)
    
    results.append({
        'trial': idx + 1,
        'candidates': candidates,
        'n_candidates': len(candidates),
        'n_duplicates': len(candidates) - len(set(candidates)),
        'decoded_hints': [n for n, z in top5_clean],
        'top_zscores': [(n, round(z, 2)) for n, z in top5_clean],
        'components_are': [A, B, C],
    })

# Verify no substring leaks
print(f"\n=== Substring leak check ===")
leaks = 0
for r in results:
    hints_text = ' '.join(r['decoded_hints']).lower()
    for comp in r['components_are']:
        if comp.lower() in hints_text:
            leaks += 1
            print(f"  Trial {r['trial']}: LEAK '{comp}' in hints")
print(f"Total leaks: {leaks}/60 component-hint pairs")

# Verify no cross-trial hint overlap
print(f"\n=== Cross-trial hint overlap ===")
hint_sets = [set(r['decoded_hints']) for r in results]
for i in range(len(results)):
    for j in range(i+1, len(results)):
        overlap = hint_sets[i] & hint_sets[j]
        if overlap:
            print(f"  Trial {i+1} & Trial {j+1} share hints: {overlap}")

# Save public packet
output = {
    'task': 'Caixa de Pandora — Avaliação Cega v2 (Sem vazamento)',
    'note': 'Vocabulário limpo: sem conceitos "abstract_*". Dicas filtradas para não conter substrings dos componentes. Anisotropy baseline medido no vocabulário limpo.',
    'anisotropy': {'mean': round(mean_sim, 4), 'std': round(std_sim, 4)},
    'invention_pairwise_similarity': {
        'mean': round(float(torch.tensor([torch.cosine_similarity(inventions[i].unsqueeze(0), inventions[j].unsqueeze(0)) for i in range(len(inventions)) for j in range(i+1, len(inventions))]).mean()), 4)
    },
    'vocab_size': len(decoder.vocab_names),
    'instructions': [
        '20 trials. Cada trial: 9 candidatos, exatamente 3 são componentes.',
        'Dicas decodificadas do HDV (z-score normalizado, filtradas para não conter componentes).',
        'Avaliador precisa ser alguém SEM contexto do projeto (você tem viés de expectativa).',
    ],
    'trials': [{
        'trial': r['trial'],
        'candidates': r['candidates'],
        'hints': r['decoded_hints'],
    } for r in results]
}

out_path = r'evals/results/pandora_blind_v2_public.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# Answer key
key_path = r'evals/results/pandora_blind_v2_key.json'
with open(key_path, 'w', encoding='utf-8') as f:
    json.dump([{'trial': r['trial'], 'components': r['components_are']} for r in results],
              f, ensure_ascii=False, indent=2)

print(f"\nSaved: {out_path}")
print(f"Key: {key_path}")
print(f"\n=== Sample trial 1 (clean) ===")
print(f"  Candidates: {results[0]['candidates']}")
print(f"  Hints: {results[0]['decoded_hints']}")
print(f"  Answer: {results[0]['components_are']}")
