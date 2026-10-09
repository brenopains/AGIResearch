"""Caixa de Pandora — Pacote para Avaliação Cega"""
import sys, os, json, pickle, torch, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(sys.path[0])

from core.bge_encoder import BGEEncoder
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.cognition.hdc_decoder import HDCDecoder

# Load system
state = pickle.load(open(r'save/agi_state_live.pkl', 'rb'))
bge = BGEEncoder()
cge = CognitiveGraphEngine(bge_encoder=bge)
decoder = HDCDecoder(bge, cge)
for nid, nd in state.get('cge_nodes', {}).items():
    if isinstance(nid, str) and isinstance(nd, dict) and 'vector' in nd:
        if isinstance(nd['vector'], torch.Tensor):
            cge.nodes[nid] = nd

# Build vocabulary with English words extracted from concept names
extra_words = []
for nid in list(state.get('cge_nodes', {}).keys()):
    if isinstance(nid, str):
        for part in nid.replace('_', ' ').replace('-', ' ').split():
            if len(part) > 2 and part.isalpha():
                extra_words.append(part.lower())
        cleaned = re.sub(r'[^a-zA-Z\s]', '', nid.replace('_', ' '))
        if cleaned.strip() and len(cleaned) > 2:
            extra_words.append(cleaned.strip())
extra_words = list(set(extra_words))[:2000]
decoder.build_vocabulary(extra_words=extra_words, max_vocab=3000)

# Move vocab to CPU for memory efficiency
decoder.vocab_vectors = decoder.vocab_vectors.to('cpu') if decoder.vocab_vectors is not None else None

# Compute anisotropy baseline
all_vecs = decoder.vocab_vectors
if all_vecs is not None and len(all_vecs) > 10:
    n = min(500, len(all_vecs))
    vecs = all_vecs[:n]
    sim_samples = []
    for i in range(min(200, n)):
        for j in range(i+1, min(200, n)):
            sim = torch.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)).item()
            sim_samples.append(sim)
    mean_sim = sum(sim_samples)/len(sim_samples)
    std_sim = (sum((s-mean_sim)**2 for s in sim_samples)/len(sim_samples))**0.5
else:
    mean_sim, std_sim = 0.018, 0.091

print(f"Anisotropy baseline: mean={mean_sim:.4f} +- {std_sim:.4f}")

# Load pool
pool = json.load(open(r'evals/pool/pandora_pool.json'))
results = []
n_trials = min(20, len(pool))

# For each triple, present 9 candidates (3 correct + 6 distractors) 
# and ask evaluator to identify the 3 components
all_concepts = list(decoder.vocab_names)
import random
random.seed(20260712)

for idx, item in enumerate(pool[:n_trials]):
    A, B, C = item['A'], item['B'], item['C']
    
    # Create invention
    inv = decoder.create_invention([A, B, C])
    if inv is None: continue
    inv_cpu = inv.cpu()
    
    # Decode with z-score
    decoded = decoder.decode(inv_cpu, exclude=[A, B, C], top_k=20)
    z_scored = [(name, (sim - mean_sim) / std_sim) for name, sim in decoded]
    
    # Get top-8 decoded concepts as "decoded features"
    decoded_features = [name for name, z in z_scored[:8]]
    
    # Generate 6 distractors (random concepts NOT the components)
    distractors = []
    while len(distractors) < 6:
        candidate = random.choice(all_concepts)
        if candidate not in [A, B, C] and candidate not in distractors:
            distractors.append(candidate)
    
    # Shuffle: 3 components + 6 distractors = 9 candidates
    candidates = [A, B, C] + distractors
    random.shuffle(candidates)
    
    results.append({
        'trial': idx + 1,
        'n_total_candidates': 9,
        'n_components': 3,
        'n_distractors': 6,
        'candidates': candidates,
        'components_are': [A, B, C],  # ANSWER KEY (for evaluator later)
        'decoded_features': decoded_features,
        'top_5_zscores': [(name, round(z, 2)) for name, z in z_scored[:5]],
    })

# Save evaluation packet
output = {
    'task': 'Caixa de Pandora — Avaliação Cega',
    'instructions': [
        'Para cada trial, você recebe 9 conceitos candidatos.',
        'Exatamente 3 deles são os componentes originais da invenção.',
        'Os outros 6 são distratores aleatórios.',
        'Sua tarefa: identificar os 3 componentes corretos.',
        'Depois de responder, revelaremos a resposta.',
    ],
    'anisotropy_note': f'Similaridade média entre conceitos aleatórios: {mean_sim:.4f} ± {std_sim:.4f}. Z-scores normalizados. Vocabulário: {len(decoder.vocab_names)} conceitos.',
    'decoder_note': 'Os "decoded features" são os conceitos mais similares ao vetor invenção no espaço HDV 10000D (z-score normalizado). Ajudam a guiar a escolha mas não são os componentes.',
    'trials': []
}

for r in results:
    output['trials'].append({
        'trial': r['trial'],
        'candidates': r['candidates'],
        'decoded_hints_features': r['decoded_features'],
        'top_hdc_matches': r['top_5_zscores'],
        # ANSWER KEY hidden in separate file
    })

# Save public packet (without answer key)
out_path = r'evals/results/pandora_blind_public.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# Save answer key separately
answer_key = [{
    'trial': r['trial'],
    'components': r['components_are'],
} for r in results]

key_path = r'evals/results/pandora_blind_key.json'
with open(key_path, 'w', encoding='utf-8') as f:
    json.dump(answer_key, f, ensure_ascii=False, indent=2)

print(f"\nBlind evaluation packet: {out_path}")
print(f"Answer key: {key_path}")
print(f"\nTotal trials: {len(results)}")
print(f"Each trial: 9 candidates (3 components + 6 distractors)")
print(f"\nSample trial 1:")
print(f"  Candidates: {results[0]['candidates']}")
print(f"  Decoded hints: {results[0]['decoded_features']}")
print(f"  Answer: {results[0]['components_are']}")
