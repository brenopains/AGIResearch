"""
Option 3: Caixa de Pandora decoder via DistilGPT-2
===================================================
Pipeline:
  1. bind(A, B, C) -> invention HDV
  2. nearest-neighbor in CGE vocabulary (z-score normalized)
  3. LM generates description from nearest concepts
  
Honest documentation: cognition = HDC, production = small neural LM (81M params)
"""
import sys, os, json, pickle, torch, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(sys.path[0])

from core.bge_encoder import BGEEncoder
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.cognition.hdc_decoder import HDCDecoder
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load system
print("[LOAD] System...")
state = pickle.load(open(r'save/agi_state_live.pkl', 'rb'))
bge = BGEEncoder()
cge = CognitiveGraphEngine(bge_encoder=bge)
decoder = HDCDecoder(bge, cge)

for nid, nd in state.get('cge_nodes', {}).items():
    if isinstance(nid, str) and isinstance(nd, dict) and 'vector' in nd:
        if isinstance(nd['vector'], torch.Tensor):
            cge.nodes[nid] = nd

# 2. Build vocabulary from CGE concepts + English words extracted from them
extra_words = []
for nid in list(state.get('cge_nodes', {}).keys()):
    if isinstance(nid, str):
        # Extract English words from concept names like "abstract_communication_consists"
        for part in nid.replace('_', ' ').replace('-', ' ').split():
            if len(part) > 3 and part.isalpha():
                extra_words.append(part.lower())
        # Also add the full name
        cleaned = re.sub(r'[^a-zA-Z\s]', '', nid.replace('_', ' '))
        if cleaned.strip() and len(cleaned) > 3:
            extra_words.append(cleaned.strip())

extra_words = list(set(extra_words))[:2000]
print(f"[VOCAB] Extra words: {len(extra_words)}")
decoder.build_vocabulary(extra_words=extra_words, max_vocab=3000)

# 3. Load generator LM
print("[LM] Loading DistilGPT-2...")
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('distilgpt2')
device = cge.device
model = model.to(device)
print("[LM] Ready")

# 4. Move vocabulary to CPU to save GPU memory for LM
if decoder.vocab_vectors is not None:
    decoder.vocab_vectors = decoder.vocab_vectors.to('cpu')

# 5. Compute anisotropy baseline for z-score normalization
print("[BASELINE] Computing random similarity baseline...")
all_vecs = decoder.vocab_vectors  # (N, 10000) bipolar (on CPU)
if all_vecs is not None and len(all_vecs) > 10:
    # Compute mean pairwise similarity efficiently (chunked, no NxNxD tensor)
    n = min(500, len(all_vecs))
    vecs = all_vecs[:n].to('cpu')  # Move to CPU for this computation
    sim_samples = []
    for i in range(min(200, n)):
        for j in range(i+1, min(200, n)):
            sim = torch.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)).item()
            sim_samples.append(sim)
    mean_sim = sum(sim_samples) / len(sim_samples) if sim_samples else 0.547
    std_sim = (sum((s - mean_sim)**2 for s in sim_samples) / len(sim_samples))**0.5 if sim_samples else 0.066
    print(f"  Mean random sim: {mean_sim:.4f} +- {std_sim:.4f} (n={len(sim_samples)} pairs)")
else:
    mean_sim, std_sim = 0.547, 0.066

# 5. Run Pandora with decoder + LM
pool = json.load(open(r'evals/pool/pandora_pool.json'))
results = []

for idx, item in enumerate(pool[:10]):
    A, B, C = item['A'], item['B'], item['C']
    print(f"\n[{idx+1}/10] {A} + {B} + {C}")
    
    # Create invention
    inv = decoder.create_invention([A, B, C])
    if inv is None:
        continue
    
    # Decode: nearest neighbors with z-score (move invention to CPU for vocab matching)
    inv_cpu = inv.cpu()
    decoded = decoder.decode(inv_cpu, exclude=[A, B, C], top_k=10)
    
    # Z-score normalized
    z_scored = [(name, (sim - mean_sim) / std_sim) for name, sim in decoded]
    
    # Top features
    top_features = [name for name, z in z_scored[:5]]
    print(f"  Nearest: {', '.join(top_features)}")
    
    # Generate description via LM
    prompt = f"Describe a novel invention that combines these elements: {', '.join(top_features)}. The invention is:"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    out = model.generate(
        **inputs, 
        max_new_tokens=60, 
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    description = tokenizer.decode(out[0], skip_special_tokens=True)
    # Clean: remove the prompt part
    description = description.replace(prompt, '').strip()
    # Take first sentence
    if '.' in description:
        description = description.split('.')[0] + '.'
    description = description[:200]
    
    print(f"  Description: {description}")
    
    results.append({
        'id': item['id'],
        'components': [A, B, C],
        'nearest_features': top_features,
        'description': description,
    })

# Save for blind evaluation
print(f"\n{'='*70}")
print(f"Saving {len(results)} results for blind evaluation...")
out = {
    'note': 'Each invention = bind(A, B, C). The decoder extracts nearest CGE concepts (z-score normalized) and uses DistilGPT-2 (81M params) to generate a description from them. The user (blind) must identify components from descriptions.',
    'anisotropy_baseline': {
        'mean_random_similarity': mean_sim,
        'std_random_similarity': std_sim,
        'note': 'BGE-based cosine similarity between random concept pairs in the vocabulary. ~0.55 mean indicates strong anisotropy — z-score normalization partially corrects this.'
    },
    'trials': results
}

out_path = r'evals/results/blind_evaluation_packet.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"Saved to {out_path}")
