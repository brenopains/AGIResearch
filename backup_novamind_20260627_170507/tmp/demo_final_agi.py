"""NovaMind - AGI Signal Demo
Shows: memory recall, metacognition, world model, learning, reasoning"""
import gc, sys, os
for mod in list(sys.modules.keys()):
    if any(k in mod for k in ['core', 'novacrush', 'word_ngram']):
        del sys.modules[mod]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, time
gc.collect(); torch.cuda.empty_cache()
from core.mind import NovaMind

print("=" * 65)
print("  NovaMind Cognitive Architecture - AGI Signal Demo")
print("  7-layer: perception, world model, memory, reasoning, metacognition, goals, learning")
print("  GPU: NVIDIA GeForce RTX 3050 Laptop (4GB VRAM)")
print("=" * 65)
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
t0 = time.time()
mind = NovaMind()
print(f"  Init: {time.time()-t0:.1f}s | RAM: ~1.7GB | VRAM: 17MB")
print()

def run(q):
    r = mind.think(q)
    c = [c['label'] for c in r['layers']['perception']['concepts']]
    m = r['layers']['metacognition']
    mem = r['layers']['memory']
    wm = r['layers']['world_model']
    lr = r['layers']['learning']
    recall = [ep['content'].get('input','') for ep in mem.get('recalled_episodes',[]) if ep.get('content')]
    return r, c, m, mem, wm, lr, recall

# Phase 1: Learn
print("--- PHASE 1: Episodic Learning ---")
inputs = [
    "My dog is named Rex",
    "I live in Brazil",
    "I like pizza",
    "I am learning about neural networks",
]
for q in inputs:
    r, c, m, mem, wm, lr, recall = run(q)
    print(f"  IN:  {q}")
    print(f"  OUT: {r['response']['text'][:90]}")
    print(f"       concepts={c[:4]} confidence={m['confidence']['overall']:.0%} novel={lr['novelty_analysis']['is_novel']}")
    print()

# Phase 2: Memory
print("--- PHASE 2: Memory Recall ---")
queries = [
    ("What is my dog name", "Rex"),
    ("Where do I live", "Brazil"),
    ("What food do I like", "pizza"),
    ("What am I learning about", "neural networks"),
]
for q, expected in queries:
    r, c, m, mem, wm, lr, recall = run(q)
    recalled = any(expected.lower() in str(rec).lower() for rec in recall)
    print(f"  Q:   {q}")
    print(f"  OUT: {r['response']['text'][:90]}")
    print(f"       recall-hit={recalled} confidence={m['confidence']['overall']:.0%}")
    print()

# Phase 3: Reasoning
print("--- PHASE 3: Hybrid Reasoning ---")
r, c, m, mem, wm, lr, recall = run(
    "If machines learn from data, and neural networks are machines, can neural networks learn from data?"
)
reasoning = r['layers']['reasoning']
paradigms = reasoning.get('paradigms', {})
consensus = reasoning.get('consensus', {})
print(f"  IN:  If machines learn from data, and neural networks are machines...")
print(f"  OUT: {r['response']['text'][:90]}")
print(f"  Reasoning:")
for name, result in paradigms.items():
    print(f"       {name}: conf={result['confidence']:.0%} | {result['conclusion'][:60]}")
print(f"  Consensus: {reasoning['reasoning_time_ms']:.0f}ms | primary={consensus.get('primary_paradigm','?')}")
print()

# Phase 4: World Model
print("--- PHASE 4: World Model (Causal Graph) ---")
state = mind.get_full_state()
wm = state['world_model']
chains = mind.get_causal_chains_with_labels(min_length=2)
print(f"  Nodes: {len(wm['nodes'])}, Edges: {len(wm['edges'])}")
if chains:
    print(f"  Causal chains:")
    for chain in chains[:5]:
        print(f"    {' -> '.join(chain[:4])}")
print()

# Phase 5: Metacognition
print("--- PHASE 5: Metacognitive State ---")
meta_state = state['metacognition']['current_state']
print(f"  Confidence:  {meta_state['confidence']:.0%}")
print(f"  Coherence:   {meta_state['coherence']:.0%}")
print(f"  Surprise:    {meta_state['surprise_level']:.0%}")
print(f"  Emotions:    curiosity={meta_state['emotional_state']['curiosity']:.0%} "
      f"certainty={meta_state['emotional_state']['certainty']:.0%} "
      f"confusion={meta_state['emotional_state']['confusion']:.0%}")
print()

# Summary
print("--- SUMMARY ---")
print(f"  Cycles: {mind.cycle_count}")
print(f"  Total runtime: {state['uptime_seconds']:.0f}s")
print(f"  Concepts known: {state['perception']['total_concepts']}")
print(f"  World model: {len(wm['nodes'])} nodes, {len(wm['edges'])} causal edges")
print(f"  Memory: {state['memory']['stats']['episodic']['currently_active']} episodes, "
      f"{state['memory']['stats']['semantic']['total_rules']} semantic rules")
print(f"  Learning: {state['learning']['total_updates']} updates, "
      f"{state['learning']['unique_patterns_seen']} unique patterns")
print(f"  Goals: {len(state['goals']['active_goals'])} active")
print(f"  VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB / 4096MB")
print("=" * 65)
