"""NovaMind - Cognitive demo with memory, metacognition, and causal learning"""
import gc, sys, os
for mod in list(sys.modules.keys()):
    if any(k in mod for k in ['core', 'novacrush', 'word_ngram']):
        del sys.modules[mod]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, time
gc.collect(); torch.cuda.empty_cache()
from core.mind import NovaMind

mind = NovaMind()
t_start = time.time()

def show_cycle(q):
    r = mind.think(q)
    dt = (time.time() - t_start) * 1000
    meta = r['layers']['metacognition']
    mem = r['layers']['memory']
    wm = r['layers']['world_model']
    learn = r['layers']['learning']
    concepts = [c['label'] for c in r['layers']['perception']['concepts']]
    conf = meta['confidence']['overall']
    recall = [ep['content'].get('input','') for ep in mem.get('recalled_episodes', [])
              if ep.get('content')]
    is_novel = learn['novelty_analysis'].get('is_novel', False)

    print(f"[{mind.cycle_count:3d}] {q}")
    print(f"     concepts: {concepts[:5]}")
    print(f"     confidence: {conf:.0%} | novel: {is_novel} | episodes: {mem['stats']['episodic']['currently_active']}")
    print(f"     world model: {len(wm['nodes'])} nodes, {len(wm['edges'])} edges")
    if recall:
        print(f"     recall: {recall[-1][:50]}")
    print(f"     output: {r['response']['text'][:100]}")
    print()
    return r

print("=== NovaMind Cognitive Demo ===\n")

# Phase 1: Learn facts
print("-- PHASE 1: Learning Facts --")
show_cycle("My dog is named Rex")
show_cycle("I live in Brazil")
show_cycle("I like pizza")

# Phase 2: Query memory
print("-- PHASE 2: Memory Recall --")
show_cycle("What is my dog name")
show_cycle("Where do I live")
show_cycle("What do I like to eat")

# Phase 3: Novel information
print("-- PHASE 3: Novel Concepts --")
show_cycle("Explain quantum entanglement")
show_cycle("What is a transformer model")

# Phase 4: Knowledge about system
print("-- PHASE 4: Self-Awareness --")
show_cycle("What do you know")
state = mind.get_full_state()
print(f"\n=== FINAL STATE ===")
print(f"Runtime: {state['uptime_seconds']:.0f}s")
print(f"Cycles: {state['cycle_count']}")
print(f"Concepts: {state['perception']['total_concepts']}")
print(f"World model: {len(state['world_model']['nodes'])} nodes, {len(state['world_model']['edges'])} edges")
print(f"Memory: {state['memory']['stats']['episodic']['currently_active']} active episodes, {state['memory']['stats']['semantic']['total_rules']} rules")
print(f"Paradigm weights: {state['reasoning']['paradigm_weights']}")
print(f"Learning updates: {state['learning']['total_updates']}")
print(f"Unique patterns: {state['learning']['unique_patterns_seen']}")
goals_data = state.get('goals', {}).get('active_goals', [])
print(f"Active goals: {len(goals_data)}")
