"""NovaMind Interactive REPL
Converse with the cognitive architecture in real time.
Shows full cognitive trace for each cycle."""
import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
gc.collect(); torch.cuda.empty_cache()
from core.mind import NovaMind

os.system('cls' if os.name == 'nt' else 'clear')
print("=" * 65)
print("  NovaMind Interactive Session")
print("  Type 'quit' to exit, 'state' for cognitive state, 'wm' for world model")
print("=" * 65)

mind = NovaMind()
print(f"  Init: ~1.7GB RAM, 17MB VRAM | Ready")
print()

while True:
    try:
        q = input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); break
    if not q:
        continue
    if q.lower() == 'quit':
        break
    if q.lower() == 'state':
        s = mind.get_full_state()
        meta = s['metacognition']
        mc = meta['current_state']
        print(f"  Cycles: {s['cycle_count']} | Confidence: {mc['confidence']:.0%} | "
              f"Coherence: {mc['coherence']:.0%} | "
              f"Curiosity: {mc['emotional_state']['curiosity']:.0%}")
        print(f"  Concepts: {len(mind.perception.concept_registry)} | "
              f"WM edges: {len(s['world_model']['edges'])} | "
              f"Episodes: {s['memory']['stats']['episodic']['currently_active']}")
        print(f"  VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB / 4096MB")
        continue
    if q.lower() == 'wm':
        s = mind.get_full_state()
        wm = s['world_model']
        chains = mind.get_causal_chains_with_labels(min_length=2)
        print(f"  World Model: {len(wm['nodes'])} nodes, {len(wm['edges'])} edges")
        if chains:
            for chain in chains[:10]:
                print(f"    {' -> '.join(chain[:5])}")
        continue

    # --- Process input ---
    r = mind.think(q)
    layers = r['layers']
    concepts = [c['label'] for c in layers['perception']['concepts']]
    meta = layers['metacognition']
    learning = layers['learning']
    recall = layers['memory'].get('recalled_episodes', [])
    recalled_texts = [ep.get('content', {}).get('input', '') for ep in recall if ep.get('content')]

    # Show cognitive trace
    trace = []
    trace.append(f"  concepts={concepts[:5]}")
    trace.append(f"confidence={meta['confidence']['overall']:.0%}")
    trace.append(f"novel={learning['novelty_analysis']['is_novel']}")
    if recalled_texts:
        trace.append(f"recall={len(recalled_texts)} ep(s)")
    print(f"  [{', '.join(trace)}]")
    print(f"  >>> {r['response']['text']}")
