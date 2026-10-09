"""NovaMind - Full cognitive demo with proper fresh imports"""
import gc, sys, os

# Force clean reload
for mod in list(sys.modules.keys()):
    if any(k in mod for k in ['core', 'novacrush', 'word_ngram']):
        del sys.modules[mod]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, time, psutil
proc = psutil.Process()

gc.collect(); torch.cuda.empty_cache()
ram_start = proc.memory_info().rss / 1024**2
t0 = time.time()

from core.mind import NovaMind
mind = NovaMind()

ram_init = proc.memory_info().rss / 1024**2
print(f"INIT: {time.time()-t0:.1f}s | RAM: {ram_init:.0f}MB | VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

questions = [
    "What is machine learning",
    "How do neural networks work",
    "Explain deep learning",
    "What is artificial intelligence",
    "Define reinforcement learning",
]

print(f"\n{'#':>2} {'Time':>5} {'RAM':>6} {'Output':>60}")
print("=" * 78)
for i, q in enumerate(questions):
    gc.collect()
    t = time.time()
    r = mind.think(q)
    dt = (time.time() - t) * 1000
    ram = proc.memory_info().rss / 1024**2
    output = r['response']['text'][:55]
    conf = r['layers']['metacognition']['confidence']['overall']
    concepts = len(r['layers']['perception']['concepts'])
    print(f"{i+1:2d} {dt:5.0f}ms {ram:6.0f}MB {output}")
    print(f"    confidence={conf:.0%} concepts={concepts}")

ram_final = proc.memory_info().rss / 1024**2
vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
print("=" * 78)
print(f"RAM: {ram_final:.0f}MB | VRAM: {vram:.0f}MB | Cycles: {mind.cycle_count}")
print(f"Concepts: {len(mind.perception.concept_registry)} | Episodes: {len(mind.memory.episodic)}")
