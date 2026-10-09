import torch, time, gc, sys, os
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
import psutil
proc = psutil.Process()
gc.collect(); torch.cuda.empty_cache()

from core.mind import NovaMind
mind = NovaMind()

questions = [
    "What is machine learning",
    "How do neural networks work",
    "Explain deep learning",
    "What is artificial intelligence",
    "What is a transformer model",
]

print(f"\n{'#':>2} {'Time':>5} {'RAM':>6} {'Output':>65}")
print("=" * 85)
for i, q in enumerate(questions):
    ram_b4 = proc.memory_info().rss / 1024**2
    t = time.time()
    r = mind.think(q)
    dt = (time.time() - t) * 1000
    ram_aft = proc.memory_info().rss / 1024**2
    output = r['response']['text'][:60]
    conf = r['layers']['metacognition']['confidence']['overall']
    concepts = r['layers']['perception']['concepts']
    concept_labels = [c['label'] for c in concepts[:5]]
    mem = r['layers']['memory']['stats']
    print(f"{i+1:2d} {dt:5.0f}ms {ram_aft:6.0f}MB {output}")
    print(f"    concepts: {concept_labels}")
    print(f"    confidence: {conf:.0%} | episodes: {mem['episodic']['currently_active']}")

ram_final = proc.memory_info().rss / 1024**2
vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
print("=" * 85)
print(f"RAM: {ram_final:.0f}MB (+{ram_final-1125:.0f}) | VRAM: {vram:.0f}MB | Cycles: {mind.cycle_count}")
