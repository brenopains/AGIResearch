import torch, time, gc, sys, os
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
import psutil
proc = psutil.Process()
gc.collect(); torch.cuda.empty_cache()

from core.mind import NovaMind
mind = NovaMind()

ram_init = proc.memory_info().rss / 1024**2
print(f"RAM init: {ram_init:.0f}MB")

questions = [
    "What is machine learning",
    "How do neural networks work",
    "Explain deep learning",
    "What is artificial intelligence",
    "Remember the word quantum",
    "What did I just ask you to remember",
]

print(f"\n{'#':>2} {'Time':>6} {'RAM':>6} {'Output':>60}")
print("-" * 80)

for i, q in enumerate(questions):
    ram_b4 = proc.memory_info().rss / 1024**2
    t = time.time()
    r = mind.think(q)
    dt = (time.time() - t) * 1000
    ram_aft = proc.memory_info().rss / 1024**2
    output = r['response']['text'][:58]
    mem = r['layers']['memory']
    print(f"{i+1:2d} {dt:6.0f}ms {ram_aft:6.0f}MB {output}")
    print(f"    Episodes: {mem['stats']['episodic']['currently_active']} | Rules: {mem['stats']['semantic']['total_rules']}")

ram_final = proc.memory_info().rss / 1024**2
vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
print("-" * 80)
print(f"Final RAM: {ram_final:.0f}MB (+{ram_final-ram_init:.0f}) | VRAM: {vram:.0f}MB | Cycles: {mind.cycle_count}")
