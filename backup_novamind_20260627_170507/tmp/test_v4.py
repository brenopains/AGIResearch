import torch, time, gc, sys, os
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
import psutil
proc = psutil.Process()
gc.collect(); torch.cuda.empty_cache()

ram_before = proc.memory_info().rss / 1024**2
from core.mind import NovaMind

t0 = time.time()
mind = NovaMind()
t_init = time.time() - t0
print(f"Init: {t_init*1000:.0f}ms | RAM: {proc.memory_info().rss/1024**2:.0f}MB")

questions = [
    "Explain what is machine learning",
    "How does artificial intelligence work",
    "Tell me about neural networks",
    "What is deep learning used for",
    "Define reinforcement learning",
    "Remember that I like cats",
    "What did I tell you I like",
]

print(f"\n{'#':>2} {'Time':>5} {'RAM':>6} {'Concepts':>10} {'Output':>60}")
print("=" * 88)

for i, q in enumerate(questions):
    t = time.time()
    r = mind.think(q)
    dt = (time.time() - t) * 1000
    ram = proc.memory_info().rss / 1024**2
    concepts = r['layers']['perception']['total_concepts_known']
    conf = r['layers']['metacognition']['confidence']['overall']
    output = r['response']['text'][:55]
    mem_stats = r['layers']['memory']['stats']
    ep = mem_stats['episodic']['currently_active']
    sem = mem_stats['semantic']['total_rules']
    print(f"{i+1:2d} {dt:5.0f}ms {ram:6.0f}MB {concepts:10d} {output}")
    print(f"    confidence={conf:.0%} episodes={ep} rules={sem}")

vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
ram_final = proc.memory_info().rss / 1024**2
print("=" * 88)
print(f"RAM: {ram_final:.0f}MB | VRAM: {vram:.0f}MB | Cycles: {mind.cycle_count}")
