"""
NovaMind Cognitive Demo
Demonstrates: perception, memory, reasoning, metacognition, learning
Resource limits: VRAM < 50MB, RAM < 2.5GB, cycles < 500ms
"""
import torch, time, gc, sys, os
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
import psutil
proc = psutil.Process()

gc.collect(); torch.cuda.empty_cache()
ram_start = proc.memory_info().rss / 1024**2
t_start = time.time()

from core.mind import NovaMind
mind = NovaMind()

ram_init = proc.memory_info().rss / 1024**2
print(f"INIT: {time.time()-t_start:.1f}s | RAM: {ram_init:.0f}MB | VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB")
print()

conversation = [
    "What is machine learning",
    "How do neural networks work",
    "Explain deep learning",
    "Remember that my name is Alex",
    "What did I tell you my name is",
    "Define reinforcement learning",
    "How is machine learning different from deep learning",
    "What was the first thing I asked you",
    "Tell me about supervised learning",
    "What is a neural network",
]

print(f"{'#':>2} {'ms':>5} {'RAM':>5} {'Output':>65}")
print("-" * 82)
for i, q in enumerate(conversation):
    gc.collect()
    t = time.time()
    r = mind.think(q)
    dt = (time.time() - t) * 1000
    ram = proc.memory_info().rss / 1024**2
    conf = r['layers']['metacognition']['confidence']['overall']
    concepts = r['layers']['perception']['concepts']
    c_labels = [c['label'] for c in concepts[:4]]
    mem = r['layers']['memory']['stats']
    ep = mem['episodic']['currently_active']
    sem = mem['semantic']['total_rules']
    output = r['response']['text'][:60]

    print(f"{i+1:2d} {dt:5.0f} {ram:5.0f}MB {output}")
    print(f"    concepts: {c_labels} | confidence: {conf:.0%} | memory: {ep}ep/{sem}sem")

vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
ram_final = proc.memory_info().rss / 1024**2
print("-" * 82)
print(f"RAM: {ram_final:.0f}MB (+{ram_final-ram_start:.0f}) | VRAM: {vram:.0f}MB")
print(f"Cycles: {mind.cycle_count} | Total time: {time.time()-t_start:.1f}s")
print(f"Concepts known: {len(mind.perception.concept_registry)}")
print(f"Memory: {len(mind.memory.episodic)} episodes, {len(mind.memory.semantic)} semantic rules")
print(f"World model: {len(mind.world_model.nodes)} nodes, {len(mind.world_model.edges)} edges")
print(f"Reasoning history: {len(mind.reasoning.reasoning_history)} records")
print(f"Learning events: {mind.learning.total_updates} updates, {len(mind.learning.seen_patterns)} patterns")
