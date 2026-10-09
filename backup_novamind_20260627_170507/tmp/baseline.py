import torch, time, gc, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil
proc = psutil.Process()

gc.collect()
torch.cuda.empty_cache()
ram_before = proc.memory_info().rss / 1024**2
vram_before = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

from core.mind import NovaMind
mind = NovaMind()

ram_after_init = proc.memory_info().rss / 1024**2
vram_after_init = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

t0 = time.time()
result = mind.think("What is artificial intelligence and how does machine learning work?")
t1 = time.time()
cycle_time = (t1 - t0) * 1000

ram_after_cycle = proc.memory_info().rss / 1024**2
vram_after_cycle = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

print("=== BASELINE ===")
print(f"RAM  - before: {ram_before:.0f}MB | after init: {ram_after_init:.0f}MB | after cycle: {ram_after_cycle:.0f}MB")
print(f"VRAM - before: {vram_before:.0f}MB | after init: {vram_after_init:.0f}MB | after cycle: {vram_after_cycle:.0f}MB")
print(f"Cycle time: {cycle_time:.0f}ms")
print(f"Concepts: {result['layers']['perception']['total_concepts_known']}")
print(f"Confidence: {result['layers']['metacognition']['confidence']['overall']:.2f}")
print(f"Output: {result['response']['text'][:300]}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("Device: CPU")
