"""Measure warm latency and resource cost of embedding recall."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, gc
from core.memory import MemorySystem
from core.semantic_encoder import attach_encoder
from tmp.bench_recall import build_memory

proc_rss = lambda: __import__('psutil').Process().memory_info().rss/1e6 if _has_psutil() else -1
def _has_psutil():
    try:
        import psutil; return True
    except Exception:
        return False

def vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()/1e6
    return 0

print(f"RAM before load: {proc_rss():.0f} MB | VRAM: {vram_mb():.0f} MB")
mem = build_memory()
enc = attach_encoder(mem)

# cold: first query (loads model + encodes all episodes)
t0 = time.time(); mem.recall_episodic({'input': 'where do I live'}, top_k=1)
cold = (time.time()-t0)*1000
print(f"RAM after model load + cold query: {proc_rss():.0f} MB | VRAM: {vram_mb():.0f} MB")
print(f"Cold query (model load + encode 20 episodes): {cold:.0f} ms")

# warm: episode embeddings cached, only query is encoded
warm = []
for q in ['what is my job','which pet do I have','my favorite drink','what do I drive',
          'where do I sleep','what languages','my phone brand','weekend sport',
          'favorite color','birth year']:
    t0 = time.time(); mem.recall_episodic({'input': q}, top_k=1); warm.append((time.time()-t0)*1000)
print(f"Warm query (encode 1 new query, episodes cached): {sum(warm)/len(warm):.1f} ms avg, "
      f"min {min(warm):.1f} / max {max(warm):.1f}")
print(f"Device: {enc._device}")
