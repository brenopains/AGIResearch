"""Debug METACOG: check why queries return 0 results."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = r"D:\novamind_data\cache\huggingface"
import torch
from core.hdc_memory import HDCMemory
from core.hdc_utils import unpack_bipolar
from core.coglang import Manifold

memory = HDCMemory(device="cpu")

# 1. Check what's in the first item
for key, index in memory._indices.items():
    if len(index) > 0:
        first = index[0]
        print(f"First item from {key}:")
        for k, v in list(first.items())[:8]:
            print(f"  {k}: {v}")
        break

# 2. Query with a real packed vector from memory
for key, packed in memory._packed.items():
    if packed.shape[0] > 0:
        vec = unpack_bipolar(packed[0])  # returns float tensor
        results = memory.query_vec(vec, top_k=5, min_sim=-1.0)
        print(f"\nQuery with real vec (from {key}): {len(results)} results")
        for score, meta, src in results[:3]:
            print(f"  score={score:.4f} meta={dict(list(meta.items())[:3])}")
        break

# 3. Query with random vector
random_vec = (torch.rand(10000) * 2 - 1).sign()
rand_results = memory.query_vec(random_vec, top_k=5, min_sim=-1.0)
print(f"\nQuery with random vec: {len(rand_results)} results")
if rand_results:
    print(f"  max score: {max(r[0] for r in rand_results):.4f}")
else:
    print("  no results (min_sim filter too high)")

# 4. Try with no min_sim filter
rand_results2 = memory.query_vec(random_vec, top_k=5, min_sim=-1.0)
print(f"\nQuery no filter: {len(rand_results2)} results")
for score, meta, src in rand_results2[:3]:
    print(f"  score={score:.4f}")
