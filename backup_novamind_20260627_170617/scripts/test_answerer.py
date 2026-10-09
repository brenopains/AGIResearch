"""Test CognitiveAnswerer with existing HDV files."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.grounding_engine import GroundingEngine
from core.hdc_memory import HDCMemory
from core.cognitive_answerer import CognitiveAnswerer
from core.positional_binder import PositionalBinder
import torch

device = torch.device("cuda")
engine = GroundingEngine(device=device)
memory = HDCMemory(device="cuda")
s = memory.get_stats()
print(f"Total items: {s.get('total_items', 0)}")

binder = PositionalBinder(max_length=30, device=device)
answerer = CognitiveAnswerer(engine, binder, binder, memory, device=device)

for q in ["gravity and time", "sherlock holmes", "the universe"]:
    r = answerer.answer(q, top_k=3)
    print(f"Q: {q}")
    print(f"A: {r}")
    print()
