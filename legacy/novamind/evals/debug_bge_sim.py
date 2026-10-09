"""Debug: check BGE similarity between specific_entity and invented words"""
import sys; sys.path.insert(0, '.')
from core.bge_encoder import BGEEncoder
bge = BGEEncoder()
pairs = [
    ("specific_entity", "tronk"),
    ("specific_entity", "quarg"),
    ("specific_entity", "grenk"),
    ("specific_entity", "grom"),
    ("tronk", "contains zilch"),
    ("specific_entity", "tronk"),
]
for a, b in pairs:
    sim = bge.similarity(a, b)
    print(f"  sim({a:20s}, {b:20s}) = {sim:.4f}")
