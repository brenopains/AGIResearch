"""Debug false positive barbara: why does engine accept it?"""
import sys; sys.path.insert(0, '.')
from core.bge_encoder import BGEEncoder
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.cognition.symbolic_engine import SymbolicEngine
from evals.symbolic_vs_bge import parse_syllogism

bge = BGEEncoder()
cge = CognitiveGraphEngine(bge_encoder=bge)
se = SymbolicEngine(cge, bge)
se.register_semantic_roles(cge.EDGE_PART_OF, cge.EDGE_CAUSES, None, None, None, None)

# Item SYNTLOG-EXT-0073: premises with NO chain
premises = ["All gleep causes flarg.", "All zork prevents dorb."]
# Correct (invalid): "All gleep prevents dorb."

u, f, n, r = parse_syllogism(premises)
print(f"Universals: {u}")
print(f"Facts: {f}")
print(f"Negations: {n}")
print(f"Relations: {r}")

for subj, prop in u: se.register_universal(subj, prop)
print(f"\nUniversal rules: {se.universal_rules}")

# Try the invalid inference
result = se.infer('gleep', 'prevents dorb')
print(f"\ninfer('gleep', 'prevents dorb'): {result}")
if result:
    print(f"  Rule: {result.rule}, Conf: {result.confidence}")
    print(f"  Premises: {result.premises}")

# What about the valid one?
result2 = se.infer('gleep', 'causes flarg')
print(f"\ninfer('gleep', 'causes flarg'): {result2}")
if result2:
    print(f"  Rule: {result2.rule}, Conf: {result2.confidence}")

# Check all infer paths
print(f"\n_check_is_a('gleep', 'zork'): {se._check_is_a('gleep', 'zork')}")
print(f"_bge_sim('gleep', 'zork'): {se._bge_sim('gleep', 'zork')}")
