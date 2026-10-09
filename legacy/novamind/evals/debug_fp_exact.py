"""Debug: print exact inference list for a false positive barbara"""
import sys, json; sys.path.insert(0, '.')
from core.bge_encoder import BGEEncoder
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.cognition.symbolic_engine import SymbolicEngine
from evals.symbolic_vs_bge import parse_syllogism
import re

bge = BGEEncoder()
cge = CognitiveGraphEngine(bge_encoder=bge)
se = SymbolicEngine(cge, bge)
se.register_semantic_roles(cge.EDGE_PART_OF, cge.EDGE_CAUSES, None, None, None, None)

pool = json.load(open(r'evals/pool/syllogism_extended_pool.json'))
fp_items = [p for p in pool if not p['valid'] and p['category'] == 'barbara']

for item in fp_items[:3]:
    premises = item['premises']
    correct = item['correct']
    print(f"\n=== {item['id']} ===")
    print(f"Premises: {premises}")
    print(f"'Correct' (invalid): {correct}")
    
    se.reset()
    u, f, n, r = parse_syllogism(premises)
    print(f"Parsed: u={u}")
    for subj, prop in u: se.register_universal(subj, prop)
    for e, p in f: se.assert_fact(e, p)
    for e, prop in n: se.assert_negation(e, prop)
    for e1, rel, e2 in r:
        se.relational_rules.setdefault(rel, []).append((e1, e2, 0.9))
        for v, name in [(se._bge_vec(e1), e1), (se._bge_vec(e2), e2)]:
            if v is not None and name not in se.cge.nodes: se.cge.create_node(name, v)
        se.cge.add_edge(e1, e2, se.cge.EDGE_CAUSES)
    
    # Generate ALL inferences
    inferences = []
    first_subj = u[0][0] if u else None
    if first_subj and first_subj != 'entities':
        for subj, props in se.universal_rules.items():
            for prop in props:
                result = se.infer(first_subj, prop)
                if result:
                    prop_str = prop
                    if prop_str.startswith('a '): prop_str = prop_str[2:]
                    inferences.append((f"All {first_subj} {prop_str}", result.confidence))
    
    print(f"Inferences: {inferences}")
    
    correct_clean = correct.strip().rstrip('.').lower()
    print(f"correct_clean: '{correct_clean}'")
    for inf_text, conf in inferences:
        print(f"  Check '{correct_clean}' in '{inf_text.lower()}': {correct_clean in inf_text.lower()}")
        print(f"  Check ==: {correct_clean == inf_text.lower()}")
