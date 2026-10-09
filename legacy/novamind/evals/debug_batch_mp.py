"""Debug: track a specific MP failure in batch context"""
import sys, json; sys.path.insert(0, '.')
from core.bge_encoder import BGEEncoder
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.cognition.symbolic_engine import SymbolicEngine
from evals.symbolic_vs_bge import parse_syllogism, check_symbolic_conclusion

bge = BGEEncoder()
cge = CognitiveGraphEngine(bge_encoder=bge)
se = SymbolicEngine(cge, bge)
se.register_semantic_roles(cge.EDGE_PART_OF, cge.EDGE_CAUSES, None, None, None, None)

pool = json.load(open(r'evals/pool/syllogism_pool.json'))
mp_items = [p for p in pool if p['category'] == 'modus_ponens']

# Process first 10 items to build up some CGE state, then test MP
for idx in range(20):
    item = pool[idx]
    se.reset()
    u, f, n, r = parse_syllogism(item['premises'])
    for s, p in u: se.register_universal(s, p)
    for e, p in f: se.assert_fact(e, p)
    for e, p in n: se.assert_negation(e, p)
    for e1, rel, e2 in r:
        se.relational_rules.setdefault(rel, []).append((e1, e2, 0.9))
        for v, name in [(se._bge_vec(e1), e1), (se._bge_vec(e2), e2)]:
            if v is not None and name not in se.cge.nodes: se.cge.create_node(name, v)
        se.cge.add_edge(e1, e2, se.cge.EDGE_CAUSES)
    if item['category'] == 'modus_ponens':
        best, ok, scores = check_symbolic_conclusion(se, item['correct_conclusion'], item['distractors'])
        print(f"[{idx}] {item['id']}: correct={item['correct_conclusion']}, ok={ok}, best={best}, scores={scores}")
        # Also check manually
        u2, _, _, _ = parse_syllogism(item['premises'])
        for s, p in u2:
            print(f"  universal_rules['{s}'] = {se.universal_rules.get(s, {})}")
        # Check is_a
        from evals.symbolic_vs_bge import re as re2
        m = re2.match(r'the\s+specific\s+entity\s+(.+)', item['correct_conclusion'].strip().rstrip('.'), re2.I)
        if m:
            prop = m.group(1).strip()
            result = se.infer('specific_entity', prop)
            print(f"  direct infer(specific_entity, '{prop}'): {result is not None}")
            if result:
                print(f"    rule={result.rule}, conf={result.confidence}")
            else:
                print(f"    _check_is_a('specific_entity', ...) would return:")
                for rs in se.universal_rules:
                    sim = se._check_is_a('specific_entity', rs)
                    print(f"      _check_is_a(specific_entity, {rs}) = {sim}")
