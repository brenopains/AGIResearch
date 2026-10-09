"""Debug MP after MT fix"""
import sys, json
sys.path.insert(0, '.')
from evals.symbolic_vs_bge import parse_syllogism, check_symbolic_conclusion
from core.cognition.symbolic_engine import SymbolicEngine
from core.bge_encoder import BGEEncoder
from core.cognitive_graph_engine import CognitiveGraphEngine

bge = BGEEncoder()
cge = CognitiveGraphEngine(bge_encoder=bge)
se = SymbolicEngine(cge, bge)
se.register_semantic_roles(cge.EDGE_PART_OF, cge.EDGE_CAUSES, None, None, None, None)

pool = json.load(open(r'evals/pool/syllogism_pool.json'))

mp_items = [p for p in pool if p['category'] == 'modus_ponens']
for i in range(3):
    item = mp_items[i]
    print(f"=== MP {i} ===")
    print(f"Premises: {item['premises']}")
    print(f"Correct: {item['correct_conclusion']}")
    u, f, n, r = parse_syllogism(item['premises'])
    print(f"Parsed: u={u}, f={f}, n={n}, r={r}")
    for s, p in u: se.register_universal(s, p)
    for e, p in f: se.assert_fact(e, p)
    for entity, neg_prop in n:
        if entity not in se.cge.nodes: se.cge.create_node(entity, se._bge_vec(entity))
        neg_vec = se._bge_vec(neg_prop)
        if neg_vec is not None:
            nid = 'not_' + neg_prop.replace(' ', '_')
            if nid not in se.cge.nodes: se.cge.create_node(nid, -neg_vec)
            se.cge.add_edge(entity, nid, se.cge.EDGE_CAUSED_BY)
    best, ok, scores = check_symbolic_conclusion(se, item['correct_conclusion'], item['distractors'])
    print(f"Best: {best}, OK: {ok}")
    print(f"Scores: {scores}")
    print()
