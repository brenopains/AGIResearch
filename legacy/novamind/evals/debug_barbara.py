"""Debug: Why does Barbara fail in the runner?"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(sys.path[0])

from core.bge_encoder import BGEEncoder
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.cognition.symbolic_engine import SymbolicEngine

bge = BGEEncoder()
cge = CognitiveGraphEngine(bge_encoder=bge)
se = SymbolicEngine(cge, bge)
se.register_semantic_roles(cge.EDGE_PART_OF, cge.EDGE_CAUSES, None, None, None, None)

pool = json.load(open(r'evals/pool/syllogism_pool.json'))
barbara = [p for p in pool if p['category'] == 'barbara']

for item in barbara[:5]:
    print(f"\n--- {item['id']} ---")
    print(f"Premises: {item['premises']}")
    print(f"Correct: {item['correct_conclusion']}")

    from evals.symbolic_vs_bge import parse_syllogism, check_symbolic_conclusion
    u, f, n, r = parse_syllogism(item['premises'])
    print(f"Universals: {u}, Facts: {f}, Relations: {r}")

    for subj, prop in u:
        se.register_universal(subj, prop)

    best, is_correct, scores = check_symbolic_conclusion(se, item['correct_conclusion'], item['distractors'])
    print(f"Best: {repr(best)}")
    print(f"Is correct: {is_correct}")
    print(f"Scores: {scores}")

    if not is_correct:
        # Direct test
        m = re.match(r'all\s+(.+?)\s+(?:is|are|has|have)\s+(.+)', 
                     item['correct_conclusion'].strip().rstrip('.'), re.I)
        if m:
            subj = m.group(1).strip()
            prop = m.group(2).strip()
            print(f"  Direct infer({subj}, {prop}):")
            result = se.infer(subj, prop)
            print(f"    Result: {result}")
