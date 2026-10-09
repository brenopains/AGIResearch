"""Debug remaining Barbara failures"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(sys.path[0])

from core.bge_encoder import BGEEncoder
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.cognition.symbolic_engine import SymbolicEngine
from evals.symbolic_vs_bge import parse_syllogism, check_symbolic_conclusion

bge = BGEEncoder()
cge = CognitiveGraphEngine(bge_encoder=bge)
se = SymbolicEngine(cge, bge)
se.register_semantic_roles(cge.EDGE_PART_OF, cge.EDGE_CAUSES, None, None, None, None)

pool = json.load(open(r'evals/pool/syllogism_pool.json'))
barbara_count = 0
fail_count = 0
for item in pool[:50]:
    if item['category'] != 'barbara':
        continue
    barbara_count += 1
    u, f, n, r = parse_syllogism(item['premises'])
    if not u:
        print(f"[{item['id']}] UNPARSED: {item['premises']}")
        continue
    for subj, prop in u:
        se.register_universal(subj, prop)
    best, is_correct, scores = check_symbolic_conclusion(se, item['correct_conclusion'], item['distractors'])
    if not is_correct:
        fail_count += 1
        print(f"[{item['id']}] FAIL: {item['premises']}")
        print(f"  Parsed: u={u}")
        print(f"  Correct: {item['correct_conclusion']}")
        print(f"  Best: {best}")
        print(f"  Scores: {scores}")
        # Try direct inference
        cand = item['correct_conclusion'].strip().rstrip('.')
        m = re.match(r'all\s+(.+?)\s+(?:is|are|has|have|can|will|does)\s+(.+)', cand, re.I)
        if not m:
            m = re.match(r'all\s+(\S+)\s+(.+)', cand, re.I)
        if m:
            subj, prop = m.group(1).strip(), m.group(2).strip()
            result = se.infer(subj, prop)
            print(f"  Direct infer({subj}, {prop}): {result}")

print(f"\nBarbara total: {barbara_count}, fails: {fail_count}")
