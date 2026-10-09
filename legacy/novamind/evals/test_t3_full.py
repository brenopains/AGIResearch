"""Full T3 test with proper per-syllogism isolation."""
import sys, json, time; sys.path.insert(0, '.')
from collections import defaultdict
from core.bge_encoder import BGEEncoder
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.cognition.symbolic_engine import SymbolicEngine
from evals.symbolic_vs_bge import parse_syllogism, check_symbolic_conclusion

bge = BGEEncoder()
cge = CognitiveGraphEngine(bge_encoder=bge)
se = SymbolicEngine(cge, bge)
se.register_semantic_roles(cge.EDGE_PART_OF, cge.EDGE_CAUSES, None, None, None, None)
pool = json.load(open(r'evals/pool/syllogism_pool.json'))

t0 = time.time()
results = []
for idx, item in enumerate(pool[:200]):
    se.reset()  # Clear between syllogisms
    u, f, n, r = parse_syllogism(item['premises'])
    for subj, prop in u: se.register_universal(subj, prop)
    for e, p in f: se.assert_fact(e, p)
    for e, prop in n: se.assert_negation(e, prop)
    for e1, rel, e2 in r:
        se.relational_rules.setdefault(rel, []).append((e1, e2, 0.9))
        v1 = se._bge_vec(e1); v2 = se._bge_vec(e2)
        for v, name in [(v1, e1), (v2, e2)]:
            if v is not None and name not in se.cge.nodes: se.cge.create_node(name, v)
        se.cge.add_edge(e1, e2, se.cge.EDGE_CAUSES)
    best, is_correct, _ = check_symbolic_conclusion(se, item['correct_conclusion'], item['distractors'])
    results.append({'id': item['id'], 'category': item['category'], 'is_correct': is_correct})

elapsed = time.time() - t0
cat_stats = defaultdict(list)
for r in results: cat_stats[r['category']].append(r['is_correct'])
print(f'SYMBOLIC SyntLog (n={len(results)}) in {elapsed:.1f}s')
sc = sum(1 for r in results if r['is_correct'])
print(f'Overall: {sc}/{len(results)} = {sc/len(results)*100:.1f}%')
print(f'BGE-only T0: 35.5% (71/200)')
for cat in ['modus_ponens', 'modus_tollens', 'barbara', 'relational_chain']:
    outcomes = cat_stats.get(cat, [])
    if outcomes:
        n, acc = len(outcomes), sum(outcomes)/len(outcomes)*100
        chance = {'modus_ponens': 50, 'modus_tollens': 50, 'barbara': 12.5, 'relational_chain': 50}
        print(f'  {cat:20s}: {acc:.1f}% ({sum(outcomes)}/{n}) [chance={chance.get(cat,50)}%]')
