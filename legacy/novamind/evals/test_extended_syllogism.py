"""Test Symbolic Engine on extended pool WITH invalid syllogisms"""
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

pool = json.load(open(r'evals/pool/syllogism_extended_pool.json'))

results = []
for idx, item in enumerate(pool[:200]):
    se.reset()
    u, f, n, r = parse_syllogism(item['premises'])
    for subj, prop in u: se.register_universal(subj, prop)
    for e, p in f: se.assert_fact(e, p)
    for e, prop in n: se.assert_negation(e, prop)
    for e1, rel, e2 in r:
        se.relational_rules.setdefault(rel, []).append((e1, e2, 0.9))
        for v, name in [(se._bge_vec(e1), e1), (se._bge_vec(e2), e2)]:
            if v is not None and name not in se.cge.nodes: se.cge.create_node(name, v)
        se.cge.add_edge(e1, e2, se.cge.EDGE_CAUSES)
    # Check if engine accepts the CORRECT conclusion
    best, accepted, _ = check_symbolic_conclusion(se, item['correct'], item['distractors'])
    # Note: 'accepted' means engine found a proof for the "correct" conclusion
    # If the syllogism is INVALID, a good engine should NOT accept it
    results.append({
        'id': item['id'],
        'category': item['category'],
        'valid': item['valid'],
        'correct_conclusion': item['correct'],
        'engine_accepted': accepted,
        'verdict': 'PASS' if (item['valid'] == accepted) else 'FAIL'
    })

# Compute metrics
valid_items = [r for r in results if r['valid']]
invalid_items = [r for r in results if not r['valid']]

tp = sum(1 for r in valid_items if r['engine_accepted'])
fn = sum(1 for r in valid_items if not r['engine_accepted'])
tn = sum(1 for r in invalid_items if not r['engine_accepted'])
fp = sum(1 for r in invalid_items if r['engine_accepted'])

print(f"=== FAKE NEWS TEST: Symbolic Engine on Extended SyntLog (n={len(results)}) ===\n")
print(f"Valid syllogisms: {len(valid_items)}")
print(f"Invalid syllogisms: {len(invalid_items)}")
print(f"\nValid syllogisms accepted (TP):  {tp}/{len(valid_items)} = {tp/len(valid_items)*100:.1f}%")
print(f"Valid syllogisms rejected (FN):  {fn}/{len(valid_items)} = {fn/len(valid_items)*100:.1f}%")
print(f"Invalid syllogisms rejected (TN): {tn}/{len(invalid_items)} = {tn/len(invalid_items)*100:.1f}%")
print(f"Invalid syllogisms accepted (FP): {fp}/{len(invalid_items)} = {fp/len(invalid_items)*100:.1f}%")

if tp + fp > 0:
    precision = tp / (tp + fp)
    print(f"\nPrecision (TP/(TP+FP)):     {precision:.3f}")
if tp + fn > 0:
    recall = tp / (tp + fn)
    print(f"Recall (TP/(TP+FN)):        {recall:.3f}")
if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)
    print(f"F1 Score:                    {f1:.3f}")
if tn + fp > 0:
    specificity = tn / (tn + fp)
    print(f"Specificity (TN/(TN+FP)):   {specificity:.3f}")
if tp + fn > 0 and tn + fp > 0:
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"Accuracy:                    {accuracy:.3f}")
    print(f"Chance baseline:             0.500")
    print(f"Improvement over chance:     {(accuracy - 0.5)*100:.1f}pp")

print(f"\n=== Detail of invalid syllogisms accepted (False Positives) ===")
for r in invalid_items:
    if r['engine_accepted']:
        print(f"  [{r['id']}] {r['category']}: conclusion='{r['correct_conclusion'][:60]}'")
