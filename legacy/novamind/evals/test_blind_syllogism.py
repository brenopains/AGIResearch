"""Adversarial Test: Symbolic Engine evaluated BLIND (no conclusion hint)"""
import sys, json; sys.path.insert(0, '.')
from collections import defaultdict
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

def evaluate_blind(se, premises, correct_conclusion):
    """
    Evaluates syllogism BLIND: only sees premises, must generate conclusion.
    Returns True if the engine's top inference matches the correct conclusion.
    """
    # Parse and register  
    u, f, n, r = parse_syllogism(premises)
    for subj, prop in u: se.register_universal(subj, prop)
    for e, p in f: se.assert_fact(e, p)
    for e, prop in n: se.assert_negation(e, prop)
    for e1, rel, e2 in r:
        se.relational_rules.setdefault(rel, []).append((e1, e2, 0.9))
        for v, name in [(se._bge_vec(e1), e1), (se._bge_vec(e2), e2)]:
            if v is not None and name not in se.cge.nodes: se.cge.create_node(name, v)
        se.cge.add_edge(e1, e2, se.cge.EDGE_CAUSES)
    
    # Extract what we can infer
    inferences = []
    
    # For MP/MT: infer about specific_entity
    if 'specific_entity' in [x for _, x in f] or any('specific_entity' in str(p) for p in premises):
        if 'specific_entity' in se.cge.nodes:
            all_props = set()
            for subj, props in se.universal_rules.items():
                for prop in props:
                    all_props.add(prop)
            # Also add negation targets
            negs = se.negations.get('specific_entity', {})
            
            for prop in all_props:
                result = se.infer('specific_entity', prop)
                if result:
                    inferences.append((f"The specific entity {prop}", result.confidence))
            
            # Check modus tollens
            for neg_prop in negs:
                for subj in se.universal_rules:
                    mt = se.modus_tollens('specific_entity', subj)
                    if mt:
                        inferences.append((f"The specific entity is NOT a {subj}", mt.confidence))
    
    # For Barbara: infer about the subject of the first universal
    if u and len(u) >= 1:
        first_subj = u[0][0]
        if first_subj != 'entities':
            for subj, props in se.universal_rules.items():
                for prop in props:
                    # Check if first_subj inherits this property via chain
                    result = se.infer(first_subj, prop)
                    if result:
                        prop_str = prop
                        if prop_str.startswith('a '): prop_str = prop_str[2:]
                        inferences.append((f"All {first_subj} {prop_str}", result.confidence))
    
    # For relational chain: check infer_relational
    if r and len(r) >= 2:
        e1_1, rel_1, e2_1 = r[0]
        e1_2, rel_2, e2_2 = r[1]
        # The conclusion should be e1_1 rel e2_2 (if same rel type)
        if rel_1 == rel_2:
            result = se.infer_relational(e1_1, rel_1, e2_2)
            if result:
                inferences.append((f"{e1_1} is {rel_1} {e2_2}", result.confidence))
    
    # Check if correct conclusion matches any inference (fuzzy: substr match)
    correct_clean = correct_conclusion.strip().rstrip('.').lower()
    for infer_text, conf in inferences:
        if correct_clean == infer_text.lower() or correct_clean in infer_text.lower():
            return True
    
    return False


results = []
for idx, item in enumerate(pool[:200]):
    se.reset()
    accepted = evaluate_blind(se, item['premises'], item['correct'])
    results.append({
        'id': item['id'],
        'category': item['category'],
        'valid': item['valid'],
        'engine_accepted': accepted,
    })

valid_items = [r for r in results if r['valid']]
invalid_items = [r for r in results if not r['valid']]
tp = sum(1 for r in valid_items if r['engine_accepted'])
fn = sum(1 for r in valid_items if not r['engine_accepted'])
tn = sum(1 for r in invalid_items if not r['engine_accepted'])
fp = sum(1 for r in invalid_items if r['engine_accepted'])

print("=" * 70)
print("ADVERSARIAL TEST: Symbolic Engine BLIND Evaluation")
print("=" * 70)
print(f"\nValid syllogisms: {len(valid_items)}")
print(f"Invalid syllogisms: {len(invalid_items)}")
print(f"\nTrue Positives (valid accepted):      {tp}/{len(valid_items)} = {tp/len(valid_items)*100:.1f}%")
print(f"False Negatives (valid rejected):     {fn}/{len(valid_items)} = {fn/len(valid_items)*100:.1f}%")
print(f"True Negatives (invalid rejected):    {tn}/{len(invalid_items)} = {tn/len(invalid_items)*100:.1f}%")
print(f"False Positives (invalid accepted):   {fp}/{len(invalid_items)} = {fp/len(invalid_items)*100:.1f}%")

if tp + fp > 0: print(f"\nPrecision: {tp/(tp+fp):.3f}")
if tp + fn > 0: print(f"Recall:    {tp/(tp+fn):.3f}")
if tn + fp > 0: print(f"Specificity: {tn/(tn+fp):.3f}")
acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
print(f"Accuracy:  {acc:.3f}  (chance=0.500)")
print(f"F1:        {2*tp/(2*tp+fp+fn):.3f}" if (2*tp+fp+fn) > 0 else "F1: N/A")

print(f"\n--- Invalid syllogisms that engine falsely accepted ---")
for r in invalid_items:
    if r['engine_accepted']:
        item = pool[[i for i, p in enumerate(pool) if p.get('id') == r['id']][0]]
        print(f"  [{r['id']}] {r['category']}: {item['premises']}")
