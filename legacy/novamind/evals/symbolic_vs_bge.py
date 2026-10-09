"""
Comparador: Symbolic Engine vs BGE-only para SyntLog
=====================================================
Teste limpo: para cada silogismo, registra as premissas no motor
simbólico e verifica se a conclusão correta é inferida.
"""
import sys, os, json, re, torch
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(sys.path[0])

from core.cognitive_graph_engine import CognitiveGraphEngine
from core.bge_encoder import BGEEncoder
from core.cognition.symbolic_engine import SymbolicEngine


def parse_syllogism(premises):
    """Parseia silogismo em regras universais e fatos."""
    universals = []
    facts = []
    negations = []
    relations = []

    for p in premises:
        p = p.strip().rstrip('.')
        
        # 0. "All entities that condition result" — must check BEFORE generic All-X
        text_after_that = re.match(r'All\s+entities\s+that\s+(.+)', p, re.I)
        if text_after_that:
            after = text_after_that.group(1).strip()
            words = after.split()
            if len(words) >= 4:
                cond = ' '.join(words[:2])
                prop = ' '.join(words[2:4])
                universals.append((cond, prop))
            elif len(words) >= 2:
                cond = words[0]
                prop = ' '.join(words[1:]) if len(words) > 1 else words[0]
                universals.append((cond, prop))
            continue
        
        # 1. "All X verb_phrase" — everything after subject is the verb phrase
        #    (preserves linking verbs: "All blix is flink" → ('blix', 'is flink'))
        m = re.match(r'All\s+(\S+)\s+(.+)', p, re.I)
        if m:
            subj = m.group(1).strip()
            prop = m.group(2).strip()
            # Skip if subject is "entities" (already handled by entities that rule)
            if subj.lower() != 'entities':
                universals.append((subj, prop))
            continue
        # "A specific entity is a X"
        m = re.match(r'A\s+specific\s+entity\s+is\s+a\s+(.+)', p, re.I)
        if m:
            facts.append(("specific_entity", m.group(1).strip()))
            continue
        # "A specific entity does NOT X"
        m = re.match(r'A\s+specific\s+entity\s+does\s+NOT\s+(.+)', p, re.I)
        if m:
            negations.append(("specific_entity", m.group(1).strip()))
            continue
        # "X is rel Y"
        m = re.match(r'(.+?)\s+is\s+(.+?)\s+(.+)$', p, re.I)
        if m:
            e1 = m.group(1).strip()
            rel = m.group(2).strip()
            e2 = m.group(3).strip()
            relations.append((e1, rel, e2))
            continue

    return universals, facts, negations, relations


def check_symbolic_conclusion(engine, correct, distractors):
    """Verifica se a conclusão correta é inferida vs distractors."""
    correct_clean = correct.strip().rstrip('.')

    # Tenta inferir "specific_entity" com várias propriedades
    entity = "specific_entity"

    # Para todo answer candidate (correct + distractors)
    candidates = [correct_clean] + [d.strip().rstrip('.') for d in distractors]

    scores = {}
    for cand in candidates:
        cand_lower = cand.lower()

        # 0. NEGATION FIRST: "The specific entity is NOT a X" -> modus_tollens
        m = re.match(r'the\s+specific\s+entity\s+is\s+not\s+a?\s*(.+)', cand_lower, re.I)
        if m:
            prop = m.group(1).strip()
            result = engine.modus_tollens(entity, prop)
            if result:
                scores[cand] = result.confidence
                continue

        # 1. "The specific entity X" -> try infer(entity, X)
        m = re.match(r'the\s+specific\s+entity\s+(.+)', cand_lower, re.I)
        if m:
            prop = m.group(1).strip()
            result = engine.infer(entity, prop)
            if result:
                scores[cand] = result.confidence
                continue

        # "All X are Y" or "All X verb Y"
        m = re.match(r'all\s+(.+?)\s+(?:is|are|has|have|can|will|does)\s+(.+)', cand_lower, re.I)
        if not m:
            m = re.match(r'all\s+(\S+)\s+(.+)', cand_lower, re.I)
        if m:
            subj = m.group(1).strip()
            prop = m.group(2).strip()
            result = engine.infer(subj, prop)
            if result:
                scores[cand] = result.confidence
                continue

        # "X is rel Y" -> use infer_relational
        m = re.match(r'(.+?)\s+is\s+(.+?)\s+(.+)$', cand_lower, re.I)
        if m:
            e1 = m.group(1).strip()
            rel = m.group(2).strip()
            e2 = m.group(3).strip()
            result = engine.infer_relational(e1, rel, e2)
            if result:
                scores[cand] = result.confidence

    # Best score
    if scores:
        best = max(scores, key=scores.get)
        return best, best == correct_clean, scores
    return correct_clean, False, scores


def run_full_comparison(n=200):
    """Roda n syllogisms com SymbolicEngine + BGE-only, compara."""
    from evals.t0_runner import run_t3_syntlog

    pool_path = ('C:\\Users\\usuario\\Documents\\AGI 2.0\\novamind\\evals\\pool\\syllogism_pool.json').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~'))
    pool = json.load(open(pool_path))

    # --- Symbolic Engine results ---
    print("=" * 70)
    print("SYMBOLIC ENGINE — SyntLog")
    print("=" * 70)

    bge = BGEEncoder()
    cge = CognitiveGraphEngine(bge_encoder=bge)
    se = SymbolicEngine(cge, bge)
    se.register_semantic_roles(cge.EDGE_PART_OF, cge.EDGE_CAUSES, None, None, None, None)

    symbolic_results = []
    parsed_ok = 0
    for item in pool[:n]:
        premises = item["premises"]
        correct = item["correct_conclusion"]
        distractors = item["distractors"]
        category = item["category"]

        universals, facts, negations, relations = parse_syllogism(premises)

        # Register
        for subj, prop in universals:
            se.register_universal(subj, prop)
        for entity, prop in facts:
            se.assert_fact(entity, prop)
        for entity, negated_prop in negations:
            if entity not in se.cge.nodes:
                se.cge.create_node(entity, se._bge_vec(entity))
            # Add a "NOT property" edge for MT detection
            neg_vec = se._bge_vec(negated_prop)
            if neg_vec is not None:
                se.cge.create_node('not_' + negated_prop, -neg_vec)
                # Mark entity as NOT having this property
                se.cge.add_edge(entity, 'not_' + negated_prop, se.cge.EDGE_CAUSED_BY)
        for e1, rel, e2 in relations:
            # Register relational rules
            if rel not in se.relational_rules:
                se.relational_rules[rel] = []
            se.relational_rules[rel].append((e1, e2, 0.9))
            # Also add edges in CGE for graph traversal
            v1 = se._bge_vec(e1)
            v2 = se._bge_vec(e2)
            if v1 is not None and v2 is not None:
                if e1 not in se.cge.nodes:
                    se.cge.create_node(e1, v1)
                if e2 not in se.cge.nodes:
                    se.cge.create_node(e2, v2)
                se.cge.add_edge(e1, e2, se.cge.EDGE_CAUSES)

        best_choice, is_correct, scores = check_symbolic_conclusion(se, correct, distractors)
        if universals or facts:
            parsed_ok += 1

        symbolic_results.append({
            "id": item["id"],
            "category": category,
            "correct": correct,
            "chosen": best_choice,
            "is_correct": is_correct,
            "n_universals": len(universals),
            "n_facts": len(facts),
        })

    # Aggregate
    cat_stats = defaultdict(list)
    for r in symbolic_results:
        cat_stats[r["category"]].append(r["is_correct"])

    sym_correct = sum(1 for r in symbolic_results if r["is_correct"])
    print(f"\nSymbolic Results (n={len(symbolic_results)}, parsed={parsed_ok}):")
    print(f"  Overall: {sym_correct}/{len(symbolic_results)} = {sym_correct/len(symbolic_results)*100:.1f}%")
    chance_map = {"modus_ponens": 50, "modus_tollens": 50, "barbara": 12.5, "relational_chain": 50}
    for cat, outcomes in sorted(cat_stats.items()):
        n_cat = len(outcomes)
        acc = sum(outcomes) / n_cat * 100
        ch = chance_map.get(cat, 50)
        print(f"  {cat:20s}: {acc:.1f}% ({sum(outcomes)}/{n_cat}) [chance={ch}%]")

    # --- Compare with BGE-only (from T0) ---
    t0_report_path = ('C:\\Users\\usuario\\Documents\\AGI 2.0\\novamind\\evals\\results\\t0_report.json').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~'))
    bge_data = None
    try:
        report = json.load(open(t0_report_path))
        bge_data = report.get("tests", {}).get("T3_syntlog", {}).get("details", [])
    except:
        pass

    if bge_data:
        bge_correct = sum(1 for r in bge_data[:n] if r.get("is_correct"))
        print(f"\n  BENGELINE (BGE-only): {bge_correct}/{min(n, len(bge_data))} = {bge_correct/min(n, len(bge_data))*100:.1f}%")
        print(f"  IMPROVEMENT: +{(sym_correct/len(symbolic_results)*100) - (bge_correct/min(n, len(bge_data))*100):.1f} pp")

    print(f"\nPróximo passo: implementar decoder generativo (F3) e re-testar T1.")


if __name__ == "__main__":
    import time
    t0 = time.time()
    run_full_comparison(50)
    print(f"\nElapsed: {time.time()-t0:.1f}s")
