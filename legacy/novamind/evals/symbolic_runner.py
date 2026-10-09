"""
SymbolicSyllogismRunner — Executa SyntLog via SymbolicEngine
=============================================================
Ponte entre texto de premissas (All X are Y) e motor simbólico.
Compara resultados: Symbolic vs BGE-only.
"""
import sys, os, json, re, torch
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(sys.path[0])

from core.cognitive_graph_engine import CognitiveGraphEngine
from core.bge_encoder import BGEEncoder
from core.cognition.symbolic_engine import SymbolicEngine


class SyllogismParser:
    """Parseia premissas textuais e as registra no SymbolicEngine."""

    # Patterns: "All X are Y", "All X have Y", "A specific entity is a X",
    # "A specific entity does NOT Y", "X is rel Y", "All entities that Y have Z"
    PATTERN_UNIVERSAL = re.compile(r'All\s+(.+?)\s+(?:is|are|has|have)\s+(.+)', re.IGNORECASE)
    PATTERN_UNIVERSAL_COND = re.compile(r'All\s+entities\s+that\s+(.+?)\s+(?:is|are|have)\s+(.+)', re.IGNORECASE)
    PATTERN_ENTITY_IS = re.compile(r'A\s+specific\s+entity\s+is\s+a\s+(.+)', re.IGNORECASE)
    PATTERN_ENTITY_NOT = re.compile(r'A\s+specific\s+entity\s+does\s+NOT\s+(.+)', re.IGNORECASE)
    PATTERN_RELATION = re.compile(r'(.+?)\s+is\s+(.+?)\s+(.+?)(?:\.|$)', re.IGNORECASE)

    def parse_and_register(self, premises: list, symbolic_engine: SymbolicEngine) -> dict:
        """Parseia premissas e registra no SymbolicEngine. Retorna diagnóstico."""
        registered = {"universals": [], "facts": [], "relations": [], "entities": [], "errors": []}

        for premise in premises:
            premise = premise.strip()

            # Pattern: "All X are Y" or "All X have Y"
            m = self.PATTERN_UNIVERSAL.match(premise)
            if m:
                subj = m.group(1).strip()
                prop = m.group(2).strip().rstrip('.')
                # Handle "is a X" -> prop is "a X"
                if prop.startswith("a "):
                    prop = prop[2:]
                symbolic_engine.register_universal(subj, prop)
                registered["universals"].append((subj, prop))
                continue

            # Pattern: "All entities that Y have Z"
            m = self.PATTERN_UNIVERSAL_COND.match(premise)
            if m:
                cond = m.group(1).strip()
                prop = m.group(2).strip().rstrip('.')
                if prop.startswith("a "):
                    prop = prop[2:]
                symbolic_engine.register_universal(cond, prop)
                registered["universals"].append((cond, prop))
                continue

            # Pattern: "A specific entity is a X"
            m = self.PATTERN_ENTITY_IS.match(premise)
            if m:
                cat = m.group(1).strip().rstrip('.')
                entity = "specific_entity"
                symbolic_engine.assert_fact(entity, cat)
                registered["facts"].append((entity, cat))
                if entity not in registered["entities"]:
                    registered["entities"].append(entity)
                continue

            # Pattern: "A specific entity does NOT ..." (Modus Tollens)
            m = self.PATTERN_ENTITY_NOT.match(premise)
            if m:
                prop = m.group(1).strip().rstrip('.')
                registered["negations"].append(("specific_entity", prop))
                continue

            # Pattern: "X is rel Y" (relational chain)
            m = self.PATTERN_RELATION.match(premise)
            if m:
                e1 = m.group(1).strip()
                rel = m.group(2).strip()
                e2 = m.group(3).strip().rstrip('.')
                registered["relations"].append((e1, rel, e2))
                if e1 not in registered["entities"]:
                    registered["entities"].append(e1)
                if e2 not in registered["entities"]:
                    registered["entities"].append(e2)
                continue

            registered["errors"].append(f"Unparseable: {premise}")

        return registered

    def infer_conclusion(self, symbolic_engine, correct_conclusion: str) -> dict:
        """Tenta inferir a conclusão correta e alternativas."""
        results = {}

        # Extract entity-property from conclusion "The specific entity X"
        conclusion_clean = correct_conclusion.strip().rstrip('.')
        entity = "specific_entity"

        # Try entity inference
        # Parse: "The specific entity X" or "All X Y"
        m = re.match(r'The\s+specific\s+entity\s+(.*)', conclusion_clean, re.IGNORECASE)
        if m:
            prop_phrase = m.group(1).strip()
            result = symbolic_engine.infer(entity, prop_phrase)
            if result:
                results["entity_prop"] = result.to_dict()

        # Parse: "All X are Y" type conclusions
        m = re.match(r'All\s+(.+?)\s+(?:is|are|has|have)\s+(.+)', conclusion_clean, re.IGNORECASE)
        if m:
            subj = m.group(1).strip()
            prop = m.group(2).strip().rstrip('.')
            result = symbolic_engine.infer(subj, prop)
            if result:
                results["all_prop"] = result.to_dict()

        # Try Modus Tollens: if conclusion is negative "X is NOT a Y"
        m = re.match(r'The\s+specific\s+entity\s+is\s+(?:NOT|not)\s+a\s+(.+)', conclusion_clean, re.IGNORECASE)
        if m:
            cat = m.group(1).strip()
            result = symbolic_engine.modus_tollens(entity, cat)
            if result:
                results["modus_tollens"] = result.to_dict()

        return results


def run_symbolic_syllogism(pool, n=100):
    """Roda syllogisms via SymbolicEngine e retorna resultados comparativos."""
    print(f"[SYMBOLIC] Running {n}/{len(pool)} syllogisms...")

    bge = BGEEncoder()
    cge = CognitiveGraphEngine(bge_encoder=bge)
    se = SymbolicEngine(cge, bge)
    se.register_semantic_roles(cge.EDGE_PART_OF, cge.EDGE_CAUSES, None, None, None, None)
    parser = SyllogismParser()

    results = []
    for item in pool[:n]:
        premises = item["premises"]
        correct = item["correct_conclusion"]
        category = item["category"]
        item_id = item["id"]

        # Register in symbolic engine
        parse_info = parser.parse_and_register(premises, se)

        # Check if parsing succeeded
        if parse_info["errors"]:
            continue

        # Infer conclusion
        inferences = parser.infer_conclusion(se, correct)

        # Determine if the symbolic inference matched the correct conclusion
        symbolic_correct = False
        inferred_conf = 0.0
        for inf_type, inf_data in inferences.items():
            concl = inf_data.get("conclusion", "")
            if "not_" in concl:
                symbolic_correct = (concl.replace("not_", "") == 
                    correct.replace("The specific entity is NOT a ", "").replace("The specific entity is not a ", "").strip().rstrip('.').lower())
            elif "specific_entity_" in inf_data.get("conclusion", ""):
                inferred_prop = inf_data["conclusion"].replace("specific_entity_", "")
                expected_prop = re.sub(r'.*entity\s+(.*?)$', r'\1', correct, flags=re.IGNORECASE).strip()
                symbolic_correct = (inferred_prop.lower() == expected_prop.lower())
            inferred_conf = max(inferred_conf, inf_data.get("confidence", 0))

        results.append({
            "id": item_id,
            "category": category,
            "correct": correct,
            "symbolic_correct": symbolic_correct,
            "symbolic_confidence": inferred_conf,
            "parsed_universals": len(parse_info.get("universals", [])),
            "parsed_facts": len(parse_info.get("facts", [])),
        })

    return results


def compare_with_bge(results, bge_only_t3_results):
    """Compara resultados simbólicos com BGE-only."""
    sym_correct = sum(1 for r in results if r["symbolic_correct"])
    print(f"\n[COMPARISON]")
    print(f"  Symbolic:   {sym_correct}/{len(results)} = {sym_correct/len(results)*100:.1f}%")
    if bge_only_t3_results:
        bge_correct = sum(1 for r in bge_only_t3_results[:len(results)] if r["is_correct"])
        print(f"  BGE-only:   {bge_correct}/{len(results)} = {bge_correct/len(results)*100:.1f}%")

    # Per category
    from collections import defaultdict
    sym_cats = defaultdict(list)
    for r in results:
        sym_cats[r["category"]].append(r["symbolic_correct"])
    for cat, outcomes in sorted(sym_cats.items()):
        n = len(outcomes)
        acc = sum(outcomes) / n * 100
        print(f"  {cat:20s}: {acc:.1f}% ({sum(outcomes)}/{n})")


if __name__ == "__main__":
    pool_path = ('C:\\Users\\usuario\\Documents\\AGI 2.0\\novamind\\evals\\pool\\syllogism_pool.json').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~'))
    pool = json.load(open(pool_path))
    n = min(100, len(pool))
    results = run_symbolic_syllogism(pool, n)
    compare_with_bge(results, None)
