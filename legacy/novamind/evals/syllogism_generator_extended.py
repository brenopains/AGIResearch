"""
Extended Syllogism Generator — INCLUDING INVALID syllogisms.
Tests whether the Symbolic Engine can REJECT invalid conclusions,
not just confirm valid ones.

Each category has both VALID and INVALID entries.
An engine that says "yes" to everything will get 50% at best.
"""
import random, json, hashlib
from typing import List, Dict

random.seed(20260712)

SUBJECTS = ["grom", "fex", "blarg", "snorp", "quib", "zork", "mlep", "tronk",
    "flarg", "snerp", "gleep", "bork", "zilp", "quarg", "freb", "slin",
    "dronk", "plerg", "snirk", "torb", "glurp", "frend", "blix", "treek",
    "sprog", "murk", "grenk", "florp", "snert", "trill", "gorb", "pleen",
    "zang", "fring", "blorp", "trog", "snerk", "grink", "flarn"]

PREDICATES = ["has zorb", "is flink", "can greb", "loves snorg", "hates blig",
    "sees kronk", "hears plig", "eats glarn", "drinks frob", "fears dring",
    "contains zilch", "produces sprin", "absorbs trum", "reflects gorp",
    "amplifies flerb", "reduces zert", "follows blurn", "precedes mlep",
    "causes flarg", "prevents dorb", "enhances plick", "inhibits trerb",
    "transforms glurn", "creates zerb", "destroys flint"]

RELATIONS = [("greater than", "less than"), ("above", "below"),
    ("faster than", "slower than"), ("larger than", "smaller than"),
    ("heavier than", "lighter than"), ("hotter than", "colder than"),
    ("louder than", "quieter than"), ("brighter than", "dimmer than")]


# ============= VALID SYLLOGISMS =============
# (same as before, organized by type)

def _valid_mp():
    s = random.choice(SUBJECTS); p = random.choice(PREDICATES)
    return {"premises": [f"All {s} {p}.", f"A specific entity is a {s}."],
        "correct": f"The specific entity {p}.",
        "distractors": [f"The specific entity does not {p}.", f"All entities {p}.", f"The specific entity is not a {s}."],
        "valid": True, "category": "modus_ponens"}

def _valid_mt():
    s = random.choice(SUBJECTS); p = random.choice(PREDICATES)
    return {"premises": [f"All {s} {p}.", f"A specific entity does NOT {p}."],
        "correct": f"The specific entity is NOT a {s}.",
        "distractors": [f"The specific entity {p}.", f"All {s} do not {p}.", f"The specific entity is a {s}."],
        "valid": True, "category": "modus_tollens"}

def _valid_barbara():
    s = random.choice(SUBJECTS); p = random.choice(PREDICATES)
    q = random.choice([pr for pr in PREDICATES if pr != p])
    return {"premises": [f"All {s} {p}.", f"All entities that {p} {q}."],
        "correct": f"All {s} {q}.",
        "distractors": [f"All {s} do not {q}.", f"Some {s} {q}.", f"No {s} {q}."],
        "valid": True, "category": "barbara"}

def _valid_rc():
    a = random.choice(SUBJECTS)
    b = random.choice([s for s in SUBJECTS if s != a])
    c = random.choice([s for s in SUBJECTS if s not in (a, b)])
    rel = random.choice(RELATIONS)[0]
    return {"premises": [f"{a} is {rel} {b}.", f"{b} is {rel} {c}."],
        "correct": f"{a} is {rel} {c}.",
        "distractors": [f"{c} is {rel} {a}.", f"{a} is not {rel} {c}.", f"{b} is {rel} {a}."],
        "valid": True, "category": "relational_chain"}


# ============= INVALID SYLLOGISMS =============
# Premises that look well-formed but the conclusion does NOT follow.

def _invalid_mp_wrong_conclusion():
    """MP-like structure but conclusion doesn't match the premise."""
    s = random.choice(SUBJECTS); p1 = random.choice(PREDICATES)
    p2 = random.choice([pr for pr in PREDICATES if pr != p1 and not pr.startswith(p1[:3])])
    return {"premises": [f"All {s} {p1}.", f"A specific entity is a {s}."],
        "correct": f"The specific entity {p2}.",  # WRONG: should be p1
        "distractors": [f"The specific entity {p1}.", f"The specific entity does not {p2}."],
        "valid": False, "category": "modus_ponens"}

def _invalid_mt_wrong_direction():
    """MT-like: All X are Y, entity is NOT X -> NOT Y (INVALID: denial of antecedent)."""
    s = random.choice(SUBJECTS); p = random.choice(PREDICATES)
    return {"premises": [f"All {s} {p}.", f"A specific entity is NOT a {s}."],
        "correct": f"The specific entity does not {p}.",  # INVALID: denying antecedent
        "distractors": [f"The specific entity {p}.", f"The specific entity is a {s}."],
        "valid": False, "category": "modus_tollens"}

def _invalid_barbara_broken_chain():
    """Barbara-like: All X are Y. All Z are W. -> All X are W. (INVALID: no chain)"""
    s1 = random.choice(SUBJECTS); s2 = random.choice([s for s in SUBJECTS if s != s1])
    p1 = random.choice(PREDICATES); p2 = random.choice([pr for pr in PREDICATES if pr != p1])
    return {"premises": [f"All {s1} {p1}.", f"All {s2} {p2}."],
        "correct": f"All {s1} {p2}.",  # INVALID: s1 and s2 are different, p1 != p2
        "distractors": [f"No {s1} {p2}.", f"Some {s2} {p1}."],
        "valid": False, "category": "barbara"}

def _invalid_rc_affirming_consequent():
    """RC-like: A rel B, C rel A -> B rel C (INVALID: no chain)."""
    a = random.choice(SUBJECTS)
    b = random.choice([s for s in SUBJECTS if s != a])
    c = random.choice([s for s in SUBJECTS if s not in (a, b)])
    rel = random.choice(RELATIONS)[0]
    return {"premises": [f"{a} is {rel} {b}.", f"{c} is {rel} {a}."],
        "correct": f"{b} is {rel} {c}.",  # INVALID: no chain, c->a->b doesn't give b->c
        "distractors": [f"{a} is not {rel} {c}.", f"{c} is {rel} {b}."],
        "valid": False, "category": "relational_chain"}

def _invalid_mp_affirming_consequent():
    """All X are Y. Entity is Y. -> Entity is X (INVALID: affirming consequent)."""
    s = random.choice(SUBJECTS); p = random.choice(PREDICATES)
    return {"premises": [f"All {s} {p}.", f"A specific entity {p}."],
        "correct": f"The specific entity is a {s}.",  # INVALID: affirming consequent
        "distractors": [f"The specific entity is not a {s}.", f"The specific entity does not {p}."],
        "valid": False, "category": "modus_ponens"}


VALID_GENERATORS = {
    "modus_ponens": _valid_mp,
    "modus_tollens": _valid_mt,
    "barbara": _valid_barbara,
    "relational_chain": _valid_rc,
}

INVALID_GENERATORS = [
    _invalid_mp_wrong_conclusion,
    _invalid_mt_wrong_direction,
    _invalid_barbara_broken_chain,
    _invalid_rc_affirming_consequent,
    _invalid_mp_affirming_consequent,
]


def generate_extended_pool(n_valid_per_category: int = 100,
                           n_invalid_total: int = 100) -> List[Dict]:
    pool = []
    # Valid
    for cat, gen in VALID_GENERATORS.items():
        for _ in range(n_valid_per_category):
            pool.append(gen())
    # Invalid
    for _ in range(n_invalid_total):
        pool.append(random.choice(INVALID_GENERATORS)())

    random.shuffle(pool)
    for i, item in enumerate(pool):
        h = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()[:12]
        item["id"] = f"SYNTLOG-EXT-{i:04d}-{h}"
    return pool


if __name__ == "__main__":
    pool = generate_extended_pool(100, 100)  # 400 valid + 100 invalid = 500 total
    with open("evals/pool/syllogism_extended_pool.json", "w") as f:
        json.dump(pool, f, indent=2)
    valid_count = sum(1 for p in pool if p["valid"])
    invalid_count = sum(1 for p in pool if not p["valid"])
    print(f"Generated {len(pool)} syllogisms ({valid_count} valid, {invalid_count} invalid)")
    cats = {}
    for p in pool:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    for c, n in sorted(cats.items()):
        print(f"  {c}: {n}")
