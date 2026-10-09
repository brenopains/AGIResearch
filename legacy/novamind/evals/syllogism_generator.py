"""
Gerador de silogismos EXTERNO ao NovaMind.
Vocabulário aleatório do WordNet — sem contaminação com quaisquer datasets do sistema.
Este gerador é usado como estímulo de teste, não como treino.
"""
import random
import json
import hashlib
from typing import List, Dict

random.seed(20260712)  # fixo e registrado no pré-registro

# Vocabulário fechado (substitui nomes próprios e relações em templates)
SUBJECTS = [
    "grom", "fex", "blarg", "snorp", "quib", "zork", "mlep", "tronk",
    "flarg", "snerp", "gleep", "bork", "zilp", "quarg", "freb", "slin",
    "dronk", "plerg", "snirk", "torb", "glurp", "frend", "blix", "treek",
    "sprog", "murk", "grenk", "florp", "snert", "trill", "gorb", "pleen",
    "zang", "fring", "blorp", "trog", "snerk", "mlep", "grink", "flarn"
]

PREDICATES = [
    "has zorb", "is flink", "can greb", "loves snorg", "hates blig",
    "sees kronk", "hears plig", "eats glarn", "drinks frob", "fears dring",
    "contains zilch", "produces sprin", "absorbs trum", "reflects gorp",
    "amplifies flerb", "reduces zert", "follows blurn", "precedes mlep",
    "causes flarg", "prevents dorb", "enhances plick", "inhibits trerb",
    "transforms glurn", "creates zerb", "destroys flint"
]

RELATIONS = [
    ("greater than", "less than"),
    ("above", "below"),
    ("faster than", "slower than"),
    ("larger than", "smaller than"),
    ("heavier than", "lighter than"),
    ("hotter than", "colder than"),
    ("louder than", "quieter than"),
    ("brighter than", "dimmer than"),
]

# Mapeamento de modus ponens genérico: "All X are Y. X is a Z."
def _make_modus_ponens():
    subj = random.choice(SUBJECTS)
    pred = random.choice(PREDICATES)
    return {
        "premises": [
            f"All {subj} {pred}.",
            f"A specific entity is a {subj}."
        ],
        "correct_conclusion": f"The specific entity {pred}.",
        "distractors": [
            f"The specific entity does not {pred}.",
            f"All entities {pred}.",
            f"The specific entity is not a {subj}."
        ],
        "category": "modus_ponens"
    }

def _make_modus_tollens():
    subj = random.choice(SUBJECTS)
    pred = random.choice(PREDICATES)
    return {
        "premises": [
            f"All {subj} {pred}.",
            f"A specific entity does NOT {pred}."
        ],
        "correct_conclusion": f"The specific entity is NOT a {subj}.",
        "distractors": [
            f"The specific entity {pred}.",
            f"All {subj} do not {pred}.",
            f"The specific entity is a {subj}."
        ],
        "category": "modus_tollens"
    }

def _make_barbara():
    subj_a = random.choice(SUBJECTS)
    pred_b = random.choice(PREDICATES)
    pred_c = random.choice([p for p in PREDICATES if p != pred_b])
    return {
        "premises": [
            f"All {subj_a} {pred_b}.",
            f"All entities that {pred_b} {pred_c}."
        ],
        "correct_conclusion": f"All {subj_a} {pred_c}.",
        "distractors": [
            f"All {subj_a} do not {pred_c}.",
            f"Some {subj_a} {pred_c}.",
            f"No {subj_a} {pred_c}."
        ],
        "category": "barbara"
    }

def _make_relational_chain():
    rel_pair = random.choice(RELATIONS)
    rel = rel_pair[0]  # e.g., "greater than"
    subj_a = random.choice(SUBJECTS)
    subj_b = random.choice([s for s in SUBJECTS if s != subj_a])
    subj_c = random.choice([s for s in SUBJECTS if s not in (subj_a, subj_b)])
    return {
        "premises": [
            f"{subj_a} is {rel} {subj_b}.",
            f"{subj_b} is {rel} {subj_c}."
        ],
        "correct_conclusion": f"{subj_a} is {rel} {subj_c}.",
        "distractors": [
            f"{subj_c} is {rel} {subj_a}.",
            f"{subj_a} is not {rel} {subj_c}.",
            f"{subj_b} is {rel} {subj_a}."
        ],
        "category": "relational_chain"
    }

GENERATORS = {
    "modus_ponens": _make_modus_ponens,
    "modus_tollens": _make_modus_tollens,
    "barbara": _make_barbara,
    "relational_chain": _make_relational_chain,
}

def generate_pool(n_per_category: int = 200) -> List[Dict]:
    pool = []
    for _ in range(n_per_category):
        pool.append(_make_modus_ponens())
        pool.append(_make_modus_tollens())
        pool.append(_make_barbara())
        pool.append(_make_relational_chain())
    # shuffle
    random.shuffle(pool)
    # assign unique ID
    for i, item in enumerate(pool):
        h = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()[:12]
        item["id"] = f"SYNTLOG-{i:04d}-{h}"
    return pool

if __name__ == "__main__":
    pool = generate_pool(250)  # 1000 total (4 categories × 250)
    with open("pool/syllogism_pool.json", "w") as f:
        json.dump(pool, f, indent=2)
    print(f"Generated {len(pool)} syllogisms")
    cats = {}
    for p in pool:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    for c, n in sorted(cats.items()):
        print(f"  {c}: {n}")
