"""
Multi-hop compositional inference benchmark.

Tests whether the system can answer questions that require CHAINING facts it
was never told directly. This probes reasoning generalization, not recall.

Example: told "Rex is a dog", "dogs are mammals", "mammals are animals",
ask "Is Rex an animal?" -> requires 3-hop transitive chain (is_a closure).

Usage: python tmp/bench_inference.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.mind import NovaMind

# Each case: facts to teach, then query. A CORRECT answer must mention BOTH the
# queried subject AND the inferred property/class (else it's a false positive:
# surfacing an intermediate fact like "Mammal is an animal" for "Is Rex an animal").
CASES = [
    {
        'facts': ["Rex is a dog", "A dog is a mammal", "A mammal is an animal"],
        'query': "Is Rex an animal",
        'subject': 'rex', 'expect': 'animal', 'hops': 3,
    },
    {
        'facts': ["Socrates is a human", "A human is a mortal"],
        'query': "Is Socrates a mortal",
        'subject': 'socrates', 'expect': 'mortal', 'hops': 2,
    },
    {
        'facts': ["A dog is a mammal", "A mammal has a heart", "Rex is a dog"],
        'query': "Does Rex have a heart",
        'subject': 'rex', 'expect': 'heart', 'hops': 3,
    },
    {
        'facts': ["Paris is in France", "France is in Europe"],
        'query': "Is Paris in Europe",
        'subject': 'paris', 'expect': 'europe', 'hops': 2,
    },
    {
        'facts': ["A car is a vehicle", "A vehicle has wheels", "A Toyota is a car"],
        'query': "Does a Toyota have wheels",
        'subject': 'toyota', 'expect': 'wheel', 'hops': 3,
    },
]


def run():
    passed = 0
    for i, case in enumerate(CASES, 1):
        mind = NovaMind(use_semantic_recall=False)  # isolate inference, not recall
        for f in case['facts']:
            mind.think(f)
        resp = mind.think(case['query'])['response']['text'].lower()
        # STRICT: must mention the queried subject AND the inferred target
        ok = case['subject'] in resp and case['expect'].lower() in resp
        passed += ok
        print(f"  [{'OK ' if ok else 'XX '}] ({case['hops']}-hop) {case['query']:28s} -> {resp[:60]}")
    print(f"{'-'*72}\n  multi-hop inference (strict): {passed}/{len(CASES)} = {100*passed/len(CASES):.0f}%")


if __name__ == '__main__':
    run()
