"""
Negative control for inference: the system must NOT affirm chains that don't
exist. Guards against the dishonest failure mode of "say yes to everything".
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.mind import NovaMind

# facts, query, and the WRONG affirmation that must NOT appear
CASES = [
    {
        'facts': ["Rex is a dog", "A cat is a mammal"],
        'query': "Is Rex a cat",
        'must_not_contain': 'rex is a cat',
    },
    {
        'facts': ["A dog is a mammal", "A bird has wings"],
        'query': "Does a dog have wings",
        'must_not_contain': 'dog has wings',
    },
    {
        'facts': ["Paris is in France", "Tokyo is in Japan"],
        'query': "Is Paris in Japan",
        'must_not_contain': 'paris is in japan',
    },
    {
        'facts': ["Rex is a dog"],
        'query': "Is Rex a fish",
        'must_not_contain': 'rex is a fish',
    },
]


def run():
    passed = 0
    for case in CASES:
        mind = NovaMind(use_semantic_recall=False)
        for f in case['facts']:
            mind.think(f)
        resp = mind.think(case['query'])['response']['text'].lower()
        ok = case['must_not_contain'] not in resp
        passed += ok
        print(f"  [{'OK ' if ok else 'XX '}] {case['query']:22s} -> {resp[:55]}")
    print(f"{'-'*70}\n  correctly refused false chains: {passed}/{len(CASES)} = {100*passed/len(CASES):.0f}%")


if __name__ == '__main__':
    run()
