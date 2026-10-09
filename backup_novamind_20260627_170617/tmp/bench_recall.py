"""
Paraphrase recall benchmark.

Stores a set of facts as episodes, then queries with PARAPHRASES that share
few or no content words with the stored fact. Measures recall@1: does the
memory return the correct fact as the top result?

This isolates the recall component (MemorySystem.recall_episodic) from the
rest of the pipeline, so we measure exactly what the embedding change affects.

Usage:
    python tmp/bench_recall.py            # current method (Jaccard/NCD)
    python tmp/bench_recall.py --emb      # embedding-based recall
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import MemorySystem

# (fact stored, paraphrased question with little/no lexical overlap)
DATASET = [
    ("My dog is named Rex",                 "What should I call my canine companion"),
    ("I live in Brazil",                    "Which country am I a resident of"),
    ("I like pizza",                        "What is my favorite dish to eat"),
    ("I am learning about neural networks", "Which subject am I currently studying"),
    ("My car is a Toyota",                  "What brand is my automobile"),
    ("I work as a software engineer",       "What is my profession"),
    ("My favorite color is blue",           "Which hue do I prefer most"),
    ("I was born in 1995",                  "What year did I come into the world"),
    ("My sister is a doctor",               "What does my sibling do for a living"),
    ("I play the guitar",                   "Which musical instrument can I perform with"),
    ("I drink coffee every morning",        "What beverage do I consume at the start of the day"),
    ("My apartment is on the fifth floor",  "Which level of the building is my home on"),
    ("I speak Portuguese and English",      "What languages am I able to communicate in"),
    ("My phone is an iPhone",               "What kind of mobile device do I own"),
    ("I go running on weekends",            "What physical activity do I do on Saturdays"),
]

# Distractor facts (stored but never the correct answer) to make recall harder
DISTRACTORS = [
    "The weather today is sunny",
    "Paris is the capital of France",
    "Water boils at 100 degrees",
    "The meeting is at 3pm",
    "Cats are mammals",
]


def build_memory():
    mem = MemorySystem()
    # store correct facts
    for i, (fact, _) in enumerate(DATASET):
        mem.store_episode(
            content={'input': fact, 'concepts': [], 'relations': 0, 'type': 'interaction'},
            episode_type='interaction', importance=0.6,
        )
    # store distractors
    for d in DISTRACTORS:
        mem.store_episode(
            content={'input': d, 'concepts': [], 'relations': 0, 'type': 'interaction'},
            episode_type='interaction', importance=0.5,
        )
    return mem


def run(use_emb: bool):
    mem = build_memory()
    if use_emb:
        from core.semantic_encoder import attach_encoder
        attach_encoder(mem)

    correct = 0
    total = len(DATASET)
    latencies = []
    print(f"\n{'='*70}\nMethod: {'EMBEDDING (MiniLM cosine)' if use_emb else 'BASELINE (Jaccard + NCD)'}\n{'='*70}")
    for fact, query in DATASET:
        t0 = time.time()
        results = mem.recall_episodic({'input': query}, top_k=1)
        latencies.append((time.time() - t0) * 1000)
        top = results[0].content.get('input', '') if results else '(none)'
        ok = (top == fact)
        correct += ok
        mark = 'OK ' if ok else 'XX '
        print(f"  [{mark}] q: {query[:42]:42s} -> {top[:32]}")
    print(f"{'-'*70}")
    print(f"  recall@1: {correct}/{total} = {100*correct/total:.0f}%")
    print(f"  avg latency: {sum(latencies)/len(latencies):.2f} ms/query")
    return correct, total


if __name__ == '__main__':
    use_emb = '--emb' in sys.argv
    run(use_emb)
