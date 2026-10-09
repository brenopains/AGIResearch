import os
import gc
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_mem(step_name):
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024
        print(f"[MEM] {step_name}: {mem:.2f} MB")
    except ImportError:
        print(f"[MEM] {step_name}")

print_mem("Initial startup")

from core.mind import NovaMind
mind = NovaMind(use_semantic_recall=False)
from brain_persistence import load_brain
load_brain(mind)
print_mem("After load_brain")

# Load real sentences from books
from knowledge.manager import KnowledgeManager

def extract_relevant_sentences(filepath, keywords, max_sentences=60):
    sentences = []
    if not os.path.exists(filepath):
        return sentences
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    raw_sentences = KnowledgeManager.split_into_sentences(text, max_words=30)
    for s in raw_sentences:
        s_lower = s.lower()
        if any(k in s_lower for k in keywords):
            sentences.append(s)
            if len(sentences) >= max_sentences:
                break
    return sentences

dataset_dir = r"D:\novamind_data\text_datasets"
keywords = [
    "bentinho", "capitu", "ezequiel", "sherlock", "watson", 
    "alice", "frankenstein", "monster", "vampire", "dracula",
    "london", "rio"
]
books_files = [
    "dom_casmurro_pt.txt",
    "bras_cubas_pt.txt",
    "alice_en.txt",
    "sherlock_en.txt",
    "frenstein_en.txt"
]

all_sentences = []
for f in books_files:
    path = os.path.join(dataset_dir, f)
    book_sents = extract_relevant_sentences(path, keywords, max_sentences=20)
    all_sentences.extend(book_sents)

print(f"Extracted {len(all_sentences)} real sentences.")
print_mem("After loading sentences")

print("Running think cycles on real sentences...")
import torch
for i, sentence in enumerate(all_sentences[:100]):
    mind.think(sentence)
    if i % 10 == 0:
        print_mem(f"After real cycle {i} (Sentence: '{sentence[:30]}...')")
        gc.collect()

