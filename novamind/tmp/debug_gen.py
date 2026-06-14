import torch, time, gc, sys, os
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
import psutil
proc = psutil.Process()
gc.collect(); torch.cuda.empty_cache()

# Debug: directly test the ngram
from core.word_ngram import WordNGram
from core.language_cortex import _filter_known

ng = WordNGram('tmp/wikitext_corpus.txt', max_vocab=15000, max_lines=80000)
print(f"Vocab: {ng.vocab_size} words")

vocab = set(ng.unigrams.keys())
words = _filter_known("Explain what is machine learning".split(), vocab)
print(f"Filtered words: {words}")

STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
             'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
             'can', 'could', 'may', 'might', 'shall', 'should', 'what',
             'why', 'how', 'when', 'where', 'who', 'which', 'this', 'that',
             'its', 'it', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him',
             'not', 'no', 'yes'}
content = [w for w in words if w not in STOPWORDS]
print(f"Content: {content}")
all_set = set(words)
print(f"Guide: {all_set}")

if content:
    start = content[:3]
    print(f"Start: {start}")
    raw = ng.generate(start, max_words=18, temp=0.45, guide=all_set)
    print(f"Raw: {raw!r}")
    if raw and not raw.lower().startswith('the '):
        print(f"Would return: {raw[0].upper() + raw[1:] + '.'}")
    else:
        print(f"Fell through (starts with 'the')")
else:
    print("No content words!")

print("\n--- Template fallback ---")
if content:
    main = content[0]
    others = content[1:4]
    if others:
        print(f"Return: {main.title()} is related to {' and '.join(others)}.")
    else:
        print(f"Return: I recognize the concept of {main}.")
elif words:
    print(f"Return: I am processing information about {' and '.join(words[:3])}.")
else:
    print("Return: I am thinking about your input.")
