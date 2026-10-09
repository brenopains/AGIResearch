import sys, os
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

# Direct test of the generate_fluent_thought
from core.language_cortex import NativeLanguageCortex
lc = NativeLanguageCortex()
print(f"NGram loaded: {lc.ngram is not None}")
if lc.ngram:
    print(f"Vocab: {lc.ngram.vocab_size}")

# Test the exact call from mind.py
result = lc.generate_fluent_thought(
    prompt_text="Explain what is machine learning",
    predicted_concepts=['explain', 'what', 'machine', 'learning'],
    semantic_skeleton=['machine', 'learning', 'artificial', 'intelligence'],
    confidence=0.76,
    is_novel=True
)
print(f"\nResult: {result!r}")
