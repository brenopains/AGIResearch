"""Debug: trace exactly what generate_fluent_thought receives from mind.py"""
import gc, sys, os

for mod in list(sys.modules.keys()):
    if any(k in mod for k in ['core', 'novacrush', 'word_ngram']):
        del sys.modules[mod]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, time

from core.mind import NovaMind
mind = NovaMind()

print("=== Monkey-patching generate_fluent_thought ===\n")

original = mind.language.generate_fluent_thought

def debug_generate(*args, **kwargs):
    print(f"[TRACE] generate_fluent_thought called with:")
    print(f"  prompt_text={args[0]!r}")
    print(f"  predicted_concepts={args[1]!r}")
    print(f"  semantic_skeleton={kwargs.get('semantic_skeleton')!r}")
    result = original(*args, **kwargs)
    print(f"  RESULT={result!r}")
    print()
    return result

mind.language.generate_fluent_thought = debug_generate

r = mind.think("What is machine learning")
print(f"\nFINAL OUTPUT: {r['response']['text'][:100]}")
