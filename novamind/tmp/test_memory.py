"""Memory recall demo"""
import gc, sys, os
for mod in list(sys.modules.keys()):
    if any(k in mod for k in ['core', 'novacrush', 'word_ngram']):
        del sys.modules[mod]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, time
gc.collect(); torch.cuda.empty_cache()
from core.mind import NovaMind

mind = NovaMind()

queries = [
    'My dog is named Rex',
    'I live in Brazil',
    'What is the name of my dog',
    'Where do I live',
    'What is my dog name',
    'Tell me about my dog',
]
for i, q in enumerate(queries):
    t = time.time()
    r = mind.think(q)
    dt = (time.time() - t) * 1000
    mem = r['layers']['memory']
    recalled = [ep['content'].get('input','') for ep in mem.get('recalled_episodes', [])]
    output = r['response']['text'][:120]
    print(f"Q{i+1}: {q}")
    print(f"  Response: {output}")
    if recalled:
        for rec in recalled:
            print(f"  <memory: {rec[:60]}")
    print()
