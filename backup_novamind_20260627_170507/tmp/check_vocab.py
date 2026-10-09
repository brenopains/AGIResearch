import sys, os
sys.path.insert(0, '.')
from core.word_ngram import WordNGram
ng = WordNGram('tmp/wikitext_corpus.txt', max_vocab=15000, max_lines=80000)
for w in ['explain', 'what', 'is', 'machine', 'learning', 'tell', 'about', 'neural', 'define', 'reinforcement', 'like', 'cats', 'deep', 'artificial']:
    in_vocab = 'YES' if w in ng.unigrams else 'NO'
    print(f'{w}: {in_vocab} (count={ng.unigrams.get(w, 0)})')
print()
print('Top 30 vocab:')
for i, (w, c) in enumerate(sorted(ng.unigrams.items(), key=lambda x: -x[1])[:30]):
    print(f'  {i+1}. {w}: {c}')
