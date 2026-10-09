"""
Word-level N-gram Language Generator - Optimized v2
RAM: ~80MB for 30k vocab. Train: ~15s for 100k lines.
"""
import re
import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple


class WordNGram:
    def __init__(self, corpus_path: str = None, max_vocab: int = 30000, max_lines: int = 100000):
        self.unigrams = {}
        self.bigrams = {}
        self.trigrams = {}
        self.total_words = 0
        self.vocab_size = 0
        self.start_words = []
        self.clean_pattern = re.compile(r'[^a-zA-Z0-9\'\-\s]')
        if corpus_path and os.path.exists(corpus_path):
            self.train(corpus_path, max_vocab, max_lines)

    def train(self, corpus_path: str, max_vocab: int = 30000, max_lines: int = 100000):
        uc = defaultdict(int)
        bc = defaultdict(lambda: defaultdict(int))
        tc = defaultdict(lambda: defaultdict(int))
        sc = defaultdict(int)

        with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
            for lineno, line in enumerate(f):
                if lineno >= max_lines:
                    break
                text = self.clean_pattern.sub(' ', line.strip().lower())
                words = [w for w in text.split() if len(w) > 1 or w in ('a', 'i')]
                if len(words) < 2:
                    continue
                sc[words[0]] += 1
                for i, w in enumerate(words):
                    uc[w] += 1
                    if i > 0:
                        bc[words[i-1]][w] += 1
                    if i > 1:
                        tc[(words[i-2], words[i-1])][w] += 1

        top = sorted(uc.keys(), key=lambda w: uc[w], reverse=True)[:max_vocab]
        vocab = set(top)

        self.unigrams = {w: uc[w] for w in vocab}
        self.total_words = sum(self.unigrams.values())
        self.vocab_size = len(self.unigrams)

        self.bigrams = {}
        for a in bc:
            if a in vocab:
                f = {b: bc[a][b] for b in bc[a] if b in vocab}
                if f:
                    self.bigrams[a] = (f, sum(f.values()))

        self.trigrams = {}
        for (a, b) in tc:
            if a in vocab and b in vocab:
                f = {c: tc[(a,b)][c] for c in tc[(a,b)] if c in vocab}
                if f:
                    self.trigrams[(a,b)] = (f, sum(f.values()))

        self.start_words = [(w, sc[w]) for w in sc if w in vocab]
        if not self.start_words:
            self.start_words = [(w, c) for w, c in sorted(self.unigrams.items(), key=lambda x: -x[1])[:1000]]

    def _pick(self, candidates: Dict[str, int], total: int, temp: float) -> str:
        if not candidates:
            return None
        if temp < 0.01:
            return max(candidates, key=candidates.get)
        words = list(candidates.keys())
        probs = [(candidates[w] / total) ** (1.0 / max(temp, 0.05)) for w in words]
        z = sum(probs)
        r = random.random() * z
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                return words[i]
        return words[-1]

    def generate(self, seed_words: List[str] = None, max_words: int = 20,
                 temp: float = 0.5, guide: set = None) -> str:
        if not seed_words:
            seed_words = [self._pick(dict(self.start_words),
                                     sum(c for _, c in self.start_words), temp)]

        words = list(seed_words)
        recent = set()
        guide = guide or set()

        for _ in range(max_words - len(seed_words)):
            ctx = tuple(words[-2:]) if len(words) >= 2 else (words[-1],)

            candidates = None
            total_c = 0

            if len(ctx) == 2 and ctx in self.trigrams:
                candidates, total_c = self.trigrams[ctx]
            if (not candidates or total_c == 0) and ctx[-1] in self.bigrams:
                candidates, total_c = self.bigrams[ctx[-1]]
            if not candidates or total_c == 0:
                candidates = self.unigrams
                total_c = self.total_words

            if candidates and len(recent) > 5:
                filt = {w: c for w, c in candidates.items() if w not in recent}
                if filt:
                    candidates, total_c = filt, sum(filt.values())

            if guide and candidates:
                scored = {}
                for w, c in candidates.items():
                    score = c * (3.0 if w in guide else 1.0)
                    scored[w] = score
                candidates = scored
                total_c = sum(scored.values())

            nw = self._pick(candidates, total_c, temp)
            if nw is None:
                break
            words.append(nw)
            recent.add(nw)

        return ' '.join(words)
