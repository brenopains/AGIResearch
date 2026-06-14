"""
NovaMind — Language Cortex (Broca/Wernicke's Area)
WordNGram-based coherent output seeded by cognitive state.
"""

import logging
import os
from typing import List, Dict, Optional

from .word_ngram import WordNGram

CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp', 'wikitext_corpus.txt')


def _filter_known(words: List[str], vocab: set) -> List[str]:
    seen = set()
    result = []
    for w in words:
        w = w.lower().strip(" '.,!?;:()[]{}\"")
        if w in vocab and w not in seen and len(w) > 1:
            seen.add(w)
            result.append(w)
    return result


class NativeLanguageCortex:
    def __init__(self, substrate=None):
        self.is_active = True
        self.ngram = None
        try:
            if os.path.exists(CORPUS_PATH):
                self.ngram = WordNGram(CORPUS_PATH, max_vocab=15000, max_lines=80000)
                logging.info(f"WordNGram: {self.ngram.vocab_size} words, {self.ngram.total_words} total")
            else:
                logging.warning(f"Corpus not found: {CORPUS_PATH}")
        except Exception as e:
            logging.error(f"WordNGram init failed: {e}")

    def parse_to_concepts(self, human_text: str) -> List[str]:
        words = [w.strip().lower() for w in human_text.split() if len(w) > 2]
        return words

    def generate_from_seed(self, seed_words: List[str], max_words: int = 12, temp: float = 0.4) -> str:
        if not self.ngram or not seed_words:
            return " ".join(seed_words) if seed_words else ""
        raw = self.ngram.generate(seed_words[:3], max_words=max_words, temp=temp)
        if raw:
            return raw[0].upper() + raw[1:]
        return " ".join(seed_words)

    STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
                 'can', 'could', 'may', 'might', 'shall', 'should', 'what',
                 'why', 'how', 'when', 'where', 'who', 'which', 'this', 'that',
                 'its', 'it', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him',
                 'not', 'no', 'yes'}

    def generate_fluent_thought(self, prompt_text: str, predicted_concepts: List[str],
                                semantic_skeleton: List[str] = None,
                                confidence: float = 0.5, is_novel: bool = False) -> str:
        if not self.ngram:
            return "Processing."

        vocab = set(self.ngram.unigrams.keys())
        all_words = _filter_known(prompt_text.split(), vocab)
        for src in [semantic_skeleton, predicted_concepts]:
            if src:
                all_words.extend(w for w in _filter_known(src, vocab) if w not in all_words)

        content = [w for w in all_words if w not in self.STOPWORDS]
        all_set = set(all_words)

        if content:
            start = content[:3]
            raw = self.ngram.generate(start, max_words=18, temp=0.45, guide=all_set)
            if raw and not raw.lower().startswith('the '):
                return raw[0].upper() + raw[1:] + "."

        if content:
            main = content[0]
            others = content[1:4]
            if others:
                return f"{main.title()} is related to {' and '.join(others)}."
            return f"I recognize the concept of {main}."
        if all_words:
            return f"I am processing information about {' and '.join(all_words[:3])}."
        return "I am thinking about your input."
