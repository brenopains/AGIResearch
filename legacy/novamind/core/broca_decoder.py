"""
BrocaDecoder — Decodificador de Pensamento HDC para Texto
===========================================================
Traduz manifolds de pensamento de volta para palavras com
SINTAXE via Role-Filler Unbinding.

Funciona como a Área de Broca no cérebro:
1. Tenta decodificar por papéis sintáticos (SUJEITO, VERBO, OBJETO)
2. Se falha, cai para busca plana Top-K (fala telegráfica)

A sintaxe emerge do armazenamento estruturado das frases:
  Frase = ROLE_SUBJECT * palavra + ROLE_VERB * palavra + ROLE_OBJECT * palavra
"""

import torch
import re
import unicodedata
from typing import Dict, List, Tuple, Optional
from collections import Counter

from .coglang import Manifold, ALL_ROLES, bind_roles, unbind_role, decode_roles, ROLE_SUBJECT, ROLE_VERB, ROLE_OBJECT
from .grounding_engine import GroundingEngine


class BrocaDecoder:
    """
    Decodificador de pensamento → texto via vocabulário HDC.
    Suporta decodificação plana (Top-K) e estruturada (Role-Filler).
    """

    # Ordem sintática default para montar frases
    SYNTACTIC_ORDER = ["suj", "ver", "obj", "adj", "adv", "pre", "neg", "con", "cond"]

    def __init__(self, engine: GroundingEngine, min_word_len: int = 2):
        self.engine = engine
        self.min_word_len = min_word_len
        self.word_vectors: Dict[str, Manifold] = {}
        self.word_list: List[str] = []
        self.vocab_matrix: Optional[torch.Tensor] = None
        self.device = engine.device

    def add_word(self, word: str):
        """Adiciona uma palavra ao vocabulário, codificando-a como Manifold HDC."""
        word = word.strip().lower()
        if len(word) < self.min_word_len or not word.isalpha():
            return
        if word in self.word_vectors:
            return
        mf = self.engine.encode_text(word)
        self.word_vectors[word] = mf

    def build_vocabulary_from_texts(self, texts: List[str]):
        """
        Constrói o vocabulário a partir de uma lista de textos.
        Extrai palavras únicas, codifica cada uma em HDC.
        """
        word_counter = Counter()
        for text in texts:
            text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
            words = re.findall(r'[a-zA-Z]{2,}', text.lower())
            word_counter.update(words)

        for word, _ in word_counter.most_common(8000):
            self.add_word(word)

        self._build_matrix()
        print(f"[BrocaDecoder] Vocabulario: {len(self.word_vectors)} palavras")

    def build_vocabulary_from_metadata(self, index: List[Dict]):
        """Constrói vocabulário a partir dos metadados do HDV."""
        texts = []
        for meta in index:
            t = meta.get('text', meta.get('id', ''))
            if t:
                texts.append(t)
        self.build_vocabulary_from_texts(texts)

    def _build_matrix(self):
        """Constrói matriz de vocabulário para busca rápida na GPU."""
        if not self.word_vectors:
            return
        self.word_list = list(self.word_vectors.keys())
        tensors = [self.word_vectors[w].tensor for w in self.word_list]
        self.vocab_matrix = torch.stack(tensors).to(self.device)

    def decode(self, thought_manifold: Manifold, top_k: int = 5,
               min_sim: float = 0.10) -> str:
        """
        Decodifica um manifold de pensamento em texto.

        1. Tenta decodificação estruturada (Role-Filler Unbinding)
        2. Se falha, cai para decodificação plana (Top-K)

        Returns:
            String com a frase montada.
        """
        if self.vocab_matrix is None or len(self.word_list) == 0:
            return "[...]"

        # Estratégia 1: Decodificação por papéis sintáticos
        role_results = decode_roles(thought_manifold, self.vocab_matrix,
                                     self.word_list, top_k=1, min_sim=0.12)

        if len(role_results) >= 2:
            return self._format_role_sentence(role_results)

        # Estratégia 2: Decodificação plana (Top-K original)
        return self._decode_flat(thought_manifold, top_k=top_k, min_sim=min_sim)

    def _decode_flat(self, thought_manifold: Manifold, top_k: int = 5,
                     min_sim: float = 0.10) -> str:
        """Decodificação plana: Top-K palavras mais similares."""
        query = thought_manifold.tensor.to(self.device)
        sims = torch.matmul(self.vocab_matrix, query) / Manifold.DIMENSIONALITY

        values, indices = torch.topk(sims, min(top_k * 2, len(self.word_list)))

        selected = []
        best_sim = values[0].item()
        for i in range(len(indices)):
            sim = values[i].item()
            if sim < min_sim:
                break
            word = self.word_list[indices[i]]
            if word not in selected:
                selected.append(word)

        if not selected:
            if best_sim > 0.05:
                return f"[{self.word_list[indices[0]]}?]"
            return "[...]"

        return " ".join(selected)

    def _format_role_sentence(self, role_results: list) -> str:
        """
        Monta uma frase gramatical a partir dos resultados de role unbinding.
        Ordem: suj → ver → obj → adj → adv → pre → neg

        Exemplo:
            [("suj", "man", 0.45), ("ver", "rides", 0.52), ("obj", "horse", 0.38)]
            → "man rides horse"
        """
        # Converte para dicionário para acesso rápido
        role_words = {}
        role_sims = {}
        for role_name, word, sim in role_results:
            if role_name not in role_words or sim > role_sims.get(role_name, 0):
                role_words[role_name] = word
                role_sims[role_name] = sim

        # Monta na ordem sintática
        parts = []
        for role in self.SYNTACTIC_ORDER:
            if role in role_words:
                parts.append(role_words[role])

        if not parts:
            return "[...]"

        return " ".join(parts)

    def decode_with_scores(self, thought_manifold: Manifold, top_k: int = 5
                           ) -> List[Tuple[str, float]]:
        """
        Decodifica e retorna palavras com suas similaridades (modo plano).
        Útil para debug.
        """
        if self.vocab_matrix is None or len(self.word_list) == 0:
            return []

        query = thought_manifold.tensor.to(self.device)
        sims = torch.matmul(self.vocab_matrix, query) / Manifold.DIMENSIONALITY
        values, indices = torch.topk(sims, min(top_k, len(self.word_list)))

        return [(self.word_list[indices[i]], values[i].item())
                for i in range(len(indices))]

    def decode_structured(self, thought_manifold: Manifold, min_sim: float = 0.12
                          ) -> List[Tuple[str, str, float]]:
        """
        Decodifica apenas por papéis sintáticos.
        Retorna lista de (role_name, palavra, similaridade).
        """
        if self.vocab_matrix is None or len(self.word_list) == 0:
            return []

        return decode_roles(thought_manifold, self.vocab_matrix,
                            self.word_list, top_k=1, min_sim=min_sim)
