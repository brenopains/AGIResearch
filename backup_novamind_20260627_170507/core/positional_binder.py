"""
Positional Binder — Vínculo Posicional para Fala Fluida
========================================================
Em vez de empacotar palavras num único conceito (como o Role-Filler SVO),
o PositionalBinder coloca cada palavra na SUA POSIÇÃO na sequência.

Frase = POS_1 * word_1 + POS_2 * word_2 + ... + POS_N * word_N

Isto permite:
  - Decodificar palavras na ORDEM CERTA (fala fluida)
  - Resolver o bug de conceitos multi-palavra (sort_list → "sorted" na POS_1, "list" na POS_2)
  - A AGI aprender gramática emergentemente (POS_1 = "The" ou "A")

Uso:
    binder = PositionalBinder()
    thought = binder.encode_sequence(["the", "dog", "runs"])
    speech = binder.decode_sequence(thought, vocab_matrix, word_list)
    # speech ≈ "the dog runs"
"""

import torch
import torchhd
import hashlib
from typing import List, Optional, Tuple

from .coglang import Manifold


class PositionalBinder:
    """
    Codifica e decodifica sequências de palavras no espaço HDC
    usando vetores de posição POS_1 .. POS_N.
    
    Cada palavra é vinculada à sua posição:
        sentence = POS_0 * word_0 + POS_1 * word_1 + ...
    
    Para decodificar, unbind cada posição e ache a palavra mais próxima.
    """

    def __init__(self, max_length: int = 50, device: Optional[str] = None):
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_obj = torch.device(self.device)
        self.dim = Manifold.DIMENSIONALITY  # 10000

        # Gera vetores de posição determinísticos (mesmo esquema dos role vectors)
        self.position_vectors = self._generate_positions()

    def _generate_positions(self) -> torch.Tensor:
        """Gera max_length vetores de posição determinísticos."""
        positions = []
        g = torch.Generator(device=self.device_obj)
        for i in range(self.max_length):
            name = f"POS_{i}"
            seed = int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2**31)
            g.manual_seed(seed)
            vec = torchhd.random(1, self.dim, "MAP",
                                 generator=g, device=self.device_obj).squeeze(0)
            positions.append(vec)
        return torch.stack(positions)  # [max_length, DIM]

    def encode_sequence(self, words: List[str], grounding_engine=None) -> Manifold:
        """
        Codifica uma lista de palavras como uma sequência posicional HDC.
        
        Args:
            words: lista de palavras em ordem
            grounding_engine: GroundingEngine para encode_text (se None, só retorna o bundle)
        
        Returns:
            Manifold bipolar representando a sequência completa
        """
        if not words:
            raise ValueError("encode_sequence precisa de ao menos uma palavra")

        accumulated = None
        n = min(len(words), self.max_length)

        if grounding_engine is None:
            # Sem engine: espera que words sejam Manifolds
            import warnings
            warnings.warn("encode_sequence sem grounding_engine requer Manifolds em words")

        for i in range(n):
            word = words[i]
            # Codifica a palavra
            if grounding_engine is not None:
                word_mf = grounding_engine.encode_text(word)
            else:
                word_mf = word if isinstance(word, Manifold) else None
            if word_mf is None:
                continue

            # Vincula: POS_i * word_i
            pos_vec = Manifold(self.position_vectors[i], device=self.device_obj)
            bound = word_mf * pos_vec

            # Acumula
            accumulated = bound if accumulated is None else accumulated + bound

        if accumulated is None:
            raise ValueError("Nenhuma palavra válida para codificar")

        # Binariza para manter bipolaridade
        binary = torch.sign(accumulated.tensor)
        binary[binary == 0] = 1
        return Manifold(binary, device=self.device_obj)

    def decode_sequence(self, manifold: Manifold, vocab_matrix: torch.Tensor,
                        word_list: List[str], min_sim: float = 0.12,
                        max_words: int = 20) -> List[Tuple[str, float]]:
        """
        Decodifica uma sequência posicional de volta para palavras.
        
        Percorre POS_0, POS_1, ... desvinculando cada posição e
        encontrando a palavra mais próxima no vocabulário.
        
        Args:
            manifold: o Manifold da sequência codificada
            vocab_matrix: [N, DIM] tensor do vocabulário
            word_list: lista de palavras correspondente às linhas da matriz
            min_sim: similaridade mínima para considerar que há uma palavra
            max_words: máximo de palavras a decodificar
        
        Returns:
            Lista de (palavra, similaridade) na ordem da sequência
        """
        result = []
        mf_tensor = manifold.tensor.to(self.device_obj)

        for i in range(min(self.max_length, max_words)):
            pos_vec = self.position_vectors[i]
            # Unbinding: frase * POS_i ≈ word_i
            unbound = mf_tensor * pos_vec
            # Cos similaridade com vocabulário
            sims = torch.matmul(vocab_matrix, unbound) / self.dim
            best_sim, best_idx = sims.max(0)
            best_sim = best_sim.item()
            best_word = word_list[best_idx.item()]

            if best_sim < min_sim:
                break  # Silêncio — fim da frase

            result.append((best_word, best_sim))

        return result

    def decode_to_text(self, manifold: Manifold, vocab_matrix: torch.Tensor,
                       word_list: List[str], min_sim: float = 0.12,
                       max_words: int = 20) -> str:
        """Decodifica sequência e retorna como string."""
        seq = self.decode_sequence(manifold, vocab_matrix, word_list,
                                   min_sim=min_sim, max_words=max_words)
        if not seq:
            return "[...]"
        return " ".join(w for w, _ in seq)

    # ------------------------------------------------------------------ #
    #  CHUNKING — Lei de Miller: blocos de 3-4 palavras cada            #
    #  Resolve o crosstalk: SNR permanece ~50:1 em vez de cair para 5:1 #
    # ------------------------------------------------------------------ #

    def encode_chunked_sequence(self, words: List[str],
                                grounding_engine=None,
                                chunk_size: int = 4,
                                min_word_len: int = 1) -> List[Manifold]:
        """
        Quebra a sequência em chunks de `chunk_size` palavras cada e
        codifica cada chunk independentemente.

        Cada chunk usa POS_0..POS_{chunk_size-1} — o mesmo conjunto de
        vetores de posição — porque cada chunk é um manifold separado.

        Args:
            words: lista de palavras em ordem
            grounding_engine: para encode_text
            chunk_size: máx palavras por chunk (default 4 — Lei de Miller)
            min_word_len: ignora palavras mais curtas que isto
                         (default 1; usar 2 para filtrar "a", "an")

        Returns:
            Lista de Manifolds, um por chunk
        """
        if min_word_len > 1:
            words = [w for w in words if len(w) >= min_word_len]
        if not words:
            return []

        chunks = [words[i:i + chunk_size]
                  for i in range(0, len(words), chunk_size)]

        return [self.encode_sequence(chunk, grounding_engine)
                for chunk in chunks]

    def decode_chunked_sequence(self, chunk_manifolds: List[Manifold],
                                vocab_matrix: torch.Tensor,
                                word_list: List[str],
                                min_sim: float = 0.12,
                                max_words_per_chunk: int = 6
                                ) -> List[List[Tuple[str, float]]]:
        """
        Decodifica cada chunk individualmente.

        Returns:
            Lista de listas de (palavra, similaridade), um por chunk
        """
        results = []
        for mf in chunk_manifolds:
            chunk_seq = self.decode_sequence(
                mf, vocab_matrix, word_list,
                min_sim=min_sim, max_words=max_words_per_chunk
            )
            if chunk_seq:
                results.append(chunk_seq)
        return results

    def decode_chunked_to_text(self, chunk_manifolds: List[Manifold],
                               vocab_matrix: torch.Tensor,
                               word_list: List[str],
                               min_sim: float = 0.12,
                               max_words_per_chunk: int = 6,
                               sep: str = " ") -> str:
        """
        Decodifica todos os chunks e junta o texto final na ordem.

        Args:
            sep: separador entre chunks (default " " — junta tudo)
                 Use " " para frases, " | " para debug visual dos chunks

        Returns:
            String com todas as palavras na ordem, ou "[...]" se vazio
        """
        chunk_seqs = self.decode_chunked_sequence(
            chunk_manifolds, vocab_matrix, word_list,
            min_sim=min_sim, max_words_per_chunk=max_words_per_chunk
        )
        if not chunk_seqs:
            return "[...]"
        parts = [" ".join(w for w, _ in seq) for seq in chunk_seqs]
        return sep.join(parts)
