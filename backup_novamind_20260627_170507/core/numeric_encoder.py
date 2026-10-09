"""
Numeric Encoder — Números como Geometria no Hiperespaço
========================================================
Dual representation:
  1) Level encoding (magnitude intuition via Jaccard dos blocos ativos)
  2) Text encoding (reconhecimento exato via GroundingEngine n-gram)

Similaridade (Jaccard dos ativos):
    sim(5, 6)   ≈ 0.91  (quase mesmo tamanho)
    sim(5, 100) ≈ 0.17  (tamanhos diferentes)
"""

import torch
from typing import Union, Optional

from .coglang import Manifold


class NumericEncoder:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_obj = torch.device(self.device)
        self.dim = Manifold.DIMENSIONALITY
        self._cache = {}

    def encode(self, value: Union[int, float]) -> Manifold:
        """
        Level encoding: ativa proporcionalmente ao valor.
        v=0 → 0 ativas, v=grande → dim/2 ativas (máx 50%).
        """
        if value in self._cache:
            return self._cache[value]

        v = abs(float(value))
        fraction = 1.0 - 1.0 / (1.0 + v * 0.01)
        n_active = max(1, int(fraction * self.dim))
        n_active = min(n_active, self.dim // 2)

        vec = torch.ones(self.dim, device=self.device_obj)
        if n_active < self.dim:
            vec[n_active:] = -1.0

        if value < 0:
            vec = -vec

        mf = Manifold(vec, device=self.device_obj)
        self._cache[value] = mf
        return mf

    def encode_text(self, value: Union[int, float],
                    engine) -> Manifold:
        """Codifica número como texto (ex: '42')."""
        return engine.encode_text(str(value))

    def similarity(self, a: Union[int, float, Manifold],
                   b: Union[int, float, Manifold],
                   method: str = 'jaccard') -> float:
        """
        Similaridade entre números.

        method='jaccard': Jaccard dos blocos ativos (magnitude)
        method='cosine': cosseno bipolar padrão
        """
        va = a if isinstance(a, Manifold) else self.encode(a)
        vb = b if isinstance(b, Manifold) else self.encode(b)

        if method == 'cosine':
            return float((va.tensor * vb.tensor).sum() / self.dim)

        # Jaccard: 2 * |Ativos_A ∩ Ativos_B| / (|Ativos_A| + |Ativos_B|)
        active_a = va.tensor > 0
        active_b = vb.tensor > 0
        overlap = (active_a & active_b).sum().item()
        total_a = active_a.sum().item()
        total_b = active_b.sum().item()
        if total_a + total_b == 0:
            return 1.0
        return 2.0 * overlap / (total_a + total_b)
