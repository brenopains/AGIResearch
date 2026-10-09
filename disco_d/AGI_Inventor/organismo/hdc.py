"""
organismo.hdc — campo associativo (vetores hiperdimensionais bipolares)
========================================================================
O HDC aqui NAO guarda a ideia: guarda o CONTEXTO em que cada estrutura nasceu,
pagou ou falhou, e mede semelhanca entre contextos. Programas, leis, dados e
evidencias ficam inteiros fora do vetor (desenho de ARQUITETURA.md e da proposta
de conectomas: "HDC ordena e referencia").
"""
from __future__ import annotations

import hashlib

import numpy as np

D = 2048
_cache: dict[str, np.ndarray] = {}


def vetor(nome: str) -> np.ndarray:
    """Vetor bipolar aleatorio, deterministico pelo nome."""
    v = _cache.get(nome)
    if v is None:
        sem = int.from_bytes(hashlib.sha256(nome.encode()).digest()[:8], "little")
        v = np.where(np.random.default_rng(sem).random(D) < 0.5, -1.0, 1.0)
        _cache[nome] = v
    return v


def codifica(feats: dict) -> np.ndarray:
    """Contexto = soma de (papel ⊙ valor) normalizada."""
    if not feats:
        return np.zeros(D)
    s = np.zeros(D)
    for k, val in feats.items():
        s += vetor("papel:" + k) * vetor(f"valor:{k}={val}")
    return s / np.sqrt(len(feats))


def sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))
