"""
Broca V2 — Decodificador Generativo de Texto HDC → Fluente
=============================================================
Traduz vetores HDV (10.000D) em texto coerente palavra por palavra
via LSTM treinada em corpus de texto. Sem LLM. Sem API externa.

Uso:
    from core.broca_v2 import BrocaV2
    decoder = BrocaV2()
    decoder.load("D:/temp/broca_v2.pt")
    texto = decoder.generate(hdv_vector, max_length=100)
"""
from .decoder import BrocaV2
from .model import BrocaV2Model
from .tokenizer import BPETokenizer
