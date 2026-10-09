"""
Testa geracao de codigo com o novo python_brain.hdv
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
torch.set_grad_enabled(False)

from core.grounding_engine import GroundingEngine
from core.positional_binder import PositionalBinder
from core.broca_decoder import BrocaDecoder
from core.autoregressive_generator import AutoregressiveGenerator
from core.hdc_memory import HDCMemory

print("=== TESTE GERACAO DE CODIGO ===\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
engine = GroundingEngine(device=device)
binder = PositionalBinder(max_length=30, device=device)
decoder = BrocaDecoder(engine=engine)

# Carrega HDV (code_brain + python_brain + text_library + msvd_brain)
hdc_memory = HDCMemory()
print(f"Fontes HDV: {list(hdc_memory.matrices.keys())}\n")

gen = AutoregressiveGenerator(
    engine=engine, binder=binder, decoder=decoder,
    hdc_memory=hdc_memory,
)

# Testes de codigo
tests = [
    "def soma a b",
    "def is_even n",
    "def bubble_sort arr",
    "def fibonacci n",
    "def is_palindrome s",
]

for test in tests:
    print(f"\n--- GENERATE: {test} ---")
    result = gen.generate(test, max_chunks=10, verbose=True)
    print(f"\n>>> RESULTADO: {result}")
