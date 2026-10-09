"""
Teste final: geracao de codigo com CodeSanitizer + Completion Drop + python_brain
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
torch.set_grad_enabled(False)

from core.grounding_engine import GroundingEngine
from core.positional_binder import PositionalBinder
from core.broca_decoder import BrocaDecoder
from core.autoregressive_generator import AutoregressiveGenerator
from core.hdc_memory import HDCMemory

print("=" * 60)
print("TESTE FINAL: GERACAO DE CODIGO")
print("CodeSanitizer + Completion Drop + python_brain")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
engine = GroundingEngine(device=device)
binder = PositionalBinder(max_length=30, device=device)
decoder = BrocaDecoder(engine=engine)

hdc_memory = HDCMemory()
gen = AutoregressiveGenerator(
    engine=engine, binder=binder, decoder=decoder,
    hdc_memory=hdc_memory,
)

tests = [
    "def soma(a, b):",
    "def is_even(n):",
    "def bubble_sort(arr):",
    "def fibonacci(n):",
    "def is_palindrome(s):",
    "def is_prime(n):",
]

for test in tests:
    print(f"\n--- GENERATE: {test} ---")
    result = gen.generate(test, max_chunks=8, verbose=True)
    print(f">>> {result}")
