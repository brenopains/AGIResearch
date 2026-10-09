"""Teste cientifico de composicao: divisivel por 7 e 11"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from core.grounding_engine import GroundingEngine
from core.hdc_memory import HDCMemory
from core.autoregressive_generator import AutoregressiveGenerator
from core.positional_binder import PositionalBinder
from core.broca_decoder import BrocaDecoder
from core.temporal_buffer import TemporalBuffer
from core.code_extractor import sanitize_code, desanitize_code

engine = GroundingEngine(device='cuda')
binder = PositionalBinder()
decoder = BrocaDecoder(engine)
buffer = TemporalBuffer(engine)
memory = HDCMemory()

gen = AutoregressiveGenerator(engine, binder, decoder, hdc_memory=memory, temporal_buffer=buffer)

# ENSINA conceitos-chave separadamente via working HDV
print("=" * 60)
print("ENSINANDO CONCEITOS-CHAVE")
print("=" * 60)
training = [
    # MOD com numeros
    "def is_even(n): return n % 2 == 0",
    "def is_multiple_of_7(n): return n % 7 == 0",
    "def is_multiple_of_11(n): return n % 11 == 0",
    # AND logico
    "def both_positive(a, b): return a > 0 and b > 0",
    "def is_in_range(n): return n >= 0 and n <= 100",
    # MOD + AND combinados
    "def is_divisible_by_2_and_3(n): return n % 2 == 0 and n % 3 == 0",
    "def is_divisible_by_3_and_5(n): return n % 3 == 0 and n % 5 == 0",
    # Outras funcoes de apoio
    "def is_valid_index(i, size): return i >= 0 and i < size",
    "def is_between(a, x, b): return a < x and x < b",
    "def has_remainder(n, d): return n % d != 0",
    "def is_divisible(n, d): return n % d == 0",
]
for func in training:
    gen.add_to_working_hdv(func)
    print(f"  [+] {func}")

# VERIFICA se working_hdv tem os conceitos
print()
print(f"Working HDV: {len(gen.working_hdv)} itens")
print(f"Working HDV keys: {len(gen.working_hdv_keys)} keys")

# TESTE PRINCIPAL
print()
print("=" * 60)
print("TESTE: GERAR def is_divisible_by_7_and_11(n):")
print("(NUNCA viu esta funcao antes)")
print("=" * 60)

result = gen.generate(
    "def is_divisible_by_7_and_11(n):",
    max_chunks=12,
    chunk_size=4,
    min_sim=0.0,
    verbose=True
)

print()
print(">>> RESULTADO FINAL:")
print(result)

# VALIDA
print()
print("=" * 60)
print("VALIDACAO")
print("=" * 60)
try:
    compile(result, '<test>', 'exec')
    print("  ✅ COMPILA!")
except SyntaxError as e:
    print(f"  ❌ SYNTAX ERROR: {e}")

if 'def is_divisible_by_7_and_11' in result:
    try:
        namespace = {}
        exec(result, namespace)
        fn = namespace.get('is_divisible_by_7_and_11')
        if fn:
            print()
            print("TESTES LOGICOS:")
            tests = [(7, False), (11, False), (77, True), (154, True), (1, False)]
            all_pass = True
            for n, exp in tests:
                actual = fn(n)
                status = "✅" if actual == exp else "❌"
                if actual != exp: all_pass = False
                print(f"  f({n:3d}) = {str(actual):5s}  esperado={str(exp):5s}  {status}")
            print()
            print("  ✅ TODOS PASSARAM!" if all_pass else "  ⚠ ALGUNS FALHARAM")
    except Exception as e:
        print(f"  ⚠ ERRO NA EXECUCAO: {e}")
