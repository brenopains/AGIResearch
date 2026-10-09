"""Adiciona funcoes de treino que ensinam mod + and separadamente"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from core.grounding_engine import GroundingEngine
from core.hdc_memory import HDCMemory, merge_fallback
from core.code_extractor import sanitize_code

engine = GroundingEngine(device='cuda')
mem = HDCMemory(engine)

# Funcoes que ensinam MOD com diferentes numeros
training_functions = [
    # mod basico com diferentes numeros
    "def is_even(n): return n % 2 == 0",
    "def is_odd(n): return n % 2 != 0",
    "def is_multiple_of_3(n): return n % 3 == 0",
    "def is_multiple_of_5(n): return n % 5 == 0",
    "def is_multiple_of_7(n): return n % 7 == 0",
    "def is_multiple_of_11(n): return n % 11 == 0",
    "def is_multiple_of_10(n): return n % 10 == 0",
    # mod + comparacao
    "def has_remainder(n, d): return n % d != 0",
    "def is_divisible(n, d): return n % d == 0",
    # and logico com condicoes
    "def is_between(a, x, b): return a < x and x < b",
    "def is_positive_even(n): return n > 0 and n % 2 == 0",
    "def is_in_range(n): return n >= 0 and n <= 100",
    "def is_valid_index(i, size): return i >= 0 and i < size",
    "def both_positive(a, b): return a > 0 and b > 0",
    "def is_divisible_by_2_and_3(n): return n % 2 == 0 and n % 3 == 0",
    "def is_divisible_by_3_and_5(n): return n % 3 == 0 and n % 5 == 0",
    # if/else com multiplas condicoes
    "def is_leap_year(y): return (y % 400 == 0) or (y % 4 == 0 and y % 100 != 0)",
    "def classify_number(n): return 'positive' if n > 0 else 'negative' if n < 0 else 'zero'",
    "def max_of_three(a, b, c): return a if a > b and a > c else b if b > c else c",
    # funcoes com return bool e condicoes
    "def is_teenager(age): return age >= 13 and age <= 19",
    "def can_vote(age, citizen): return age >= 16 and citizen",
    "def is_valid_password(pw): return len(pw) >= 8 and any(c.isdigit() for c in pw)",
]

# Carregar python_brain existente
mem.load_all()
device = 'cuda'

# Codificar e adicionar cada funcao
added = 0
for func in training_functions:
    sanitized = sanitize_code(func)
    # Verificar se ja existe similar
    q = engine.encode_text(sanitized).tensor.float()
    existing = mem.query(q.cpu(), top_k=1)
    if existing and existing[0][1] > 0.5:
        # Ja existe algo similar
        continue
    # Codificar e adicionar
    vec = engine.encode_text(sanitized).tensor.float().cpu()
    mem.add_item(sanitized, vec[0])
    added += 1
    print(f"  [+] {func}")

mem.save_all()
print(f"\nAdicionadas {added} novas funcoes de treino ao python_brain.hdv")
print(f"Total python_brain: {mem.memories.get('python_brain', {}).get('size', 0)} itens")
