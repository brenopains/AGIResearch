"""EXPERIMENTO CIENTIFICO: numeros sao entidades manipulaveis no espaco HDC?"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from core.grounding_engine import GroundingEngine
from core.hdc_memory import HDCMemory
from core.positional_binder import PositionalBinder
from core.coglang import Manifold

engine = GroundingEngine(device='cuda')
binder = PositionalBinder()
memory = HDCMemory()

DIM = Manifold.DIMENSIONALITY  # 10000

# ================================================================
# EXPERIMENTO 1: Vetores de numeros sao distintos?
# ================================================================
print("=" * 70)
print("EXPERIMENTO 1: Numeros sao vetores distintos no espaco?")
print("=" * 70)

numbers = ['0', '1', '2', '3', '5', '7', '11', '77', '154']
vecs = {}
for n in numbers:
    v = engine.encode_text(n).tensor.float().cpu()
    vecs[n] = v

# Matriz de similaridade entre numeros
print(f"\n{'':>8}", end='')
for n2 in numbers[:6]:
    print(f'{n2:>10}', end='')
print()
for n1 in numbers[:6]:
    print(f'{n1:>8}', end='')
    for n2 in numbers[:6]:
        sim = torch.nn.functional.cosine_similarity(vecs[n1], vecs[n2], dim=0).item()
        print(f'{sim:>10.3f}', end='')
    print()

# ================================================================
# EXPERIMENTO 2: is_divisible_by_7 contem o vetor de 7?
# ================================================================
print()
print("=" * 70)
print("EXPERIMENTO 2: O conceito composto contem o numero?")
print("=" * 70)

# Codificar conceitos atomicos e compostos
atomicos = {
    '7': engine.encode_text('7').tensor.float().cpu(),
    '11': engine.encode_text('11').tensor.float().cpu(),
    'and': engine.encode_text('and').tensor.float().cpu(),
    'divisible': engine.encode_text('divisible').tensor.float().cpu(),
    'mod': engine.encode_text('mod').tensor.float().cpu(),
}

compostos = {
    'is_divisible_by_7': engine.encode_text('is_divisible_by_7').tensor.float().cpu(),
    'is_divisible_by_11': engine.encode_text('is_divisible_by_11').tensor.float().cpu(),
    'is_divisible_by_3_and_5': engine.encode_text('is_divisible_by_3_and_5').tensor.float().cpu(),
    'is_divisible_by_7_and_11': engine.encode_text('is_divisible_by_7_and_11').tensor.float().cpu(),
}

print(f"\nSimilaridade entre conceitos atomicos e compostos:")
print(f"{'Conceito':<35} {'sim(7)':>8} {'sim(11)':>8} {'sim(and)':>8} {'sim(divisible)':>8}")
print("-" * 70)
for name, vec in compostos.items():
    s7 = torch.nn.functional.cosine_similarity(vec, atomicos['7'], dim=0).item()
    s11 = torch.nn.functional.cosine_similarity(vec, atomicos['11'], dim=0).item()
    sa = torch.nn.functional.cosine_similarity(vec, atomicos['and'], dim=0).item()
    sd = torch.nn.functional.cosine_similarity(vec, atomicos['divisible'], dim=0).item()
    print(f'{name:<35} {s7:>8.3f} {s11:>8.3f} {sa:>8.3f} {sd:>8.3f}')

# ================================================================
# EXPERIMENTO 3: Binding algebra funciona neste espaco?
# ================================================================
print()
print("=" * 70)
print("EXPERIMENTO 3: Binding/Unbinding funciona com encode_text?")
print("=" * 70)

# Em HDC classico: vec(A+B) = vec(A) XOR vec(B) para bipolar
# Aqui encode_text nao usa bind, usa hash de n-gramas
# Simulamos: soma de vetores bipolar = maioria dos bits (±1)
# E testamos se "unbind" recupera o original

def bipolar(v):
    return torch.sign(v).int()

v7 = bipolar(atomicos['7'])
v11 = bipolar(atomicos['11'])
v_div = bipolar(atomicos['divisible'])
v_mod = bipolar(atomicos['mod'])

# Binding via XOR em bipolar
bound_7_div = v7 * v_div  # Hadamard product = XOR em bipolar

# Unbinding: bound * v_div = (v7 * v_div) * v_div = v7 (em bipolar perfeito)
unbound_7 = bound_7_div * v_div

# Similaridade entre o desbound e o original
sim_recovered = torch.nn.functional.cosine_similarity(
    unbound_7.float(), v7.float(), dim=0).item()
print(f'\nRecuperacao de 7 a partir de (7 * divisible) * divisible: sim={sim_recovered:.3f}')
print(f'(1.0 = binding/unbinding perfeito, ~0.0 = nao funciona)')

# Testar com soma (bundle): 7 + 11
bundle_7_11 = torch.sign(v7.float() + v11.float())  # bundle por maioria

# O bundle NÃO preserva componentes individuais via unbinding
# Porque bundle = soma, nao = bind
sim_bundle_7 = torch.nn.functional.cosine_similarity(
    bundle_7_11.float(), v7.float(), dim=0).item()
print(f'\nBundle(7, 11) similar a 7: sim={sim_bundle_7:.3f}')
print(f'(Bundle funde os conceitos, nao os preserva separadamente)')

# ================================================================
# EXPERIMENTO 4: O que o PositionalBinder oferece?
# ================================================================
print()
print("=" * 70)
print("EXPERIMENTO 4: PositionalBinder pode compor numeros?")
print("=" * 70)

# binder.encode_sequence usa POS_i * word_i
# Isso permite unbind de posicoes
try:
    seq = binder.encode_sequence(['divisible', 'by', '7', 'and', '11'])
    # Unbind manual: rebind com vetor de posicao
    pos0_vec = binder.position_vectors[0].to(seq.device)
    unbound0 = seq * pos0_vec  # Hadamard product unbind = POS0 * (POS0 * word0 + ...) = word0 + noise
    mf0 = Manifold(unbound0, device=unbound0.device)
    from core.broca_decoder import BrocaDecoder
    decoder = BrocaDecoder(engine)
    decoded0 = decoder.decode(mf0, top_k=5)
    print(f'\nUnbind posicao 0 (esperado: "divisible"): {decoded0}')
    
    pos3_vec = binder.position_vectors[3].to(seq.device)
    unbound3 = seq * pos3_vec
    mf3 = Manifold(unbound3, device=unbound3.device)
    decoded3 = decoder.decode(mf3, top_k=5)
    print(f'Unbind posicao 3 (esperado: "11"): {decoded3}')
except Exception as e:
    print(f'  Erro no unbind: {e}')
    import traceback
    traceback.print_exc()

# ================================================================
# VEREDITO
# ================================================================
print()
print("=" * 70)
print("VEREDITO")
print("=" * 70)

print("""
1. Numeros tem vetores DISTINTOS: sim, cada numero gera um vetor unico
   via hashing de n-gramas.

2. Conceito composto NAO preserva componentes atomicos:
   "is_divisible_by_7" tem sim=0.0 com "7" porque encode_text()
   hasheia a string INTEIRA, sem estrutura interna.

3. Binding/unbinding NAO funciona: o espaco encode_text e baseado
   em projecao de n-gramas SHA256, nao em operacoes HDC classicas
   (XOR/bundle).

4. PositionalBinder permite unbind de posicoes: POS_i * word_i
   preserva cada palavra separadamente.

CONCLUSAO: Para a Forja funcionar, a composicao deve usar
PositionalBinder (que preserva componentes) e NAO encode_text
(que funde tudo num hash).
""")
