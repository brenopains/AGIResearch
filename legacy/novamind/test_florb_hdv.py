"""
Teste Florb: valida que os 3 fixes (buffer L1, working HDV, anti-loop) funcionam.
Nao precisa de REPL — testa o IntentionDrivenGenerator diretamente.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
torch.set_grad_enabled(False)

from core.grounding_engine import GroundingEngine
from core.positional_binder import PositionalBinder
from core.broca_decoder import BrocaDecoder
from core.autoregressive_generator import AutoregressiveGenerator, IntentionDrivenGenerator
from core.temporal_buffer import TemporalBuffer
from core.hdc_memory import HDCMemory

print("=" * 60)
print("TESTE FLORB — Working HDV + Buffer L1 + Anti-loop")
print("=" * 60)

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[SETUP] Device: {device}")

engine = GroundingEngine(device=device)
binder = PositionalBinder(max_length=30, device=device)
decoder = BrocaDecoder(engine=engine)

# HDCMemory já carrega na importacao
hdc_memory = HDCMemory()
print(f"[SETUP] HDC Memory: {list(hdc_memory.matrices.keys())}")

# Temporal buffer (vazio inicialmente)
buffer = TemporalBuffer(capacity=10, decay=0.8)

# 2. Cria generator com buffer
gen = AutoregressiveGenerator(
    engine=engine, binder=binder, decoder=decoder,
    hdc_memory=hdc_memory, temporal_buffer=buffer
)

# 3. Teste 1: Memoria (conceito novo -> working HDV -> recuperacao)
print(f"\n{'='*60}")
print("TESTE 1: Memoria — ensinar e recuperar Florb")
print(f"{'='*60}")

# Ensina o conceito (simula o que o REPL faz apos think())
conceito_florb = "A Florb is a blue animal that lives underground"
gen.add_to_working_hdv(conceito_florb)

print(f"\nWorking HDV apos ensinar: {len(gen.working_hdv)} itens")
for k in gen.working_hdv:
    print(f"  - '{k[:60]}...'")

# Testa geracao
print(f"\nGerando com intencao: 'What color is a Florb?'")
result = gen.generate("What color is a Florb?", max_chunks=10, verbose=True)
print(f"\n>>> RESULTADO 1: {result}")

# 4. Teste 2: Buffer temporal L1
print(f"\n{'='*60}")
print("TESTE 2: Buffer temporal L1")
print(f"{'='*60}")

# Alimenta buffer com encode_text (mesmo espaco do residual)
mf = engine.encode_text(conceito_florb[:200])
buffer.push(mf.tensor)
print(f"Buffer apos push: {len(buffer.buffer)} chunks, usage={buffer.usage:.0%}")

# Testa geracao que deveria pegar do buffer primeiro
print(f"\nGerando com intencao: 'Tell me about Florb'")
result2 = gen.generate("Tell me about a Florb", max_chunks=10, verbose=True)
print(f"\n>>> RESULTADO 2: {result2}")

# 5. Teste 3: Anti-loop com repeticoes forçadas
print(f"\n{'='*60}")
print("TESTE 3: Anti-loop — repeticoes forcadas")
print(f"{'='*60}")

# Cria um IntentionDrivenGenerator isolado com HDV que tem poucos itens
# (para forçar loop) e sem working_hdv / buffer
gen_isolado = IntentionDrivenGenerator(
    engine, binder, decoder,
    hdc_memory=hdc_memory
)

print(f"\nGerando (sem buffer, sem working HDV — propenso a loop):")
result3 = gen_isolado.generate("once upon a time", max_chunks=15, verbose=True)
print(f"\n>>> RESULTADO 3 (anti-loop ativo): {result3[:200] if len(result3) > 200 else result3}")

# Conta repeticoes no resultado
chunks = result3.split(" ")
unique = set(chunks)
if len(chunks) > 0:
    print(f"  Total palavras: {len(chunks)}, unicas: {len(unique)}, repetidas: {len(chunks)-len(unique)}")

# 6. Teste 4: Atualizacao de crenca (contradicao)
print(f"\n{'='*60}")
print("TESTE 4: Atualizacao de crenca")
print(f"{'='*60}")

# Ensina cor original
gen.add_to_working_hdv("A Florb is a blue animal")
print(f"Ensinado: 'A Florb is a blue animal'")
print(f"Working HDV agora: {[k[:40] for k in gen.working_hdv_keys]}")

# Depois ensina contradicao (a Florb é na verdade red)
gen.add_to_working_hdv("A Florb is actually red")
print(f"Atualizado: 'A Florb is actually red'")
print(f"Working HDV agora: {[k[:40] for k in gen.working_hdv_keys]}")

print(f"\nGerando com intencao: 'What color is a Florb?'")
result4 = gen.generate("What color is a Florb?", max_chunks=5, verbose=True)
print(f"\n>>> RESULTADO 4: {result4}")

# 7. Teste 5: Dois conceitos distintos nao vazam
print(f"\n{'='*60}")
print("TESTE 5: Dois conceitos — Florb (blue) + Glorp (yellow), sem vazamento")
print(f"{'='*60}")

gen2 = AutoregressiveGenerator(
    engine=engine, binder=binder, decoder=decoder,
    hdc_memory=hdc_memory
)
gen2.add_to_working_hdv("A Florb is a blue animal that lives underground")
gen2.add_to_working_hdv("A Glorp is a yellow flying creature that lives in the sky")

print(f"Working HDV: {[k[:40] for k in gen2.working_hdv_keys]}")

print(f"\nGerando: 'Tell me about a Glorp'")
result5 = gen2.generate("Tell me about a Glorp", max_chunks=8, verbose=True)
print(f"\n>>> RESULTADO 5 (Glorp): {result5}")
print()

print(f"Gerando: 'Tell me about a Florb'")
result5b = gen2.generate("Tell me about a Florb", max_chunks=8, verbose=True)
print(f"\n>>> RESULTADO 5b (Florb): {result5b}")

# Verifica vazamento
florb_result = result5b.lower()
glorp_result = result5.lower()
florb_leak = "yellow" in florb_result or "fly" in florb_result
glorp_leak = "blue" in glorp_result or "underground" in glorp_result
if florb_leak:
    print(f"\n! VAZAMENTO: 'yellow'/'fly' apareceu na historia do Florb")
if glorp_leak:
    print(f"\n! VAZAMENTO: 'blue'/'underground' apareceu na historia do Glorp")
if not florb_leak and not glorp_leak:
    print(f"\nOK Sem vazamento entre conceitos")

print(f"\n{'='*60}")
print("TESTE CONCLUIDO")
print(f"{'='*60}")
