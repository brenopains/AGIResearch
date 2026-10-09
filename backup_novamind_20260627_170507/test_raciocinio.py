"""
Teste de Raciocínio vs Memória
===============================
Teste A: Narrativa aberta - "once upon a time there was a purple dragon"
Teste B: Composição criativa - ensinar "Florb" e depois pedir história
"""
import sys, os, warnings, time
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))
warnings.filterwarnings('ignore')

from core.mind import NovaMind
from core.broca_decoder import BrocaDecoder
from core.sandbox_engine import SandboxEngine
from core.positional_binder import PositionalBinder
from core.autoregressive_generator import AutoregressiveGenerator, IntentionDrivenGenerator
from core.temporal_buffer import TemporalBuffer
from core.coglang import Manifold

print('=== TESTE A: NARRATIVA ABERTA ===')
print('Prompt: "once upon a time there was a purple dragon"')
print('Hipotese: Se for memoria pura, retorna Bentinho. Se for narrativa, gera algo sobre dragao.\n')

mind = NovaMind()
engine = mind.hdc_memory.engine
decoder = BrocaDecoder(engine)
all_idx = []
for v in mind.hdc_memory.indices.values():
    all_idx.extend(v)
decoder.build_vocabulary_from_metadata(all_idx)
decoder.min_word_len = 0
sandbox = SandboxEngine(engine)
binder = PositionalBinder(max_length=30, device=engine.device)
gen = AutoregressiveGenerator(engine=engine, binder=binder, decoder=decoder,
                               sandbox=sandbox, hdc_memory=mind.hdc_memory)
conv_buffer = TemporalBuffer(capacity=10, decay=0.8)

# TESTE A
print('=' * 60)
print('TESTE A: once upon a time there was a purple dragon')
print('=' * 60)
prompt_a = 'once upon a time there was a purple dragon'
r = gen.generate(prompt_a, max_chunks=12, verbose=True)
print(f'\nRESULTADO A: {r}\n')

# Alimenta no buffer
if r != "[...]":
    words = r.split()[:8]
    if words:
        mf = binder.encode_sequence(words, engine)
        conv_buffer.push(mf.tensor)

# TESTE B: Ensinar Florb
print('=' * 60)
print('TESTE B: Ensinar conceito novo (Florb) e pedir historia')
print('=' * 60)

# Ensina Florb
frase_ensino = 'A Florb is a blue animal that lives underground'
t1 = time.time()
thought = mind.think(frase_ensino)
dt = time.time() - t1
print(f'Ensino: "{frase_ensino}" ({dt*1000:.0f}ms)')

# Alimenta no buffer
words_in = frase_ensino.split()[:8]
if words_in:
    mf_in = binder.encode_sequence(words_in, engine)
    conv_buffer.push(mf_in.tensor)

# Pergunta sobre Florb
prompt_b = 'tell me a story about a Florb'
print(f'\nPrompt: "{prompt_b}"')
r2 = gen.generate(prompt_b, max_chunks=12, verbose=True)
print(f'\nRESULTADO B: {r2}\n')

print('=' * 60)
print('RESUMO')
print('=' * 60)
print(f'Teste A (purple dragon): Memoria ou narrativa?')
print(f'  Saida: {r[:100]}')
print(f'Teste B (Florb): Composicao ou repeticao?')
print(f'  Saida: {r2[:100]}')
