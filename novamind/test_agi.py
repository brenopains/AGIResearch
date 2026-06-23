"""
Teste interativo da AGI — conversa + geração + cognição
========================================================
Envia comandos para o REPL e captura respostas.
"""
import sys, os, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
sys.stdout = open(sys.stdout.fileno(), 'w', encoding='utf-8', buffering=1)

from core.mind import NovaMind
from core.broca_decoder import BrocaDecoder
from core.sandbox_engine import SandboxEngine
from core.positional_binder import PositionalBinder
from core.autoregressive_generator import AutoregressiveGenerator
from core.numeric_encoder import NumericEncoder

print("=" * 60)
print("  TESTE DE CAPACIDADES DA AGI (PRÉ-TREINO)")
print("=" * 60)

print("\nInicializando NovaMind...")
t0 = time.time()
mind = NovaMind()
print(f"Pronta em {time.time()-t0:.1f}s")

# Setup componentes
decoder = None
if hasattr(mind, 'hdc_memory') and mind.hdc_memory is not None:
    stats = mind.hdc_memory.get_stats()
    print(f"Memoria HDV: {stats['total_items']} itens, {stats['total_vram_kb']:.0f} KB VRAM")

    all_index = []
    for idx_list in mind.hdc_memory.indices.values():
        all_index.extend(idx_list)

    if all_index:
        decoder = BrocaDecoder(mind.hdc_memory.engine)
        decoder.build_vocabulary_from_metadata(all_index)
        extras = (
            "o que e um uma voce eu ele ela nos eles isso aquila esta este sim nao ser ter estar poder fazer "
            "a an the is are was were to of in on for with by at and or this that it be has have "
            "do does did will would could should may might shall can not no but so if as up down out "
            "off over under again further then once here there where when why how all each every "
            "both few more most other some such only own same too very just also"
        )
        decoder.build_vocabulary_from_texts([extras])
        decoder.min_word_len = 0
        print(f"BrocaDecoder: {len(decoder.word_vectors)} palavras")

    sandbox = SandboxEngine(mind.hdc_memory.engine)
    binder = PositionalBinder(max_length=30, device=mind.hdc_memory.engine.device)

    gen_engine = mind.hdc_memory.engine
    gen_sandbox = SandboxEngine(gen_engine)
    generator = AutoregressiveGenerator(
        engine=gen_engine, binder=binder, decoder=decoder,
        sandbox=gen_sandbox, hdc_memory=mind.hdc_memory
    )

    num_encoder = NumericEncoder(device=mind.hdc_memory.engine.device)

print("\n" + "=" * 60)
print("TESTE 1: Conversa Basica (input -> think -> decode)")
print("=" * 60)

for frase in [
    "hello world",
    "what is the meaning of life",
    "tell me about artificial intelligence",
    "my name is AGI test",
    "what is my name",
]:
    print(f"\n>>> Input: '{frase}'")
    t1 = time.time()
    thought = mind.think(frase)
    dt = (time.time() - t1) * 1000

    hdc_ctx = thought.get('layers', {}).get('hdc_memory', {}).get('context', '')
    pred_error = thought.get('predictive_coding', {}).get('prediction_error', 0)
    meta = thought.get('layers', {}).get('metacognition', {})

    spoken = ""
    if decoder and binder is not None:
        words_input = frase.split()
        chunks = binder.encode_chunked_sequence(
            words_input, decoder.engine, chunk_size=4, min_word_len=2
        )
        if chunks:
            pos_seq = binder.decode_chunked_to_text(
                chunks, decoder.vocab_matrix, decoder.word_list, min_sim=0.08
            )
            if pos_seq != "[...]" and len(pos_seq) >= 2:
                spoken = pos_seq

    if not spoken and decoder and sandbox is not None:
        composed_mf = sandbox.compose_concept(frase)
        if composed_mf is not None:
            structured = sandbox.decode_roles(
                composed_mf, decoder.vocab_matrix, decoder.word_list, min_sim=0.12
            )
            if len(structured) >= 2:
                order = {'suj': 0, 'ver': 1, 'obj': 2, 'adj': 3, 'adv': 4, 'pre': 5, 'neg': 6}
                sorted_res = sorted(structured, key=lambda x: order.get(x[0], 99))
                words_only = [w for _, w, _ in sorted_res]
                spoken = " ".join(words_only)
            else:
                spoken = decoder.decode(composed_mf, top_k=5, min_sim=0.08)
    elif not spoken and decoder:
        thought_mf = decoder.engine.encode_text(hdc_ctx[:500] if hdc_ctx else frase[:500])
        spoken = decoder.decode(thought_mf, top_k=5, min_sim=0.08)

    if not spoken or spoken == "[...]":
        if hdc_ctx:
            lines = hdc_ctx.split('\n')
            parts = []
            for line in lines[:3]:
                if 'sim=' in line:
                    m = line.split('sim=')[-1].split('|')
                    if len(m) > 1:
                        parts.append(m[1].strip()[:60])
            if parts:
                spoken = " | ".join(parts)
            else:
                spoken = "(calada)"
        else:
            spoken = "(calada)"

    print(f"  AGI ({dt:.0f}ms): {spoken}")
    print(f"  [pe={pred_error:.3f}]")
    print()

print("=" * 60)
print("TESTE 2: Geracao Autoregressiva (/generate)")
print("=" * 60)

for prompt in [
    "once upon a time there was a",
    "the cat sat on the",
    "artificial intelligence is",
    "def fibonacci",
    "def hello world",
    "return",
]:
    print(f"\n>>> /generate {prompt}")
    t1 = time.time()
    try:
        result = generator.generate(prompt, max_chunks=10, verbose=False)
        dt = time.time() - t1
        print(f"  AGI ({dt:.1f}s, {len(result)} chars):")
        print(f"  {result[:300]}")
    except Exception as e:
        print(f"  ERRO: {e}")
    print()

print("=" * 60)
print("TESTE 3: Memoria de Contexto (temporal buffer)")
print("=" * 60)

print("\n>>> Alimentando frases sequenciais...")
for frase in [
    "My name is Nova",
    "I am an artificial intelligence",
    "I live in a computer",
    "What is my name",
]:
    print(f"  Input: '{frase}'")
    thought = mind.think(frase)
    pe = thought.get('predictive_coding', {}).get('prediction_error', 0)
    print(f"  PE={pe:.3f}")
print()

print("=" * 60)
print("TESTE 4: Intuicao Numerica (NumericEncoder)")
print("=" * 60)
print(f"  sim(5,6)   = {num_encoder.similarity(5,6):.3f}")
print(f"  sim(5,10)  = {num_encoder.similarity(5,10):.3f}")
print(f"  sim(5,100) = {num_encoder.similarity(5,100):.3f}")

print("\n" + "=" * 60)
print("  TESTE CONCLUIDO")
print("=" * 60)
