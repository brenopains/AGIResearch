"""
Teste pos-treino: o vies "the" quebrou?
========================================
Testa o AutoregressiveGenerator diretamente.
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from core.mind import NovaMind
from core.broca_decoder import BrocaDecoder
from core.sandbox_engine import SandboxEngine
from core.positional_binder import PositionalBinder
from core.autoregressive_generator import AutoregressiveGenerator
from core.numeric_encoder import NumericEncoder

print("Inicializando...", flush=True)
t0 = time.time()
mind = NovaMind()
engine = mind.hdc_memory.engine
decoder = None
all_index = []
for idx_list in mind.hdc_memory.indices.values():
    all_index.extend(idx_list)
if all_index:
    decoder = BrocaDecoder(engine)
    decoder.build_vocabulary_from_metadata(all_index)
    extras = "o que e um uma voce eu ele ela nos eles isso aquila esta este sim nao ser ter estar poder fazer a an the is are was were to of in on for with by at and or this that it be has have do does did will would could should may might shall can not no but so if as up down out off over under again further then once here there where when why how all each every both few more most other some such only own same too very just also"
    decoder.build_vocabulary_from_texts([extras])
    decoder.min_word_len = 0
    print(f"Decoder: {len(decoder.word_vectors)} palavras", flush=True)

binder = PositionalBinder(max_length=30, device=engine.device)
sandbox = SandboxEngine(engine)
generator = AutoregressiveGenerator(engine=engine, binder=binder, decoder=decoder, sandbox=sandbox, hdc_memory=mind.hdc_memory)

print(f"Pronto em {time.time()-t0:.1f}s\n", flush=True)

# TESTE 1: Geracao variada
print("=" * 50)
print("TESTE: GERACAO VARIADA")
print("=" * 50)
for prompt in [
    "once upon a time there was a",
    "the cat sat on the",
    "artificial intelligence is",
    "def fibonacci",
    "def hello",
]:
    print(f"\n>>> /generate {prompt}")
    t1 = time.time()
    try:
        result = generator.generate(prompt, max_chunks=8, verbose=False)
        dt = time.time() - t1
        palavras = result.split()[:8]
        pct_the = palavras.count("the") / max(len(palavras), 1) * 100
        print(f"  ({dt:.1f}s, {len(result)} chars): {result[:200]}")
        print(f"  >> 'the' rate: {pct_the:.0f}%  {'VIES QUEBRADO!' if pct_the < 40 else 'AINDA COM VIES'}")
    except Exception as e:
        print(f"  ERRO: {e}")

# TESTE 2: Conversa basica
print("\n" + "=" * 50)
print("TESTE: CONVERSA BASICA")
print("=" * 50)
for frase in [
    "what is your name",
    "hello how are you",
    "tell me about yourself",
]:
    print(f"\n>>> {frase}")
    t1 = time.time()
    thought = mind.think(frase)
    hdc_ctx = thought.get('layers', {}).get('hdc_memory', {}).get('context', '')
    spoken = ""
    if decoder and binder is not None:
        words_input = frase.split()
        chunks = binder.encode_chunked_sequence(words_input, decoder.engine, chunk_size=4, min_word_len=2)
        if chunks:
            pos_seq = binder.decode_chunked_to_text(chunks, decoder.vocab_matrix, decoder.word_list, min_sim=0.08)
            if pos_seq != "[...]" and len(pos_seq) >= 2:
                spoken = pos_seq
    if not spoken and decoder and sandbox is not None:
        mf = sandbox.compose_concept(frase)
        if mf is not None:
            spoken = decoder.decode(mf, top_k=5, min_sim=0.08)
    if not spoken or spoken == "[...]":
        spoken = "(calada)"
    dt = time.time() - t1
    print(f"  AGI ({dt*1000:.0f}ms): {spoken[:200]}")
    print()

print("=" * 50)
print("FIM DO TESTE")
print("=" * 50)
