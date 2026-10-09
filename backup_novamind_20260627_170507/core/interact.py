"""
Interact — Interface de Texto com a NovaMind
==============================================
REPL interativo com BrocaDecoder (fala emergente HDC).

Cada ciclo think() executa 13 camadas cognitivas + memória HDV.
A resposta da AGI é gerada pelo BrocaDecoder, que traduz o
manifold de pensamento em palavras do vocabulário aprendido.

Nenhum chatbot hardcoded. A fala emerge geometricamente.
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mind import NovaMind
from core.broca_decoder import BrocaDecoder
from core.sandbox_engine import SandboxEngine
from core.positional_binder import PositionalBinder
from core.autoregressive_generator import AutoregressiveGenerator, IntentionDrivenGenerator
from core.numeric_encoder import NumericEncoder
from core.temporal_buffer import TemporalBuffer


def main():
    print("=" * 60)
    print("  NovaMind — Interface de Texto")
    print("  Fala emergente via BrocaDecoder (HDC unbinding)")
    print("  13 camadas cognitivas + memoria HDV compilada")
    print("=" * 60)
    print()

    print("Inicializando NovaMind...")
    start = time.time()

    # Silencia warnings
    import warnings
    warnings.filterwarnings("ignore")

    mind = NovaMind()
    elapsed = time.time() - start
    print(f"NovaMind pronta em {elapsed:.1f}s")

    # BrocaDecoder: constroi vocabulario a partir da memoria HDV
    decoder = None
    if hasattr(mind, 'hdc_memory') and mind.hdc_memory is not None:
        stats = mind.hdc_memory.get_stats()
        print(f"Memoria HDV: {stats['total_items']} itens, {stats['total_vram_kb']:.0f} KB VRAM")

        # Constrói vocabulário do BrocaDecoder a partir dos metadados HDV
        all_index = []
        for idx_list in mind.hdc_memory.indices.values():
            all_index.extend(idx_list)

        if all_index:
            decoder = BrocaDecoder(mind.hdc_memory.engine)
            decoder.build_vocabulary_from_metadata(all_index)
            # Adiciona palavras extras comuns (incluindo curtas/stopwords
            # que o GroundingEngine filtra mas o PositionalBinder precisa)
            extras = (
                "o que e um uma voce eu ele ela nos eles isso aquila esta este sim nao ser ter estar poder fazer "
                "a an the is are was were to of in on for with by at and or this that it be has have "
                "do does did will would could should may might shall can not no but so if as up down out "
                "off over under again further then once here there where when why how all each every "
                "both few more most other some such only own same too very just also"
            )
            decoder.build_vocabulary_from_texts([extras])
            # Garante min_word_len=0 para incluir letras isoladas (ex: "a")
            decoder.min_word_len = 0
            print(f"BrocaDecoder pronto: {len(decoder.word_vectors)} palavras")

        # SandboxEngine: composição generativa de conceitos
        sandbox = SandboxEngine(mind.hdc_memory.engine)
        print(f"SandboxEngine pronto: composição generativa HDC")

        # PositionalBinder: fala fluida (sequência ordenada)
        binder = PositionalBinder(max_length=30, device=mind.hdc_memory.engine.device)
        print(f"PositionalBinder pronto: fala fluida posicional")

        # TemporalBuffer: memória de conversa (criado ANTES do generator)
        conv_buffer = TemporalBuffer(capacity=10, decay=0.8)
        print(f"TemporalBuffer pronto: memoria de conversa (cap=10, decay=0.8)")

        # AutoregressiveGenerator: geração chunk-a-chunk (recebe buffer temporal)
        gen_engine = mind.hdc_memory.engine
        gen_sandbox = SandboxEngine(gen_engine)
        generator = AutoregressiveGenerator(
            engine=gen_engine,
            binder=binder,
            decoder=decoder,
            sandbox=gen_sandbox,
            hdc_memory=mind.hdc_memory,
            temporal_buffer=conv_buffer
        )
        print(f"AutoregressiveGenerator pronto: geracao chunk-a-chunk + working HDV")

        # NumericEncoder: matemática no hiperespaço
        num_encoder = NumericEncoder(device=mind.hdc_memory.engine.device)
        print(f"NumericEncoder pronto: matematica geometrica HDC")
    else:
        sandbox = None
        binder = None
        generator = None
        num_encoder = None
        decoder = None
        print("Nenhum .hdv encontrado. Rode ingest_all() primeiro.")

    print()
    print("Comandos:")
    print("  /stats       — estatisticas internas")
    print("  /memory      — memoria HDV carregada")
    print("  /vocab       — palavras do vocabulario do BrocaDecoder")
    print("  /decode TEXTO — debug: mostra palavras do decoder para um texto")
    print("  /cycle N     — N ciclos de pensamento sem input")
    print("  /generate TEXTO — geracao chunk-a-chunk (IntentionDriven)")
    print("  /context     — estado do buffer temporal")
    print("  /sleep       — ciclo de sono forçado")
    print("  /math        — teste do NumericEncoder")
    print("  /quit        — sair")
    print()

    while True:
        try:
            user_input = input("voce: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAte logo!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Ate logo!")
            break

        if user_input == "/stats":
            print(json.dumps(mind.memory.get_stats(), indent=2, default=str))
            if hasattr(mind, 'predictor'):
                ps = mind.predictor.get_state()
                print(f"Pred error: {ps['prediction_error']:.3f}, precision: {ps['precision']:.2f}")
            continue

        if user_input == "/memory":
            if hasattr(mind, 'hdc_memory') and mind.hdc_memory is not None:
                stats = mind.hdc_memory.get_stats()
                print(f"Fontes: {stats['sources']}")
                print(f"Itens: {stats['total_items']}")
                print(f"VRAM: {stats['total_vram_kb']:.0f} KB")
                # Amostra dos indices
                for src, idx_list in mind.hdc_memory.indices.items():
                    print(f"\n  {src} ({len(idx_list)} itens):")
                    for m in idx_list[:3]:
                        txt = m.get('text', m.get('id', ''))[:80]
                        print(f"    - {txt}")
            else:
                print("HDCMemory nao carregado")
            continue

        if user_input == "/vocab":
            if decoder:
                words = sorted(decoder.word_vectors.keys())
                print(f"Vocabulario ({len(words)} palavras):")
                for i in range(0, len(words), 20):
                    print("  " + " ".join(words[i:i+20]))
            else:
                print("BrocaDecoder nao inicializado")
            continue

        if user_input.startswith("/decode"):
            text = user_input[8:].strip()
            if not text:
                print("Uso: /decode TEXTO")
                continue
            if decoder:
                results = decoder.decode_with_scores(text, top_k=5)
                print(f"Decodificando: '{text}'")
                for word, sim in results:
                    print(f"  {word}: {sim:.3f}")
            else:
                print("BrocaDecoder nao inicializado")
            continue

        if user_input.startswith("/cycle"):
            n = 1
            try:
                n = int(user_input.split()[1])
            except (IndexError, ValueError):
                pass
            for i in range(n):
                thought = mind.think("")
                pe = thought.get('predictive_coding', {}).get('prediction_error', 0)
                print(f"  ciclo {thought['cycle']}: pred_error={pe:.3f}")
            continue

        if user_input == "/context":
            if 'conv_buffer' in dir() and conv_buffer is not None:
                print(f"Buffer temporal: {len(conv_buffer.buffer)}/{conv_buffer.capacity} chunks")
                for i, (tick, _) in enumerate(conv_buffer.buffer):
                    print(f"  [{i}] tick={tick}")
            else:
                print("TemporalBuffer nao inicializado")
            continue

        if user_input == "/sleep":
            if hasattr(mind, 'archetypes'):
                try:
                    result = mind.archetypes.consolidate()
                    print(f"[SLEEP] Consolidacao concluida: {result}")
                except Exception as e:
                    print(f"[SLEEP] Erro: {e}")
            else:
                print("ArchetypeEngine nao disponivel")
            continue

        if user_input.startswith("/generate"):
            text = user_input[10:].strip()
            if not text:
                print("Uso: /generate TEXTO (ex: /generate write a poem about stars)")
                continue
            if 'generator' not in dir() or generator is None:
                print("AutoregressiveGenerator nao inicializado")
                continue

            # Enriquece intenção com contexto do buffer temporal
            enriched = text
            if 'conv_buffer' in dir() and conv_buffer is not None and conv_buffer.buffer:
                ctx_vec = conv_buffer.get_decayed()
                ctx_text = decoder.decode(
                    Manifold(ctx_vec, device=gen_engine.device),
                    top_k=3, min_sim=0.08
                )
                if ctx_text and ctx_text != "[...]":
                    enriched = f"{ctx_text} {text}"
                    if 'verbose' not in dir() or True:
                        print(f"  [CONTEXTO] Buffer: '{ctx_text}' -> intencao enriquecida")

            print(f"\n[GENERATOR] Alvo: '{text}'")
            print(f"  Intencao enriquecida: '{enriched}'")
            print("Gerando chunk-a-chunk (IntentionDriven -> Sandbox)...")
            result = generator.generate(enriched, max_chunks=30, verbose=True)
            print(f"\n[GENERATOR] Resultado ({len(result)} chars):")
            print(result)

            # Alimenta resultado no buffer temporal + working HDV
            if result != "[...]":
                generator.add_to_working_hdv(result)
                if 'conv_buffer' in dir() and conv_buffer is not None:
                    mf = gen_engine.encode_text(result[:200])
                    conv_buffer.push(mf.tensor)
            continue

        if user_input == "/math":
            if 'num_encoder' not in dir() or num_encoder is None:
                print("NumericEncoder nao inicializado")
                continue
            print("\n[NUMERIC ENCODER] Intuicao de magnitude (Jaccard ativos):")
            print(f"  sim(5, 5)   = {num_encoder.similarity(5, 5):.3f}  (1.0 = mesmo)")
            print(f"  sim(5, 6)   = {num_encoder.similarity(5, 6):.3f}  (~0.9 = quase igual)")
            print(f"  sim(5, 10)  = {num_encoder.similarity(5, 10):.3f}  (~0.7)")
            print(f"  sim(5, 20)  = {num_encoder.similarity(5, 20):.3f}  (~0.45)")
            print(f"  sim(5, 50)  = {num_encoder.similarity(5, 50):.3f}  (~0.25)")
            print(f"  sim(5, 100) = {num_encoder.similarity(5, 100):.3f}  (~0.17 = distante)")
            continue

        # Processa input normalmente
        start_t = time.time()

        # Alimenta input do usuário no buffer temporal (encode_text)
        if 'conv_buffer' in dir() and conv_buffer is not None:
            mf_in = gen_engine.encode_text(user_input[:200])
            conv_buffer.push(mf_in.tensor)

        thought = mind.think(user_input)
        elapsed_t = time.time() - start_t

        # Consolida imediatamente o output do think() na working HDV do generator
        response_text = thought.get('response', {}).get('text', '')
        if response_text and 'generator' in dir() and generator is not None:
            generator.add_to_working_hdv(response_text)
            generator.add_to_working_hdv(user_input)

        # Estado interno
        meta = thought.get('layers', {}).get('metacognition', {})
        emotion = meta.get('emotional_state', {})
        satisfaction = emotion.get('satisfaction', 0.5)
        confusion = emotion.get('confusion', 0.0)
        confidence = meta.get('confidence', {}).get('overall', 0.0)
        pred_error = thought.get('predictive_coding', {}).get('prediction_error', 0)

        # Contexto HDV
        hdc = thought.get('layers', {}).get('hdc_memory', {})
        hdc_ctx = hdc.get('context', '')

        # PositionalBinder: fala fluida chunked (Lei de Miller)
        spoken = ""
        if decoder and binder is not None:
            words_input = user_input.split()
            chunks = binder.encode_chunked_sequence(
                words_input, decoder.engine,
                chunk_size=4, min_word_len=2  # filtra "a" (1 letra)
            )
            if chunks:
                pos_seq = binder.decode_chunked_to_text(
                    chunks, decoder.vocab_matrix,
                    decoder.word_list, min_sim=0.08
                )
                if pos_seq != "[...]" and len(pos_seq) >= 2:
                    spoken = pos_seq

        # Fallback SVO/estruturado se positional não deu resultado
        if not spoken and decoder and sandbox is not None:
            composed_mf = sandbox.compose_concept(user_input)
            if composed_mf is not None:
                structured = sandbox.decode_roles(
                    composed_mf, decoder.vocab_matrix,
                    decoder.word_list, min_sim=0.12
                )
                if len(structured) >= 2:
                    order = {'suj': 0, 'ver': 1, 'obj': 2, 'adj': 3, 'adv': 4, 'pre': 5, 'neg': 6}
                    sorted_res = sorted(structured, key=lambda x: order.get(x[0], 99))
                    words_only = [w for _, w, _ in sorted_res]
                    spoken = " ".join(words_only)
                else:
                    spoken = decoder.decode(composed_mf, top_k=5, min_sim=0.08)
        elif not spoken and decoder:
            thought_mf = decoder.engine.encode_text(
                hdc_ctx[:500] if hdc_ctx else user_input[:500]
            )
            spoken = decoder.decode(thought_mf, top_k=5, min_sim=0.08)

        # Resposta padrao se BrocaDecoder nao disponivel/calou
        if not spoken or spoken == "[...]":
            spoken = format_thought_simple(thought, hdc_ctx)

        print(f"\nAGI ({elapsed_t*1000:.0f}ms):")
        print(f"  {spoken}")
        buf_status = ""
        if 'conv_buffer' in dir() and conv_buffer is not None:
            buf_status = f" buf={conv_buffer.usage*100:.0f}%"
        print(f"  [conf={confidence:.2f} sat={satisfaction:.2f} confusao={confusion:.2f} erro={pred_error:.3f}{buf_status}]")

        if confusion > 0.6:
            print("  >> ALTA CONFUSAO: AGI nao reconhece este conceito")

        print()


def format_thought_simple(thought, hdc_ctx):
    """Fallback quando BrocaDecoder nao tem palavras."""
    meta = thought.get('layers', {}).get('metacognition', {})
    emotion = meta.get('emotional_state', {})
    confusion = emotion.get('confusion', 0.0)

    if hdc_ctx:
        # Extrai as palavras mais similares do contexto HDV
        lines = hdc_ctx.split('\n')
        parts = []
        for line in lines[:3]:
            if 'sim=' in line:
                m = line.split('sim=')[-1].split('|')
                if len(m) > 1:
                    text_part = m[1].strip()[:60]
                    parts.append(text_part)
        if parts:
            return " | ".join(parts)

    if confusion > 0.6:
        return "Nao sei."
    else:
        return "Processando..."


if __name__ == "__main__":
    main()
