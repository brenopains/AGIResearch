"""
Autoregressive Generator — Sistema 1 (Intuição) + Sistema 2 (Verificação)
==========================================================================
Duas estratégias complementares:

ESTRATÉGIA A: IntentionDrivenGenerator (PREFERIDA)
  - Fixa a intenção (target concept)
  - A cada passo, calcula ENERGIA RESIDUAL = intenção - já_dito
  - Busca chunk mais próximo do residual: buffer L1 → working HDV → HDV
  - Para quando completion > 0.85
  - Anti-loop: se repete, injeta ruído no residual

ESTRATÉGIA B: Verificador Sandbox (FALLBACK)
  - W matrix / MLP propõe chunk
  - Sandbox mede Free Energy
  - Se FE < 0.5, aceita; senão, fallback para chunk alvo
"""
import os, math, torch, torch.nn as nn
from typing import Optional, List, Tuple

from .coglang import Manifold, calculate_free_energy
from .grounding_engine import GroundingEngine
from .positional_binder import PositionalBinder
from .broca_decoder import BrocaDecoder
from .code_extractor import sanitize_code, desanitize_code

MLP_PATH = r'D:\novamind_data\transition_mlp.pt'
W_PATH = r'D:\novamind_data\transition_matrix.pt'
FE_ACCEPT = 0.5
COMPLETION_THRESHOLD = 0.85
ANTILOOP_NOISE = 0.1    # magnitude de ruído a cada repetição
WM_SIZE = 200            # max itens na working memory HDV


class HDCSequencePredictor(nn.Module):
    def __init__(self, dim=10000):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, 4000), nn.ReLU(), nn.Linear(4000, dim))
    def forward(self, x):
        return self.network(x)


# ═══════════════════════════════════════════════════════════════════
# IntentionDrivenGenerator — com buffer L1, working HDV, anti-loop
# ═══════════════════════════════════════════════════════════════════

class IntentionDrivenGenerator:
    """
    Geração orientada por Intenção (Energia Residual).

    Ordem de busca: buffer temporal L1 → working memory HDV → HDV
    Anti-loop: ruído crescente no residual a cada repetição.
    Consolidação: chunks gerados vão para working HDV.
    """

    def __init__(self, engine: GroundingEngine,
                 binder: PositionalBinder,
                 decoder: BrocaDecoder,
                 hdc_memory=None,
                 temporal_buffer=None,
                 working_hdv: dict = None,
                 working_hdv_keys: list = None):
        self.engine = engine
        self.binder = binder
        self.decoder = decoder
        self.device = engine.device
        self.dim = Manifold.DIMENSIONALITY
        self.hdc_memory = hdc_memory
        self.temporal_buffer = temporal_buffer

        # Working memory HDV: dicionário texto → vetor (compartilhada ou local)
        self.working_hdv = working_hdv if working_hdv is not None else {}
        self.working_hdv_keys = working_hdv_keys if working_hdv_keys is not None else []

    def _add_to_working_hdv(self, text: str):
        """Consolida chunk na working memory HDV.
        Usa engine.encode_text() para garantir mesmo espaço vetorial do residual.
        """
        key = text.lower().strip()
        if key in self.working_hdv:
            return
        if len(key) < 3:
            return
        try:
            mf = self.engine.encode_text(key)
            self.working_hdv[key] = mf.tensor.float().cpu()
            self.working_hdv_keys.append(key)
            if len(self.working_hdv_keys) > WM_SIZE:
                old = self.working_hdv_keys.pop(0)
                self.working_hdv.pop(old, None)
        except Exception:
            pass

    def _search_working_hdv(self, residual_vec: torch.Tensor,
                            top_k: int = 3) -> List[Tuple[float, str]]:
        """Busca na working memory HDV os chunks mais próximos do residual."""
        results = []
        for key, vec in self.working_hdv.items():
            sim = float(torch.nn.functional.cosine_similarity(
                residual_vec.unsqueeze(0),
                vec.to(self.device).unsqueeze(0)))
            if sim > 0.1:
                results.append((sim, key))
        results.sort(reverse=True)
        return results[:top_k]

    def generate(self, target_text: str,
                 max_chunks: int = 20,
                 chunk_size: int = 4,
                 min_sim: float = 0.08,
                 verbose: bool = True) -> str:
        # Sanitiza a intencao para alinhar com o espaco do HDV (code impls)
        sanitized_target = sanitize_code(target_text)
        intention_mf = self.engine.encode_text(sanitized_target)
        intention_vec = intention_mf.tensor.float()
        spoken_vec = torch.zeros(self.dim, device=self.device)
        generated = []
        generated_keys = []   # para anti-loop
        repetition_count = 0

        if verbose:
            print(f'  [INTENCAO] "{target_text[:50]}"')

        for step in range(max_chunks):
            residual_vec = intention_vec - spoken_vec
            residual_vec = torch.sign(residual_vec)

            # Métrica de SATURAÇÃO: o quanto da intenção ainda resta no residual
            # Quando residual ≈ ruído, cosine(residual, intention) ≈ 0 → intenção esgotada
            residual_energy = float(torch.nn.functional.cosine_similarity(
                residual_vec.unsqueeze(0), intention_vec.unsqueeze(0)))
            if residual_energy < 0.15:
                if verbose:
                    print(f'  [Chunk {step+1}] Intenção saturada (residual_energy={residual_energy:.2f})')
                break

            # Busca chunk: buffer L1 → working HDV → HDV
            chunk_text = self._search(residual_vec, chunk_size, min_sim)

            if not chunk_text:
                if verbose:
                    print(f'  [Chunk {step+1}] (busca vazia, residual_energy={residual_energy:.2f})')
                break

            # Anti-loop: se repetiu, injeta ruído no residual e repete busca
            if chunk_text.lower() in generated_keys:
                repetition_count += 1
                noise = torch.randn(self.dim, device=self.device) * ANTILOOP_NOISE * repetition_count
                residual_vec = torch.sign(residual_vec + noise)
                if verbose:
                    print(f'  [Chunk {step+1}] (repetição, ruído={ANTILOOP_NOISE * repetition_count:.2f})')
                chunk_text = self._search(residual_vec, chunk_size, min_sim)
                if not chunk_text or chunk_text.lower() in generated_keys:
                    if verbose:
                        print(f'  [Chunk {step+1}] (loop persistente, parando)')
                    break
                repetition_count = 0  # reset se quebrou o loop

            # Consolida na working HDV
            self._add_to_working_hdv(chunk_text)

            generated.append(chunk_text)
            generated_keys.append(chunk_text.lower())

            if verbose:
                print(f'  [Chunk {step+1}] {chunk_text}  [residual_energy={residual_energy:.2f}]')

            # Atualiza spoken_energy (mesmo espaco: encode_text)
            chunk_intention = self.engine.encode_text(
                " ".join(chunk_text.split()[:chunk_size]))
            spoken_vec = spoken_vec + chunk_intention.tensor.float()
            spoken_vec = torch.sign(spoken_vec)

        if not generated:
            return "[...]"
        sep = "\n" if ('def ' in target_text or 'class ' in target_text) else " "
        result = sep.join(generated)
        return desanitize_code(result)

    def _search(self, residual_vec: torch.Tensor,
                chunk_size: int,
                min_sim: float) -> Optional[str]:
        """
        Busca em cascata: buffer temporal L1 → working HDV → HDV.
        Retorna o texto do primeiro candidato encontrado.
        """
        # 1. BUFFER TEMPORAL L1 (mesmo espaço: encode_text)
        if self.temporal_buffer is not None and self.temporal_buffer.buffer:
            best_text, best_sim = None, -1
            for tick, buf_vec in self.temporal_buffer.buffer:
                buf_vec = buf_vec.to(self.device)
                sim = float(torch.nn.functional.cosine_similarity(
                    residual_vec.unsqueeze(0), buf_vec.unsqueeze(0)))
                if sim > 0.3 and sim > best_sim:
                    best_sim = sim
                    mf = Manifold(buf_vec, device=self.device)
                    decoded = self.decoder.decode(
                        mf, top_k=chunk_size, min_sim=min_sim)
                    if decoded and decoded != "[...]":
                        best_text = decoded
            if best_text:
                return best_text

        # 2. WORKING MEMORY HDV (threshold baixo para capturar conceitos recentes)
        wm_results = self._search_working_hdv(residual_vec)
        if wm_results:
            best_sim, best_key = wm_results[0]
            if best_sim > 0.15:
                return best_key

        # 3. HDV LONGO PRAZO (usa metadado completo, nao chunk_size)
        if self.hdc_memory is not None and self.hdc_memory.matrices:
            best_text, best_sim = None, -1
            for key, matrix in self.hdc_memory.matrices.items():
                index = self.hdc_memory.indices[key]
                sims = torch.matmul(matrix, residual_vec) / self.dim
                top_sim, top_idx = torch.topk(sims, min(5, len(index)))
                for i, idx_val in enumerate(top_idx):
                    sim = top_sim[i].item()
                    if sim < 0.1:
                        continue
                    meta = index[idx_val]
                    text = meta.get('text', meta.get('id', ''))
                    if not text or len(text) < 3:
                        continue
                    # Usa o texto completo do metadado (preserva informacao)
                    if sim > best_sim:
                        best_sim = sim
                        best_text = text
            if best_text:
                return best_text

        return None


# ═══════════════════════════════════════════════════════════════════
# AutoregressiveGenerator — tenta IntentionDriven, depois fallback
# ═══════════════════════════════════════════════════════════════════

class AutoregressiveGenerator:

    def __init__(self, engine: GroundingEngine,
                 binder: PositionalBinder,
                 decoder: BrocaDecoder,
                 **kwargs):
        self.engine = engine
        self.binder = binder
        self.decoder = decoder
        self.device = engine.device
        self.dim = Manifold.DIMENSIONALITY
        self.sandbox = kwargs.get('sandbox')
        if self.sandbox is None:
            from .sandbox_engine import SandboxEngine
            self.sandbox = SandboxEngine(engine)
        self.hdc_memory = kwargs.get('hdc_memory')
        self.temporal_buffer = kwargs.get('temporal_buffer')

        # Working memory HDV persistente (compartilhada entre generates)
        self.working_hdv = {}
        self.working_hdv_keys = []

    def add_to_working_hdv(self, text: str):
        """Consolida conceito na working HDV usando encode_text().
        Chamado pelo REPL após think() para que o generator veja conceitos novos."""
        key = text.lower().strip()
        if key in self.working_hdv or len(key) < 3:
            return
        try:
            mf = self.engine.encode_text(key)
            self.working_hdv[key] = mf.tensor.float().cpu()
            self.working_hdv_keys.append(key)
            if len(self.working_hdv_keys) > WM_SIZE:
                old = self.working_hdv_keys.pop(0)
                self.working_hdv.pop(old, None)
        except Exception:
            pass

        self.model = None
        self.W = None

        if os.path.exists(W_PATH):
            try:
                data = torch.load(W_PATH, map_location=self.device,
                                  weights_only=False)
                raw = data['W'] if isinstance(data, dict) else data
                self.W = raw.to(device=self.device, dtype=torch.float32)
                print(f'[GENERATOR] W carregada: {list(self.W.shape)}')
            except Exception as e:
                print(f'[GENERATOR] Erro ao carregar W: {e}')

        if os.path.exists(MLP_PATH):
            try:
                m = HDCSequencePredictor().to(self.device)
                m.load_state_dict(
                    torch.load(MLP_PATH, map_location=self.device,
                               weights_only=False)['model'])
                m.eval()
                self.model = m
                print(f'[GENERATOR] MLP carregada de {MLP_PATH}')
            except Exception as e:
                print(f'[GENERATOR] Erro ao carregar MLP: {e}')

    def generate(self, target_text: str,
                 max_chunks: int = 20,
                 chunk_size: int = 4,
                 min_sim: float = 0.08,
                 verbose: bool = True) -> str:
        """Tenta IntentionDriven primeiro. Se falhar, usa fallback W/MLP."""
        intent_gen = IntentionDrivenGenerator(
            self.engine, self.binder, self.decoder,
            self.hdc_memory, self.temporal_buffer,
            self.working_hdv, self.working_hdv_keys)
        result = intent_gen.generate(
            target_text, max_chunks, chunk_size, min_sim, verbose)
        if result != "[...]":
            return desanitize_code(result)
        return desanitize_code(self._fallback_generate(
            target_text, max_chunks, chunk_size, min_sim, verbose))

    def _fallback_generate(self, target_text: str,
                           max_chunks: int, chunk_size: int,
                           min_sim: float, verbose: bool) -> str:
        target_words = [w for w in target_text.split() if len(w) >= 2]
        target_chunks = self._split(target_words, chunk_size)
        if not target_chunks:
            return "[...]"

        chunk_idx = 0
        target_mf = self.binder.encode_sequence(
            target_chunks[chunk_idx], self.engine)
        working_memory = Manifold(
            torch.ones(1, self.dim, device=self.device))
        generated = []
        seen = set()

        for step in range(max_chunks):
            state = Manifold(
                torch.sign((working_memory + target_mf).tensor),
                device=self.device)

            proposed_text, proposed_mf = self._intuicao(state, chunk_size, min_sim)
            accepted_text, fe = self._verifica(
                proposed_text, proposed_mf, target_chunks, chunk_idx,
                target_text, chunk_size, min_sim)

            if accepted_text is None:
                if verbose:
                    print(f'  [Chunk {step+1}] (FE={fe:.2f} > {FE_ACCEPT}, fallback target)')
                accepted_text = " ".join(target_chunks[chunk_idx])

            key = accepted_text.lower()
            if key in seen:
                if verbose:
                    print(f'  [Chunk {step+1}] (dedup: {accepted_text})')
                chunk_idx += 1
                if chunk_idx >= len(target_chunks):
                    break
                target_mf = self.binder.encode_sequence(
                    target_chunks[chunk_idx], self.engine)
                continue
            seen.add(key)
            generated.append(accepted_text)
            if verbose:
                print(f'  [Chunk {step+1}] {accepted_text}  [FE={fe:.2f}]')

            p_mf = self.binder.encode_sequence(
                accepted_text.split()[:chunk_size], self.engine)
            working_memory = Manifold(
                torch.sign((working_memory + p_mf).tensor),
                device=self.device)
            chunk_idx += 1
            if chunk_idx >= len(target_chunks):
                break
            target_mf = self.binder.encode_sequence(
                target_chunks[chunk_idx], self.engine)

        if not generated:
            return "[...]"
        sep = "\n" if ('def ' in target_text or 'class ' in target_text) else " "
        return sep.join(generated)

    def _intuicao(self, state, chunk_size, min_sim):
        if self.W is not None:
            raw = self.W @ state.tensor
            p = Manifold(torch.sign(raw), device=self.device)
            words = self.binder.decode_sequence(
                p, self.decoder.vocab_matrix, self.decoder.word_list,
                min_sim=min_sim, max_words=chunk_size)
            if words:
                text = " ".join(w for w, _ in words)
                mf = self.binder.encode_sequence(text.split()[:chunk_size], self.engine)
                return (text, mf)
        if self.model is not None:
            with torch.no_grad():
                raw = self.model(state.tensor.float())
                raw = raw / raw.norm() * math.sqrt(self.dim)
            p = Manifold(raw, device=self.device)
            words = self.binder.decode_sequence(
                p, self.decoder.vocab_matrix, self.decoder.word_list,
                min_sim=min_sim, max_words=chunk_size)
            if words:
                text = " ".join(w for w, _ in words)
                mf = self.binder.encode_sequence(text.split()[:chunk_size], self.engine)
                return (text, mf)
        return (None, None)

    def _verifica(self, proposed_text, proposed_mf,
                  target_chunks, chunk_idx, target_text,
                  chunk_size, min_sim):
        context_words = []
        for ci in range(max(0, chunk_idx-2), min(len(target_chunks), chunk_idx+3)):
            context_words.extend(target_chunks[ci])
        context = " ".join(context_words) if context_words else target_text[:50]
        target_concept = self.sandbox.compose_concept(context)
        if proposed_text is None or proposed_mf is None:
            if target_concept is not None:
                return (None, 1.0)
            return (None, 0.0)
        if target_concept is not None:
            fe = calculate_free_energy(proposed_mf, target_concept)
        else:
            fe = 0.0
        if fe < FE_ACCEPT:
            return (proposed_text, fe)
        return (None, fe)

    @staticmethod
    def _split(words, chunk_size):
        return [words[i:i+chunk_size]
                for i in range(0, len(words), chunk_size)]
