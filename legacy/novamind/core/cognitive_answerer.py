"""
CognitiveAnswerer — Answerer Composicional.
Substitui o gerador estatístico (W/MLP) por composição direta via bind HDC.

Pipeline:
  [Pergunta] → encode_text → vetor A
             → extrai conceitos atômicos → vetor B
             → compose(A, B) = sign(bundle)
             → query HDV com composed → conceitos-na-interseção
             → compõe resposta textual
"""

import torch, re, unicodedata, time
from typing import List


class CognitiveAnswerer:

    def __init__(self, engine, binder, decoder, memory, device='cuda'):
        self.engine = engine
        self.binder = binder
        self.decoder = decoder
        self.memory = memory
        self.device = device
        self.dim = 10000

        self.stopwords = frozenset({
            'the','a','an','is','are','was','were','be','been','has','have','had',
            'do','does','did','will','would','can','could','may','might','shall',
            'to','of','in','on','for','with','by','as','at','from','into','through',
            'and','or','but','if','because','so','than','that','this','these','those',
            'i','you','he','she','it','we','they','me','him','her','us','them',
            'my','your','his','its','our','their','no','not','very','just',
            'what','when','where','why','how','who','which','a','about','after','also',
            'be','been','being','between','both','each','few','has','have','its',
            'more','most','other','some','such','than','too','up','very','was','were',
        })

    def extract_keywords(self, text: str, max_kw: int = 4) -> List[str]:
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        words = re.findall(r'[a-zA-Z]{4,}', text.lower())
        return [w for w in words if w not in self.stopwords][:max_kw]

    def encode(self, text: str):
        return self.engine.encode_text(text).tensor.float()

    def answer(self, question: str, top_k: int = 5) -> str:
        """
        answer() padrão: busca direta no HDV, fallback textual.
        """
        t0 = time.time()
        q_mf = self.engine.encode_text(question)
        results = self.memory.query_manifold(q_mf, top_k=top_k, min_sim=0.08)
        t1 = time.time()
        print(f"[answer] query={t1-t0:.3f}s results={len(results)}")

        if not results:
            return "Não encontrei conceitos relacionados na memória."

        used = set()
        parts = []
        for sim, meta, src in results:
            text = meta.get('text', '')
            if not text or text in used:
                continue
            used.add(text)
            words = ' '.join(text.split()[:8])
            parts.append(words)
            if len(parts) >= top_k:
                break

        return '. '.join(parts) + '.'

    def answer_compositional(self, question: str, top_k: int = 5) -> str:
        """
        Composição vetorial limpa:
          1. Codifica a pergunta INTEIRA como vetor semântico
          2. Extrai keywords e codifica cada uma como vetor
          3. Bundle da pergunta + keywords → vetor composto
          4. Query HDV com o vetor composto
          5. Retorna conceitos na interseção
        """
        # 1. Codifica a pergunta inteira (preserva contexto completo)
        q_vec = self.encode(question)

        # 2. Extrai e codifica keywords
        keywords = self.extract_keywords(question, max_kw=3)
        kw_vecs = {}
        for kw in keywords:
            try:
                kw_vecs[kw] = self.encode(kw)
            except Exception:
                continue

        # 3. Bundle: pergunta + keywords
        composed = q_vec.clone()
        for vec in kw_vecs.values():
            composed = composed + vec
        composed = torch.sign(composed)

        from core.coglang import Manifold
        composed_mf = Manifold(composed, device=torch.device(self.device))

        # 4. Query HDV com o vetor composto
        results = self.memory.query_manifold(composed_mf, top_k=top_k * 2, min_sim=0.06)

        if not results:
            return self.answer(question, top_k=top_k)

        # 5. Pontua cada resultado: similaridade com a pergunta + média das keywords
        scored = []
        for sim, meta, src in results:
            text = meta.get('text', '')
            if not text:
                continue
            c_vec = self.encode(text[:200])
            sim_q = torch.dot(c_vec, q_vec).item() / self.dim
            kw_avg = 0.0
            if kw_vecs:
                kw_sims = [torch.dot(c_vec, v).item() / self.dim for v in kw_vecs.values()]
                kw_avg = sum(kw_sims) / len(kw_sims)
            # Score = similaridade com a pergunta + bônus por tocar em keywords
            score = sim_q + kw_avg * 0.5
            scored.append((score, sim_q, kw_avg, text))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 6. Monta resposta
        used = set()
        parts = []
        for score, sim_q, kw_avg, text in scored:
            if not text or text in used:
                continue
            used.add(text)
            words = ' '.join(text.split()[:8])
            parts.append(words)
            if len(parts) >= top_k:
                break

        if not parts:
            return self.answer(question, top_k=top_k)

        return '. '.join(parts) + '.'

    def decode_vector(self, vec):
        """Decodifica um vetor HDC de volta para palavras.
        Usa o vocabulário do HDV: encontra os conceitos mais similares e extrai palavras-chave."""
        from core.coglang import Manifold
        mf = Manifold(vec, device=torch.device(self.device))
        results = self.memory.query_manifold(mf, top_k=10, min_sim=0.0)

        # Peso = similaridade
        word_scores = {}
        for sim, meta, src in results:
            text = meta.get('text', '')
            if not text:
                continue
            import unicodedata, re
            text_ascii = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
            words = re.findall(r'[a-zA-Z]{4,}', text_ascii.lower())
            for w in set(words):
                if w not in self.stopwords:
                    word_scores[w] = word_scores.get(w, 0) + sim

        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return [w for w, s in sorted_words[:12]]

    def analogical_transfer(self, concept_a: str, concept_b: str, top_k: int = 5) -> str:
        """
        Transferência Analógica Pura:
          a ⊕ b → query HDV + decode vocabulary → o que está na interseção?

        Estratégia:
          1. Busca top-K conceitos de A e B separadamente
          2. Para cada conceito em A, mede similaridade com B
          3. Para cada conceito em B, mede similaridade com A
          4. Rankeia por harmônica (sim_A, sim_B) → conceitos na interseção
          5. Decodifica o vetor composto via vocabulário
        """
        a_mf = self.engine.encode_text(concept_a)
        b_mf = self.engine.encode_text(concept_b)
        a_vec = a_mf.tensor.float()
        b_vec = b_mf.tensor.float()

        # 1. Busca top-K de cada conceito individualmente
        results_a = self.memory.query_manifold(a_mf, top_k=15, min_sim=0.05)
        results_b = self.memory.query_manifold(b_mf, top_k=15, min_sim=0.05)

        # 2. Monta candidatos: conceitos similares a A ou B
        candidates = {}  # text -> (sim_a, sim_b, source)
        for sim, meta, src in results_a:
            text = meta.get('text', '')
            if text:
                candidates[text] = [sim, 0.0, src]
        for sim, meta, src in results_b:
            text = meta.get('text', '')
            if text:
                if text in candidates:
                    candidates[text][1] = sim
                else:
                    candidates[text] = [0.0, sim, src]

        # 3. Calcula similaridade de cada candidato com o OUTRO conceito
        candidate_list = []
        for text, (sim_a, sim_b, src) in candidates.items():
            if not text:
                continue
            c_vec = self.encode(text[:200])

            if sim_a == 0.0:
                sim_a = torch.dot(c_vec, a_vec).item() / self.dim
            if sim_b == 0.0:
                sim_b = torch.dot(c_vec, b_vec).item() / self.dim

            harmonic = 2 * sim_a * sim_b / (sim_a + sim_b + 1e-8)
            product = sim_a * sim_b
            candidate_list.append((harmonic, product, sim_a, sim_b, text[:200], src))

        # 4. Rankeia por harmônica (similaridade com AMBOS)
        candidate_list.sort(key=lambda x: x[0], reverse=True)

        print(f"[analogical] {concept_a} (+) {concept_b}:")
        for h, prod, sa, sb, t, src in candidate_list[:8]:
            print(f"  H={h:.3f} P={prod:.3f} A={sa:.3f} B={sb:.3f} | {t[:60]}")

        # 5. Decodifica o vetor composto (bundle) para palavras
        bundle_vec = torch.sign(a_vec + b_vec)
        decoded_words = self.decode_vector(bundle_vec)
        print(f"[analogical] decoded: {decoded_words[:8]}")

        # 6. Monta resposta
        used = set()
        parts = []

        # Primeiro: conceitos que tocam em AMBOS (harmonic > threshold)
        for harmonic, prod, sim_a, sim_b, text, src in candidate_list:
            if not text or text in used:
                continue
            used.add(text)
            if harmonic > 0.06 and prod > 0.01:
                words = ' '.join(text.split()[:8])
                parts.append(words)
            if len(parts) >= top_k:
                break

        # Se não achou interseção, mostra os conceitos mais similares de cada lado
        if not parts:
            for harmonic, prod, sim_a, sim_b, text, src in candidate_list[:top_k]:
                if text and text not in used:
                    used.add(text)
                    words = ' '.join(text.split()[:8])
                    parts.append(words)

        # Adiciona palavras decodificadas como fechamento
        if decoded_words:
            kw_summary = ', '.join(decoded_words[:6])
            parts.append(f"(analogical keywords: {kw_summary})")

        return '. '.join(parts) + '.'
