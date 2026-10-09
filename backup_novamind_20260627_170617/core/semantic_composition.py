"""
SemanticCompositionEngine — Motor de Composição Semântica Puramente HDC
========================================================================
Substitui a Broca V2 (LSTM). Nenhuma estatística, nenhum next-token prediction.
Pura álgebra vetorial no hiperespaço.

Como funciona:
  1. RECEBE um vetor HDC (pensamento), mesmo que inédito
  2. DECOMPÕE em palavras conhecidas via cache de palavras × similaridade HDC
  3. ATRIBUI papéis sintáticos (Sujeito, Verbo, Objeto, Modificador)
     testando qual papel maximiza ressonância com o vetor original
  4. MONTAGEM: ordem canônica → Modificador + Sujeito + Verbo + Objeto
  5. ESCRITA LONGA: subtrai frases geradas da intenção original
     (energia residual = intention − accumulator), garantindo coerência

Referência: sent.txt + "A SUA INTUIÇÃO SOBRE A BROCA É TÃO" + 
"AQUI ESTÁ O MOTOR DE COMPOSIÇÃO SEM"
"""

import os
import re
import json
import torch
from typing import List, Optional, Tuple, Dict

DIM = 10000
CACHE_PATH = "D:/temp/composition_word_cache.json"


class SemanticCompositionEngine:
    def __init__(self, hdc_memory, engine, device: str = "cuda"):
        self.memory = hdc_memory
        self.engine = engine
        self.device = torch.device(device)

        self._word_cache: Dict[str, torch.Tensor] = {}
        self._cache_loaded = False

        self.ROLE_SUBJECT = torch.randn(DIM, device=self.device).sign()
        self.ROLE_VERB = torch.randn(DIM, device=self.device).sign()
        self.ROLE_OBJECT = torch.randn(DIM, device=self.device).sign()
        self.ROLE_MODIFIER = torch.randn(DIM, device=self.device).sign()

        print(f"[COMPOSE] Motor de Composição Semântica ativo (device={device})")

    def translate_thought(self, thought_vec: torch.Tensor) -> str:
        tv = thought_vec.flatten().to(self.device)
        ingredients = self._extract_ingredients(tv)
        if not ingredients:
            return "[conceito irredutivel]"
        return self._mount_sentence(tv, ingredients)

    def write_report(self, intention_text: str, num_sentences: int = 10) -> str:
        intention_vec = self.engine.encode_text(intention_text).tensor.flatten().to(self.device)
        accumulator = torch.zeros(DIM, device=self.device)
        full_text = []

        for i in range(num_sentences):
            residual = torch.sign(intention_vec - accumulator)
            completion = torch.cosine_similarity(
                accumulator.unsqueeze(0), intention_vec.unsqueeze(0)
            ).item()
            if completion > 0.90:
                break

            sentence = self.translate_thought(residual)
            if not sentence or sentence.startswith("[conceito"):
                break
            full_text.append(sentence)

            sentence_vec = self.engine.encode_text(sentence).tensor.flatten().to(self.device)
            accumulator = torch.sign(accumulator + sentence_vec)

        return " ".join(full_text)

    # ─── Cache de Palavras ──────────────────────────────────

    def build_word_cache(self, force_rebuild: bool = False):
        """
        Constrói cache de palavras a partir dos textos no HDV.
        Cada palavra única é codificada como HDC e armazenada.
        """
        if self._cache_loaded and not force_rebuild:
            return

        if not force_rebuild and self._load_cache():
            self._cache_loaded = True
            print(f"[COMPOSE] Cache carregado: {len(self._word_cache)} palavras")
            return

        seen = set()
        word_limit = 10000  # máximo de palavras únicas (evita OOM/Cache gigante)

        for src_key, index in self.memory.indices.items():
            for meta in index:
                text = meta.get("text") or meta.get("id") or ""
                for word in text.replace("\n", " ").split():
                    clean = word.strip(".,;:!?()[]\"'\"").lower()
                    clean = re.sub(r"[^a-z0-9\-\']", "", clean)
                    if len(clean) > 2 and clean not in seen:
                        seen.add(clean)
                        if len(seen) >= word_limit:
                            break
                if len(seen) >= word_limit:
                    break
            if len(seen) >= word_limit:
                break

        print(f"[COMPOSE] Indexando {len(seen)} palavras no hiperespaço...")
        for word in seen:
            try:
                wv = self.engine.encode_text(word).tensor
                self._word_cache[word] = wv
            except Exception:
                pass

        self._cache_loaded = True
        print(f"[COMPOSE] Cache construído: {len(self._word_cache)} palavras (sem persistência JSON)")

    def _extract_ingredients(self, thought_vec: torch.Tensor) -> List:
        if not self._cache_loaded:
            self.build_word_cache()

        if not self._word_cache:
            return []

        tv = thought_vec.flatten().float()
        scored = []

        for word, wv in self._word_cache.items():
            sim = float(torch.cosine_similarity(
                tv.unsqueeze(0), wv.flatten().float().unsqueeze(0)))
            if sim > 0.20:
                scored.append((word, sim))

        scored.sort(key=lambda x: -x[1])

        deduped = []
        seen_vecs = []
        for word, sim in scored:
            wv = self._word_cache[word]
            is_dup = False
            for existing_vec in seen_vecs:
                dup_sim = float(torch.cosine_similarity(
                    wv.flatten().float().unsqueeze(0),
                    existing_vec.flatten().float().unsqueeze(0)))
                if dup_sim > 0.92:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append((word, sim))
                seen_vecs.append(wv)
            if len(deduped) >= 5:
                break

        return [(w, self._word_cache[w], s) for w, s in deduped]

    def _mount_sentence(self, thought_vec: torch.Tensor,
                        ingredients: List) -> str:
        assigned = {}

        for word_text, word_vec, sim in ingredients:
            wv = word_vec.flatten().to(self.device)
            sim_sub = float(torch.cosine_similarity(
                thought_vec.unsqueeze(0), (wv * self.ROLE_SUBJECT).unsqueeze(0)))
            sim_verb = float(torch.cosine_similarity(
                thought_vec.unsqueeze(0), (wv * self.ROLE_VERB).unsqueeze(0)))
            sim_obj = float(torch.cosine_similarity(
                thought_vec.unsqueeze(0), (wv * self.ROLE_OBJECT).unsqueeze(0)))
            sim_mod = float(torch.cosine_similarity(
                thought_vec.unsqueeze(0), (wv * self.ROLE_MODIFIER).unsqueeze(0)))

            roles = [("subject", sim_sub), ("verb", sim_verb),
                     ("object", sim_obj), ("modifier", sim_mod)]
            best_role, best_sim = max(roles, key=lambda x: x[1])

            if best_role not in assigned or best_sim > assigned[best_role][1]:
                assigned[best_role] = (word_text, best_sim)

        parts = []
        for role in ["modifier", "subject", "verb", "object"]:
            if role in assigned:
                parts.append(assigned[role][0])

        return " ".join(parts) if parts else (ingredients[0][0] if ingredients else "")

    # ─── Persistência ───────────────────────────────────────

    def _save_cache(self):
        try:
            data = {word: vec.cpu().tolist() for word, vec in self._word_cache.items()}
            os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
            with open(CACHE_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[COMPOSE] Erro salvando cache: {e}")

    def _load_cache(self) -> bool:
        if not os.path.exists(CACHE_PATH):
            return False
        try:
            with open(CACHE_PATH) as f:
                data = json.load(f)
            for word, vec_list in data.items():
                self._word_cache[word] = torch.tensor(vec_list, device=self.device)
            return True
        except Exception:
            return False

    def _mount_sentence(self, thought_vec: torch.Tensor,
                        ingredients: List[Tuple[str, torch.Tensor, float]]) -> str:
        """
        Atribui papéis sintáticos por ressonância e monta a frase.
        """
        assigned = {}

        for word_text, word_vec, sim in ingredients:
            wv = word_vec.flatten().to(self.device)

            sim_sub = torch.cosine_similarity(thought_vec.unsqueeze(0),
                                              (wv * self.ROLE_SUBJECT).unsqueeze(0)).item()
            sim_verb = torch.cosine_similarity(thought_vec.unsqueeze(0),
                                               (wv * self.ROLE_VERB).unsqueeze(0)).item()
            sim_obj = torch.cosine_similarity(thought_vec.unsqueeze(0),
                                              (wv * self.ROLE_OBJECT).unsqueeze(0)).item()
            sim_mod = torch.cosine_similarity(thought_vec.unsqueeze(0),
                                              (wv * self.ROLE_MODIFIER).unsqueeze(0)).item()

            best_role = max(
                ("subject", sim_sub), ("verb", sim_verb),
                ("object", sim_obj), ("modifier", sim_mod),
                key=lambda x: x[1]
            )

            if best_role[0] not in assigned:
                assigned[best_role[0]] = (word_text, best_role[1])

        parts = []
        if "modifier" in assigned:
            parts.append(assigned["modifier"][0])
        if "subject" in assigned:
            parts.append(assigned["subject"][0])
        if "verb" in assigned:
            parts.append(assigned["verb"][0])
        if "object" in assigned:
            parts.append(assigned["object"][0])

        return " ".join(parts) if parts else ingredients[0][0] if ingredients else ""
