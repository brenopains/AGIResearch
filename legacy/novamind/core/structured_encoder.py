"""
StructuredHDVEncoder — Converte texto em HDV estruturado via parser sintatico.

Pipeline:
  Texto
    -> spacy parser (dep, POS, entity)
    -> extracao de papeis: sujeito, verbo, objeto, modifiers, negacao
    -> bind de cada papel com HDV do conceito
    -> bundle de todos os binds em HDV estruturado

Diferenca do encoder anterior (media spacy):
  - "cao morde homem" != "homem morde cao" (ordem preservada)
  - Suporta negacao ("nao", "nunca")
  - Suporta quantificadores ("todo", "algum", "nenhum")
  - Suporta condicionais ("se...entao")
  - Polissemia via contexto (bank+river != bank+finance)
"""
import numpy as np
import spacy
import torch
from typing import List, Optional, Tuple

# ─── Palavras que NUNCA devem virar conceitos ─────────────
_SKIP_WORDS = frozenset({
    "what", "why", "how", "when", "where", "who", "which", "whom",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "he", "she", "we", "you", "i", "me", "him", "her", "us",
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "on", "at", "by", "for", "with", "without",
    "from", "into", "through", "during", "before", "after",
    "and", "but", "or", "nor", "not", "so", "yet", "if", "because",
    "as", "until", "while", "about", "no", "yes", "just", "very",
})
_CONTENT_POS = frozenset({"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"})


def is_content_word(token) -> bool:
    return (token.pos_ in _CONTENT_POS
            and token.lemma_.lower() not in _SKIP_WORDS
            and len(token.lemma_) > 1)

# ─── ROLES (vetores aleatorios fixos para papeis semanticos) ──
np.random.seed(47)
ROLES = {}
for role_name in ["SUBJECT", "VERB", "OBJECT", "MODIFIER", "NEGATION",
                  "QUANTIFIER", "TENSE", "CAUSE", "CONDITION",
                  "LOCATION", "TIME", "POSSESSOR", "POSSESSED",
                  "ATTRIBUTE", "QUESTION", "COMPARATIVE", "PURPOSE", "MANNER"]:
    ROLES[role_name] = np.sign(np.random.randn(10000))


# ─── Config global (carregado uma vez) ─────────────────────
_NLP = None
_WORD_PROJ = None

def _ensure_nlp():
    global _NLP, _WORD_PROJ
    if _NLP is None:
        _NLP = spacy.load("en_core_web_lg")
        np.random.seed(46)
        _WORD_PROJ = np.random.randn(10000, 300).astype(np.float32) / np.sqrt(300)


def word_to_hdv(word: str) -> np.ndarray:
    """Converte palavra em HDV semantico via spacy word vector."""
    _ensure_nlp()
    doc = _NLP(word.lower().strip())
    vecs = [t for t in doc if t.has_vector and t.vector_norm > 0]
    if not vecs:
        return np.zeros(10000, dtype=np.float32)
    avg = np.mean([t.vector for t in doc if t.has_vector and t.vector_norm > 0], axis=0).astype(np.float32)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    hdv = np.sign(_WORD_PROJ @ avg)
    hdv[hdv == 0] = 1
    return hdv.astype(np.float32)


def bind(role_vec: np.ndarray, filler_hdv: np.ndarray) -> np.ndarray:
    """Bind: HDV papel * HDV conteudo (multiplicacao de Hadamard)."""
    return np.sign(role_vec * filler_hdv).astype(np.float32)


def unbind(bound_hdv: np.ndarray, role_vec: np.ndarray) -> np.ndarray:
    """Unbind: HDV ligado * HDV papel (recupera filler aproximado)."""
    return np.sign(bound_hdv * role_vec).astype(np.float32)


def bundle(hdvs: List[np.ndarray]) -> np.ndarray:
    """Bundle: soma + sign (media dos HDVs)."""
    if not hdvs:
        return np.zeros(10000, dtype=np.float32)
    result = np.sign(np.sum(hdvs, axis=0))
    result[result == 0] = 1
    return result.astype(np.float32)


class StructuredHDVEncoder:
    """Encoder que converte texto em HDV estruturado via parser."""

    def __init__(self):
        _ensure_nlp()
        self.nlp = _NLP
        self._word_cache = {}

    def encode(self, text: str) -> Tuple[np.ndarray, dict]:
        """Texto -> HDV estruturado + metadados.

        Returns:
            hdv: HDV estruturado (10000D bipolar)
            meta: dict com estrutura extraida (sujeito, verbo, objeto, etc.)
        """
        doc = self.nlp(text.strip())  # NAO usa lower() para preservar parser

        binds = []
        meta = {
            "raw": text,
            "tokens": [],
            "subject": None, "verb": None, "object": None,
            "modifiers": [], "negation": False,
            "quantifier": None, "tense": None,
            "entities": [],
        }

        for token in doc:
            meta["tokens"].append({
                "text": token.text, "lemma": token.lemma_,
                "pos": token.pos_, "dep": token.dep_,
                "head": token.head.text,
            })

        # ─── Extrair sujeito (apenas palavras de conteudo) ──
        subjects = [t for t in doc if t.dep_ in ("nsubj", "nsubjpass", "csubj") and is_content_word(t)]
        if subjects:
            subj = subjects[0]
            subj_hdv = self._get_word_hdv(subj.lemma_)
            binds.append(bind(ROLES["SUBJECT"], subj_hdv))
            meta["subject"] = subj.lemma_.lower()

        # ─── Extrair verbo principal (apenas se for conteudo) ──
        verbs = [t for t in doc if t.pos_ == "VERB" and is_content_word(t)]
        if verbs:
            verb = verbs[0]
            verb_hdv = self._get_word_hdv(verb.lemma_)
            binds.append(bind(ROLES["VERB"], verb_hdv))
            meta["verb"] = verb.lemma_.lower()
            meta["tense"] = verb.tag_

        # ─── Extrair objeto (apenas palavras de conteudo) ─────
        objs = [t for t in doc if t.dep_ in ("dobj", "pobj", "attr") and is_content_word(t)]
        if objs:
            obj = objs[0]
            obj_hdv = self._get_word_hdv(obj.lemma_)
            binds.append(bind(ROLES["OBJECT"], obj_hdv))
            meta["object"] = obj.lemma_.lower()

        # ─── Negacao ──────────────────────────────────────
        negs = [t for t in doc if t.dep_ == "neg"]
        if negs:
            binds.append(ROLES["NEGATION"])  # HDV de negacao
            meta["negation"] = True

        # ─── Quantificadores ──────────────────────────────
        for token in doc:
            if token.lemma_ in ("all", "every", "each", "some", "any", "no", "none"):
                q_hdv = self._get_word_hdv(token.lemma_)
                binds.append(bind(ROLES["QUANTIFIER"], q_hdv))
                meta["quantifier"] = token.lemma_

        # ─── Modificadores (adjetivos, adverbios) ──────────
        mods = [t for t in doc if t.pos_ in ("ADJ", "ADV") and t.dep_ in ("amod", "advmod", "acomp")]
        for mod in mods:
            mod_hdv = self._get_word_hdv(mod.lemma_)
            binds.append(bind(ROLES["MODIFIER"], mod_hdv))
            meta["modifiers"].append(mod.lemma_)

        # ─── Causa (because, since, therefore) ───────────────
        for token in doc:
            if token.lemma_ in ("because", "since", "therefore", "thus", "hence"):
                cause_hdv = self._get_word_hdv(token.lemma_)
                binds.append(bind(ROLES["CAUSE"], cause_hdv))

        # ─── Condicional (if, unless, provided) ──────────────
        for token in doc:
            if token.lemma_ in ("if", "unless", "provided", "whenever"):
                cond_hdv = self._get_word_hdv(token.lemma_)
                binds.append(bind(ROLES["CONDITION"], cond_hdv))

        # ─── Se nao tem estrutura, fallback para media simples ──
        if not binds:
            # Fallback: media dos vetores (comportamento antigo)
            vecs = [t.vector for t in doc if t.has_vector and t.vector_norm > 0]
            if vecs:
                avg = np.mean(vecs, axis=0).astype(np.float32)
                norm = np.linalg.norm(avg)
                if norm > 0:
                    avg = avg / norm
                hdv = np.sign(_WORD_PROJ @ avg)
                hdv[hdv == 0] = 1
                return hdv.astype(np.float32), meta
            return np.zeros(10000, dtype=np.float32), meta

        # Bundle final: soma todos os binds
        result = bundle(binds)
        return result, meta

    def _get_word_hdv(self, word: str) -> np.ndarray:
        """Retorna HDV para uma palavra (com cache)."""
        word = word.lower().strip()
        if word not in self._word_cache:
            self._word_cache[word] = word_to_hdv(word)
        return self._word_cache[word]

    def extract_facts(self, text: str) -> List[Tuple[str, str, str]]:
        """Extrai triplas (sujeito, verbo, objeto) do texto.

        Permite alimentar o CGE com relacoes estruturadas.
        """
        doc = self.nlp(text.lower().strip())
        facts = []
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subj = token.lemma_
                verb = token.head.lemma_
                obj = None
                for child in token.head.children:
                    if child.dep_ in ("dobj", "pobj", "attr", "prep"):
                        obj = child.lemma_
                        break
                facts.append((subj, verb, obj or ""))
        return facts


# ─── Teste ─────────────────────────────────────────────────
if __name__ == "__main__":
    encoder = StructuredHDVEncoder()

    tests = [
        "cat chases mouse",
        "mouse chases cat",
        "every mammal breathes air",
        "whale is a mammal",
        "if water boils it becomes steam",
        "the king and queen ruled the kingdom",
        "cats and dogs are mammals",
        "I do not like green eggs and ham",
    ]

    print("=== ENCODER ESTRUTURAL ===")
    for t in tests:
        hdv, meta = encoder.encode(t)
        facts = encoder.extract_facts(t)
        print(f"\n{t}")
        print(f"  Sujeito: {meta['subject']}")
        print(f"  Verbo:   {meta['verb']}")
        print(f"  Objeto:  {meta['object']}")
        print(f"  Negacao: {meta['negation']}")
        print(f"  Fatos:   {facts}")
        print(f"  HDV[:5]: {hdv[:5]}")

    # Teste: "cao morde homem" vs "homem morde cao"
    hdv1, _ = encoder.encode("dog bites man")
    hdv2, _ = encoder.encode("man bites dog")
    cos = np.dot(hdv1, hdv2) / (DIM)
    print(f"\n=== ORDEM PRESERVADA? ===")
    print(f"  'dog bites man' vs 'man bites dog'")
    print(f"  Cosine: {cos:.4f} (baixo = ordens diferentes)")

    # Teste: "bank" com contextos diferentes (polissemia)
    hdv3, m3 = encoder.encode("bank river water")
    hdv4, m4 = encoder.encode("bank money finance")
    cos2 = np.dot(hdv3, hdv4) / DIM
    print(f"\n=== POLISSEMIA ===")
    print(f"  'bank river water' vs 'bank money finance'")
    print(f"  Cosine: {cos2:.4f} (baixo = contextos diferentes)")
    print(f"  Contexto 1: subj={m3['subject']}, obj={m3['object']}")
    print(f"  Contexto 2: subj={m4['subject']}, obj={m4['object']}")
