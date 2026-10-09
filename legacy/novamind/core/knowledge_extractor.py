"""
KnowledgeExtractor — Transforma texto em conhecimento estruturado.

Pipeline:
  Texto → sentencas → parser → entidades + relacoes + hipoteses + excecoes
  → triplas (sujeito, verbo, objeto, contexto)
  → nos CGE + arestas CAUSES/IS_A/CONTRADICTS
  → deteccao de conflitos com conhecimento existente

Diferenca do encoder simples (structured_encoder):
  - Processa multiplas sentencas
  - Detecta hipoteses (if X then Y, because, therefore)
  - Detecta excecoes (unless, except, however)
  - Extrai contexto alem da frase individual
  - Detecta conflitos com o CGE existente
"""
import sys, os, json, time, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from .structured_encoder import StructuredHDVEncoder, word_to_hdv, bind, bundle, ROLES


class KnowledgeExtractor:
    """Extrai conhecimento estruturado de textos e integra ao CGE."""

    def __init__(self):
        self.encoder = StructuredHDVEncoder()
        self._seen_facts = set()

    # ─── Pipeline principal ─────────────────────────────────

    def process(self, text: str) -> dict:
        """Texto completo → conhecimento estruturado.

        Returns:
            dict com facts, hypotheses, exceptions, conflicts
        """
        sentences = self._split_sentences(text)
        all_facts = []
        all_hypotheses = []
        all_exceptions = []
        context = {"current_topic": None, "entities": {}}

        for sent in sentences:
            if not sent.strip():
                continue

            # Extrai triplas da sentenca
            triples = self._extract_triples(sent, context)

            # Detecta se e hipotese
            is_hypothesis = any(w in sent.lower() for w in
                                ["if ", "then ", "therefore", "because", "since", "implies",
                                 "if and only if", "iff ", "→", "=>"])

            # Detecta se e excecao
            is_exception = any(w in sent.lower() for w in
                               ["unless", "except", "however", "although", "but ",
                                "nevertheless", "nonetheless", "contrary"])

            for triple in triples:
                if is_hypothesis:
                    all_hypotheses.append(triple)
                elif is_exception:
                    all_exceptions.append(triple)
                else:
                    all_facts.append(triple)

            # Atualiza contexto
            for triple in triples:
                if triple[0]:  # sujeito
                    context["current_topic"] = triple[0]
                    context["entities"][triple[0]] = True
                if triple[2]:  # objeto
                    context["entities"][triple[2]] = True

        return {
            "facts": all_facts,
            "hypotheses": all_hypotheses,
            "exceptions": all_exceptions,
            "sentences": len([s for s in sentences if s.strip()]),
            "context": context,
        }

    # ─── Extração de triplas ───────────────────────────────

    def _split_sentences(self, text: str) -> list:
        """Divide texto em sentencas."""
        text = text.replace("\n", " ")
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def _extract_triples(self, sentence: str, context: dict) -> list:
        """Extrai triplas (sujeito, verbo, objeto) de uma sentenca."""
        hdv, meta = self.encoder.encode(sentence)
        triples = []

        # Tripla principal da sentenca
        subj = meta.get("subject") or context.get("current_topic")
        verb = meta.get("verb")
        obj = meta.get("object")

        if subj and verb and obj:
            triples.append((subj, verb, obj, "fact"))
        elif subj and verb:
            triples.append((subj, verb, "", "fact"))
        elif subj and not verb and obj:
            # "X is Y" → relacao IS_A
            triples.append((subj, "is", obj, "is_a"))

        # Relacoes compostas (and, also)
        if "and" in sentence.lower():
            parts = sentence.lower().split("and")
            for part in parts:
                hdv2, meta2 = self.encoder.encode(part)
                s2 = meta2.get("subject")
                v2 = meta2.get("verb")
                o2 = meta2.get("object")
                if s2 and v2 and o2 and (s2, v2, o2) != (subj, verb, obj):
                    triples.append((s2, v2, o2, "compound"))

        return triples

    # ─── Integracao com CGE ────────────────────────────────

    def integrate_into_cge(self, knowledge: dict, cge) -> dict:
        """Integra conhecimento extraido no CGE.

        Cria nos e arestas. Detecta conflitos.
        """
        results = {"nodes_created": 0, "edges_created": 0, "conflicts": 0, "conflict_details": []}

        for triple in knowledge.get("facts", []):
            subj, verb, obj, rel_type = triple
            results["nodes_created"] += self._ensure_node(cge, subj)
            if obj:
                results["nodes_created"] += self._ensure_node(cge, obj)
                # Verifica conflito com conhecimento existente
                conflict = self._detect_conflict(cge, subj, verb, obj)
                if conflict:
                    results["conflicts"] += 1
                    results["conflict_details"].append(conflict)
                # Cria aresta
                if self._add_edge(cge, subj, obj, verb, rel_type):
                    results["edges_created"] += 1

        # Hipotese: arestas com confianca menor
        for triple in knowledge.get("hypotheses", []):
            subj, verb, obj, rel_type = triple
            results["nodes_created"] += self._ensure_node(cge, subj)
            if obj:
                results["nodes_created"] += self._ensure_node(cge, obj)
                if self._add_edge(cge, subj, obj, verb, rel_type, confidence=0.3):
                    results["edges_created"] += 1

        # Excecoes: arestas de CONTRADICAO
        for triple in knowledge.get("exceptions", []):
            subj, verb, obj, rel_type = triple
            if subj and obj and cge and hasattr(cge, "EDGE_CAUSES"):
                try:
                    # Marca como excecao (flag no node)
                    if subj in cge.nodes:
                        cge.nodes[subj]["exception"] = True
                    if obj in cge.nodes:
                        cge.nodes[obj]["exception"] = True
                except:
                    pass

        return results

    def _ensure_node(self, cge, name: str) -> int:
        """Cria no CGE se nao existir."""
        if cge is None or not name or name in cge.nodes:
            return 0
        try:
            hdv = word_to_hdv(name)
            import torch
            vec = torch.from_numpy(hdv).float()
            cge.create_node(name, semantic_vec=vec)
            cge.nodes[name]["confidence"] = 0.5
            return 1
        except Exception:
            return 0

    def _add_edge(self, cge, subj, obj, verb, rel_type, confidence=0.5) -> bool:
        """Adiciona aresta no CGE."""
        if cge is None or subj not in cge.nodes or obj not in cge.nodes:
            return False
        try:
            edge_type = cge.EDGE_CAUSES if hasattr(cge, "EDGE_CAUSES") else None
            if edge_type is None:
                return False
            # Nao duplica arestas
            for e in cge.edges[-200:]:
                if e.get("from") == subj and e.get("to") == obj:
                    return False
            cge.add_edge(subj, obj, edge_type)
            # Confianca baseada no tipo de relacao
            if subj in cge.nodes:
                cge.nodes[subj]["confidence"] = max(
                    cge.nodes[subj].get("confidence", 0.5), confidence)
            return True
        except Exception:
            return False

    def _detect_conflict(self, cge, subj, verb, obj) -> dict:
        """Detecta se o novo conhecimento conflita com o existente."""
        if cge is None or subj not in cge.nodes:
            return None

        # Verifica se ja existe uma aresta contraditoria
        for e in cge.edges:
            if e.get("from") == subj:
                existing_obj = e.get("to")
                if existing_obj and existing_obj != obj:
                    # Conflito: mesmo sujeito, objetos diferentes
                    if verb in ("is", "has", "contains", "causes"):
                        return {
                            "subject": subj,
                            "existing": existing_obj,
                            "new": obj,
                            "verb": verb,
                        }
        return None

    # ─── Processamento de artigo completo ──────────────────

    def process_article(self, title: str, body: str, cge=None) -> dict:
        """Processa um artigo completo (titulo + corpo) e integra no CGE."""
        full_text = f"{title}. {body}"
        knowledge = self.process(full_text)

        results = {
            "title": title,
            "knowledge": knowledge,
            "cge_integration": {"nodes": 0, "edges": 0, "conflicts": 0},
        }

        if cge:
            integration = self.integrate_into_cge(knowledge, cge)
            results["cge_integration"] = integration

        return results

    # ─── Conflito entre artigos ────────────────────────────

    def compare_articles(self, article1: dict, article2: dict) -> list:
        """Compara dois artigos e retorna conflitos entre eles."""
        k1 = article1.get("knowledge", {})
        k2 = article2.get("knowledge", {})
        conflicts = []

        # Mesmo sujeito, objetos diferentes
        facts1 = {(f[0], f[1], f[2]) for f in k1.get("facts", [])}
        facts2 = {(f[0], f[1], f[2]) for f in k2.get("facts", [])}

        for f1 in facts1:
            for f2 in facts2:
                if f1[0] == f2[0] and f1[1] == f2[1] and f1[2] != f2[2]:
                    conflicts.append({
                        "type": "contradiction",
                        "subject": f1[0], "verb": f1[1],
                        "article1_value": f1[2],
                        "article2_value": f2[2],
                    })
        return conflicts
