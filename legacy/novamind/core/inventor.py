"""
Inventor — Gerador de Problemas Novos / Perguntas Absurdas

O maior salto cientifico veio porque alguem perguntou
o que ninguem perguntava.

O Inventor NAO resolve.
O Inventor CRIA PERGUNTAS.

Perguntas absurdas, contraintuitivas, fora-da-caixa.

Como:
  "E se a gravidade fosse repulsiva em algumas condicoes?"
  "O que acontece se combinarmos duas teorias que se contradizem?"
  "Ha algum fenomeno que observamos mas nao conseguimos explicar?"
  "Qual seria a relacao entre X e Y se descobrissemos que Z?"
  "Se eu trocar a representacao, a mesma relacao aparece?"

O sistema usa o UnknownSpace + ExplainabilitySpace + CognitiveTension
para gerar perguntas onde ha lacunas de conhecimento.
"""
import time, uuid, hashlib, random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


PRIMORDIAL_QUESTIONS = [
    "E se a gravidade fosse repulsiva em algumas condicoes?",
    "Qual fenomeno observamos mas nao conseguimos explicar?",
    "O que acontece se combinarmos duas teorias que se contradizem?",
    "Ha um padrao que aparece em dominios sem relacao aparente?",
    "Se a representacao mudar, a mesma relacao se mantem?",
    "O que aconteceria se invertermos completamente o problema?",
]


class Inventor:
    """Gerador de problemas e perguntas novas.

    Nao resolve. Inventa.
    """

    def __init__(self, cge=None, unknown=None, explainability=None,
                 tension=None):
        self.cge = cge
        self.unknown = unknown
        self.explainability = explainability
        self.tension = tension
        self.questions: List[Dict] = []
        self._question_seeds = list(PRIMORDIAL_QUESTIONS)

    def generate_question(self) -> Optional[Dict]:
        """Gera uma nova pergunta baseada em lacunas observadas."""
        candidates = []

        # De UnknownSpace: gaps estruturais
        if self.unknown is not None:
            frontier = self.unknown.get_frontier(3)
            for region in frontier:
                a, b = region.get("concept_a", "?"), region.get("concept_b", "?")
                if a and b and a != "?":
                    candidates.append(f"Qual a relacao entre {a} e {b}?")
                    candidates.append(f"E se {a} fosse uma instancia de {b}?")

        # De ExplainabilitySpace: fenomenos previstos mas nao explicados
        if self.explainability is not None:
            gaps = self.explainability.get_open_gaps(3)
            for gap in gaps:
                candidates.append(f"Por que {gap.phenomenon[:30]} acontece?")

        # De CognitiveTension: contradicoes
        if self.tension is not None:
            press = self.tension.get_pressing_tensions(3)
            for t in press:
                candidates.append(f"Como resolver: {t.description[:40]}?")

        # De CGE: nos com poucas conexoes
        if self.cge is not None and len(self.cge.nodes) > 10:
            isolated = sorted(self.cge.nodes.items(),
                              key=lambda x: len(x[1].get("vector", [])))[:3]
            for nid, _ in isolated:
                if nid and not nid.startswith("_"):
                    candidates.append(f"O que conecta {nid} ao resto do conhecimento?")

        if not candidates:
            return None

        # Escolhe uma pergunta com exploracao
        q = random.choice(candidates)
        return {"question": q, "source": "inventor",
                "created_at": time.time()}

    def invent_problem(self) -> Optional[Dict]:
        """Inventa um PROBLEMA novo (diferente de pergunta).

        Um problema tem:
          - objetivo (o que queremos)
          - restricoes (o que nao podemos)
          - estado inicial
        """
        if self.cge is None:
            return None
        # Pega um topico do CGE como estado inicial
        nodes = [n for n in self.cge.nodes.keys()
                 if not n.startswith("_")]
        if len(nodes) < 2:
            return None
        a, b = random.sample(nodes, 2)
        return {
            "type": "invented_problem",
            "initial_state": a,
            "target": b,
            "constraint": "minimizar recursos cognitivos",
            "created_at": time.time(),
        }

    def store_question(self, question: Dict):
        """Armazena uma pergunta no log."""
        self.questions.append(question)
        if len(self.questions) > 100:
            self.questions = self.questions[-100:]

    def get_unanswered_questions(self, max_age_seconds: float = 86400) -> List[Dict]:
        """Retorna perguntas ainda nao respondidas (menos de 1 dia)."""
        now = time.time()
        return [q for q in self.questions
                if "answered" not in q and (now - q.get("created_at", 0)) < max_age_seconds]

    def mark_answered(self, question_text: str):
        """Marca uma pergunta como respondida."""
        for q in self.questions:
            if q.get("question") == question_text:
                q["answered"] = True
                q["answered_at"] = time.time()
                break

    def get_state(self) -> Dict:
        return {
            "n_questions": len(self.questions),
            "unanswered": len(self.get_unanswered_questions()),
            "recent": [q.get("question", "")[:60] for q in self.questions[-3:]],
        }
