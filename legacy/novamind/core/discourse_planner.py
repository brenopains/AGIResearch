"""
DiscoursePlanner — Planejador de Discurso

Antes de falar, planeja:
  - O que dizer (selecao de conteudo do CGE + quotes)
  - Em que ordem (narrativa)
  - Com que formato (texto, analogia, codigo)
  - Com que profundidade (baseado no audience model)

Nao gera texto. Planeja a ESTRUTURA.
Quem gera texto e o MentalTranslator / Broca.
"""
import time, uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class DiscourseSegment:
    """Um segmento do discurso planejado."""
    type: str = ""           # "definition", "analogy", "example", "quote", "visual", "code"
    concept: str = ""
    content: str = ""
    source: str = ""         # "cge", "quote_memory", "teaching_engine"
    order: int = 0

    def to_dict(self) -> Dict:
        return {"type": self.type, "concept": self.concept[:30],
                "content": self.content[:80], "order": self.order}


@dataclass
class DiscoursePlan:
    """Um plano de discurso completo."""
    id: str = ""
    topic: str = ""
    audience_profile: str = "leigo"
    segments: List[DiscourseSegment] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_segment(self, type: str, concept: str, content: str, source: str = ""):
        seg = DiscourseSegment(type=type, concept=concept, content=content,
                               source=source, order=len(self.segments))
        self.segments.append(seg)

    def to_dict(self) -> Dict:
        return {"topic": self.topic, "audience": self.audience_profile,
                "segments": [s.to_dict() for s in self.segments]}


class DiscoursePlanner:
    """Planeja a estrutura de uma comunicacao.

    Pipeline:
      1. Recebe intencao (explicar X)
      2. Consulta audience model
      3. Consulta teaching engine para estrategia
      4. Busca quotes relevantes
      5. Estrutura segmentos na ordem ideal
      6. Entrega plano para o Broca/MentalTranslator
    """

    def __init__(self, cge=None):
        self.cge = cge

    def plan(self, topic: str, profile_id: str = "leigo",
             teaching_strategy: str = "analogy",
             quotes: Optional[List[str]] = None) -> DiscoursePlan:
        """Cria um plano de discurso para explicar um topico."""
        plan = DiscoursePlan(
            id=f"plan_{uuid.uuid4().hex[:8]}",
            topic=topic, audience_profile=profile_id,
        )

        # 1. Definicao do conceito
        plan.add_segment("definition", topic, f"Explicando {topic}")

        # 2. Analogia ou exemplo
        if teaching_strategy == "analogy":
            plan.add_segment("analogy", topic, f"Analogia para {topic}")
        elif teaching_strategy == "code":
            plan.add_segment("code", topic, f"Exemplo de codigo para {topic}")
        elif teaching_strategy == "visual":
            plan.add_segment("visual", topic, f"Descricao visual de {topic}")

        # 3. Citacao relevante
        if quotes:
            for q in quotes[:1]:
                plan.add_segment("quote", topic, q, source="quote_memory")

        # 4. Conexao com outros conceitos (do CGE)
        if self.cge and topic in self.cge.nodes:
            neighbors = []
            for e in self.cge.edges[-100:]:
                if e["from"] == topic and e["to"] not in neighbors:
                    neighbors.append(e["to"])
                elif e["to"] == topic and e["from"] not in neighbors:
                    neighbors.append(e["from"])
            for n in neighbors[:3]:
                plan.add_segment("connection", n, f"Relacionado a {n}")

        return plan

    def get_state(self) -> Dict:
        return {"status": "ready"}
