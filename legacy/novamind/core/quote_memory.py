"""
QuoteMemory — Memoria de Frases Exatas

O CGE guarda SIGNIFICADO.
QuoteMemory guarda a FRASE ORIGINAL com contexto.

Diferenca:
  CGE: "Einstein defendia a relatividade" (significado)
  QuoteMemory: "Einstein escreveu: 'A imaginacao e mais importante que o conhecimento'" (citacao)

Cada quote tem:
  - texto original
  - autor/persona
  - fonte (livro, artigo, conversa)
  - timestamp
  - confianca
  - ligacoes a conceitos no CGE
"""
import time, uuid, hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class Quote:
    """Uma citacao exata."""
    id: str = ""
    text: str = ""
    author: str = ""
    source: str = ""           # "wikipedia", "book", "conversation", "article"
    source_url: str = ""
    context: str = ""          # contexto em que foi dita/escrita
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.5
    importance: float = 0.5    # 0..1, quao importante
    cge_links: List[str] = field(default_factory=list)  # conceitos no CGE

    def to_dict(self) -> Dict:
        return {"id": self.id[:8], "text": self.text[:120],
                "author": self.author, "source": self.source,
                "confidence": round(self.confidence, 2)}


PRIMORDIAL_QUOTES = [
    Quote(text="A imaginacao e mais importante que o conhecimento", author="Albert Einstein",
          source="entrevista", importance=0.9, confidence=0.95,
          cge_links=["imagination", "knowledge"]),
    Quote(text="Deus nao joga dados", author="Albert Einstein",
          source="carta a Max Born", importance=0.8, confidence=0.9,
          cge_links=["quantum", "determinism"]),
    Quote(text="A coisa mais bela que podemos experimentar e o misterioso", author="Albert Einstein",
          source="o mundo como eu vejo", importance=0.85, confidence=0.9,
          cge_links=["mystery", "beauty"]),
    Quote(text="Nao e que eu seja tao inteligente, e que eu fico mais tempo com os problemas", author="Albert Einstein",
          source="entrevista", importance=0.85, confidence=0.85,
          cge_links=["persistence", "intelligence"]),
    Quote(text="Se eu vi mais longe, foi por estar sobre ombros de gigantes", author="Isaac Newton",
          source="carta a Robert Hooke", importance=0.9, confidence=0.95,
          cge_links=["science", "progress"]),
    Quote(text="A duvida e o principio da sabedoria", author="Aristoteles",
          source="metafora", importance=0.85, confidence=0.85,
          cge_links=["doubt", "wisdom"]),
    Quote(text="So sei que nada sei", author="Socrates",
          source="citacao classica", importance=0.9, confidence=0.9,
          cge_links=["knowledge", "humility"]),
    Quote(text="Penso, logo existo", author="Rene Descartes",
          source="discurso do metodo", importance=0.9, confidence=0.95,
          cge_links=["thought", "existence"]),
    Quote(text="O que nao me mata me fortalece", author="Friedrich Nietzsche",
          source="crepusculo dos idolos", importance=0.85, confidence=0.9,
          cge_links=["strength", "adversity"]),
    Quote(text="A liberdade consiste em poder fazer tudo que nao prejudica os outros", author="John Stuart Mill",
          source="sobre a liberdade", importance=0.8, confidence=0.85,
          cge_links=["freedom", "ethics"]),
]


class QuoteMemory:
    """Banco de citacoes.

    Cada quote e uma frase exata com metadata.
    A AGI consulta QuoteMemory quando precisa citar alguem textualmente.
    """

    def __init__(self, cge=None):
        self.cge = cge
        self.quotes: Dict[str, Quote] = {}
        for q in PRIMORDIAL_QUOTES:
            qid = f"quote_{hashlib.md5(q.text.encode()).hexdigest()[:8]}"
            q.id = qid
            self.quotes[qid] = q

    def add(self, text: str, author: str = "", source: str = "?",
            context: str = "", importance: float = 0.5,
            confidence: float = 0.5, cge_links=None) -> str:
        """Adiciona uma citacao."""
        qid = f"quote_{hashlib.md5(text.encode()).hexdigest()[:8]}"
        if qid in self.quotes:
            self.quotes[qid].confidence = max(self.quotes[qid].confidence, confidence)
            return qid
        quote = Quote(id=qid, text=text, author=author, source=source,
                     context=context, importance=importance, confidence=confidence,
                     cge_links=cge_links or [])
        self.quotes[qid] = quote
        return qid

    def search(self, query: str, top_k: int = 3) -> List[Quote]:
        """Busca quotes por texto, autor ou conceito."""
        q = query.lower()
        results = []
        for quote in self.quotes.values():
            score = 0
            if q in quote.text.lower():
                score += 2
            if q in quote.author.lower():
                score += 1.5
            if any(q in link.lower() for link in quote.cge_links):
                score += 1
            if score > 0:
                results.append((score, quote))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:top_k]]

    def quote(self, concept: str) -> Optional[Quote]:
        """Retorna a melhor citacao para um conceito."""
        best = [q for q in self.quotes.values() if concept in q.cge_links]
        if not best:
            return None
        return max(best, key=lambda q: q.importance * q.confidence)

    def get_state(self) -> Dict:
        return {"n_quotes": len(self.quotes),
                "recent": [q.to_dict() for q in list(self.quotes.values())[-3:]]}
