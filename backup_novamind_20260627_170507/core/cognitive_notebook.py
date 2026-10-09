"""
CognitiveNotebook — Quadro Branco Externalizado

Diferenca entre pensar e externalizar o pensamento.
Humanos desenham porque desenhar MUDA o pensamento.
Nao e comunicacao. E computacao.

Este modulo e o "bloco de anotacoes" da AGI.
Ela escreve, relê, e a Forja revisita.

Cada pagina:
  Problema -> Hipoteses -> Experimentos -> Resultados -> Insights
"""
import time, uuid, json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class NotebookEntry:
    type: str = ""  # "hypothesis", "experiment", "result", "insight", "question"
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.5
    source: str = ""  # que modulo gerou


@dataclass
class NotebookPage:
    id: str = ""
    title: str = ""
    problem: str = ""
    entries: List[NotebookEntry] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    cge_nodes: List[str] = field(default_factory=list)

    def add_entry(self, entry: NotebookEntry):
        self.entries.append(entry)
        self.last_accessed = time.time()

    def get_insights(self) -> List[NotebookEntry]:
        return [e for e in self.entries if e.type == "insight"]

    def get_hypotheses(self) -> List[NotebookEntry]:
        return [e for e in self.entries if e.type == "hypothesis"]

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "title": self.title, "problem": self.problem[:200],
            "entries": [{"type": e.type, "content": e.content[:200],
                        "confidence": e.confidence} for e in self.entries[-10:]],
            "tags": self.tags, "n_entries": len(self.entries),
            "created_at": self.created_at,
        }

    def summarize(self) -> str:
        n_hyps = len(self.get_hypotheses())
        n_insights = len(self.get_insights())
        return (f"[Page:{self.id[:6]}] {self.title} | "
                f"{n_hyps} hipoteses, {n_insights} insights, "
                f"{len(self.entries)} entradas")


class CognitiveNotebook:
    """O quadro branco / bloco de anotacoes da AGI.

    Paginas sao armazenadas no CGE como nos type='notebook_page'.
    A Forja pode revisitar paginas antigas para minerar padroes.
    """

    def __init__(self, cge=None):
        self.cge = cge
        self.pages: Dict[str, NotebookPage] = {}
        self.current_page_id: Optional[str] = None

    def new_page(self, title: str = "", problem: str = "") -> str:
        """Cria uma nova pagina no caderno."""
        page_id = f"nb_{uuid.uuid4().hex[:8]}"
        page = NotebookPage(
            id=page_id, title=title or f"Page {len(self.pages)+1}",
            problem=problem,
        )
        self.pages[page_id] = page
        self.current_page_id = page_id
        return page_id

    def write(self, type: str, content: str, tags=None,
              confidence: float = 0.5, source: str = ""):
        """Escreve uma entrada na pagina atual."""
        if self.current_page_id is None:
            self.new_page("auto", content[:60])
        page = self.pages.get(self.current_page_id)
        if page is None:
            return
        entry = NotebookEntry(
            type=type, content=content, tags=tags or [],
            confidence=confidence, source=source,
        )
        page.add_entry(entry)
        self._sync_to_cge(type, content, tags or [])

    def _sync_to_cge(self, type: str, content: str, tags: List[str]):
        """Sincroniza entrada com CGE para persistencia e mineracao."""
        if self.cge is None:
            return
        import hashlib
        node_id = f"_nb_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        if node_id not in self.cge.nodes:
            import torch
            g = torch.Generator()
            g.manual_seed(int(hashlib.md5(content.encode()).hexdigest()[:8], 16))
            vec = torch.sign(torch.randn(10000, generator=g))
            self.cge.create_node(node_id, vec)
        if node_id in self.cge.nodes:
            self.cge.nodes[node_id]["notebook_entry"] = {
                "type": type, "content": content[:500], "tags": tags,
                "timestamp": time.time(),
            }
            self.cge.nodes[node_id]["type"] = "notebook"

    def add_hypothesis(self, hypothesis: str, confidence: float = 0.3):
        self.write("hypothesis", hypothesis, confidence=confidence, source="self")

    def add_experiment(self, experiment: str):
        self.write("experiment", experiment, source="self")

    def add_result(self, result: str, confidence: float = 0.5):
        self.write("result", result, confidence=confidence, source="self")

    def add_insight(self, insight: str, confidence: float = 0.7):
        self.write("insight", insight, confidence=confidence, source="self")

    def add_question(self, question: str):
        self.write("question", question, source="self")

    def switch_page(self, page_id: str) -> bool:
        if page_id in self.pages:
            self.current_page_id = page_id
            return True
        return False

    def get_current_page(self) -> Optional[NotebookPage]:
        if self.current_page_id:
            return self.pages.get(self.current_page_id)
        return None

    def get_page(self, page_id: str) -> Optional[NotebookPage]:
        return self.pages.get(page_id)

    def search(self, query: str) -> List[NotebookPage]:
        q = query.lower()
        results = []
        for page in self.pages.values():
            if q in page.title.lower() or q in page.problem.lower():
                results.append(page)
                continue
            for entry in page.entries:
                if q in entry.content.lower():
                    results.append(page)
                    break
        return results

    def mine_insights(self, min_confidence: float = 0.6) -> List[str]:
        """Extrai todos os insights com confianca minima."""
        insights = []
        for page in self.pages.values():
            for entry in page.entries:
                if entry.type == "insight" and entry.confidence >= min_confidence:
                    insights.append(entry.content)
        return insights

    def get_state(self) -> Dict:
        return {
            "n_pages": len(self.pages),
            "current_page": self.get_current_page().to_dict() if self.get_current_page() else None,
            "total_entries": sum(len(p.entries) for p in self.pages.values()),
            "total_insights": len(self.mine_insights()),
            "page_titles": [p.title for p in self.pages.values()][-10:],
        }

    def persist_to_cge(self):
        """Salva indice do caderno no CGE."""
        if self.cge is None:
            return
        import torch
        idx_id = "_notebook_index"
        g = torch.Generator(); g.manual_seed(42)
        vec = torch.sign(torch.randn(10000, generator=g))
        self.cge.create_node(idx_id, vec)
        if idx_id in self.cge.nodes:
            self.cge.nodes[idx_id]["notebook_index"] = {
                "pages": [p.to_dict() for p in self.pages.values()],
                "updated_at": time.time(),
            }
