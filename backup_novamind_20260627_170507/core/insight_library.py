"""
InsightLibrary — Biblioteca de Meta-Regras (Insights)

Nao armazena FATOS ("Terra orbita Sol").
Armazena PRINCIPIOS ("decompor antes de atacar aumenta taxa de sucesso").

Cada insight e um no especial no CGE com type="meta_rule".
A Forja descobre insights minerando CognitiveTraces.
O MetaController consulta insights para guiar estrategias.
"""
import time, uuid, hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


PRIMORDIAL_INSIGHTS = [
    {
        "principle": "Coisas grandes sao feitas de coisas pequenas",
        "tags": ["decomposition", "complexity"],
        "confidence": 0.95,
    },
    {
        "principle": "Coisas complexas sao feitas de coisas simples",
        "tags": ["complexity", "simplicity"],
        "confidence": 0.90,
    },
    {
        "principle": "Quando dois sistemas convergem para a mesma solucao, existe uma restricao fundamental",
        "tags": ["convergence", "constraint"],
        "confidence": 0.75,
    },
    {
        "principle": "Uma excecao recorrente revela uma variavel oculta",
        "tags": ["exception", "hidden_variable"],
        "confidence": 0.70,
    },
    {
        "principle": "Causalidade e mais util que correlacao",
        "tags": ["causality", "reasoning"],
        "confidence": 0.85,
    },
    {
        "principle": "Decompor o problema antes de atacar aumenta a taxa de sucesso",
        "tags": ["decomposition", "strategy"],
        "confidence": 0.80,
    },
    {
        "principle": "Modelos simples que explicam muito sao melhores que modelos complexos que explicam tudo",
        "tags": ["parsimony", "occam"],
        "confidence": 0.85,
    },
    {
        "principle": "Explorar antes de explorar produz conhecimento mais robusto",
        "tags": ["exploration", "strategy"],
        "confidence": 0.75,
    },
    {
        "principle": "Erro de predicao alto significa que o modelo precisa mudar, nao os dados",
        "tags": ["prediction_error", "model_update"],
        "confidence": 0.80,
    },
    {
        "principle": "O que e util em um contexto provavelmente nao e em outro",
        "tags": ["context", "generalization"],
        "confidence": 0.70,
    },
]


@dataclass
class Insight:
    principle: str
    confidence: float = 0.5
    tags: List[str] = field(default_factory=list)
    source_trace_id: str = ""
    success_rate: float = 0.0
    n_applications: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    domain: str = "general"

    @property
    def id(self) -> str:
        return f"insight_{hashlib.md5(self.principle.encode()).hexdigest()[:8]}"

    def to_dict(self) -> Dict:
        return {"id": self.id, "principle": self.principle[:120],
                "confidence": self.confidence, "tags": self.tags,
                "success_rate": self.success_rate,
                "n_applications": self.n_applications,
                "domain": self.domain}

    def summarize(self) -> str:
        return (f"[{self.id}] \"{self.principle[:60]}...\" | "
                f"conf={self.confidence:.2f} success={self.success_rate:.2f} "
                f"({self.n_applications}x)")


class InsightLibrary:
    """Biblioteca de principios abstratos acima do CGE.

    A Forja descobre insights minerando CognitiveTraces.
    Apos N aplicacoes com sucesso, o insight ganha confianca.
    """

    def __init__(self, cge=None):
        self.cge = cge
        self.insights: Dict[str, Insight] = {}
        self._history: List[Dict] = []

        # Carregar insights primordiais
        for data in PRIMORDIAL_INSIGHTS:
            ins = Insight(
                principle=data["principle"],
                confidence=data["confidence"],
                tags=data["tags"],
            )
            self.insights[ins.id] = ins

    def add_insight(self, principle: str, confidence: float = 0.3,
                    tags=None, source_trace: str = "",
                    domain: str = "general") -> Insight:
        """Adiciona um novo insight (descoberto pela Forja)."""
        ins = Insight(
            principle=principle, confidence=confidence,
            tags=tags or [], source_trace_id=source_trace,
            domain=domain,
        )
        if ins.id in self.insights:
            existing = self.insights[ins.id]
            existing.confidence = max(existing.confidence, confidence)
            existing.last_used = time.time()
            if source_trace:
                existing.source_trace_id = source_trace
            return existing
        self.insights[ins.id] = ins
        self._sync_to_cge(ins)
        self._history.append({"action": "add", "id": ins.id,
                              "principle": principle[:80],
                              "time": time.time()})
        return ins

    def apply(self, insight_id: str, success: bool = True) -> float:
        """Registra uma aplicacao do insight e atualiza metrics."""
        ins = self.insights.get(insight_id)
        if ins is None:
            return 0.0
        ins.n_applications += 1
        ins.last_used = time.time()
        if success:
            ins.success_rate = ((ins.success_rate * (ins.n_applications - 1)) + 1.0) / ins.n_applications
        else:
            ins.success_rate = (ins.success_rate * (ins.n_applications - 1)) / ins.n_applications
        ins.confidence = min(1.0, ins.confidence + 0.05 if success else ins.confidence - 0.05)
        return ins.success_rate

    def query(self, context: str = "", tags=None,
              min_confidence: float = 0.3, top_k: int = 5) -> List[Insight]:
        """Busca insights relevantes para um contexto."""
        candidates = list(self.insights.values())
        # Filtrar por confianca
        candidates = [i for i in candidates if i.confidence >= min_confidence]
        # Filtrar por tags se fornecidas
        if tags:
            candidates = [i for i in candidates
                         if any(t in i.tags for t in tags)]
        # Ordenar por confianca * success_rate
        candidates.sort(key=lambda i: i.confidence * (0.5 + i.success_rate), reverse=True)
        return candidates[:top_k]

    def query_by_domain(self, domain: str, top_k: int = 3) -> List[Insight]:
        """Insights especificos de um dominio."""
        matched = [i for i in self.insights.values() if i.domain == domain]
        matched.sort(key=lambda i: -i.confidence)
        return matched[:top_k]

    def discover_from_traces(self, traces: List[Dict]) -> List[Insight]:
        """A Forja chama isso para minerar novos insights de CognitiveTraces."""
        discovered = []
        if len(traces) < 5:
            return discovered

        # Analisar padroes: modos que mais geram sucesso
        mode_success: Dict[str, List[float]] = {}
        for td in traces:
            steps = td.get("steps", []) if isinstance(td, dict) else []
            for s in steps:
                mode = s.get("mode", "") if isinstance(s, dict) else getattr(s, "mode", "")
                reward = s.get("reward", 0) if isinstance(s, dict) else getattr(s, "reward", 0)
                if mode:
                    mode_success.setdefault(mode, []).append(reward)

        for mode, rewards in mode_success.items():
            if len(rewards) >= 5 and sum(rewards) / len(rewards) > 0.6:
                new = self.add_insight(
                    principle=f"Usar modo {mode.upper()} quando o contexto atual se alinha com seus pontos fortes",
                    confidence=0.4, tags=[mode, "meta_policy"],
                    source_trace="forja_mining",
                )
                discovered.append(new)

        # Analisar: decomposicao precede sucesso
        decomp_count = 0
        for td in traces:
            steps_text = str(td)
            if "decompos" in steps_text.lower():
                decomp_count += 1
        if decomp_count > len(traces) * 0.3 and decomp_count >= 3:
            discovered.append(self.add_insight(
                principle="Decompor problemas complexos ANTES de atacar aumenta significativamente a taxa de sucesso",
                confidence=0.5, tags=["decomposition", "strategy"],
                source_trace="forja_mining"))

        return discovered

    def get_state(self) -> Dict:
        return {
            "n_insights": len(self.insights),
            "top": [i.summarize() for i in self.query("", top_k=5)],
            "history_count": len(self._history),
        }

    def _sync_to_cge(self, insight: Insight):
        """Armazena insight no CGE como no type='meta_rule'."""
        if self.cge is None:
            return
        import torch
        nid = insight.id
        g = torch.Generator()
        g.manual_seed(int(hashlib.md5(insight.principle.encode()).hexdigest()[:8], 16))
        vec = torch.sign(torch.randn(10000, generator=g))
        self.cge.create_node(nid, vec)
        if nid in self.cge.nodes:
            self.cge.nodes[nid]["type"] = "meta_rule"
            self.cge.nodes[nid]["insight_data"] = insight.to_dict()
            self.cge.nodes[nid]["confidence"] = insight.confidence
