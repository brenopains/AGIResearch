"""Tipos e utilidades comuns aos agentes."""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json, time, pathlib, uuid

@dataclass
class Hypothesis:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    title: str = ""
    statement: str = ""               # afirmacao testavel em 1-3 frases
    rationale: str = ""               # justificativa baseada em literatura
    mechanism: str = ""               # via molecular/celular proposta
    proposed_intervention: str = ""    # composto/abordagem
    evidence_pmids: List[str] = field(default_factory=list)
    novelty: float = 0.0              # 0..1 (1 = nada parecido publicado)
    plausibility: float = 0.0         # 0..1 mecanistica
    tractability: float = 0.0         # 0..1 (existe forma de testar/realizar)
    safety: float = 0.0               # 0..1 (1=baixo risco)
    evidence_strength: float = 0.0    # 0..1
    overall: float = 0.0
    elo: float = 1200.0               # rank do torneio
    parent_ids: List[str] = field(default_factory=list)  # genealogia (evolution)
    critiques: List[str] = field(default_factory=list)
    born_at: float = field(default_factory=time.time)

    def to_dict(self): return asdict(self)

def save_run(name: str, payload: Dict[str, Any]):
    base = pathlib.Path(__file__).parent.parent / "runs"
    base.mkdir(parents=True, exist_ok=True)
    p = base / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p

def load_run(name: str) -> Optional[Dict]:
    p = pathlib.Path(__file__).parent.parent / "runs" / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
