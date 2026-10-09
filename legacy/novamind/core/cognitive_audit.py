"""
CognitiveAudit — Rastreabilidade profunda do raciocinio.

Registra para cada ciclo:
  - Multiplas hipoteses concorrentes com evidencias
  - Atencao seletiva (por que X venceu Y)
  - Origem de cada ativacao (WordNet, curriculo, episodio)
  - Cadeia causal completa
  - O que foi descartado e por que
  - Prediction error por no (nao apenas global)
  - Snapshot do estado mental interno

Cada ciclo vira um relatorio cientifico completo e auditavel.
"""
import time, json, os, copy
from typing import Dict, List, Optional, Any
from collections import defaultdict

AUDIT_DIR = r"D:\AGI2\cognitive_audit"
os.makedirs(AUDIT_DIR, exist_ok=True)


class Hypothesis:
    """Uma hipotese concorrente com suas evidencias."""
    def __init__(self, name: str, confidence: float = 0.0):
        self.name = name
        self.confidence = confidence
        self.evidence: List[Dict] = []  # [{source, strength, text}]
        self.supporting_nodes: List[str] = []
        self.contradicting_nodes: List[str] = []
        self.created_at = time.time()
    
    def add_evidence(self, source: str, strength: float, text: str = ""):
        self.evidence.append({
            "source": source, "strength": round(strength, 3), "text": text[:100],
        })
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "confidence": round(self.confidence, 3),
            "evidences": self.evidence[-5:],  # ultimas 5 evidencias
            "supporting": self.supporting_nodes[:5],
            "contradicting": self.contradicting_nodes[:5],
        }


class AttentionRecord:
    """Registro de atencao seletiva: candidatos -> selecionados -> foco."""
    def __init__(self):
        self.candidates: List[Dict] = []  # todos os considerados
        self.shortlisted: List[Dict] = []  # top N
        self.focus: str = ""  # o vencedor
        self.discarded: List[Dict] = []  # rejeitados com motivo
    
    def add_candidate(self, name: str, score: float, source: str = ""):
        self.candidates.append({"name": name, "score": round(score, 3), "source": source})
    
    def select_focus(self, name: str, reason: str = ""):
        self.focus = name
        # Move de candidates para shortlisted/discarded
        for c in self.candidates:
            if c["name"] == name:
                self.shortlisted.append(c)
            elif c not in self.shortlisted and c not in self.discarded:
                entry = {**c, "reason_discarded": reason or "lower score"}
                self.discarded.append(entry)
    
    def to_dict(self) -> dict:
        return {
            "candidates_considered": len(self.candidates),
            "shortlisted": self.shortlisted[:5],
            "focus": self.focus,
            "discarded": self.discarded[:8],
        }


class NodeActivation:
    """Ativacao de um no com origem e PE local."""
    def __init__(self, node_id: str, activation: float, source: str = "", pe: float = 0.0):
        self.node_id = node_id
        self.activation = activation
        self.source = source  # "WordNet", "Science", "Recall", "Episodio", "EpisodioN"
        self.pe = pe  # prediction error LOCAL deste no
        self.connected_to: List[str] = []


class CausalChain:
    """Cadeia causal completa: A -> B -> C -> conclusao."""
    def __init__(self):
        self.steps: List[Dict] = []
    
    def add_step(self, from_node: str, relation: str, to_node: str, confidence: float):
        self.steps.append({
            "from": from_node, "relation": relation, "to": to_node,
            "confidence": round(confidence, 3),
        })
    
    def to_dict(self) -> list:
        return self.steps[-10:]  # ultimos 10 passos


class ForjaRecord:
    """Registro de criacao de conceitos/procedimentos pela Forja."""
    def __init__(self):
        self.new_concepts: List[Dict] = []
        self.new_procedures: List[Dict] = []
    
    def add_concept(self, name: str, frequency: int, compression_ratio: float,
                    description: str = ""):
        self.new_concepts.append({
            "name": name, "frequency": frequency,
            "compression": round(compression_ratio, 3),
            "description": description,
            "created_at": time.time(),
        })
    
    def add_procedure(self, name: str, steps: list, confidence: float,
                      description: str = ""):
        self.new_procedures.append({
            "name": name, "steps": steps, "confidence": round(confidence, 3),
            "description": description, "created_at": time.time(),
        })
    
    def to_dict(self) -> dict:
        return {
            "new_concepts": self.new_concepts[-3:],
            "new_procedures": self.new_procedures[-3:],
        }


class CognitiveAudit:
    """Auditoria completa de um ciclo cognitivo.
    
    Uso:
        audit = CognitiveAudit()
        audit.start_cycle(input_text)
        audit.add_hypothesis("whale=mammal", 0.81)
        audit.add_evidence("whale=mammal", "WordNet", 0.85, "whale is-a mammal")
        audit.record_attention("whale", candidates=[...])
        audit.add_causal_step("whale", "is_a", "mammal", 0.9)
        audit.discard_hypothesis("whale=fish", "low energy, contradicts mammal")
        audit.end_cycle(output)
        audit.save()
    """
    
    def __init__(self):
        self.current: Dict = {}
        self.history: List[str] = []  # lista de IDs salvos
    
    def start_cycle(self, input_text: str = "", modality: str = "text"):
        audit_id = f"audit_{int(time.time()*1000)}_{len(self.history)}"
        self.current = {
            "audit_id": audit_id,
            "timestamp": time.time(),
            "input": input_text,
            "modality": modality,
            "hypotheses": {},  # nome -> Hypothesis
            "hypotheses_ordered": [],  # ordem de confianca
            "attention": AttentionRecord(),
            "causal_chain": CausalChain(),
            "cge_activations": [],  # lista de NodeActivation
            "forja": ForjaRecord(),
            "mental_state": {},
            "output": "",
            "duration": 0.0,
            "iterations": 0,
        }
    
    # ─── Hipotese ──────────────────────────────────────────
    
    def add_hypothesis(self, name: str, confidence: float = 0.0) -> Hypothesis:
        if name not in self.current["hypotheses"]:
            self.current["hypotheses"][name] = Hypothesis(name, confidence)
        return self.current["hypotheses"][name]
    
    def add_evidence(self, hypothesis_name: str, source: str, strength: float,
                     text: str = ""):
        hyp = self.add_hypothesis(hypothesis_name)
        hyp.add_evidence(source, strength, text)
        # Atualiza confianca baseada na nova evidencia
        weights = [e["strength"] for e in hyp.evidence]
        hyp.confidence = sum(weights) / max(1, len(weights))
    
    def discard_hypothesis(self, name: str, reason: str):
        if name in self.current["hypotheses"]:
            hyp = self.current["hypotheses"].pop(name)
            self.current["attention"].discarded.append({
                "name": name,
                "confidence": hyp.confidence,
                "reason": reason,
            })
    
    def get_best_hypothesis(self) -> Optional[str]:
        hyps = self.current["hypotheses"].values()
        if not hyps:
            return None
        best = max(hyps, key=lambda h: h.confidence)
        return best.name
    
    # ─── Atencao ───────────────────────────────────────────
    
    def add_candidate(self, name: str, score: float, source: str = ""):
        self.current["attention"].add_candidate(name, score, source)
    
    def select_focus(self, name: str, reason: str = ""):
        self.current["attention"].select_focus(name, reason)
        self.current["mental_state"]["focus"] = name
    
    # ─── Cadeia causal ─────────────────────────────────────
    
    def add_causal_step(self, from_node: str, relation: str, to_node: str,
                        confidence: float = 0.5):
        self.current["causal_chain"].add_step(from_node, relation, to_node, confidence)
    
    # ─── CGE activations ──────────────────────────────────
    
    def record_activation(self, node_id: str, activation: float, source: str = "",
                          pe: float = 0.0):
        act = NodeActivation(node_id, activation, source, pe)
        self.current["cge_activations"].append(act)
    
    # ─── Forja ─────────────────────────────────────────────
    
    def add_concept(self, name: str, frequency: int, compression: float,
                    description: str = ""):
        self.current["forja"].add_concept(name, frequency, compression, description)
    
    def add_procedure(self, name: str, steps: list, confidence: float,
                      description: str = ""):
        self.current["forja"].add_procedure(name, steps, confidence, description)
    
    # ─── Estado mental ─────────────────────────────────────
    
    def set_mental_state(self, **kwargs):
        self.current["mental_state"].update(kwargs)
    
    # ─── Finalizacao ──────────────────────────────────────
    
    def end_cycle(self, output: str = "", iterations: int = 0):
        self.current["output"] = output
        self.current["iterations"] = iterations
        self.current["duration"] = round(time.time() - self.current["timestamp"], 3)
        
        # Ordena hipoteses por confianca
        hyps = list(self.current["hypotheses"].values())
        hyps.sort(key=lambda h: h.confidence, reverse=True)
        self.current["hypotheses_ordered"] = [h.to_dict() for h in hyps[:5]]
        
        # Salva
        self.save()
    
    def save(self):
        path = os.path.join(AUDIT_DIR, f"{self.current['audit_id']}.json")
        with open(path, "w") as f:
            json.dump(self._serialize(), f, indent=1, default=str)
        self.history.append(self.current["audit_id"])
    
    def _serialize(self) -> dict:
        c = copy.deepcopy(self.current)
        # Converte objetos para dict
        c["attention"] = c["attention"].to_dict() if hasattr(c["attention"], "to_dict") else {}
        c["causal_chain"] = c["causal_chain"].to_dict() if hasattr(c["causal_chain"], "to_dict") else []
        c["forja"] = c["forja"].to_dict() if hasattr(c["forja"], "to_dict") else {}
        c["cge_activations"] = [
            {"node": a.node_id, "act": round(a.activation, 3),
             "source": a.source, "pe": round(a.pe, 3)}
            for a in c["cge_activations"][:20]
        ]
        c["hypotheses"] = {k: v.to_dict() for k, v in c["hypotheses"].items()}
        return c
    
    def load(self, audit_id: str) -> Optional[dict]:
        path = os.path.join(AUDIT_DIR, f"{audit_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None
    
    def list_audits(self, n: int = 20) -> List[Dict]:
        files = sorted(os.listdir(AUDIT_DIR), reverse=True)[:n]
        audits = []
        for fname in files:
            if fname.endswith(".json"):
                path = os.path.join(AUDIT_DIR, fname)
                try:
                    with open(path) as f:
                        data = json.load(f)
                        audits.append({
                            "id": data.get("audit_id", fname),
                            "input": data.get("input", "")[:50],
                            "output": data.get("output", "")[:50],
                            "iterations": data.get("iterations", 0),
                            "duration": data.get("duration", 0),
                            "n_hypotheses": len(data.get("hypotheses", {})),
                        })
                except:
                    pass
        return audits


class AuditReplay:
    """Reproduz uma auditoria completa como relatorio cientifico."""
    
    def __init__(self, audit: CognitiveAudit):
        self.audit = audit
    
    def show(self, audit_id: str) -> str:
        data = self.audit.load(audit_id)
        if not data:
            return f"[Auditoria {audit_id} nao encontrada]"
        
        lines = []
        lines.append("=" * 70)
        lines.append(f"  AUDITORIA COGNITIVA — {data.get('audit_id', audit_id)[:40]}")
        lines.append("=" * 70)
        
        # Input
        lines.append(f"\nINPUT: {data.get('input', '')}")
        lines.append(f"MODALITY: {data.get('modality', 'text')}")
        
        # Hipoteses
        hyps = data.get("hypotheses_ordered", []) or list(data.get("hypotheses", {}).values())
        if hyps:
            lines.append(f"\nHIPOTESES CONCORRENTES ({len(hyps)}):")
            for h in hyps[:5]:
                name = h.get("name", "?")
                conf = h.get("confidence", 0)
                evs = h.get("evidences", [])
                lines.append(f"  [{conf:.2f}] {name}")
                for ev in evs[:3]:
                    lines.append(f"         evidence: {ev.get('source','?')} ({ev.get('strength',0):.2f}) {ev.get('text','')[:60]}")
        
        # Descartadas
        attn = data.get("attention", {})
        discarded = attn.get("discarded", [])
        if discarded:
            lines.append(f"\nDESCARTADAS ({len(discarded)}):")
            for d in discarded[:5]:
                lines.append(f"  [{d.get('confidence',0):.2f}] {d.get('name','')} — motivo: {d.get('reason','')[:80]}")
        
        # Cadeia causal
        chain = data.get("causal_chain", [])
        if chain:
            lines.append(f"\nCADEIA CAUSAL:")
            for step in chain[:8]:
                lines.append(f"  {step.get('from','')} --[{step.get('relation','')}]--> {step.get('to','')} (conf={step.get('confidence',0):.2f})")
        
        # CGE activations (com PE local)
        acts = data.get("cge_activations", [])
        if acts:
            lines.append(f"\nATIVACOES CGE (com PE local):")
            for a in acts[:10]:
                lines.append(f"  {a.get('node','')}: act={a.get('act',0):.3f} pe={a.get('pe',0):.3f} source={a.get('source','')}")
        
        # Forja
        forja = data.get("forja", {})
        concepts = forja.get("new_concepts", [])
        if concepts:
            lines.append(f"\nFORJA — NOVOS CONCEITOS:")
            for c in concepts:
                lines.append(f"  {c.get('name','')}: freq={c.get('frequency',0)} compression={c.get('compression',0):.3f}")
        procs = forja.get("new_procedures", [])
        if procs:
            lines.append(f"\nFORJA — NOVOS PROCEDIMENTOS:")
            for p in procs:
                steps = ", ".join(p.get("steps", [])) if isinstance(p.get("steps"), list) else str(p.get("steps", ""))
                lines.append(f"  {p.get('name','')}: {steps} (conf={p.get('confidence',0):.2f})")
        
        # Estado mental
        ms = data.get("mental_state", {})
        if ms:
            lines.append(f"\nESTADO MENTAL:")
            for k, v in ms.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.3f}")
                else:
                    lines.append(f"  {k}: {str(v)[:60]}")
        
        # Output
        lines.append(f"\nRESPOSTA: {data.get('output', '')}")
        lines.append(f"\nDuracao: {data.get('duration',0):.1f}s | Iteracoes: {data.get('iterations',0)}")
        lines.append("=" * 70)
        
        return "\n".join(lines)
