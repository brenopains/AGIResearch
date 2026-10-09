"""
CommunicationMemory — Memoria de Comunicacoes Passadas

Registra:
  - O que foi explicado
  - Para quem
  - Como (formato, estrategia)
  - A pessoa entendeu?
  - Onde travou?
  - O que funcionou?

Com o tempo, a AGI aprende a se comunicar melhor.
"""
import time, uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class CommunicationRecord:
    id: str = ""
    concept: str = ""
    profile_id: str = ""
    format: str = ""
    content_summary: str = ""
    understood: bool = False
    struggle_point: str = ""
    what_worked: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {"concept": self.concept, "profile": self.profile_id,
                "format": self.format, "understood": self.understood}


class CommunicationMemory:
    def __init__(self, max_records: int = 200):
        self.records: List[CommunicationRecord] = []
        self.max_records = max_records

    def add(self, concept: str, profile_id: str, format: str,
            understood: bool, struggle: str = "", worked: str = "") -> str:
        rec = CommunicationRecord(
            id=f"comm_{uuid.uuid4().hex[:8]}",
            concept=concept, profile_id=profile_id, format=format,
            understood=understood, struggle_point=struggle, what_worked=worked,
        )
        self.records.append(rec)
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]
        return rec.id

    def best_format_for(self, concept: str, profile_id: str) -> Optional[str]:
        """Melhor formato para este conceito + perfil."""
        relevant = [r for r in self.records
                   if r.concept == concept and r.profile_id == profile_id and r.understood]
        if not relevant:
            return None
        formats = {}
        for r in relevant:
            formats[r.format] = formats.get(r.format, 0) + 1
        return max(formats, key=formats.get)

    def get_state(self) -> Dict:
        return {"n_records": len(self.records),
                "avg_understood": sum(1 for r in self.records if r.understood) / max(1, len(self.records))}
