"""
Observation Layer — Normaliza absolutamente tudo que o sistema percebe.

Entrada: texto, imagem, video, audio, programas executados, erros, sucessos,
         feedback do CGE, traces, metas, acoes.

Saida: lista de ObservedEvent (timestamp, tipo, payload normalizado).

Nao interpreta. Nao classifica. So captura e normaliza.
"""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import time


class EventType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    PROGRAM_EXECUTED = "program_executed"
    PROGRAM_FAILED = "program_failed"
    PROGRAM_SUCCEEDED = "program_succeeded"
    CONCEPT_CREATED = "concept_created"
    CONCEPT_REFINED = "concept_refined"
    GOAL_SET = "goal_set"
    GOAL_ACHIEVED = "goal_achieved"
    ERROR_SIGNAL = "error_signal"
    IMPASSE = "impasse"
    CGE_UPDATE = "cge_update"
    MEMORY_RETRIEVAL = "memory_retrieval"
    SIMULATION_RESULT = "simulation_result"
    IMAGINATION = "imagination"
    FEEDBACK = "feedback"


@dataclass
class ObservedEvent:
    """Um evento observavel normalizado."""
    timestamp: float = field(default_factory=time.time)
    event_type: EventType = EventType.TEXT
    source: str = "unknown"         # "cge", "program_synthesizer", "world", "meta"
    modality: str = "text"          # "text", "visual", "audio", "motor", "conceptual"
    payload: Dict[str, Any] = field(default_factory=dict)
    domain: str = "general"         # "arc", "biology", "geometry", "language", etc
    
    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 3),
            "type": self.event_type.value,
            "source": self.source,
            "modality": self.modality,
            "domain": self.domain,
            "payload": {k: str(v)[:60] for k, v in self.payload.items()},
        }


class ObservationLayer:
    """Receptor universal de observacoes.
    
    Uso:
        ol = ObservationLayer()
        ol.observe("text", {"content": "a ball bounces back"}, domain="physics")
        ol.observe("program_executed", {"program": "flip_horizontal", "success": True})
        
        for event in ol.get_recent(n=10):
            print(event.to_dict())
    """
    
    def __init__(self, max_events: int = 10000):
        self.events: List[ObservedEvent] = []
        self.max_events = max_events
    
    def observe(self, event_type: str, payload: dict, source: str = "unknown",
                modality: str = "text", domain: str = "general") -> ObservedEvent:
        """Registra uma observacao."""
        try:
            et = EventType(event_type)
        except ValueError:
            et = EventType.TEXT
        
        event = ObservedEvent(
            event_type=et,
            source=source,
            modality=modality,
            payload=payload,
            domain=domain,
        )
        self.events.append(event)
        
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        return event
    
    def get_recent(self, n: int = 100, event_type: Optional[str] = None,
                   domain: Optional[str] = None) -> List[ObservedEvent]:
        """Retorna eventos recentes com filtros opcionais."""
        filtered = self.events
        if event_type:
            filtered = [e for e in filtered if e.event_type.value == event_type]
        if domain:
            filtered = [e for e in filtered if e.domain == domain]
        return filtered[-n:]
    
    def get_by_domain(self, domain: str) -> List[ObservedEvent]:
        """Retorna todos os eventos de um dominio."""
        return [e for e in self.events if e.domain == domain]
    
    def get_stats(self) -> dict:
        """Estatisticas das observacoes."""
        type_counts = {}
        domain_counts = {}
        for e in self.events:
            type_counts[e.event_type.value] = type_counts.get(e.event_type.value, 0) + 1
            domain_counts[e.domain] = domain_counts.get(e.domain, 0) + 1
        return {
            "total_events": len(self.events),
            "by_type": type_counts,
            "by_domain": domain_counts,
            "time_span": (self.events[-1].timestamp - self.events[0].timestamp) / 3600
            if len(self.events) >= 2 else 0,
        }
