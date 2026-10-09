"""
SelfModel — Modelo de Si Mesmo

Divide-se em dois subsistemas que competem e cooperam:

SelfIdentity: quem sou (crencas, historia, capacidades, perfil de aprendizado)
  - conhecido / desconhecido
  - confianca nas crencas
  - autobiografia (experiencias marcantes)
  - capability_profile: no que e bom/ruim
  - learning_profile: como aprende melhor

SelfProcess: o que estou fazendo agora
  - operador/programa ativo
  - estado de atencao, fadiga, emocao
  - erro atual
  - modo cognitivo

O SelfModel e atualizado pelo Maestro a cada ciclo.
A Forja pode abstrair padroes sobre a propria cognicao a partir dele.
"""
import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Belief:
    """Uma crenca — conceito que o sistema considera verdadeiro."""
    concept: str
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_revised: float = 0.0
    source_reliability: float = 0.5
    times_reinforced: int = 0

    def reinforce(self, evidence: str, reliability: float = 0.5):
        """Reforca esta crenca com nova evidencia."""
        self.evidence.append(evidence)
        self.times_reinforced += 1
        self.last_revised = time.time()
        # Confianca aumenta com evidencias consistentes
        self.confidence = min(1.0, self.confidence + reliability * 0.1)
        self.source_reliability = (self.source_reliability + reliability) / 2

    def weaken(self, factor: float = 0.5):
        """Enfraquece esta crenca (evidencia contraditoria)."""
        self.confidence *= factor
        if self.confidence < 0.05:
            self.confidence = 0.0

    @property
    def is_active(self) -> bool:
        return self.confidence > 0.1


@dataclass
class AutobiographicalMemory:
    """Memoria autobiografica — experiencias marcantes."""
    episodes: List[Dict] = field(default_factory=lambda: deque(maxlen=200))

    def record(self, episode: Dict):
        """Registra um episodio marcante."""
        self.episodes.append({
            **episode,
            "time": time.time(),
        })

    def recall(self, query: str, n: int = 5) -> List[Dict]:
        """Recupera episodios similares a um conceito."""
        results = []
        for ep in reversed(self.episodes):
            if query.lower() in str(ep.get("concept", "")).lower():
                results.append(ep)
                if len(results) >= n:
                    break
        return results

    def get_recent(self, n: int = 5) -> List[Dict]:
        return list(self.episodes)[-n:]


@dataclass
class CapabilityProfile:
    """Perfil de capacidades — no que o sistema e bom/ruim."""
    domain_scores: Dict[str, float] = field(default_factory=dict)
    # dominio -> score 0..1

    def update(self, domain: str, score: float):
        """Atualiza score para um dominio."""
        current = self.domain_scores.get(domain, 0.5)
        self.domain_scores[domain] = current * 0.8 + score * 0.2

    def get_strengths(self, threshold: float = 0.6) -> List[str]:
        return [d for d, s in self.domain_scores.items() if s >= threshold]

    def get_weaknesses(self, threshold: float = 0.4) -> List[str]:
        return [d for d, s in self.domain_scores.items() if s <= threshold]


@dataclass
class LearningProfile:
    """Perfil de aprendizado — como o sistema aprende melhor."""
    modality_scores: Dict[str, float] = field(default_factory=lambda: {
        "video": 0.5, "text": 0.5, "audio": 0.5, "interactive": 0.5, "reflection": 0.5
    })
    # modalidade -> eficiencia 0..1

    def update(self, modality: str, efficiency: float):
        current = self.modality_scores.get(modality, 0.5)
        self.modality_scores[modality] = current * 0.9 + efficiency * 0.1

    def best_modality(self) -> str:
        return max(self.modality_scores, key=self.modality_scores.get)

    def worst_modality(self) -> str:
        return min(self.modality_scores, key=self.modality_scores.get)


class SelfIdentity:
    """Quem sou — crencas, historia, capacidades, perfil."""

    def __init__(self):
        self.beliefs: Dict[str, Belief] = {}
        self.autobiography = AutobiographicalMemory()
        self.capabilities = CapabilityProfile()
        self.learning_profile = LearningProfile()
        self.goals: List[str] = []
        self.preferences: List[str] = []
        self.creation_time: float = time.time()

    def add_belief(self, concept: str, evidence: str = "",
                   confidence: float = 0.5, reliability: float = 0.5) -> str:
        """Adiciona ou reforca uma crenca."""
        if concept in self.beliefs:
            self.beliefs[concept].reinforce(evidence, reliability)
        else:
            self.beliefs[concept] = Belief(
                concept=concept, confidence=confidence,
                evidence=[evidence] if evidence else [],
                source_reliability=reliability,
            )
        return concept

    def revise_belief(self, concept: str, contradicting_evidence: str,
                      reliability: float = 0.3):
        """Revisa uma crenca a luz de evidencia contraditoria."""
        belief = self.beliefs.get(concept)
        if belief:
            belief.weaken(reliability)
            self.autobiography.record({
                "type": "belief_revision",
                "concept": concept,
                "evidence": contradicting_evidence,
                "old_confidence": belief.confidence,
            })

    def get_confident_beliefs(self, threshold: float = 0.6) -> List[str]:
        return [c for c, b in self.beliefs.items() if b.confidence >= threshold]

    def get_uncertain_topics(self, threshold: float = 0.3) -> List[str]:
        """Topicos com baixa confianca — candidatos a exploracao."""
        return [c for c, b in self.beliefs.items() if b.confidence < threshold]

    def get_state(self) -> Dict:
        return {
            "beliefs": len(self.beliefs),
            "confident": len(self.get_confident_beliefs()),
            "uncertain": len(self.get_uncertain_topics()),
            "autobiography_size": len(self.autobiography.episodes),
            "strengths": self.capabilities.get_strengths()[:3],
            "weaknesses": self.capabilities.get_weaknesses()[:3],
            "best_modality": self.learning_profile.best_modality(),
            "age_hours": round((time.time() - self.creation_time) / 3600, 1),
        }


class SelfProcess:
    """O que estou fazendo agora — estado cognitivo momentaneo."""

    def __init__(self):
        self.active_operator: Optional[str] = None
        self.active_program: Optional[str] = None
        self.attention_focus: Optional[str] = None
        self.fatigue: float = 0.0  # 0..1
        self.current_emotion: str = "neutral"
        self.current_error: float = 0.0
        self.cognitive_mode: str = "idle"
        self.problems_active: List[Dict] = []
        self.step_count: int = 0
        self.busy_since: float = time.time()

    def update(self, operator: Optional[str] = None, program: Optional[str] = None,
               attention: Optional[str] = None, error: float = 0.0,
               mode: str = "idle", emotion: str = "neutral"):
        """Atualiza estado do processo a cada ciclo do Maestro."""
        self.step_count += 1
        if operator:
            self.active_operator = operator
        if program:
            self.active_program = program
        if attention:
            self.attention_focus = attention
        self.current_error = error
        self.cognitive_mode = mode
        self.current_emotion = emotion

        # Fadiga: acumula devagar, recupera mais devagar
        if mode in ("explore", "exploit", "test", "hypothesize"):
            self.fatigue = min(1.0, self.fatigue + 0.02)
        elif mode in ("reflect", "consolidate", "idle"):
            self.fatigue = max(0.0, self.fatigue - 0.02)
        # Modos de alta intensidade cansam mais
        if error > 0.5:
            self.fatigue = min(1.0, self.fatigue + 0.01)

        if self.fatigue > 0.8:
            self.busy_since = time.time()

    def needs_rest(self) -> bool:
        return self.fatigue > 0.7

    def should_switch_mode(self) -> bool:
        return self.fatigue > 0.6 and self.cognitive_mode not in ("reflect", "consolidate")

    def get_state(self) -> Dict:
        return {
            "active_operator": self.active_operator,
            "active_program": self.active_program,
            "attention": self.attention_focus,
            "fatigue": round(self.fatigue, 2),
            "emotion": self.current_emotion,
            "error": round(self.current_error, 2),
            "mode": self.cognitive_mode,
            "problems_active": len(self.problems_active),
            "steps": self.step_count,
        }


class SelfModel:
    """Modelo completo de si mesmo — identidade + processo.

    O Maestro consulta o SelfModel para decidir:
      - O que estudar (lacunas no SelfIdentity)
      - Quando descansar (fadiga no SelfProcess)
      - Qual modalidade usar (LearningProfile)
      - Se deve revisar crencas (Belief com baixa confianca)
    """

    def __init__(self):
        self.identity = SelfIdentity()
        self.process = SelfProcess()

    def tick(self, operator: Optional[str] = None, program: Optional[str] = None,
             attention: Optional[str] = None, error: float = 0.0,
             mode: str = "idle", emotion: str = "neutral"):
        """Ciclo de atualizacao do SelfModel."""
        self.process.update(operator, program, attention, error, mode, emotion)

    def learn_from_experience(self, concept: str, evidence: str,
                              reliability: float = 0.5, domain: str = "general",
                              modality: str = "text", success: bool = True):
        """Aprende com uma experiencia — atualiza crencas, capacidades, perfil."""
        self.identity.add_belief(concept, evidence, reliability=reliability)
        self.identity.capabilities.update(domain, 0.5 + 0.5 * success)
        self.identity.learning_profile.update(modality, 0.5 + 0.5 * success)
        self.identity.autobiography.record({
            "type": "learning",
            "concept": concept,
            "domain": domain,
            "modality": modality,
            "success": success,
        })

    def get_state(self) -> Dict:
        return {
            "identity": self.identity.get_state(),
            "process": self.process.get_state(),
        }

    def should_rest(self) -> bool:
        return self.process.needs_rest()

    def should_revise(self) -> List[str]:
        """Crencas que deveriam ser revisadas."""
        return self.identity.get_uncertain_topics()

    def get_curiosity_targets(self, n: int = 3) -> List[str]:
        """Topicos que o sistema deveria explorar."""
        return self.identity.get_uncertain_topics()[:n]
