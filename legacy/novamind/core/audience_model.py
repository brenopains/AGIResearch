"""
AudienceModel — Modelo de Audiência

Antes de responder, a AGI estima quem esta ouvindo:
  - Perfil (leigo, tecnico, cientista, crianca, programador...)
  - Vocabulario estimado
  - Conhecimento previo sobre o topico
  - Preferencia de formato (texto, video, analogia, equacao)

Depois adapta: vocabulario, profundidade, exemplos, velocidade.
"""
import time, hashlib, random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class Profile:
    id: str = ""
    name: str = "default"
    vocabulary_level: float = 0.5       # 0=basico, 1=tecnico
    math_level: float = 0.3             # 0=nenhuma, 1=avancada
    preferred_format: str = "text"       # "text", "analogy", "visual", "video", "code"
    patience: float = 0.5               # 0=quer direto ao ponto, 1=aceita detalhes
    known_concepts: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)


PRIMORDIAL_PROFILES = [
    Profile(id="leigo", name="Leigo", vocabulary_level=0.2, math_level=0.0,
            preferred_format="analogy", patience=0.3, interests=["exemplos", "analogias"]),
    Profile(id="estudante", name="Estudante", vocabulary_level=0.4, math_level=0.3,
            preferred_format="visual", patience=0.6, interests=["explicacao passo a passo"]),
    Profile(id="engenheiro", name="Engenheiro", vocabulary_level=0.7, math_level=0.7,
            preferred_format="code", patience=0.7, interests=["aplicacoes praticas"]),
    Profile(id="cientista", name="Cientista", vocabulary_level=0.9, math_level=0.9,
            preferred_format="text", patience=0.9, interests=["detalhes tecnicos", "equacoes"]),
    Profile(id="crianca", name="Crianca", vocabulary_level=0.1, math_level=0.0,
            preferred_format="visual", patience=0.2, interests=["historias", "desenhos"]),
    Profile(id="programador", name="Programador", vocabulary_level=0.7, math_level=0.5,
            preferred_format="code", patience=0.6, interests=["algoritmos", "exemplos de codigo"]),
]


class AudienceModel:
    """Modelo de audiência.

    Estima o perfil do interlocutor e adapta a comunicacao.
    """

    def __init__(self):
        self.profiles: Dict[str, Profile] = {}
        for p in PRIMORDIAL_PROFILES:
            self.profiles[p.id] = p

    def identify(self, clues: Dict) -> str:
        """Identifica o perfil mais provavel baseado em pistas."""
        best_profile = "leigo"
        best_score = 0.0

        for pid, profile in self.profiles.items():
            score = 0.5  # base
            if "vocabulary" in clues:
                score += 1.0 - abs(profile.vocabulary_level - clues["vocabulary"])
            if "math" in clues:
                score += 1.0 - abs(profile.math_level - clues["math"])
            if "format" in clues and clues["format"] == profile.preferred_format:
                score += 1.0
            if score > best_score:
                best_score = score
                best_profile = pid

        return best_profile

    def get_profile(self, profile_id: str) -> Optional[Profile]:
        return self.profiles.get(profile_id)

    def get_state(self) -> Dict:
        return {"n_profiles": len(self.profiles),
                "profiles": list(self.profiles.keys())}
