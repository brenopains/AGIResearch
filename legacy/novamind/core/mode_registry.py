"""
CognitiveMode — Modo Cognitivo DINAMICO (nao enum)

Diferenca do enum anterior:
  - Novos modos podem ser criados em RUNTIME pela ForjaEvolution
  - Cada modo tem perfil comportamental proprio
  - O MetaController opera sobre instancias, nao sobre valores fixos

A ForjaEvolution descobre novos modos minerando CognitiveTraces
e os registra aqui. O MetaController passa a usar automaticamente.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ModeProfile:
    """Perfil de um modo cognitivo."""
    name: str
    description: str = ""
    requires_external_input: bool = False
    generates_knowledge: bool = False
    is_active: bool = True

    # Parametros comportamentais (o que este modo faz)
    search_bias: str = "balanced"  # "explore", "exploit", "reflect"
    error_sensitivity: float = 0.5
    simulation_depth: float = 0.5
    forge_rate: float = 0.5

    # Metadados
    primordial: bool = False
    confidence: float = 0.5
    created_by: str = ""  # "primordial", "forja_evolution", "meta_forja"


# Modos primordiais (os mesmos 7 do enum antigo, agora como data)
PRIMORDIAL_MODES = {
    "explore": ModeProfile("explore", "Busca novidade, expande conhecimento",
                          requires_external_input=True, search_bias="explore",
                          forge_rate=0.3, primordial=True, confidence=0.95),
    "hypothesize": ModeProfile("hypothesize", "Gera candidatos de explicacao",
                              generates_knowledge=True, search_bias="reflect",
                              simulation_depth=0.7, primordial=True, confidence=0.9),
    "predict": ModeProfile("predict", "Simula consequencias de hipoteses",
                          simulation_depth=0.9, primordial=True, confidence=0.85),
    "test": ModeProfile("test", "Executa acao para refutar hipotese",
                       requires_external_input=True, search_bias="explore",
                       primordial=True, confidence=0.85),
    "consolidate": ModeProfile("consolidate", "Comprime, abstrai, esquece irrelevante",
                              generates_knowledge=True, search_bias="exploit",
                              forge_rate=0.8, primordial=True, confidence=0.9),
    "exploit": ModeProfile("exploit", "Usa conhecimento consolidado",
                          search_bias="exploit", primordial=True, confidence=0.85),
    "reflect": ModeProfile("reflect", "Reavalia estrategia atual",
                          generates_knowledge=True, search_bias="reflect",
                          error_sensitivity=0.8, primordial=True, confidence=0.85),
    "idle": ModeProfile("idle", "Modo inativo (pausa)",
                       is_active=False, primordial=True, confidence=0.99),
}


class ModeRegistry:
    """Registro DINAMICO de modos cognitivos.

    Diferenca do CognitiveMode(Enum):
      - Qualquer codigo pode registrar um novo modo via register()
      - O MetaController itera sobre o registro, nao sobre um enum fixo
      - A ForjaEvolution pode criar modos especializados para dominios
    """

    def __init__(self):
        self._modes: Dict[str, ModeProfile] = {}
        self._creation_log: List[Dict] = []

        for name, profile in PRIMORDIAL_MODES.items():
            self._modes[name] = profile

    def register(self, name: str, profile: ModeProfile) -> str:
        """Registra um novo modo em runtime."""
        key = name.lower().replace(" ", "_")
        if key in self._modes:
            # Atualizar existencia
            existing = self._modes[key]
            if profile.confidence > existing.confidence:
                existing.confidence = profile.confidence
            return key
        self._modes[key] = profile
        self._creation_log.append({
            "name": key, "description": profile.description[:60],
            "created_by": profile.created_by,
        })
        return key

    def get(self, name: str) -> Optional[ModeProfile]:
        return self._modes.get(name.lower().replace(" ", "_"))

    def get_all(self, active_only: bool = True) -> Dict[str, ModeProfile]:
        if active_only:
            return {k: v for k, v in self._modes.items() if v.is_active}
        return dict(self._modes)

    def get_active_names(self) -> List[str]:
        return [k for k, v in self._modes.items() if v.is_active]

    def get_primordial_names(self) -> List[str]:
        return [k for k, v in self._modes.items() if v.primordial]

    def get_discovered_names(self) -> List[str]:
        return [k for k, v in self._modes.items() if not v.primordial]

    def get_state(self) -> Dict:
        return {
            "total": len(self._modes),
            "primordial": len(self.get_primordial_names()),
            "discovered": len(self.get_discovered_names()),
            "active": self.get_active_names(),
        }


# Instancia global (substitui CognitiveMode enum)
MODE_REGISTRY = ModeRegistry()
