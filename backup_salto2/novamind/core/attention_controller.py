"""
AttentionController — Atencao Competitiva via Saliencia

Sistema de spotlight que compete recursos cognitivos.
Similar ao cerebro: multiplas regioes competem via inibicao lateral,
e a que "ganha" recebe foco.

Componentes:
  - SalienceMap: mapa de quais regioes estao "quentes"
  - SpotlightController: decide onde focar
  - ChangeDetector: qualquer mudanca no input gera spike de saliencia
"""
import time
import math
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field


@dataclass
class SalienceNode:
    """Um nodo/regiao com nivel de saliencia."""
    node_id: str
    salience: float = 0.0  # 0..1, quao "quente"
    urgency: float = 0.0  # 0..1, quao urgente (decai rapido)
    persistence: float = 0.0  # 0..1, quanto tempo dura
    source: str = "perception"  # perception | prediction | error | memory
    last_updated: float = field(default_factory=time.time)

    def decay(self, dt: float = 1.0):
        """Decaimento natural da saliencia."""
        decay_rate = 0.05 * (1.0 - self.persistence)
        self.salience *= (1.0 - decay_rate * dt)
        self.urgency *= 0.8 ** dt
        if self.salience < 0.01:
            self.salience = 0.0
        if self.urgency < 0.01:
            self.urgency = 0.0


class SalienceMap:
    """Mapa de saliencia global.

    Mantem nodos competindo por atencao.
    Usa inibicao lateral entre nodos similares para evitar
    que multiplos nodos dominem simultaneamente.
    """

    def __init__(self, lateral_inhibition_fn: Optional[Callable] = None):
        self.nodes: Dict[str, SalienceNode] = {}
        self.lateral_inhibition_fn = lateral_inhibition_fn
        self.history: List[Dict] = []

    def add_salience(self, node_id: str, amount: float, urgency: float = 0.0,
                     persistence: float = 0.3, source: str = "perception"):
        """Adiciona saliencia a um nodo (cria se nao existir)."""
        if node_id not in self.nodes:
            self.nodes[node_id] = SalienceNode(
                node_id=node_id, source=source,
                persistence=persistence
            )
        self.nodes[node_id].salience = min(1.0,
            self.nodes[node_id].salience + amount)
        self.nodes[node_id].urgency = max(self.nodes[node_id].urgency, urgency)
        self.nodes[node_id].source = source
        self.nodes[node_id].last_updated = time.time()

        # Inibicao lateral: nodos similares perdem saliencia
        if self.lateral_inhibition_fn:
            similar_nodes = self.lateral_inhibition_fn(node_id)
            if similar_nodes:
                inhibition = amount * 0.5
                for nid in similar_nodes:
                    if nid in self.nodes and nid != node_id:
                        self.nodes[nid].salience = max(0.0,
                            self.nodes[nid].salience - inhibition)

    def get_focus(self) -> Optional[str]:
        """Retorna o nodo com maior saliencia."""
        if not self.nodes:
            return None
        active = {k: v for k, v in self.nodes.items() if v.salience > 0.05}
        if not active:
            return None
        return max(active, key=lambda k: active[k].salience * (0.5 + active[k].urgency))

    def get_focus_score(self, node_id: str) -> float:
        """Retorna score de foco para um nodo especifico."""
        node = self.nodes.get(node_id)
        if not node:
            return 0.0
        return node.salience * (0.5 + node.urgency)

    def tick(self, dt: float = 1.0):
        """Um passo de tempo: decai todas as saliencias."""
        to_remove = []
        for nid, node in self.nodes.items():
            node.decay(dt)
            if node.salience <= 0 and node.urgency <= 0:
                to_remove.append(nid)
        for nid in to_remove:
            del self.nodes[nid]

        # Log
        focus = self.get_focus()
        self.history.append({
            "time": time.time(),
            "n_active": len(self.nodes),
            "focus": focus,
            "focus_salience": self.nodes[focus].salience if focus and focus in self.nodes else 0,
        })

    def get_state(self) -> Dict:
        return {
            "n_nodes": len(self.nodes),
            "focus": self.get_focus(),
            "top_saliences": sorted(
                [(k, v.salience) for k, v in self.nodes.items()],
                key=lambda x: -x[1]
            )[:5],
        }


class ChangeDetector:
    """Detecta mudancas no fluxo perceptual e gera spikes de saliencia.

    Qualquer mudanca no input:
      - novo objeto aparece → spike
      - objeto se move → spike
      - cor/forma muda → spike
      - energia muda abruptamente → spike
    """

    def __init__(self, threshold: float = 0.15):
        self.previous_state: Dict[str, float] = {}
        self.threshold = threshold
        self.changes: List[Dict] = []

    def detect(self, current_state: Dict[str, float]) -> List[Dict[str, Any]]:
        """Compara estado atual com anterior, retorna mudancas."""
        spikes = []
        all_keys = set(self.previous_state.keys()) | set(current_state.keys())

        for key in all_keys:
            prev = self.previous_state.get(key, 0.0)
            curr = current_state.get(key, 0.0)
            diff = abs(curr - prev)

            if diff > self.threshold:
                spike = {
                    "key": key,
                    "delta": diff,
                    "direction": "up" if curr > prev else "down",
                    "old_value": prev,
                    "new_value": curr,
                    "time": time.time(),
                }
                spikes.append(spike)
                self.changes.append(spike)

        self.previous_state = dict(current_state)
        return spikes

    def get_recent_changes(self, n: int = 5) -> List[Dict]:
        return self.changes[-n:]


class SpotlightController:
    """Controla o foco atencional.

    Integra SalienceMap + ChangeDetector e decide:
      - ONDE focar agora
      - QUANDO mudar de foco
      - O QUE ignorar (supressao)
    """

    def __init__(self, salience_map: SalienceMap,
                 change_detector: ChangeDetector):
        self.salience_map = salience_map
        self.change_detector = change_detector
        self.current_focus: Optional[str] = None
        self.focus_duration: float = 0.0
        self.focus_history: List[Dict] = []
        self.suppressed: Set[str] = set()  # nodos sendo ignorados
        self.min_focus_time: float = 2.0  # segundos minimos de foco

    def update(self, perceptual_state: Dict[str, float], dt: float = 1.0):
        """Ciclo de atualizacao do spotlight.

        1. Detecta mudancas → spikes de saliencia
        2. Decai saliencias existentes
        3. Decide se mantem ou troca foco
        """
        # Detectar mudancas
        changes = self.change_detector.detect(perceptual_state)
        for c in changes:
            self.salience_map.add_salience(
                c["key"], min(1.0, c["delta"] * 2),
                urgency=0.8, source="change"
            )

        # Decair
        self.salience_map.tick(dt)
        self.focus_duration += dt

        # Decidir foco
        candidate = self.salience_map.get_focus()

        # Trocar se:
        # 1. Nao tem foco atual
        # 2. Candidato e diferente e passou tempo minimo
        # 3. Candidato tem urgencia muito alta
        if candidate and candidate not in self.suppressed:
            if (self.current_focus is None or
                (candidate != self.current_focus and
                 self.focus_duration >= self.min_focus_time) or
                (self.salience_map.nodes.get(candidate, SalienceNode("")).urgency > 0.7)):

                if self.current_focus:
                    self.focus_history.append({
                        "focus": self.current_focus,
                        "duration": self.focus_duration,
                        "time": time.time(),
                    })

                self.current_focus = candidate
                self.focus_duration = 0.0

        return self.current_focus

    def suppress(self, node_id: str):
        """Ignora um nodo (usado quando ja foi processado)."""
        self.suppressed.add(node_id)
        if node_id in self.salience_map.nodes:
            del self.salience_map.nodes[node_id]

    def unsuppress(self, node_id: str):
        """Permite um nodo novamente."""
        self.suppressed.discard(node_id)

    def get_state(self) -> Dict:
        return {
            "focus": self.current_focus,
            "focus_duration": round(self.focus_duration, 1),
            "suppressed": len(self.suppressed),
            "salience_map": self.salience_map.get_state(),
        }
