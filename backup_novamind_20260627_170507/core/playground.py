import random
import time
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque


class PlaygroundState:
    """
    Simulated environment for embodied learning.

    The AGI's actions produce real sensory consequences here.
    This closes the perception-action loop that's essential for
    genuine learning.

    Like a child's playpen — safe space to try things and see
    what happens. No external rewards, just the intrinsic
    satisfaction of predicting correctly.
    """

    def __init__(self):
        self.position = (0, 0, 0)
        self.held_object = None
        self.nearby_objects = ["pedra", "galho", "folha", "agua", "areia"]
        self.sound_scape = []
        self.visual_scape = ["ceu_azul", "arvore", "chao"]
        self.last_action_result = ""
        self.action_history = deque(maxlen=20)
        self.surprise_history = deque(maxlen=20)

    def apply_action(self, action_primitive: str) -> str:
        """
        Apply a motor primitive to the environment and return
        the sensory consequence (what the AGI would see/hear/feel).

        This is the key: actions → real consequences.
        """
        result = ""
        if action_primitive == "vocalizar_aa":
            result = self._vocalize("aa")
        elif action_primitive == "vocalizar_ee":
            result = self._vocalize("ee")
        elif action_primitive == "vocalizar_mmm":
            result = self._vocalize("mmm")
        elif action_primitive == "apontar":
            result = self._point()
        elif action_primitive == "agarrar":
            result = self._grab()
        elif action_primitive == "soltar":
            result = self._release()
        elif action_primitive == "empurrar":
            result = self._push()
        elif action_primitive == "puxar":
            result = self._pull()
        elif action_primitive == "virar_esquerda":
            result = self._turn("left")
        elif action_primitive == "virar_direita":
            result = self._turn("right")
        elif action_primitive == "avancar":
            result = self._move_forward()
        elif action_primitive == "recuar":
            result = self._move_back()
        elif action_primitive == "riscar_linha":
            result = self._draw("line")
        elif action_primitive == "riscar_circulo":
            result = self._draw("circle")
        elif action_primitive == "riscar_ponto":
            result = self._draw("dot")
        else:
            result = f"[{action_primitive}]: nada acontece"

        self.last_action_result = result
        self.action_history.append({
            "action": action_primitive,
            "result": result,
            "time": time.time(),
        })
        return result

    def _vocalize(self, sound: str) -> str:
        """Vocalization produces a sound in the environment."""
        self.sound_scape.append(sound)
        if len(self.sound_scape) > 10:
            self.sound_scape.pop(0)
        # The sound echoes and interacts with nearby objects
        echo = f"[Audio: {sound}] som ecoa no ambiente"
        return echo

    def _point(self) -> str:
        obj = random.choice(self.nearby_objects)
        return f"[Visual: dedo aponta para {obj}]"

    def _grab(self) -> str:
        if self.held_object:
            return f"[Tato: ja segurando {self.held_object}]"
        obj = random.choice(self.nearby_objects)
        self.held_object = obj
        self.nearby_objects.remove(obj)
        return f"[Tato: segurando {obj}]"

    def _release(self) -> str:
        if not self.held_object:
            return "[Tato: mao vazia]"
        obj = self.held_object
        self.nearby_objects.append(obj)
        self.held_object = None
        return f"[Tato: soltou {obj}]"

    def _push(self) -> str:
        obj = self.held_object or random.choice(self.nearby_objects + ["nada"])
        if obj == "nada":
            return "[Propriocepcao: empurra o ar]"
        return f"[Tato: empurra {obj}, ele se move {random.choice(['pouco', 'muito', 'rapidamente'])}]"

    def _pull(self) -> str:
        obj = self.held_object or "nada"
        if obj == "nada":
            return "[Propriocepcao: puxa o ar]"
        return f"[Tato: puxa {obj} em sua direcao]"

    def _turn(self, direction: str) -> str:
        x, y, z = self.position
        if direction == "left":
            new_pos = (x - 1, y, z)
        else:
            new_pos = (x + 1, y, z)
        self.position = new_pos
        return f"[Visual: virou para {direction}. Agora ve {random.choice(self.visual_scape)}]"

    def _move_forward(self) -> str:
        x, y, z = self.position
        self.position = (x, y + 1, z)
        obj = random.choice(self.nearby_objects + ["nada"])
        if obj == "nada":
            return f"[Visual: avanca. Nada de novo a frente]"
        return f"[Visual: avanca. Ve {obj} a frente]"

    def _move_back(self) -> str:
        x, y, z = self.position
        self.position = (x, y - 1, z)
        return f"[Propriocepcao: recuou. Posicao: {self.position}]"

    def _draw(self, shape: str) -> str:
        descriptions = {
            "line": "Uma linha reta aparece no chao",
            "circle": "Um circulo imperfeito se forma na areia",
            "dot": "Um ponto marca a superficie",
        }
        return f"[Visual: {descriptions.get(shape, shape)}]"

    def observe(self) -> str:
        """Return current sensory state as a text description."""
        parts = [f"[Posicao: {self.position}]"]
        if self.held_object:
            parts.append(f"[Tato: segurando {self.held_object}]")
        if self.sound_scape:
            parts.append(f"[Audio: sons recentes: {', '.join(self.sound_scape[-3:])}]")
        parts.append(f"[Visual: ve {', '.join(self.visual_scape)}]")
        parts.append(f"[Propriocepcao: corpo presente no espaco]")
        return " ".join(parts)

    def reset(self):
        self.__init__()
