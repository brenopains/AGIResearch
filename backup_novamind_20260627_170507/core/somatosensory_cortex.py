"""
SomatosensoryCortex — Propriocepção do Trato Vocal
===================================================
Um bebê SENTE a língua no céu da boca, a mandíbula abrindo,
as pregas vibrando. Este córtex modela essa consciência corporal
do aparelho fonador.

Fornece feedback somatossensorial para o DIVAController:
  - Onde está cada articulador AGORA?
  - Há contato entre partes? (ex: língua no palato)
  - Qual a tensão muscular estimada?
  - Descrição textual para auto-percepção
"""

import numpy as np
from typing import Dict, Optional


CONTACT_THRESHOLD = 0.85


class SomatosensoryCortex:
    """
    Córtex Somatossensorial — mapeia o estado do trato vocal.

    Cada articulador tem uma representação proprioceptiva:
      tongue_dorsum_x: float [0,1]  (0=posterior, 1=anterior)
      tongue_dorsum_y: float [0,1]  (0=baixa, 1=alta)
      lip_rounding:    float [0,1]  (0=esticado, 1=arredondado)
      jaw_open:        float [0,1]  (0=fechado, 1=aberto)
      velum:           float [0,1]  (0=fechado/oral, 1=aberto/nasal)
      vocal_fold_tension: float [0,1] (0=relaxado, 1=tenso)

    Gera sensações táteis:
      - contato língua-palato (quando língua alta o suficiente)
      - contato labial (quando lábios se tocam)
      - vibração das pregas vocais (detectada por pitch/f0)
    """

    def __init__(self):
        self.last_proprioception: Dict = {}
        self.last_tactile: Dict = {}
        self.last_description: str = ""
        self.history: list = []

    def sense(self, articulatory_config: Dict, acoustic_feedback: Dict) -> Dict:
        """
        Processa a configuração articulatória atual e retorna
        a percepção somatossensorial completa.

        Args:
            articulatory_config: dict com {tongue_dorsum_x, tongue_dorsum_y, ...}
            acoustic_feedback: dict com {f0, f1, f2, f3, loudness}

        Returns:
            dict com proprioception, tactile, self_description
        """
        tx = articulatory_config.get("tongue_dorsum_x", 0.5)
        ty = articulatory_config.get("tongue_dorsum_y", 0.5)
        lip = articulatory_config.get("lip_rounding", 0.0)
        jaw = articulatory_config.get("jaw_open", 0.5)
        velum_cfg = articulatory_config.get("velum", 0.0)
        tension = articulatory_config.get("vocal_fold_tension", 0.6)
        f0 = acoustic_feedback.get("f0", 120)

        # Propriocepção: onde os articuladores estão
        proprioception = {
            "tongue_position": {
                "front_back": float(tx),
                "high_low": float(ty),
                "description": self._describe_tongue(tx, ty),
            },
            "lip_aperture": float(1.0 - lip),
            "lip_rounded": lip > 0.5,
            "jaw_position": {
                "openness": float(jaw),
                "description": "aberta" if jaw > 0.6 else "fechada" if jaw < 0.3 else "semi-aberta",
            },
            "velum_position": {
                "openness": float(velum_cfg),
                "description": "nasal" if velum_cfg > 0.5 else "oral",
            },
            "vocal_fold_state": {
                "tension": float(tension),
                "description": "tensa" if tension > 0.6 else "relaxada",
            },
        }

        # Tacto: detecção de contato entre articuladores
        tactile = {}
        has_palatal_contact = ty >= CONTACT_THRESHOLD
        has_lip_contact = lip >= CONTACT_THRESHOLD
        is_vibrating = f0 > 60

        tactile["palatal_contact"] = {
            "detected": bool(has_palatal_contact),
            "location": "palato duro" if tx > 0.5 else "palato mole",
            "intensity": float(max(0, (ty - CONTACT_THRESHOLD) / (1 - CONTACT_THRESHOLD))),
        }
        tactile["lip_contact"] = {
            "detected": bool(has_lip_contact),
            "intensity": float(max(0, (lip - CONTACT_THRESHOLD) / (1 - CONTACT_THRESHOLD))),
        }
        tactile["vocal_fold_vibration"] = {
            "detected": bool(is_vibrating),
            "frequency_hz": float(f0),
            "intensity": float(min(1.0, tension * (f0 / 300))),
        }
        tactile["airflow"] = {
            "oral": float(jaw * (1.0 - velum_cfg * 0.5)),
            "nasal": float(velum_cfg * 0.5),
        }

        self.last_proprioception = proprioception
        self.last_tactile = tactile

        desc = self._generate_description(proprioception, tactile)
        self.last_description = desc
        self.history.append({
            "description": desc,
            "timestamp": __import__("time").time(),
        })
        if len(self.history) > 50:
            self.history = self.history[-50:]

        return {
            "proprioception": proprioception,
            "tactile": tactile,
            "self_description": desc,
        }

    def _describe_tongue(self, tx: float, ty: float) -> str:
        height = "alta" if ty > 0.7 else "baixa" if ty < 0.3 else "media"
        front = "anterior" if tx > 0.7 else "posterior" if tx < 0.3 else "central"
        return f"lingua {height} {front}"

    def _generate_description(self, prop: Dict, tact: Dict) -> str:
        parts = []
        tongue_desc = prop.get("tongue_position", {}).get("description", "")
        if tongue_desc:
            parts.append(tongue_desc)
        jaw_desc = prop.get("jaw_position", {}).get("description", "")
        if jaw_desc:
            parts.append(f"mandibula {jaw_desc}")
        if prop.get("lip_rounded"):
            parts.append("labios arredondados")
        velum_desc = prop.get("velum_position", {}).get("description", "")
        if velum_desc and velum_desc == "nasal":
            parts.append("passagem nasal aberta")
        tension_desc = prop.get("vocal_fold_state", {}).get("description", "")
        if tension_desc:
            parts.append(f"pregas vocais {tension_desc}")
        palatal = tact.get("palatal_contact", {})
        if palatal.get("detected"):
            parts.append("lingua tocando o palato")
        vibration = tact.get("vocal_fold_vibration", {})
        if vibration.get("detected"):
            freq = vibration.get("frequency_hz", 0)
            parts.append(f"pregas vibrando a {freq:.0f}Hz")
        return f"[Sensacao: {', '.join(parts)}]"

    def get_state(self) -> Dict:
        return {
            "last_description": self.last_description,
            "history_count": len(self.history),
        }
