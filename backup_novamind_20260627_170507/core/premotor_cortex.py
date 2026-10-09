"""
PremotorCortex — Planejamento Hierárquico Fonema → Sílaba → Gesto
==================================================================
O córtex pré-motor ventral (área 6 rostral) codifica sílabas como
unidades motoras. O córtex pré-frontal dorsolateral codifica
sequências de sílabas.

Este módulo DECOMPÕE um alvo acústico (ex: [f0,f1,f2,f3] extraído
de um vídeo do YouTube) em:
  1. Fonemas (segmentos)
  2. Estrutura silábica (CV, CVC, V, etc.)
  3. Gestos articulatórios para cada fonema
  4. Timing entre gestos (via Cerebellum)

Referência: "Single-neuronal elements of speech production" (Nature 2024)
Hickok (2012) "Computational neuroanatomy of speech production"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


PHONEME_GESTURE_MAP = {
    # Vowels — usam ARTICULATORY_MAP do MotorCortex
    "aa": {"type": "vowel", "articulators": {"tongue_dorsum_x": 0.3, "tongue_dorsum_y": 0.3, "lip_rounding": 0.0, "jaw_open": 0.8, "velum": 0.0, "vocal_fold_tension": 0.5}},
    "ee": {"type": "vowel", "articulators": {"tongue_dorsum_x": 0.65, "tongue_dorsum_y": 0.4, "lip_rounding": 0.0, "jaw_open": 0.5, "velum": 0.0, "vocal_fold_tension": 0.5}},
    "ii": {"type": "vowel", "articulators": {"tongue_dorsum_x": 0.85, "tongue_dorsum_y": 0.8, "lip_rounding": 0.0, "jaw_open": 0.2, "velum": 0.0, "vocal_fold_tension": 0.6}},
    "oo": {"type": "vowel", "articulators": {"tongue_dorsum_x": 0.2, "tongue_dorsum_y": 0.5, "lip_rounding": 0.7, "jaw_open": 0.5, "velum": 0.0, "vocal_fold_tension": 0.5}},
    "uu": {"type": "vowel", "articulators": {"tongue_dorsum_x": 0.1, "tongue_dorsum_y": 0.7, "lip_rounding": 0.8, "jaw_open": 0.3, "velum": 0.0, "vocal_fold_tension": 0.6}},
    "nn": {"type": "vowel", "articulators": {"tongue_dorsum_x": 0.5, "tongue_dorsum_y": 0.5, "lip_rounding": 0.0, "jaw_open": 0.5, "velum": 0.0, "vocal_fold_tension": 0.5}},

    # Plosives
    "p": {"type": "plosive", "place": "bilabial", "closure": {"lip_rounding": 0.9, "jaw_open": 0.1}, "voiced": False},
    "b": {"type": "plosive", "place": "bilabial", "closure": {"lip_rounding": 0.9, "jaw_open": 0.1}, "voiced": True},
    "t": {"type": "plosive", "place": "alveolar", "closure": {"tongue_dorsum_x": 0.8, "tongue_dorsum_y": 0.9, "jaw_open": 0.1}, "voiced": False},
    "d": {"type": "plosive", "place": "alveolar", "closure": {"tongue_dorsum_x": 0.8, "tongue_dorsum_y": 0.9, "jaw_open": 0.1}, "voiced": True},
    "k": {"type": "plosive", "place": "velar", "closure": {"tongue_dorsum_x": 0.1, "tongue_dorsum_y": 0.8, "jaw_open": 0.1}, "voiced": False},
    "g": {"type": "plosive", "place": "velar", "closure": {"tongue_dorsum_x": 0.1, "tongue_dorsum_y": 0.8, "jaw_open": 0.1}, "voiced": True},

    # Fricatives
    "f": {"type": "fricative", "place": "labiodental", "articulators": {"lip_rounding": 0.3, "jaw_open": 0.3}},
    "v": {"type": "fricative", "place": "labiodental", "articulators": {"lip_rounding": 0.3, "jaw_open": 0.3}, "voiced": True},
    "s": {"type": "fricative", "place": "alveolar", "articulators": {"tongue_dorsum_x": 0.8, "tongue_dorsum_y": 0.7, "jaw_open": 0.2}},
    "z": {"type": "fricative", "place": "alveolar", "articulators": {"tongue_dorsum_x": 0.8, "tongue_dorsum_y": 0.7, "jaw_open": 0.2}, "voiced": True},

    # Nasals
    "n": {"type": "nasal", "place": "alveolar", "closure": {"tongue_dorsum_x": 0.8, "tongue_dorsum_y": 0.9}, "velum": 0.9},
    "mmm": {"type": "nasal", "place": "bilabial", "closure": {"lip_rounding": 0.9}, "velum": 0.9},

    # Special
    "silence": {"type": "silence"},
    "schwa": {"type": "vowel", "articulators": {"tongue_dorsum_x": 0.5, "tongue_dorsum_y": 0.5, "jaw_open": 0.5, "lip_rounding": 0.0, "velum": 0.0, "vocal_fold_tension": 0.4}},
}

SYLLABLE_PATTERNS = ["V", "CV", "VC", "CVC", "CVV", "CCV"]

# Mapa de forma acústica para vogal mais próxima
ACOUSTIC_VOWEL_MAP = {
    "aa": (700, 1100), "ee": (500, 1800),
    "ii": (300, 2200), "oo": (500, 900),
    "uu": (300, 900),
}


class PremotorCortex:
    def __init__(self):
        self.phoneme_gesture_map = PHONEME_GESTURE_MAP
        self.syllable_patterns = SYLLABLE_PATTERNS
        self.syllable_examples: Dict[str, List[str]] = defaultdict(list)
        self.plan_history: List[Dict] = []
        self._max_history = 50

    def estimate_phonemes(self, f1: float, f2: float, f0: float = 0) -> List[str]:
        """
        Estima a sequência de fonemas mais provável a partir de parâmetros acústicos.
        Usado para IDENTIFICAR o que foi ouvido no dorsal stream.
        """
        best_vowel = "schwa"
        best_dist = float("inf")
        for name, (vf1, vf2) in ACOUSTIC_VOWEL_MAP.items():
            dist = abs(f1 - vf1) / 500 + abs(f2 - vf2) / 1500
            if dist < best_dist:
                best_dist = dist
                best_vowel = name

        if best_dist < 0.5:
            return [best_vowel]
        elif best_dist < 1.0:
            return [best_vowel]
        return ["schwa"]

    def plan_target(self, f0: float, f1: float, f2: float, f3: float) -> List[Dict]:
        """
        Planeja uma sequência de gestos para IMITAR um som ouvido.

        Returns:
            lista de gestos, onde cada gesto é um dict com:
              - type: "vowel" | "plosive" | "fricative" | "nasal" | "silence"
              - articulators: dict com configuração articulatória
              - duration_ms: duração planejada
              - consonant_type: para synthesize_articulatory
              - consonant_place: para synthesize_articulatory
        """
        phonemes = self.estimate_phonemes(f1, f2, f0)
        if not phonemes:
            return []

        # Constrói sílaba simples (assume CV para consistência)
        syllable = phonemes  # já é uma lista de fonemas

        gestures = []
        for ph in phonemes:
            gesture = self.phoneme_to_gesture(ph)
            if gesture:
                gestures.append(gesture)

        syllable_type = "V"
        if gestures:
            first = gestures[0]
            if first["type"] == "vowel":
                syllable_type = "V"
            else:
                syllable_type = "CV"

        self.plan_history.append({
            "target_f1f2": (f1, f2),
            "phonemes": phonemes,
            "syllable_type": syllable_type,
            "gesture_count": len(gestures),
        })
        if len(self.plan_history) > self._max_history:
            self.plan_history = self.plan_history[-self._max_history:]

        return gestures

    def phoneme_to_gesture(self, phoneme: str) -> Optional[Dict]:
        """Mapeia um fonema para um gesto articulatório executável."""
        data = self.phoneme_gesture_map.get(phoneme)
        if not data:
            return None

        if data["type"] == "vowel":
            art = data["articulators"]
            return {
                "type": "vowel",
                "articulators": art,
                "consonant_type": None,
                "consonant_place": None,
                "duration_ms": 200,
            }
        elif data["type"] == "plosive":
            closure = data["closure"]
            gesture = {
                "type": "plosive",
                "place": data["place"],
                "articulators": {**closure, "vocal_fold_tension": 0.3},
                "consonant_type": "plosive",
                "consonant_place": data["place"],
                "duration_ms": 150,
                "voiced": data.get("voiced", False),
            }
            return gesture
        elif data["type"] == "fricative":
            art = data.get("articulators", {"jaw_open": 0.3})
            return {
                "type": "fricative",
                "place": data["place"],
                "articulators": {**art, "vocal_fold_tension": 0.3},
                "consonant_type": "fricative",
                "consonant_place": data["place"],
                "duration_ms": 200,
            }
        elif data["type"] == "nasal":
            closure = data["closure"]
            return {
                "type": "nasal",
                "place": data["place"],
                "articulators": {**closure, "velum": 0.9, "jaw_open": 0.1, "vocal_fold_tension": 0.3},
                "consonant_type": "nasal",
                "consonant_place": data["place"],
                "duration_ms": 180,
            }
        elif data["type"] == "silence":
            return {"type": "silence", "articulators": {}, "duration_ms": 100}
        return None

    def apply_timing(self, gestures: List[Dict], cerebellum) -> List[Dict]:
        """Aplica correções de timing do cerebelo a cada gesto."""
        timed = []
        for i, g in enumerate(gestures):
            g = g.copy()
            timing = cerebellum.predict_timing(g.get("articulators", {}))
            if "jaw_open_ms" in timing:
                g["duration_ms"] = timing["jaw_open_ms"] + timing.get("tongue_ms", 0) * 0.5
            elif "tongue_ms" in timing:
                g["duration_ms"] = timing["tongue_ms"]
            g["duration_ms"] = max(30, min(500, g.get("duration_ms", 150)))
            timed.append(g)
        return timed

    def get_state(self) -> Dict:
        return {
            "phonemes_mapped": len(self.phoneme_gesture_map),
            "syllables_planned": len(self.plan_history),
            "last_plan": self.plan_history[-1] if self.plan_history else None,
        }
