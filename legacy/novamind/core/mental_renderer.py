"""
MentalDecoder — Interface dos Tres Decodificadores

Nao desenha. Nao renderiza.
DECODIFICA: traduz estado cognitivo em midia para humanos.

Decoders em core/scene_decoder.py:
  Reconstruction: "foi isso que voce viu?"
  Imagination: "o que voce imaginou?"
  Communication: "melhor explicacao para humano"
"""
from typing import Dict, Optional, Any
from .scene_decoder import MentalDecoder as _MentalDecoder


class MentalDecoder:
    """Interface unificada dos tres decodificadores.

    Nao renderiza (desenho procedural). Decodifica (traducao aprendida).
    """

    def __init__(self, visual_pipeline=None, vocoder=None):
        self._core = _MentalDecoder(reconstruction_kwargs={"visual_pipeline": visual_pipeline})

    def reconstruct(self, state: Dict, mode: str = "reconstruction") -> Any:
        if mode == "reconstruction":
            return self._core.reconstruct(state)
        elif mode == "imagination":
            return self._core.imagine(state)
        elif mode == "communication":
            return self._core.communicate(state)
        return self._core.reconstruct(state)

    def get_state(self) -> Dict:
        return self._core.get_state()
