"""
FusionLayer — Fusao de HDVs Modality-Especificos em HDV Compartilhado

Cada modalidade tem seu proprio encoder:
  Visual: VisualPipeline (ResNet18 → 512D → proj → 10000D)
  Audio: EnCodec → codes → binding → 10000D
  Texto: BGE/spaCy → 300/768D → proj → 10000D

Cada um produz seu proprio HDV. O FusionLayer combina todos em
um UNICO HDV semantico que representa o conceito multimoda.

A hipotese: HDV(imagem_cachorro) ≈ HDV(latido) ≈ HDV(texto_cachorro)
"""
import time, hashlib, math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FusionLayer:
    """Fusao de HDVs modality-especificos em HDV compartilhado.

    Arquitetura:
      HDV_visual (10000D) ─┐
      HDV_audio  (10000D) ─┼──→ Fusion → HDV_compartilhado (10000D)
      HDV_text   (10000D) ─┘

    A fusao pode ser:
      - bundle (soma + sign): simples, preserva info de todos
      - media ponderada: pesos aprendidos por modalidade
      - atencao: pesos dinamicos baseados em confianca

    NAO hardcoded: a estrategia de fusao e aprendida.
    """

    def __init__(self, strategy: str = "bundle", device: str = "cpu"):
        self.device = torch.device(device)
        self.strategy = strategy
        self._weights = {"visual": 1.0, "audio": 1.0, "text": 1.0}  # aprendidos
        self._bias = {"visual": 0.0, "audio": 0.0, "text": 0.0}

    def fuse(self, hdv_visual: Optional[torch.Tensor] = None,
             hdv_audio: Optional[torch.Tensor] = None,
             hdv_text: Optional[torch.Tensor] = None,
             confidences: Optional[Dict[str, float]] = None) -> Optional[torch.Tensor]:
        """Funde multiplos HDVs em um unico HDV compartilhado.

        Args:
            hdv_visual: HDV 10000D bipolar (±1) do VisualPipeline
            hdv_audio: HDV 10000D bipolar do AudioPipeline
            hdv_text: HDV 10000D bipolar do BGEEncoder/GroundingEngine
            confidences: confianca em cada modalidade (0..1)

        Returns:
            HDV 10000D bipolar fundido, ou None se nenhum disponivel
        """
        available = {}
        if hdv_visual is not None:
            available["visual"] = hdv_visual.flatten().to(self.device)
        if hdv_audio is not None:
            available["audio"] = hdv_audio.flatten().to(self.device)
        if hdv_text is not None:
            available["text"] = hdv_text.flatten().to(self.device)

        if not available:
            return None

        # Pesos por modalidade (aprendidos, ajustados por confianca)
        weights = {}
        for mod in available:
            w = self._weights.get(mod, 1.0)
            if confidences:
                w *= confidences.get(mod, 0.5) * 2
            weights[mod] = max(0.1, w)

        if self.strategy == "bundle":
            # Soma ponderada + sign (bundle classico)
            acc = torch.zeros(10000, device=self.device)
            for mod, hdv in available.items():
                acc += hdv * weights[mod]
            return torch.sign(acc)

        elif self.strategy == "attention":
            # Atencao: pesos baseados em similaridade com HDV medio
            all_hdvs = torch.stack(list(available.values()))
            mean_hdv = torch.sign(all_hdvs.mean(dim=0))
            sims = torch.tensor([
                F.cosine_similarity(hdv.unsqueeze(0), mean_hdv.unsqueeze(0))
                for hdv in available.values()
            ])
            attn = F.softmax(sims * 2.0, dim=0)
            acc = torch.zeros(10000, device=self.device)
            for i, hdv in enumerate(available.values()):
                acc += hdv * attn[i]
            return torch.sign(acc)

        else:
            # Media simples
            acc = torch.zeros(10000, device=self.device)
            n = len(available)
            for hdv in available.values():
                acc += hdv
            return torch.sign(acc / n)

    def learn_weights(self, modality: str, correction: float = 0.05):
        """Ajusta o peso de uma modalidade baseado em feedback."""
        old = self._weights.get(modality, 1.0)
        self._weights[modality] = max(0.1, min(3.0, old + correction))

    def get_state(self) -> Dict:
        return {
            "strategy": self.strategy,
            "weights": self._weights,
        }


# ─── MODALITY MATCHING (para consistencia multimodal) ──────

class ModalityMatcher:
    """Verifica se duas modalidades convergem para o mesmo HDV.

    Usado no treinamento de consistencia multimodal:
      HDV(imagem_cachorro) ≈ HDV(texto_cachorro) ≈ HDV(audio_latido)
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def similarity(self, hdv_a: torch.Tensor, hdv_b: torch.Tensor) -> float:
        """Similaridade entre dois HDVs (cosine)."""
        a = hdv_a.flatten().float().to(self.device)
        b = hdv_b.flatten().float().to(self.device)
        sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
        return float(sim.item())

    def are_same_concept(self, hdv_a: torch.Tensor, hdv_b: torch.Tensor,
                         threshold: float = 0.5) -> bool:
        """Verifica se dois HDVs representam o mesmo conceito."""
        return self.similarity(hdv_a, hdv_b) >= threshold
