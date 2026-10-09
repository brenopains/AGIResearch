"""
TemporalEncoder — Codificacao de Eventos/Sequencias em HDVs Temporais

Video nao e apenas sequencia de imagens.
E uma sequencia de EVENTOS.

Eventos tem:
  - tipo: "queda", "colisao", "movimento", "fragmentacao"
  - duracao: inicio + fim
  - objetos envolvidos
  - causalidade: evento A → evento B

O TemporalEncoder recebe uma sequencia de estados (frames/vetores)
e produz um HDV que representa o EVENTO inteiro.

Diferenca de simplesmente juntar frames:
  O TemporalEncoder extrai RELACOES entre estados consecutivos:
    diff: o que mudou
    diff2: aceleracao
    padroes: repeticao, ritmo, ciclo

Uso:
    encoder = TemporalEncoder()
    frames = [...]  # lista de HDVs ou states
    hdv_evento = encoder.encode(frames)
"""
import time, hashlib, math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalEncoder:
    """Codifica sequencias de estados em HDVs de evento.

    Arquitetura:
      [state_t0, state_t1, ..., state_tN]
          │          │               │
          ▼          ▼               ▼
        diff(0,1)  diff(1,2)   diff(N-1,N)
          │          │               │
          └──────────┼───────────────┘
                     ▼
              bundle temporal
                     ▼
               HDV_evento
    """

    def __init__(self, dim: int = 10000, device: str = "cpu"):
        self.dim = dim
        self.device = torch.device(device)

    def encode(self, states: List[torch.Tensor],
               timestamps: Optional[List[float]] = None) -> Optional[torch.Tensor]:
        """Codifica uma sequencia de estados em HDV de evento.

        Args:
            states: lista de tensores 10000D bipolares (±1)
            timestamps: timestamps de cada estado (opcional)

        Returns:
            HDV 10000D bipolar do evento
        """
        if len(states) < 2:
            return states[0] if states else None

        # Leva todos ao device
        hdvs = [s.flatten().float().to(self.device) for s in states]

        # Computa diferencas entre estados consecutivos (movimento)
        diffs = []
        for i in range(1, len(hdvs)):
            diff = torch.sign(hdvs[i] - hdvs[i-1])  # o que mudou
            diffs.append(diff)

        # Computa aceleracao (diferenca das diferencas)
        accels = []
        for i in range(1, len(diffs)):
            accel = torch.sign(diffs[i] - diffs[i-1])
            accels.append(accel)

        # Bundle: soma todos + sign
        bundle = hdvs[0].clone()
        for d in diffs:
            bundle = bundle + d
        for a in accels[:3]:  # limita aceleracoes
            bundle = bundle + a * 0.5

        # Timestamps: se fornecidos, codifica ritmo
        if timestamps and len(timestamps) >= 2:
            intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            mean_interval = sum(intervals) / len(intervals)
            # Codifica ritmo como modulacao do HDV
            rhythm_seed = int(mean_interval * 1000) % (2**31)
            g = torch.Generator(device=self.device)
            g.manual_seed(rhythm_seed)
            rhythm_vec = torch.sign(torch.randn(self.dim, generator=g, device=self.device))
            bundle = bundle + rhythm_vec * 0.3

        return torch.sign(bundle)

    def encode_with_diff_awareness(self, states: List[torch.Tensor],
                                    return_diff: bool = False) -> Any:
        """Codifica com consciencia de diff (mudanca vs permanencia)."""
        if len(states) < 2:
            return self.encode(states)

        hdvs = [s.flatten().float().to(self.device) for s in states]

        # O que PERMANECE (intersecao dos estados)
        stable = hdvs[0].clone()
        for h in hdvs[1:]:
            stable = torch.sign(stable * h)  # bind = permanece similar

        diff_dict = {}
        for i in range(1, len(hdvs)):
            diff = torch.sign(hdvs[i] - hdvs[i-1])
            diff_dict[f"diff_{i-1}"] = diff

        hdv_evento = self.encode(states)
        return hdv_evento, stable, diff_dict if return_diff else hdv_evento
