"""
BrocaV2Model — Decodificador LSTM Generativo de HDC para Texto
================================================================
Arquitetura:
  1. Projection: HDC 10000D → 512D (contexto da "intenção")
  2. Embedding: vocab_size × 256
  3. LSTM: 2 layers, hidden=512, dropout=0.3
  4. Output: Linear(512, vocab_size)

Total: ~15M parâmetros (~200MB em fp32, ~50MB em fp16)

Input: hdc_vector (10000D) + tokens gerados anteriormente
Output: logits sobre o vocabulário (next token prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BrocaV2Model(nn.Module):
    def __init__(self, vocab_size: int, hdc_dim: int = 10000,
                 embed_dim: int = 256, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hdc_dim = hdc_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Projeção HDC → hidden state inicial da LSTM
        self.hdc_proj = nn.Sequential(
            nn.Linear(hdc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        # Embedding de tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Projeção de saída
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Inicialização Xavier para estabilidade."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        hdc_vector: torch.Tensor,
        input_ids: torch.Tensor,
        hidden: Optional[tuple] = None,
    ) -> tuple:
        """
        Forward pass com teacher forcing.

        Args:
            hdc_vector: (batch, 10000) — vetor HDC de contexto
            input_ids: (batch, seq_len) — tokens de entrada (deslocados)
            hidden: estado inicial opcional da LSTM

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: estado final da LSTM
        """
        batch_size = hdc_vector.shape[0]

        # Projetar HDC para hidden state inicial
        if hidden is None:
            h0 = self.hdc_proj(hdc_vector).unsqueeze(0)  # (1, batch, hidden)
            h0 = h0.expand(self.num_layers, -1, -1).contiguous()
            c0 = torch.zeros_like(h0)
            hidden = (h0, c0)

        # Embedding
        emb = self.embedding(input_ids)  # (batch, seq_len, embed)
        emb = self.dropout(emb)

        # LSTM
        lstm_out, hidden = self.lstm(emb, hidden)

        # Projeção de saída
        logits = self.output_proj(lstm_out)  # (batch, seq_len, vocab_size)

        return logits, hidden

    def generate(
        self,
        hdc_vector: torch.Tensor,
        max_length: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_id: int = 2,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Geração autoregressiva.

        Args:
            hdc_vector: (1, 10000) — vetor HDC de contexto
            max_length: máximo de tokens a gerar
            temperature: escala da softmax (>1 mais criativo, <1 mais determinístico)
            top_k: amostragem apenas dos top_k tokens
            eos_id: ID do token de fim de sequência

        Returns:
            generated_ids: (1, seq_len) — tokens gerados
        """
        if device is None:
            device = hdc_vector.device

        self.eval()
        with torch.no_grad():
            batch_size = hdc_vector.shape[0]

            # Estado inicial da LSTM a partir do HDC
            h0 = self.hdc_proj(hdc_vector).unsqueeze(0)
            h0 = h0.expand(self.num_layers, -1, -1).contiguous()
            c0 = torch.zeros_like(h0)
            hidden = (h0, c0)

            # Token inicial: <bos>
            input_id = torch.tensor([[1]], device=device)  # BOS_ID = 1
            generated_ids = [1]
            penalty_cache = {}

            for _ in range(max_length):
                emb = self.embedding(input_id)
                lstm_out, hidden = self.lstm(emb, hidden)
                logits = self.output_proj(lstm_out.squeeze(1))

                # Repetition penalty
                for tid in generated_ids:
                    logits[0, tid] = logits[0, tid] * 0.5

                # Temperature
                logits = logits / temperature

                # Top-K sampling
                if top_k > 0:
                    values, _ = torch.topk(logits, top_k)
                    threshold = values[:, -1].unsqueeze(1)
                    logits[logits < threshold] = float("-inf")

                # Softmax e amostragem
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                next_tok = next_id.item()
                generated_ids.append(next_tok)

                if next_tok == eos_id:
                    break

                input_id = next_id

            return torch.tensor([generated_ids], device=device)
