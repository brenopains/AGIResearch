"""
BrocaV2 — Interface de Inferência do Decodificador HDC → Texto
================================================================
Usa o modelo treinado (LSTM) para gerar texto fluente a partir de
vetores HDC. Integra com o GroundingEngine para codificar conceitos
como contexto.

Uso:
    decoder = BrocaV2()
    decoder.load("D:/temp/broca_v2/broca_v2.pt")

    # A partir de um conceito HDC
    texto = decoder.generate_from_hdv(hdv_vector, max_length=100)

    # A partir de um texto (codifica automaticamente)
    texto = decoder.generate_from_text("dog running in park", max_length=80)

    # Geração livre (sem contexto HDC)
    texto = decoder.generate(prompt="Once upon a time", max_length=200)
"""

import os
import json
from typing import Optional, List
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import BrocaV2Model
from .tokenizer import BPETokenizer

# Caminhos padrão
DEFAULT_MODEL_DIR = "D:/temp/broca_v2"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "broca_v2.pt")
DEFAULT_TOKENIZER_PATH = os.path.join(DEFAULT_MODEL_DIR, "tokenizer.json")


class BrocaV2:
    """
    Decodificador de texto fluente a partir de vetores HDC.

    Carrega o modelo LSTM treinado + tokenizador BPE.
    Gera texto autoregressivamente condicionado a um vetor HDC.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[BrocaV2Model] = None
        self.tokenizer: Optional[BPETokenizer] = None
        self.engine = None  # GroundingEngine, setado externamente
        self._loaded = False

    def load(self, model_path: str = DEFAULT_MODEL_PATH,
             tokenizer_path: str = DEFAULT_TOKENIZER_PATH):
        """
        Carrega o modelo treinado e tokenizador do disco.

        Args:
            model_path: caminho para o .pt do modelo
            tokenizer_path: caminho para o .json do tokenizador
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo nao encontrado: {model_path}. "
                f"Execute core.broca_v2.train primeiro.")

        # Tokenizador
        self.tokenizer = BPETokenizer.load(tokenizer_path)

        # Modelo
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        config = ckpt.get("config", {})

        self.model = BrocaV2Model(
            vocab_size=config.get("vocab_size", self.tokenizer.vocab_size_real),
            hdc_dim=config.get("hdc_dim", 10000),
            embed_dim=config.get("embed_dim", 256),
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 2),
            dropout=0.0,  # dropout desligado em inferência
        ).to(self.device)

        # Tenta carregar state_dict do checkpoint ou direto
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)

        self.model.eval()
        self._loaded = True
        print(f"[BrocaV2] Modelo carregado: {model_path}")
        print(f"[BrocaV2] Vocabulario: {len(self.tokenizer)} tokens")
        print(f"[BrocaV2] Device: {self.device}")

    def set_engine(self, engine):
        """Conecta o GroundingEngine para codificar textos como HDC."""
        self.engine = engine

    # ─── Geração a partir de HDC ────────────────────────────────

    @torch.no_grad()
    def generate_from_hdv(
        self,
        hdv_vector: torch.Tensor,
        max_length: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """
        Gera texto a partir de um vetor HDC.

        Args:
            hdv_vector: (10000,) ou (1, 10000) — vetor HDC de contexto
            max_length: máximo de tokens a gerar
            temperature: escala softmax (0.5=deterministico, 1.5=criativo)
            top_k: amostragem dos top_k tokens

        Returns:
            texto gerado
        """
        self._require_loaded()

        if hdv_vector.dim() == 1:
            hdv_vector = hdv_vector.unsqueeze(0)
        hdv_vector = hdv_vector.to(self.device).float()

        if hdv_vector.shape[1] != 10000:
            # Padding ou truncamento
            if hdv_vector.shape[1] < 10000:
                pad = torch.zeros(1, 10000 - hdv_vector.shape[1], device=self.device)
                hdv_vector = torch.cat([hdv_vector, pad], dim=1)
            else:
                hdv_vector = hdv_vector[:, :10000]

        generated_ids = self.model.generate(
            hdv_vector,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            eos_id=self.tokenizer.EOS_ID,
            device=self.device,
        )

        text = self.tokenizer.decode(generated_ids[0].tolist())
        return text

    @torch.no_grad()
    def generate_from_text(
        self,
        text: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """
        Gera texto a partir de um prompt textual.
        Codifica o prompt como HDC e usa como contexto.

        Args:
            text: prompt de texto
            max_length: máximo de tokens a gerar
            temperature: escala softmax
            top_k: amostragem

        Returns:
            texto gerado (continuação do prompt)
        """
        if self.engine is None:
            # Fallback: geração livre sem contexto HDC
            ids = self.tokenizer.encode(text, add_special=True)
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            hdv_vec = torch.zeros(1, 10000, device=self.device)
            hdv_vec[:, 0] = 1.0
        else:
            # Codifica o prompt como HDC
            mf = self.engine.encode_text(text)
            hdv_vec = mf.tensor.unsqueeze(0).float()
            input_ids = torch.tensor([[self.tokenizer.BOS_ID]], device=self.device)

        # Gera continuação
        generated_ids = self.model.generate(
            hdv_vec,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            eos_id=self.tokenizer.EOS_ID,
            device=self.device,
        )

        full_text = self.tokenizer.decode(generated_ids[0].tolist())

        # Se usou prompt como texto, mostra só a parte gerada
        return full_text

    def generate(
        self,
        prompt: str = "",
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """
        Geração livre (sem contexto HDC).

        Args:
            prompt: texto inicial opcional
            max_length: máximo de tokens
            temperature: escala softmax
            top_k: amostragem

        Returns:
            texto gerado
        """
        self._require_loaded()

        if prompt:
            return self.generate_from_text(prompt, max_length, temperature, top_k)

        # Geração do zero (só <bos>)
        input_ids = torch.tensor([[self.tokenizer.BOS_ID]], device=self.device)
        hdv_vec = torch.zeros(1, 10000, device=self.device)
        hdv_vec[:, 0] = 1.0

        generated_ids = self.model.generate(
            hdv_vec,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            eos_id=self.tokenizer.EOS_ID,
            device=self.device,
        )

        return self.tokenizer.decode(generated_ids[0].tolist())

    # ─── Estado ─────────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _require_loaded(self):
        if not self._loaded:
            raise RuntimeError(
                "BrocaV2 nao carregado. Chame .load() primeiro, "
                "ou execute core.broca_v2.train para treinar o modelo.")

    def get_stats(self) -> dict:
        if not self._loaded:
            return {"loaded": False}
        return {
            "loaded": True,
            "vocab_size": len(self.tokenizer) if self.tokenizer else 0,
            "device": str(self.device),
            "params": sum(p.numel() for p in self.model.parameters()),
        }
