"""
VisualRenderer — Decodificador Genérico HDV → .mp4 (não hardcoded)
=====================================================================
VQ-VAE (Vector Quantized Variational Autoencoder) que comprime frames
de vídeo num codebook discreto. O decoder reconstrói frames dos índices.

A AGI aprende a mapear vetores HDC → sequências de índices do codebook.
Qualquer conceito HDC pode ser renderizado como vídeo.

Treino:
    renderer = VisualRenderer()
    renderer.train(video_corpus_dir, epochs=10)

Geração:
    frames = renderer.render_from_hdv(hdv_vector, num_frames=30)
    renderer.save_mp4(frames, "output.mp4")

Arquitetura:
    Encoder: Conv3D → Quantizer → índices
    Decoder: índices → Conv3D Transpose → frames
    HDC Bridge: MLP(hdv_10000d) → codebook_indices_sequence
"""

import os
import time
import json
from typing import List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class VectorQuantizer(nn.Module):
    """Quantização vetorial: mapeia embeddings contínuos para índices discretos."""

    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z: (B, D, H, W) → quantized, indices, loss_vq"""
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)
        emb = self.embedding.weight
        dist = torch.cdist(z_flat, emb)
        indices = dist.argmin(dim=-1)
        z_q = emb[indices].reshape(B, H, W, D).permute(0, 3, 1, 2)
        loss = F.mse_loss(z_q.detach(), z) + 0.25 * F.mse_loss(z_q, z.detach())
        z_q = z + (z_q - z).detach()
        return z_q, indices, loss


class VisualRenderer(nn.Module):
    """
    Renderizador genérico HDV → frames → .mp4.

    NÃO é específico para um domínio. Treinado uma vez em qualquer
    corpus de vídeos. Qualquer conceito HDC pode ser renderizado.

    Args:
        codebook_size: número de embeddings discretos (512 = boa qualidade)
        embedding_dim: dimensão de cada embedding (64)
        hdc_dim: dimensão do vetor HDC (10000)
        hidden_dim: dimensão da MLP bridge (512)
    """

    def __init__(
        self,
        codebook_size: int = 512,
        embedding_dim: int = 64,
        hdc_dim: int = 10000,
        hidden_dim: int = 512,
        frame_size: int = 64,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.hdc_dim = hdc_dim
        self.hidden_dim = hidden_dim
        self.frame_size = frame_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder: (B,3,H,W) → (B,D,H/4,W/4)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, 3, stride=1, padding=1),
        )

        # Vector Quantizer
        self.quantizer = VectorQuantizer(codebook_size, embedding_dim)

        # Decoder: (B,D,H/4,W/4) → (B,3,H,W)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        # HDC Bridge: mapeia HDV 10000D → sequência de índices do codebook
        # A AGI treina esta bridge para gerar vídeos de QUALQUER conceito
        self.hdc_bridge = nn.Sequential(
            nn.Linear(hdc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Projeção para índices (por posição temporal)
        self.index_proj = nn.Linear(hidden_dim, codebook_size)

        self.to(self.device)
        self._trained = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: imagem → quantized → reconstrução."""
        z = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z)
        recon = self.decoder(z_q)
        return recon, indices, vq_loss

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Codifica frames como índices do codebook."""
        self.eval()
        with torch.no_grad():
            frames = frames.to(self.device)
            if frames.dim() == 3:
                frames = frames.unsqueeze(0)
            z = self.encoder(frames)
            _, indices, _ = self.quantizer(z)
            return indices  # (B, H', W')

    def render_from_hdv(
        self,
        hdv_vector: torch.Tensor,
        num_frames: int = 16,
        temperature: float = 0.8,
    ) -> np.ndarray:
        """
        Renderiza frames a partir de um vetor HDC.

        A HDC Bridge mapeia o vetor HDC para índices do codebook,
        e o decoder reconstrói os frames.

        Args:
            hdv_vector: (10000,) — vetor HDC do conceito
            num_frames: número de frames a gerar
            temperature: temperatura da amostragem dos índices

        Returns:
            frames: numpy array (num_frames, H, W, 3) uint8
        """
        if not self._trained:
            return self._fallback_render(hdv_vector, num_frames)

        self.eval()
        with torch.no_grad():
            hdv = hdv_vector.flatten().float().to(self.device)
            if hdv.dim() == 1:
                hdv = hdv.unsqueeze(0)

            # Gera features via bridge
            features = self.hdc_bridge(hdv)  # (1, hidden_dim)

            # Expande para múltiplos frames
            # Cada frame é uma "leitura" diferente da mesma intenção
            feat_seq = features.unsqueeze(0).expand(num_frames, -1, -1)

            # Adiciona variação temporal (sinusoidal positional encoding)
            t_enc = torch.zeros(num_frames, self.hidden_dim, device=self.device)
            for t in range(num_frames):
                t_enc[t] = torch.sin(torch.tensor(t * 0.1))
            feat_seq = feat_seq + t_enc.unsqueeze(1)

            # Projeta para logits do codebook
            logits = self.index_proj(feat_seq) / temperature  # (num, 1, codebook)
            probs = F.softmax(logits, dim=-1)

            # Shape alvo: (num_frames, H' * W', codebook_size)
            h_prime = self.frame_size // 4
            w_prime = self.frame_size // 4
            n_tokens = h_prime * w_prime

            # Expande probs para todos os tokens
            probs_exp = probs.expand(-1, n_tokens, -1)  # (num, tokens, codebook)

            # Amostra índices
            flat_probs = probs_exp.reshape(-1, self.codebook_size)
            indices = torch.multinomial(flat_probs, 1).reshape(num_frames, h_prime, w_prime)

            # Decodifica índices → pixels
            z_q = self.quantizer.embedding(indices)
            z_q = z_q.permute(0, 3, 1, 2).float()  # (B, D, H', W')
            recon = self.decoder(z_q)  # (B, 3, H, W)
            recon = recon.permute(0, 2, 3, 1).cpu().numpy()
            recon = (recon * 255).astype(np.uint8)

            return recon

    def _fallback_render(self, hdv_vector: torch.Tensor, num_frames: int) -> np.ndarray:
        """Renderização fallback quando modelo não está treinado."""
        print("[VISREN] Modelo nao treinado. Gerando frames procedurais.")
        frames = []
        h = hdv_vector.cpu().numpy().flatten()
        for t in range(num_frames):
            frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            # Gera padrão baseado no vetor HDC (determinístico)
            seed = int(h[0] * 1000 + h[1] * 100 + t)
            np.random.seed(seed % 2**31)
            for _ in range(20):
                x = int(np.random.rand() * self.frame_size)
                y = int(np.random.rand() * self.frame_size)
                r = int(abs(h[(t + _) % 10000]) * 255)
                g = int(abs(h[(t + _ + 100) % 10000]) * 255) 
                b = int(abs(h[(t + _ + 200) % 10000]) * 255)
                cv2.circle(frame, (x, y), max(1, int(np.random.rand() * 8)),
                          (r, g, b), -1)
            frames.append(frame)
        return np.array(frames)

    def save_mp4(self, frames: np.ndarray, output_path: str, fps: int = 20):
        """Salva frames como .mp4."""
        if not HAS_CV2:
            print("[VISREN] OpenCV necessario para salvar MP4")
            return
        if frames.size == 0:
            print("[VISREN] Sem frames para salvar")
            return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame
            out.write(bgr)
        out.release()
        print(f"[VISREN] Video salvo: {output_path} ({len(frames)} frames, {fps}fps)")

    def train_on_videos(
        self,
        video_dir: str,
        epochs: int = 10,
        batch_size: int = 8,
        lr: float = 1e-3,
        save_path: str = "D:/temp/visual_renderer/vqvae.pt",
    ):
        """
        Treina o VQ-VAE num diretório de vídeos.
        Extrai frames aleatórios de cada vídeo.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        print(f"[VISREN] Treinando VQ-VAE em: {video_dir}")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Coleta frames de vídeos
        frames = self._collect_frames(video_dir)
        if not frames:
            print("[VISREN] Nenhum frame encontrado. Use dataset como Kinetics-400.")
            return

        n = len(frames)
        print(f"[VISREN] {n} frames coletados")

        for epoch in range(epochs):
            np.random.shuffle(frames)
            total_loss = 0.0
            total_recon = 0.0
            total_vq = 0.0
            n_batches = 0

            for i in range(0, n, batch_size):
                batch = frames[i:i + batch_size]
                x = torch.stack([self._preprocess(f) for f in batch]).to(self.device)

                recon, indices, vq_loss = self.forward(x)
                recon_loss = F.mse_loss(recon, x)

                loss = recon_loss + vq_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_vq += vq_loss.item()
                n_batches += 1

            avg = total_loss / max(1, n_batches)
            print(f"  Epoch {epoch+1}/{epochs} loss={avg:.4f} "
                  f"(recon={total_recon/n_batches:.4f} vq={total_vq/n_batches:.4f})")

        # Salva
        torch.save({
            "model_state_dict": self.state_dict(),
            "codebook_size": self.codebook_size,
            "embedding_dim": self.embedding_dim,
            "epochs": epochs,
        }, save_path)
        self._trained = True
        print(f"[VISREN] Modelo salvo: {save_path}")

    def load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if "model_state_dict" in ckpt:
            self.load_state_dict(ckpt["model_state_dict"])
        else:
            self.load_state_dict(ckpt)
        self._trained = True
        print(f"[VISREN] Modelo carregado: {path}")

    def _collect_frames(self, video_dir: str) -> List[np.ndarray]:
        """Extrai frames de vídeos no diretório."""
        frames = []
        paths = list(Path(video_dir).glob("*.mp4")) + list(Path(video_dir).glob("*.avi"))
        for vp in paths[:10]:  # limita a 10 vídeos
            cap = cv2.VideoCapture(str(vp))
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if count % 10 == 0:  # 1 frame a cada 10
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb)
                count += 1
                if count > 500:  # max 500 frames por video
                    break
            cap.release()
        return frames[:2000]  # max 2000 frames total

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        h, w = frame.shape[:2]
        # Central crop quadrado + resize
        s = min(h, w)
        y = (h - s) // 2
        x = (w - s) // 2
        cropped = frame[y:y+s, x:x+s]
        resized = cv2.resize(cropped, (self.frame_size, self.frame_size))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        return tensor
