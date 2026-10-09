"""
AudioVocoder — Decodificador Genérico HDV → .wav (não hardcoded)
==================================================================
MLP que mapeia vetor HDC → parâmetros acústicos (F0, aperiodicidade,
spectral envelope). World Vocoder (matemática pura, sem IA) sintetiza
o .wav final.

A AGI APRENDE o mapeamento treinando em pares (HDV_conceito, áudio).
Qualquer conceito com componente sonoro pode ser "ouvido".

Não é TTS. Não é API externa. É um vocoder matemático
controlado por parâmetros decodificados do HDC.

Fluxo:
  1. encode(concept_text) → hdv_vector
  2. hdv_to_params(hdv) → F0, aperiodicity, spectrogram
  3. world_synthesize(F0, ap, sp) → .wav
  4. save_wav(waveform, "output.wav")
"""

import os
import struct
import time
from typing import Optional, Tuple, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import pyworld as pw
    HAS_WORLD = True
except ImportError:
    HAS_WORLD = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


class HDVToAcousticParams(nn.Module):
    """
    MLP que mapeia HDV 10000D → parâmetros acústicos.

    Saída:
      - F0 (pitch): 1 valor por frame
      - Aperiodicidade: 1-5 valores por frame
      - Spectral envelope: 1-24 valores por frame (MFCC-like)
    """

    def __init__(self, hdc_dim: int = 10000, hidden_dim: int = 512,
                 n_frames: int = 100, n_fft: int = 24):
        super().__init__()
        self.n_frames = n_frames
        self.n_fft = n_fft

        self.net = nn.Sequential(
            nn.Linear(hdc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Heads separados para cada parâmetro
        self.f0_head = nn.Linear(hidden_dim, n_frames)
        self.ap_head = nn.Linear(hidden_dim, n_frames * 5)  # 5-band aperiodicity
        self.sp_head = nn.Linear(hidden_dim, n_frames * n_fft)  # spectral envelope

    def forward(self, hdv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hdv: (batch, 10000)

        Returns:
            f0: (batch, n_frames) — pitch contorno
            ap: (batch, n_frames, 5) — aperiodicidade
            sp: (batch, n_frames, n_fft) — envelope espectral
        """
        h = self.net(hdv)
        f0 = self.f0_head(h)
        ap = self.ap_head(h).reshape(-1, self.n_frames, 5)
        sp = self.sp_head(h).reshape(-1, self.n_frames, self.n_fft)
        # Garante valores no range correto
        f0 = torch.sigmoid(f0) * 400.0 + 50.0  # 50-450 Hz
        ap = torch.sigmoid(ap) * 0.5            # 0-0.5
        sp = F.softplus(sp)                     # > 0
        return f0, ap, sp


class AudioVocoder:
    """
    Decodificador de áudio nativo a partir de HDC.

    Usa pyworld (World Vocoder) para síntese — matemática pura,
    sem redes neurais na síntese final. A MLP só controla os
    parâmetros acústicos.

    Uso:
        vocoder = AudioVocoder()
        audio = vocoder.synthesize(hdv_vector)  # audio array
        vocoder.save_wav(audio, "output.wav")
    """

    def __init__(self, sample_rate: int = 16000, frame_period: float = 5.0,
                 device: Optional[str] = None):
        self.sample_rate = sample_rate
        self.frame_period = frame_period  # ms entre frames
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HDVToAcousticParams(n_frames=100, n_fft=24).to(self.device)
        self._trained = False

    def synthesize(self, hdv_vector: torch.Tensor,
                   duration: float = 2.0) -> np.ndarray:
        """
        Síntese de áudio a partir de vetor HDC.

        1. MLP → parâmetros acústicos
        2. World Vocoder → waveform

        Args:
            hdv_vector: (10000,) tensor
            duration: duração em segundos

        Returns:
            waveform: (samples,) float32 array
        """
        hdv = hdv_vector.flatten().float().to(self.device)
        if hdv.dim() == 1:
            hdv = hdv.unsqueeze(0)

        if self._trained:
            f0, ap, sp = self.model(hdv)
        else:
            # Fallback: parâmetros baseados no vetor HDC
            f0, ap, sp = self._fallback_params(hdv, duration)

        # Interpola para o número correto de frames
        n_frames = int(duration * 1000 / self.frame_period)
        f0_np = self._interp(f0[0].cpu().numpy(), n_frames)
        ap_np = self._interp_2d(ap[0].cpu().numpy(), n_frames)
        sp_np = self._interp_2d(sp[0].cpu().numpy(), n_frames)

        if HAS_WORLD:
            waveform = pw.synthesize(
                f0_np.astype(np.float64),
                ap_np.astype(np.float64),
                sp_np.astype(np.float64),
                self.sample_rate,
                self.frame_period,
            )
            return waveform.astype(np.float32)
        else:
            print("[VOCODER] pyworld nao disponivel. Gerando tom sintetico.")
            return self._tone_fallback(f0_np)

    def _fallback_params(self, hdv: torch.Tensor, duration: float
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fallback: parâmetros derivados do vetor HDC."""
        h = hdv.cpu().numpy().flatten()
        n = int(duration * 1000 / self.frame_period)

        # F0: pitch derivado das primeiras componentes do HDC
        f0_base = 100.0 + abs(h[0]) * 200.0
        f0_mod = abs(h[1]) * 50.0 * np.sin(np.linspace(0, 2*np.pi*3, n))
        f0 = torch.from_numpy((f0_base + f0_mod).astype(np.float32)).unsqueeze(0)

        # AP: aperiodicidade baixa (voz mais clara)
        ap_val = 0.05 + abs(h[2]) * 0.15
        ap = torch.full((1, n, 5), ap_val, dtype=torch.float32)

        # SP: envelope espectral derivado do HDC
        sp = torch.zeros(1, n, 24, dtype=torch.float32)
        for i in range(24):
            sp[0, :, i] = torch.tensor(abs(h[(i * 400) % 10000]) * 10.0)

        return f0.to(self.device), ap.to(self.device), sp.to(self.device)

    def _interp(self, arr: np.ndarray, n_out: int) -> np.ndarray:
        x_in = np.linspace(0, 1, len(arr))
        x_out = np.linspace(0, 1, n_out)
        return np.interp(x_out, x_in, arr)

    def _interp_2d(self, arr: np.ndarray, n_out: int) -> np.ndarray:
        result = np.zeros((n_out, arr.shape[1]), dtype=arr.dtype)
        for i in range(arr.shape[1]):
            result[:, i] = self._interp(arr[:, i], n_out)
        return result

    def _tone_fallback(self, f0: np.ndarray) -> np.ndarray:
        """Gera tom senoidal quando pyworld não está disponível."""
        duration = len(f0) * self.frame_period / 1000.0
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        f0_interp = np.interp(
            t, np.linspace(0, duration, len(f0)), f0)
        phase = 2 * np.pi * np.cumsum(f0_interp) / self.sample_rate
        return np.sin(phase).astype(np.float32)

    def save_wav(self, waveform: np.ndarray, output_path: str):
        """Salva waveform como .wav."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if HAS_SOUNDFILE:
            sf.write(output_path, waveform, self.sample_rate)
        else:
            # Fallback: escreve .wav raw
            with open(output_path, "wb") as f:
                n = len(waveform)
                f.write(b"RIFF")
                f.write(struct.pack("<I", 36 + n * 2))
                f.write(b"WAVE")
                f.write(b"fmt ")
                f.write(struct.pack("<I", 16))
                f.write(struct.pack("<H", 1))
                f.write(struct.pack("<H", 1))
                f.write(struct.pack("<I", self.sample_rate))
                f.write(struct.pack("<I", self.sample_rate * 2))
                f.write(struct.pack("<H", 2))
                f.write(struct.pack("<H", 16))
                f.write(b"data")
                f.write(struct.pack("<I", n * 2))
                scaled = (waveform * 32767).astype(np.int16)
                f.write(scaled.tobytes())
        print(f"[VOCODER] Audio salvo: {output_path}")

    def train(self, audio_dir: str, epochs: int = 10, lr: float = 1e-3,
              save_path: str = "D:/temp/audio_vocoder/vocoder.pt"):
        """
        Treina a MLP em pares (texto_do_conceito, arquivo_audio).

        Para cada arquivo .wav no diretório:
          1. Extrai parâmetros acústicos via pyworld
          2. Codifica o nome do arquivo como HDC (proxy do conceito)
          3. Treina MLP: HDV → parâmetros
        """
        if not HAS_WORLD:
            print("[VOCODER] pyworld necessario para treino")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        wavs = list(Path(audio_dir).glob("*.wav"))
        if not wavs:
            print(f"[VOCODER] Nenhum .wav em {audio_dir}")
            return

        print(f"[VOCODER] Treinando em {len(wavs)} audios")

        for epoch in range(epochs):
            total_loss = 0.0
            for wav_path in wavs:
                try:
                    audio, sr = sf.read(str(wav_path))
                    if sr != self.sample_rate:
                        audio = self._resample(audio, sr, self.sample_rate)

                    # Extrai parâmetros acústicos
                    f0_gt, ap_gt, sp_gt = self._extract_params(audio)

                    # Proxy do conceito: hash do nome do arquivo como vetor
                    concept = wav_path.stem
                    hdv = self._concept_to_hdv(concept)

                    # Forward
                    f0_pred, ap_pred, sp_pred = self.model(hdv)

                    # Loss: MSE nos parâmetros
                    loss_f0 = F.mse_loss(f0_pred, f0_gt)
                    loss_ap = F.mse_loss(ap_pred, ap_gt)
                    loss_sp = F.mse_loss(sp_pred, sp_gt)
                    loss = loss_f0 + loss_ap + loss_sp

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                except Exception as e:
                    continue

            avg = total_loss / max(1, len(wavs))
            print(f"  Epoch {epoch+1}/{epochs} loss={avg:.4f}")

        torch.save(self.model.state_dict(), save_path)
        self._trained = True
        print(f"[VOCODER] Modelo salvo: {save_path}")

    def load_pretrained(self, path: str):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True))
        self._trained = True
        print(f"[VOCODER] Modelo carregado: {path}")

    def _extract_params(self, audio: np.ndarray
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extrai F0, AP, SP de um waveform via pyworld."""
        f0, t = pw.harvest(audio.astype(np.float64), self.sample_rate)
        sp = pw.cheaptrick(audio.astype(np.float64), f0, t, self.sample_rate)
        ap = pw.d4c(audio.astype(np.float64), f0, t, self.sample_rate)
        # Reduz dimensionalidade do SP
        sp_reduced = sp[:, :24]

        f0_t = torch.from_numpy(f0.astype(np.float32)).unsqueeze(0)[:, :100]
        ap_t = torch.from_numpy(ap[:, :5].astype(np.float32)).unsqueeze(0)[:, :100]
        sp_t = torch.from_numpy(sp_reduced.astype(np.float32)).unsqueeze(0)[:, :100]

        return f0_t, ap_t, sp_t

    def _concept_to_hdv(self, concept: str) -> torch.Tensor:
        """Hash do nome do conceito para vetor 10000D (proxy)."""
        np.random.seed(hash(concept) % 2**31)
        vec = np.random.randn(10000).astype(np.float32)
        return torch.from_numpy(vec).to(self.device)

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        duration = len(audio) / orig_sr
        n_target = int(duration * target_sr)
        return np.interp(
            np.linspace(0, len(audio)-1, n_target),
            np.arange(len(audio)), audio)
