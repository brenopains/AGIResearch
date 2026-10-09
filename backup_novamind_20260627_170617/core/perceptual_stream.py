"""
PerceptualStream: fluxo continuo de percepcao (video + audio) para o CGE.

Funciona como o cortex sensorial:
  1. Cada frame de video chega -> features 512D -> JL -> bipolar (CGE) + presign (buffer)
  2. Cada chunk de audio chega -> EnCodec codes -> JL -> bipolar (CGE) + codes (buffer)
  3. CGE atualiza nos visuais/auditivos em tempo real
  4. Buffer sensorial guarda presign/codes para reconstrucao fiel

A Forja pode operar nos bipolares do CGE. O buffer permite verificar
se a memoria corresponde a percepcao original (reconstrucao fiel).
"""

import torch
import cv2
import numpy as np
import time
from typing import Optional, List, Tuple
from pathlib import Path

from .coglang import Manifold
from .cognitive_graph_engine import CognitiveGraphEngine
from .visual_pipeline import VisualPipeline


class PerceptualStream:
    """
    Fluxo perceptivo continuo.

    Uso:
        stream = PerceptualStream(cge, vp, eng, device='cuda')
        stream.process_video("video.mp4")  # processa frame a frame, atualiza CGE
        stream.process_audio("audio.wav")  # processa chunk a chunk, atualiza CGE

        # A qualquer momento:
        cena = cge.get_scene_text()
        holo = cge.compress_to_hologram()
        frame_rec = vp.decode_bipolar(holo)  # o que a AGI esta vendo agora
    """

    def __init__(self, cge: CognitiveGraphEngine, visual_pipeline: VisualPipeline,
                 grounding_engine, device: str = "cuda"):
        self.cge = cge
        self.vp = visual_pipeline
        self.eng = grounding_engine
        self.device = torch.device(device)

        # Buffer sensorial: guarda presign/codes para cada frame/chunk
        self.visual_buffer: dict = {}    # {node_id: presign_tensor}
        self.audio_buffer: dict = {}     # {node_id: encodec_codes}
        self.frame_count = 0
        self.audio_chunk_count = 0

    def process_video_frame(self, frame_bgr: np.ndarray,
                             prefix: str = "frame") -> str:
        """
        Processa UM frame de video em tempo real.

        1. Converte BGR -> RGB
        2. ResNet features -> bipolar + presign
        3. Cria/atualiza no visual no CGE
        4. Guarda presign no buffer sensorial

        Retorna: node_id do no criado/atualizado
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = self.eng.transform(frame_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.eng.feature_extractor(img_tensor).view(-1)
            mf, presign, feats = self.vp.encode_frame(features)

        node_id = f"{prefix}_{self.frame_count}"
        self.cge.create_node(node_id, mf.tensor, perceptual_vec=presign)
        self.visual_buffer[node_id] = presign.cpu()
        self.frame_count += 1

        return node_id

    def process_video(self, video_path: str, max_frames: int = 0,
                      prefix: str = "frame") -> int:
        """
        Processa video inteiro frame a frame em tempo real (~25 FPS).

        Args:
            video_path: caminho do .avi/.mp4
            max_frames: 0 = todos os frames
            prefix: prefixo dos nos no CGE

        Returns: numero de frames processados
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames > 0:
            total = min(total, max_frames)

        print(f"[PerceptualStream] Video: {Path(video_path).name} "
              f"({total} frames, {fps:.0f} FPS)")

        n = 0
        t0 = time.time()
        while n < total:
            ret, frame = cap.read()
            if not ret:
                break
            self.process_video_frame(frame, prefix=prefix)
            n += 1

            if n % 30 == 0:
                elapsed = time.time() - t0
                rate = n / max(elapsed, 0.1)
                print(f"  ... {n}/{total} frames ({rate:.0f} FPS)", end="\r")

        cap.release()
        elapsed = time.time() - t0
        print(f"\n[PerceptualStream] {n} frames em {elapsed:.1f}s ({n/elapsed:.0f} FPS)")
        return n

    def process_audio_chunk(self, waveform: torch.Tensor,
                             prefix: str = "audio") -> str:
        """
        Processa UM chunk de audio em tempo real.

        Args:
            waveform: (T,) float32 em ~22kHz
            prefix: prefixo do no no CGE

        Returns: node_id
        """
        from .audio_pipeline import AudioPipeline
        ap = AudioPipeline(self.eng, device=str(self.device))
        wav = waveform.to(self.device)
        mf, codes = ap.encode_audio(wav)

        node_id = f"{prefix}_{self.audio_chunk_count}"
        self.cge.create_node(node_id, mf.tensor, perceptual_vec=wav)
        self.audio_buffer[node_id] = codes.cpu()
        self.audio_chunk_count += 1

        return node_id

    def process_video_with_audio(self, video_path: str,
                                  max_frames: int = 0) -> Tuple[int, int]:
        """
        Processa video + audio sincronizados.

        Extrai audio do video, processa em chunks de 1s,
        intercala com os frames visuais no CGE.
        """
        # Processa video
        n_video = self.process_video(video_path, max_frames=max_frames)

        # Tenta extrair audio
        try:
            audio_path = video_path.replace(".avi", ".wav").replace(".mp4", ".wav")
            if not Path(audio_path).exists():
                # Extrai audio do video com ffmpeg
                import subprocess
                audio_path = video_path + "_audio.wav"
                subprocess.run(
                    ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                     "-ar", "24000", "-ac", "1", audio_path, "-y"],
                    capture_output=True, timeout=30
                )

            import torchaudio
            wav, sr = torchaudio.load(audio_path)
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                wav = resampler(wav)

            # Processa em chunks de 1s (24000 samples)
            chunk_size = 24000
            n_audio = 0
            for start in range(0, wav.shape[-1], chunk_size):
                chunk = wav[:, start:start+chunk_size]
                if chunk.shape[-1] < 1000:  # muito curto
                    continue
                self.process_audio_chunk(chunk.squeeze(0), prefix="audio")
                n_audio += 1

            print(f"[PerceptualStream] Audio: {n_audio} chunks processados")
            return n_video, n_audio

        except Exception as e:
            print(f"[PerceptualStream] Audio nao disponivel: {e}")
            return n_video, 0

    def reconstruct_visual(self, node_id: str) -> torch.Tensor:
        """Reconstroi frame de um no visual a partir do buffer."""
        if node_id in self.visual_buffer:
            presign = self.visual_buffer[node_id].to(self.device)
            with torch.no_grad():
                f_rec = torch.matmul(self.vp.pinv, presign)
                img = self.vp.decoder(f_rec.unsqueeze(0)).squeeze(0)
            return img
        return None

    def reconstruct_audio(self, node_id: str) -> Optional[torch.Tensor]:
        """Reconstroi waveform de um no auditivo a partir do buffer."""
        if node_id in self.audio_buffer:
            from .audio_pipeline import AudioPipeline
            ap = AudioPipeline(self.eng, device=str(self.device))
            codes = self.audio_buffer[node_id].to(self.device)
            return ap.decode_from_codes(codes)
        if node_id in self.cge.nodes:
            node_vec = self.cge.nodes[node_id]["vector"]
            from .coglang import Manifold
            return self.eng.decode_audio_encodec(Manifold(node_vec, device=self.device))
        return None

    def verify_visual_fidelity(self, node_id: str,
                                frame_bgr: np.ndarray) -> dict:
        """Verifica se a memoria visual corresponde ao frame original."""
        img_rec = self.reconstruct_visual(node_id)
        if img_rec is None:
            return {"error": "node not found"}
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_orig = torch.from_numpy(
            cv2.resize(frame_rgb, (224, 224)).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        return self.vp.verify_fidelity(img_orig, img_rec)

    def clear(self):
        """Limpa buffers sensoriais (mas mantem nos no CGE)."""
        self.visual_buffer.clear()
        self.audio_buffer.clear()
