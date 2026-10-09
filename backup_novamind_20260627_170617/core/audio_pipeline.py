"""
Audio Pipeline: encoding e reconstrucao de audio usando EnCodec + HDC.

Fluxo:
  Waveform -> EnCodec encoder -> codes -> binding HDC -> bipolar [raciocinio]
  Bipolar -> unbinding HDC -> codes -> EnCodec decoder -> waveform [reconstrucao]
  
  Para reconstrucao FIEL (sem perda do unbinding):
    raw codes (guardados no buffer) -> EnCodec decoder -> waveform fiel
"""

import torch
import numpy as np
from typing import Optional, Tuple
from .coglang import Manifold


class AudioPipeline:
    """Pipeline auditivo usando EnCodec + HDC (wrapping GroundingEngine)."""

    def __init__(self, grounding_engine, device: str = "cuda"):
        self.device = torch.device(device)
        self.eng = grounding_engine
        self.encodec = self.eng.encodec_model

    def encode_audio(self, waveform: torch.Tensor) -> Tuple[Manifold, torch.Tensor]:
        """
        Codifica waveform para bipolar HDC + retorna raw codes.

        Args:
            waveform: (T,) float32 em ~22kHz

        Returns:
            mf: Manifold bipolar 10000D
            raw_codes: (2, 75) indices EnCodec (para rec fiel)
        """
        wav = waveform.to(self.device).reshape(-1)
        wav_24k = self.eng.resample_22_to_24(wav)
        wav_24k = wav_24k.reshape(1, 1, -1)

        if wav_24k.shape[-1] < 24000:
            pad = torch.zeros(1, 1, 24000 - wav_24k.shape[-1], device=self.device)
            wav_24k = torch.cat([wav_24k, pad], dim=-1)
        else:
            wav_24k = wav_24k[:, :, :24000]

        with torch.no_grad():
            encoded = self.encodec.encode(wav_24k)
            codes = encoded[0][0].squeeze(0)  # (2, 75)

            summed = torch.zeros(Manifold.DIMENSIONALITY, device=self.device)
            for q in range(2):
                for t in range(75):
                    val_idx = codes[q, t].item()
                    term = (self.eng.quant_tensors[q] *
                            self.eng.time_tensors[t] *
                            self.eng.code_value_tensors[val_idx])
                    summed = summed + term

            bipolar = torch.sign(summed)
            bipolar[bipolar == 0] = 1

        return Manifold(bipolar, device=self.device), codes

    def decode_audio(self, audio_mf: Manifold) -> torch.Tensor:
        """
        Decodifica bipolar HDC -> waveform via unbinding + EnCodec decoder.
        Usa o metodo existente do GroundingEngine.
        """
        return self.eng.decode_audio_encodec(audio_mf)

    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decodifica raw EnCodec codes diretamente -> waveform fiel.
        PERFEITO: sem perda do unbinding HDC.
        """
        codes_in = codes.unsqueeze(0).to(self.device) if codes.dim() == 2 else codes
        codes_in = codes_in.unsqueeze(0) if codes_in.dim() == 2 else codes_in
        decoded_frames = [(codes_in, None)]
        with torch.no_grad():
            wav = self.encodec.decode(decoded_frames)
        return wav.squeeze(0).squeeze(0)

    def verify_fidelity(self, original: torch.Tensor,
                        reconstructed: torch.Tensor) -> dict:
        """PSNR entre waveforms original e reconstruido."""
        o = original.float().cpu()
        r = reconstructed.float().cpu()
        min_len = min(o.shape[-1], r.shape[-1])
        o, r = o[..., :min_len], r[..., :min_len]
        import torch.nn.functional as F
        mse = float(F.mse_loss(o, r))
        psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-10))
        return {"psnr": psnr, "mse": mse}
