"""
ArticulatorySynthesizer — Modelo do Trato Vocal Baseado em Articuladores
========================================================================
Em vez de gerar formantes diretamente, a AGI controla ARTICULADORES
individuais (língua, lábios, mandíbula, véu palatino, pregas vocais)
e o modelo deriva os formantes a partir da configuração articulatória.

Isso permite que a AGI APRENDA a relação causal:
    comando motor → configuração articulatória → som

Como um bebê real que descobre: "se eu levanto a língua, o som muda".
"""

import numpy as np
import struct
import io
from typing import Dict, Tuple, Optional

SAMPLE_RATE = 22050


def articulatory_to_formants(
    tongue_dorsum_x: float = 0.5,
    tongue_dorsum_y: float = 0.5,
    lip_rounding: float = 0.0,
    jaw_open: float = 0.5,
    velum: float = 0.0,
    vocal_fold_tension: float = 0.6,
) -> Tuple[float, float, float, float]:
    """
    Mapeia configuração articulatória → (f0, f1, f2, f3).

    Baseado em modelo simplificado de tubo ressonante:
      - F1 ∝ abertura (jaw_open + (1 - tongue_dorsum_y))
      - F2 ∝ frente da língua (tongue_dorsum_x), reduzido por arredondamento labial
      - F3 ∝ comprimento do trato, reduzido por arredondamento labial
      - f0 ∝ tensão das pregas vocais

    Valores aproximados dos formantes para referência:
      /a/ (pai):    f1~700,  f2~1100 → língua BAIXA central, mandíbula ABERTA
      /e/ (pé):     f1~500,  f2~1800 → língua MÉDIA frente
      /i/ (vi):     f1~300,  f2~2200 → língua ALTA frente, mandíbula FECHADA
      /o/ (pó):     f1~500,  f2~900  → língua MÉDIA atrás, LÁBIOS arredondados
      /u/ (tu):     f1~300,  f2~900  → língua ALTA atrás, LÁBIOS arredondados
    """
    # Frequência fundamental: tensão das pregas vocais
    f0 = 80.0 + 220.0 * np.clip(vocal_fold_tension, 0.0, 1.0)

    # Abertura efetiva do trato (jaw + abaixamento da língua)
    opening = np.clip(jaw_open * 0.6 + (1.0 - tongue_dorsum_y) * 0.4, 0.0, 1.0)
    f1 = 250.0 + 550.0 * opening

    # F2: modelado como função não-linear da constrição da língua
    # A constrição frontal (tongue_x > 0.5) eleva F2 rapidamente.
    # Vogais posteriores têm F2 baixo (~800-1200), anteriores têm F2 alto (~1800-2500).
    frontness = np.clip(tongue_dorsum_x, 0.0, 1.0)
    height_factor = 1.0 + tongue_dorsum_y * 0.5  # língua alta eleva F2 em vogais frontais
    f2_raw = 900.0 + 1300.0 * (frontness ** 1.5) * height_factor
    f2 = f2_raw * (1.0 - 0.2 * np.clip(lip_rounding, 0.0, 1.0))
    f2 = np.clip(f2, 700.0, 3500.0)

    f3 = 2500.0 - 400.0 * lip_rounding - 200.0 * tongue_dorsum_y * frontness
    f3 = np.clip(f3, 2000.0, 3500.0)

    return f0, f1, f2, f3


def synthesize_articulatory(
    tongue_dorsum_x: float = 0.5,
    tongue_dorsum_y: float = 0.5,
    lip_rounding: float = 0.0,
    jaw_open: float = 0.5,
    velum: float = 0.0,
    vocal_fold_tension: float = 0.6,
    amplitude: float = 0.5,
    duration: float = 0.5,
    voiced: bool = True,
    consonant_type: Optional[str] = None,
    consonant_place: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Gera waveform a partir de configuração articulatória.

    Retorna (waveform float32 [-1,1], dict com feedback sensorial).

    O feedback sensorial inclui:
      - acoustic: formantes resultantes, pitch, loudness
      - articulatory: configuração atual dos articuladores (somatossensorial)
      - self_description: texto descritivo para a AGI
    """
    f0, f1, f2, f3 = articulatory_to_formants(
        tongue_dorsum_x, tongue_dorsum_y,
        lip_rounding, jaw_open, velum, vocal_fold_tension,
    )

    n = int(SAMPLE_RATE * duration)
    t = np.arange(n) / SAMPLE_RATE

    nasal_energy = np.clip(velum, 0.0, 1.0) * 0.3

    if consonant_type == "plosive":
        signal = _synthesize_plosive(n, SAMPLE_RATE, f0, f1, f2, f3, consonant_place)
    elif consonant_type == "fricative":
        signal = _synthesize_fricative(n, SAMPLE_RATE, f1, f2, f3, consonant_place)
    elif consonant_type == "nasal":
        signal = _synthesize_nasal(n, SAMPLE_RATE, f0, f1, f2, f3, nasal_energy)
    else:
        signal = _synthesize_vowel(n, SAMPLE_RATE, f0, f1, f2, f3, voiced)

    if nasal_energy > 0 and consonant_type is None:
        nasal_noise = np.random.randn(n) * 0.08 * nasal_energy
        signal = signal + nasal_noise

    peak = np.max(np.abs(signal))
    if peak > 1e-10:
        signal = signal / peak
    signal = np.clip(signal * amplitude * 0.7, -1.0, 1.0)

    loudness = float(np.sqrt(np.mean(signal ** 2)))

    acoustic_feedback = {
        "f0": float(f0),
        "f1": float(f1),
        "f2": float(f2),
        "f3": float(f3),
        "loudness": loudness,
        "duration": duration,
        "voiced": voiced,
        "consonant_type": consonant_type,
        "consonant_place": consonant_place,
    }

    articulatory_feedback = {
        "tongue_dorsum_x": float(tongue_dorsum_x),
        "tongue_dorsum_y": float(tongue_dorsum_y),
        "lip_rounding": float(lip_rounding),
        "jaw_open": float(jaw_open),
        "velum": float(velum),
        "vocal_fold_tension": float(vocal_fold_tension),
    }

    self_description = _describe_articulation(
        tongue_dorsum_x, tongue_dorsum_y,
        lip_rounding, jaw_open, velum,
        f0, f1, f2, f3, consonant_type, consonant_place,
    )

    return signal, {
        "acoustic": acoustic_feedback,
        "articulatory": articulatory_feedback,
        "self_description": self_description,
        "waveform": signal,
    }


def _synthesize_vowel(n: int, sr: int, f0: float, f1: float, f2: float, f3: float,
                      voiced: bool) -> np.ndarray:
    t = np.arange(n) / sr
    if voiced:
        signal = np.sin(2 * np.pi * f0 * t)
        signal += 0.5 * np.sin(2 * np.pi * f0 * 2 * t)
        signal += 0.25 * np.sin(2 * np.pi * f0 * 3 * t)
        signal += 0.125 * np.sin(2 * np.pi * f0 * 4 * t)
    else:
        signal = np.random.randn(n) * 0.3
    signal *= np.exp(-2.0 * t / (n / sr))
    signal = _apply_formants(signal, sr, f1, f2, f3)
    return signal


def _synthesize_plosive(n: int, sr: int, f0: float, f1: float, f2: float, f3: float,
                        place: Optional[str]) -> np.ndarray:
    t = np.arange(n) / sr
    closure_frames = int(n * 0.15)
    release_frames = n - closure_frames
    signal = np.zeros(n)
    if release_frames > 0:
        burst = np.random.randn(release_frames) * 0.6
        burst_env = np.exp(-10.0 * np.arange(release_frames) / release_frames)
        burst *= burst_env
        burst_sr = sr
        burst_fft = np.fft.rfft(burst)
        freqs = np.fft.rfftfreq(release_frames, d=1/burst_sr)
        if place == "bilabial":
            peak_freq = 500
        elif place == "alveolar":
            peak_freq = 4000
        elif place == "velar":
            peak_freq = 2000
        else:
            peak_freq = 1500
        burst_env_f = 1.0 / (1.0 + ((freqs - peak_freq) / (peak_freq * 0.5)) ** 4)
        burst_fft *= burst_env_f
        burst = np.fft.irfft(burst_fft, n=release_frames)
        peak_b = np.max(np.abs(burst))
        if peak_b > 1e-10:
            burst = burst / peak_b * 0.5
        signal[closure_frames:] = burst
    vowel_part = _synthesize_vowel(n // 2, sr, f0, f1 * 0.8, f2 * 0.9, f3 * 0.9,
                                    voiced=True)
    if len(vowel_part) < n:
        vowel_part = np.pad(vowel_part, (0, n - len(vowel_part)))
    signal += vowel_part * 0.3
    return signal


def _synthesize_fricative(n: int, sr: int, f1: float, f2: float, f3: float,
                          place: Optional[str]) -> np.ndarray:
    noise = np.random.randn(n) * 0.5
    noise_fft = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, d=1/sr)
    if place == "labiodental":
        peak = 8000
    elif place == "alveolar":
        peak = 6000
    elif place == "velar":
        peak = 3000
    else:
        peak = 4000
    bw = peak * 0.3
    env = 1.0 / (1.0 + ((freqs - peak) / (bw * 0.5)) ** 4)
    noise_fft *= env
    noise = np.fft.irfft(noise_fft, n=n)
    peak_n = np.max(np.abs(noise))
    if peak_n > 1e-10:
        noise = noise / peak_n * 0.4
    vowel_sig = _synthesize_vowel(n // 3, sr, 120, f1, f2, f3, voiced=True)
    if len(vowel_sig) < n:
        vowel_sig = np.pad(vowel_sig, (0, n - len(vowel_sig)))
    return noise + vowel_sig * 0.2


def _synthesize_nasal(n: int, sr: int, f0: float, f1: float, f2: float, f3: float,
                      nasal_energy: float) -> np.ndarray:
    t = np.arange(n) / sr
    signal = np.sin(2 * np.pi * f0 * t)
    signal += 0.4 * np.sin(2 * np.pi * f0 * 2 * t)
    signal *= np.exp(-1.5 * t / (n / sr))
    signal = _apply_formants(signal, sr, f1 * 0.7, f2 * 0.9, f3 * 0.85)
    nasal_resonance = np.random.randn(n) * 0.1 * nasal_energy
    signal += nasal_resonance
    return signal


def _apply_formants(signal: np.ndarray, sr: int, f1: float, f2: float, f3: float) -> np.ndarray:
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/sr)
    fft = np.fft.rfft(signal)
    def formant_envelope(f, fc, bw):
        return 1.0 / (1.0 + ((f - fc) / (bw * 0.5)) ** 4)
    envelope = formant_envelope(freqs, f1, 120)
    envelope *= formant_envelope(freqs, f2, 150)
    envelope *= formant_envelope(freqs, f3, 200)
    fft *= envelope
    return np.fft.irfft(fft, n=n)


def _describe_articulation(
    tongue_x: float, tongue_y: float, lip: float, jaw: float, velum: float,
    f0: float, f1: float, f2: float, f3: float,
    consonant_type: Optional[str], consonant_place: Optional[str],
) -> str:
    parts = []

    if tongue_y > 0.7:
        parts.append("lingua alta")
    elif tongue_y < 0.3:
        parts.append("lingua baixa")
    else:
        parts.append("lingua media")

    if tongue_x > 0.7:
        parts.append("anterior")
    elif tongue_x < 0.3:
        parts.append("posterior")
    else:
        parts.append("central")

    if lip > 0.5:
        parts.append("labios arredondados")
    else:
        parts.append("labios esticados")

    if jaw > 0.7:
        parts.append("mandibula aberta")
    elif jaw < 0.3:
        parts.append("mandibula fechada")

    if velum > 0.5:
        parts.append("velo aberto (nasal)")

    if consonant_type == "plosive":
        parts.append(f"oclusiva {consonant_place or ''}")
    elif consonant_type == "fricative":
        parts.append(f"fricativa {consonant_place or ''}")
    elif consonant_type == "nasal":
        parts.append("nasal")

    parts.append(f"{f0:.0f}Hz pitch")
    return f"[Articulacao: {', '.join(parts)}]"


def save_articulatory_wav(waveform: np.ndarray, filepath: str):
    n = len(waveform)
    signal_int = (waveform * 32767).astype(np.int16)
    buf = io.BytesIO()
    data_size = n * 2
    buf.write(struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE'))
    buf.write(struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16))
    buf.write(struct.pack('<4sI', b'data', data_size))
    buf.write(signal_int.tobytes())
    with open(filepath, 'wb') as f:
        f.write(buf.getvalue())


def random_articulation() -> Dict:
    """Gera configuração articulatória aleatória para exploração motora."""
    return {
        "tongue_dorsum_x": np.random.random(),
        "tongue_dorsum_y": np.random.random(),
        "lip_rounding": np.random.random(),
        "jaw_open": np.random.random(),
        "velum": np.random.random() * 0.3,
        "vocal_fold_tension": 0.3 + np.random.random() * 0.5,
    }
