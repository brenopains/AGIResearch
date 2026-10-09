import numpy as np
import struct
import io
import os
import tempfile

SAMPLE_RATE = 22050

FORMANT_MAP = {
    'aa': (700, 1100, 2500),
    'ee': (500, 1800, 2500),
    'ii': (300, 2200, 3000),
    'oo': (500, 900, 2500),
    'uu': (300, 900, 2300),
    'mmm': (200, 1000, 2000),
}


def synthesize(pitch=120, f1=500, f2=1500, f3=2500,
               amplitude=0.6, duration=0.5, voiced=True) -> bytes:
    """
    Gera um WAV monofônico a partir de parâmetros vocais.

    A AGI controla diretamente:
      pitch  (F0, 80-300 Hz)  — frequência fundamental (cordas vocais)
      f1-f3  (200-3500 Hz)   — formantes (trato vocal)
      amplitude (0.0-1.0)    — volume
      duration (0.1-3.0 s)   — duração
      voiced (bool)          — cordas vibrando (True) ou sussurro (False)
    """
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n) / SAMPLE_RATE

    if voiced:
        signal = np.sin(2 * np.pi * pitch * t)
        signal += 0.5 * np.sin(2 * np.pi * pitch * 2 * t)
        signal += 0.25 * np.sin(2 * np.pi * pitch * 3 * t)
    else:
        signal = np.random.randn(n) * 0.3

    signal *= np.exp(-1.5 * t / duration)

    # Formantes via multiplicação por envelope de frequência (espectral, estável)
    sr = SAMPLE_RATE
    freqs = np.fft.rfftfreq(n, d=1/sr)
    fft = np.fft.rfft(signal)

    def formant_envelope(f, fc, bw):
        return 1.0 / (1.0 + ((f - fc) / (bw/2)) ** 4)

    envelope = formant_envelope(freqs, f1, 120)
    envelope *= formant_envelope(freqs, f2, 150)
    envelope *= formant_envelope(freqs, f3, 200)

    fft *= envelope
    signal = np.fft.irfft(fft, n=n)

    peak = np.max(np.abs(signal))
    if peak > 1e-10:
        signal = signal / peak
    signal = np.clip(signal * amplitude * 0.7, -1.0, 1.0)

    signal_int = (signal * 32767).astype(np.int16)

    buf = io.BytesIO()
    data_size = n * 2
    buf.write(struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE'))
    buf.write(struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16))
    buf.write(struct.pack('<4sI', b'data', data_size))
    buf.write(signal_int.tobytes())
    return buf.getvalue()


def formant_for_vowel(vowel: str):
    base = FORMANT_MAP.get(vowel, (500, 1500, 2500))
    return base[0] + np.random.randint(-30, 30), \
           base[1] + np.random.randint(-50, 50), \
           base[2] + np.random.randint(-80, 80)


def save_wav_to(wav_bytes: bytes, filepath: str):
    with open(filepath, 'wb') as f:
        f.write(wav_bytes)
