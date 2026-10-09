"""
voz — fala local em portugues do Brasil (Kokoro-82M, roda no processador, nada sai do PC)
==========================================================================================
Uso:  echo texto | python falar.py <saida.wav> [voz] [--puro]
  voz: pm_alex | pm_santa | mistura (padrao: a do arquivo voz_escolhida.txt, ou pm_alex)
  mistura = 60% alex + 40% santa: um timbre que nao e de pessoa nenhuma.
Tratamento "IA da casa" (desligado com --puro): duas reflexoes curtas e baixas (sala
metalica discreta), um pouco de corpo nos graves e volume normalizado.
"""
import os
import re
import sys

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

AQUI = os.path.dirname(os.path.abspath(__file__))


def voz_padrao():
    arq = os.path.join(AQUI, "voz_escolhida.txt")
    return open(arq, encoding="utf-8").read().strip() if os.path.exists(arq) else "pm_alex"


def estilo(k, nome):
    if nome == "mistura":
        return 0.6 * k.get_voice_style("pm_alex") + 0.4 * k.get_voice_style("pm_santa")
    return nome


def ia_da_casa(x, sr):
    y = x.copy()
    for atraso_ms, ganho in ((17, 0.16), (36, 0.08)):          # reflexoes curtas
        d = int(sr * atraso_ms / 1000)
        y[d:] += ganho * x[:-d]
    # um pouco de corpo: soma uma versao suavizada (passa-baixa simples) do proprio sinal
    k = int(sr / 700)
    grave = np.convolve(y, np.ones(k) / k, mode="same")
    y = y + 0.25 * grave
    return y


def main():
    saida = sys.argv[1]
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    nome = args[0] if args else voz_padrao()
    texto = sys.stdin.buffer.read().decode("utf-8").strip() or "Sistemas operando, senhor."
    k = Kokoro(os.path.join(AQUI, "kokoro-v1.0.onnx"), os.path.join(AQUI, "voices-v1.0.bin"))
    frases = [f.strip() for f in re.split(r"(?<=[.!?])\s+", texto) if f.strip()]
    pedacos, sr = [], 24000
    for f in frases:
        a, sr = k.create(f, voice=estilo(k, nome), speed=0.95, lang="pt-br")
        pedacos += [a, np.zeros(int(sr * 0.14), np.float32)]
    x = np.concatenate(pedacos).astype(np.float32)
    if "--puro" not in sys.argv:
        x = ia_da_casa(x, sr)
    x = x / (np.abs(x).max() + 1e-9) * 0.89
    sf.write(saida, x, sr)


if __name__ == "__main__":
    main()
