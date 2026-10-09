"""
Roda quando o Windows liga (atalho na pasta Inicializar): acorda a mente, abre o
painel e fala o resumo com a voz Maria. Nao abre janela de terminal.
Para desligar o inicio automatico: apagar MenteViva.vbs da pasta Inicializar.
"""
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import webbrowser

import psutil

RAIZ = os.path.dirname(os.path.abspath(__file__))
PYW = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
SEM_JANELA = 0x08000000 | 0x00000008


def rodando(trecho):
    for p in psutil.process_iter(["cmdline"]):
        if trecho in " ".join(p.info["cmdline"] or []).replace("\\", "/"):
            return True
    return False


time.sleep(15)                                  # deixa a rede e a area de trabalho subirem
viva = os.path.join(RAIZ, "vidas", "viva")
if os.path.exists(os.path.join(viva, "PARAR")):
    os.remove(os.path.join(viva, "PARAR"))       # pausada antes de desligar: abrir o PC e abrir o app retomam
if os.path.exists(os.path.join(viva, "estado.pkl")):
    if not rodando("guardiao.py"):
        v = open(os.path.join(viva, "versao.txt")).read().strip() if os.path.exists(
            os.path.join(viva, "versao.txt")) else "v2"
        subprocess.Popen([PYW, "guardiao.py", os.path.join("vidas", "viva"), v], cwd=RAIZ,
                         creationflags=SEM_JANELA)
elif not rodando("decide_e_solta.py"):
    subprocess.Popen([PYW, "decide_e_solta.py"], cwd=RAIZ, creationflags=SEM_JANELA)
if not rodando("painel/servidor.py"):
    subprocess.Popen([PYW, os.path.join("painel", "servidor.py")], cwd=RAIZ, creationflags=SEM_JANELA)
time.sleep(8)
sys.path.insert(0, os.path.join(RAIZ, "painel"))
import janela  # noqa: E402
janela.abrir()
# fala ao ligar: desligada ate o usuario escolher a voz (ele nao quis a Maria)
FALAR = os.path.exists(os.path.join(RAIZ, "painel", "voz_ligada.txt"))
try:
    if not FALAR:
        raise RuntimeError("voz desligada")
    time.sleep(4)
    txt = urllib.request.urlopen("http://127.0.0.1:8765/api/fala", timeout=60).read().decode("utf-8")
    arq = os.path.join(tempfile.gettempdir(), "mente_fala.txt")
    open(arq, "w", encoding="utf-8").write(txt)
    cmd = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
           "try { $s.SelectVoice('Microsoft Maria Desktop') } catch {}; $s.Rate = 0; "
           f"$s.Speak((Get-Content -Raw -Encoding UTF8 '{arq}'))")
    subprocess.run(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", cmd],
                   creationflags=0x08000000)
except Exception:
    pass
