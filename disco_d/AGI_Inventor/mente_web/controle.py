"""
mente_web.controle — pausar e retomar a vida dela (botao do painel e abertura do app)
=====================================================================================
PAUSAR  = cria <pasta>/PARAR: ela termina o que esta fazendo (ate ~1 min, se estiver
          num video), consolida, SALVA e para; o guardiao tambem para.
RETOMAR = apaga PARAR e religa o guardiao: ela acorda exatamente de onde parou.
Abrir o app (icone 'Mente' ou o Windows ligando) tambem retoma.
"""
import os
import subprocess
import sys

import psutil

RAIZ = os.path.dirname(os.path.abspath(__file__))
PYW = os.path.join(os.path.dirname(sys.executable), "pythonw.exe") if os.name == "nt" else sys.executable
SEM_JANELA = (0x08000000 | 0x00000008) if os.name == "nt" else 0


def _rodando(trecho, pasta):
    nome = os.path.basename(os.path.normpath(pasta))
    for p in psutil.process_iter(["cmdline"]):
        try:
            cl = " ".join(p.info["cmdline"] or []).replace("\\", "/")
        except Exception:
            continue
        if trecho in cl and nome in cl:
            return True
    return False


def estado(pasta):
    """'pausando' (pediu para parar, ainda salvando), 'pausada' ou None (vivendo normal)."""
    if not os.path.exists(os.path.join(pasta, "PARAR")):
        return None
    return "pausando" if _rodando("viver.py", pasta) else "pausada"


def pausar(pasta):
    with open(os.path.join(pasta, "PARAR"), "w", encoding="utf-8") as f:
        f.write("pausada pelo painel\n")
    return estado(pasta)


def retomar(pasta):
    arq = os.path.join(pasta, "PARAR")
    if os.path.exists(arq):
        os.remove(arq)
    if os.path.exists(os.path.join(pasta, "estado.pkl")) and not _rodando("guardiao.py", pasta):
        v = "v1"
        vt = os.path.join(pasta, "versao.txt")
        if os.path.exists(vt):
            v = open(vt, encoding="utf-8").read().strip() or v
        rel = os.path.relpath(pasta, RAIZ)
        subprocess.Popen([PYW, "guardiao.py", rel, v], cwd=RAIZ, creationflags=SEM_JANELA)
    return estado(pasta)
