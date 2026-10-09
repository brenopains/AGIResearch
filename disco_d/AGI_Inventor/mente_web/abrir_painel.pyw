"""Atalho da area de trabalho: garante que o painel esta no ar e abre no navegador."""
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser

RAIZ = os.path.dirname(os.path.abspath(__file__))
PYW = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
try:
    urllib.request.urlopen("http://127.0.0.1:8765/api/organismo", timeout=3)
except Exception:
    subprocess.Popen([PYW, os.path.join("painel", "servidor.py")], cwd=RAIZ, creationflags=0x08000008)
    time.sleep(4)
sys.path.insert(0, RAIZ)
try:
    import controle  # noqa: E402
    controle.retomar(os.path.join(RAIZ, "vidas", "viva"))     # abrir o app = ela volta de onde parou
except Exception:
    pass
sys.path.insert(0, os.path.join(RAIZ, "painel"))
import janela  # noqa: E402
janela.abrir()
