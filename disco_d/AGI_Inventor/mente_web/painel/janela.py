"""Abre o painel como JANELA PROPRIA (Edge em modo aplicativo, sem barra de navegador)."""
import os
import subprocess
import webbrowser

EDGES = (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
         r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")


def abrir(url="http://127.0.0.1:8765"):
    for e in EDGES:
        if os.path.exists(e):
            subprocess.Popen([e, f"--app={url}", "--start-maximized"], creationflags=0x08000008)
            return
    webbrowser.open(url)
