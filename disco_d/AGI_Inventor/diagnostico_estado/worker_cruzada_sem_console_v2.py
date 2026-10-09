"""Metadados nativos de Windows/arquitetura antes de importar ALE/NumPy."""

import platform
import runpy
import sys
from pathlib import Path


root = Path(__file__).resolve().parent
allowed = {"avaliar_experiencia_cruzada_v2.py", "coletar_experiencia_cruzada_v2.py",
           "calibrar_importacoes_cruzada.py", "retomar_experiencia_cruzada_2.py"}
target = Path(sys.argv[1]).resolve()
if target.parent != root or target.name not in allowed:
    raise ValueError("somente trabalhadores próprios")
if sys.platform == "win32":
    platform.machine = platform._get_machine_win32
    platform.system = lambda: "Windows"
    print("native_system_queries", platform.system(), platform.machine(), flush=True)
sys.argv = [str(target), *sys.argv[2:]]
runpy.run_path(str(target), run_name="__main__")
