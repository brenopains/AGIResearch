"""Mostrar origem de qualquer novo processo, interrompendo antes de criá-lo."""

import platform
import runpy
import subprocess
import sys
import traceback
from pathlib import Path


def blocked(*args, **kwargs):
    print("unexpected_process_request", args, flush=True)
    traceback.print_stack(limit=18)
    raise RuntimeError("diagnóstico: subprocesso interrompido antes de criar")


root = Path(__file__).resolve().parent
subprocess.Popen = blocked
platform.machine = platform._get_machine_win32
sys.argv = [str(root / "avaliar_experiencia_cruzada_v2.py"), "breakout", "--ajustar"]
runpy.run_path(sys.argv[0], run_name="__main__")
