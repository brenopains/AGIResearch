"""Consulta nativa de arquitetura: evitar cmd/ver da NumPy.testing no Windows."""

import platform
import runpy
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    allowed = {"avaliar_experiencia_cruzada_v2.py", "coletar_experiencia_cruzada_v2.py", "identificar_origem_cmd.py"}
    target = Path(sys.argv[1]).resolve()
    if target.parent != root or target.name not in allowed:
        raise ValueError("somente trabalhadores próprios da bancada")
    if sys.platform == "win32":
        native_machine = getattr(platform, "_get_machine_win32", None)
        if native_machine is None:
            raise RuntimeError("consulta nativa de arquitetura indisponível")
        # This is the same implementation that platform.uname uses for machine.
        # It avoids computing unrelated release/version through a ver shell.
        platform.machine = native_machine
        print("architecture_query_direct", platform.machine(), flush=True)
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
