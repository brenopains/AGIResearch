"""Identificar somente descendentes de um trabalhador próprio de calibração."""

import json
import os
import subprocess
import sys
import time

import psutil

from experiencia_cruzada import save_json
from piloto_planos_propostos import HERE, sha


def main():
    path = HERE / "resultados" / "diagnostico_lancadores_cruzada.json"
    if path.exists():
        raise RuntimeError("diagnóstico existe; preservar")
    code = ("import avaliar_experiencia_cruzada_v2 as e; from leituras_por_programa import ConditionalReading; "
            "print('import_finished',flush=True); h,p,q,l,r,roots=e.training_data('breakout'); print('bank_loaded',flush=True); "
            "ConditionalReading().fit(h[:16],p[:16],q[:16],l[:16],r[:16],roots[:16]); print('tiny_fit_finished',flush=True)")
    environment = os.environ.copy()
    environment.update({"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    output = path.with_suffix(".log")
    seen, start = {}, time.monotonic()
    with output.open("w", encoding="utf-8") as stream:
        child = subprocess.Popen([sys.executable, "-B", "-c", code], cwd=HERE, env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
        owned = psutil.Process(child.pid)
        creation = owned.create_time()
        while child.poll() is None:
            try:
                for p in [owned, *owned.children(recursive=True)]:
                    key = str(p.pid) + ":" + str(p.create_time())
                    if key not in seen:
                        seen[key] = {"pid": p.pid, "exe": p.exe(), "name": p.name(), "parent_pid": p.ppid()}
                if time.monotonic() - start > 20:
                    for p in reversed(owned.children(recursive=True)):
                        p.terminate()
                    if owned.create_time() == creation:
                        child.terminate()
                    break
            except psutil.NoSuchProcess:
                pass
            time.sleep(.001)
        status = child.wait(timeout=5)
    record = {"status": "owned_subprocess_inventory", "exit_code": status, "processes": list(seen.values()),
              "wall_seconds": time.monotonic() - start, "all_processes_belonged_to_own_worker": True,
              "models_saved": False, "test137_outcomes_used": 0, "source_sha256": sha(HERE / "diagnosticar_lancadores_cruzada.py")}
    save_json(path, record)
    print(json.dumps(record, indent=2))
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
