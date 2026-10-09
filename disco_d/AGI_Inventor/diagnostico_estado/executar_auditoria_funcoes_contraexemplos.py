"""Bootstrap da auditoria congelada: configurar BLAS antes de importar NumPy.

O auditor original importa NumPy antes de o piloto configurar suas threads.
Este ponto de entrada preserva integralmente fontes, dados e critérios.
"""
import os
import platform
import sys
if sys.platform == "win32":
    platform.machine = platform._get_machine_win32
    platform.system = lambda: "Windows"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import json
from pathlib import Path
from auditar_funcoes_contraexemplos import audit
from piloto_funcoes_contraexemplos import HERE, sha, save_new, verify_proposal

if __name__ == "__main__":
    verify_proposal()
    manifest = HERE / "resultados" / "funcoes_contraexemplos_bootstrap_auditoria.json"
    definition = {
        "scope": "runtime_bootstrap_only_not_changed_audit_or_gate",
        "reason": "original_auditor_imports_numpy_before_thread_configuration",
        "blas_threads_before_numpy_import": 2,
        "first_direct_audit_failure": "RuntimeError: geração/origem não reproduz",
        "source_hashes": {name: sha(HERE / name) for name in (
            Path(__file__).name, "auditar_funcoes_contraexemplos.py",
            "piloto_funcoes_contraexemplos.py", "funcoes_contraexemplos.py")},
    }
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != definition:
            raise RuntimeError("bootstrap mudou; preservar registro anterior")
    else:
        save_new(manifest, definition)
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
