"""Verificar consulta nativa e contagem de processos antes de retomar ajuste."""

import json

from experiencia_cruzada import save_json
from piloto_experiencia_cruzada_v2 import PilotMonitor
from piloto_planos_propostos import HERE, sha


path = HERE / "resultados" / "calibracao_worker_cruzada.json"
if path.exists():
    raise RuntimeError("calibração já existe; preservar")
monitor = PilotMonitor(seconds=20)
stage = monitor.run([str(HERE / "worker_cruzada_sem_console.py"), str(HERE / "identificar_origem_cmd.py")], "calibracao_worker")
record = {"status": stage["status"], "maximum_processes": monitor.maximum_processes,
          "peak_sampled_aggregate_rss_bytes": monitor.peak_rss, "stage": stage,
          "source_sha256": sha(HERE / "calibrar_worker_cruzada.py"), "worker_sha256": sha(HERE / "worker_cruzada_sem_console.py"),
          "models_saved": False, "test137_outcomes_used": 0}
save_json(path, record)
print(json.dumps(record, indent=2))
if stage["status"] != "finished" or monitor.maximum_processes > 2:
    raise RuntimeError("consulta nativa não eliminou os processos auxiliares")
