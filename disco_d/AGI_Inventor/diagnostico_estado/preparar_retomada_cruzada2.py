"""Cópias de supervisor/auditor com lançador corrigido; sem tocar fonteV2."""

from piloto_planos_propostos import HERE

for name in ("retomar_experiencia_cruzada.py", "auditar_retomada_cruzada.py"):
    target = HERE / ("retomar_experiencia_cruzada_2.py" if name.startswith("retomar") else "auditar_retomada_cruzada_2.py")
    if target.exists():
        raise RuntimeError("cópia existe; preservar")
    text = (HERE / name).read_text(encoding="utf-8")
    for old, new in (
        ("worker_cruzada_sem_console.py", "worker_cruzada_sem_console_v2.py"),
        ("retomar_experiencia_cruzada.py", "retomar_experiencia_cruzada_2.py"),
        ("auditar_retomada_cruzada.py", "auditar_retomada_cruzada_2.py"),
        ("RETOMADA_CRUZADA.md", "RETOMADA_CRUZADA_2.md"),
        ("calibracao_worker_cruzada.json", "calibracao_worker_cruzada2.json"),
        ("experiencia_cruzada_retomada_", "experiencia_cruzada_retomada2_"),
        ("_retomada_ajuste131", "_retomada2_ajuste131"),
        ("_retomada_coleta137", "_retomada2_coleta137"),
        ("_retomada_avaliacao137", "_retomada2_avaliacao137"),
        ("auditoria_experiencia_cruzada_retomada.json", "auditoria_experiencia_cruzada_retomada2.json"),
    ):
        text = text.replace(old, new)
    if name.startswith("retomar"):
        old = 'earlier_seconds = old_v1["elapsed_wall_seconds"] + prior["elapsed_wall_seconds"]'
        new = ('earlier_seconds = json.loads((HERE / "resultados" / "experiencia_cruzada_retomada_recursos.json").read_text(encoding="utf-8"))["elapsed_wall_seconds_including_previous_attempts"]')
        if text.count(old) != 1:
            raise RuntimeError("âncora divergiu")
        text = text.replace(old, new)
    target.write_text(text, encoding="utf-8")
print("retomada2_sources_created_without_changing_workers_models_or_gates")
