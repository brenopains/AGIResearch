"""Continuar apenas etapas não executadas, com corpos/propostasV2 preservados."""

import json
import time

from experiencia_cruzada_v2 import PROPOSAL, TRAIN_SEED, base, manifest_path, result_path, save_json, verify_proposal
from piloto_experiencia_cruzada_v2 import PilotMonitor
from piloto_planos_propostos import HERE, sha


def main():
    verify_proposal()
    resources_path = HERE / "resultados" / "experiencia_cruzada_retomada2_recursos.json"
    freeze_path = HERE / "resultados" / "experiencia_cruzada_retomada2_manifesto.json"
    if resources_path.exists() or freeze_path.exists():
        raise RuntimeError("retomada já existe; preservar")
    prior = json.loads((HERE / "resultados" / "experiencia_cruzada_v2_recursos.json").read_text(encoding="utf-8"))
    calibration = json.loads((HERE / "resultados" / "calibracao_worker_cruzada2.json").read_text(encoding="utf-8"))
    if calibration["status"] != "finished" or calibration["maximum_processes"] > 2:
        raise RuntimeError("monitor/consulta nativa não calibrados")
    if [s["label"] for s in prior["stages"][:2]] != ["breakout_coleta131", "pong_coleta131"] or any(s["status"] != "finished" for s in prior["stages"][:2]):
        raise RuntimeError("bancos131 não foram concluídos")
    files = ["worker_cruzada_sem_console_v2.py", "retomar_experiencia_cruzada_2.py", "auditar_retomada_cruzada_2.py", "RETOMADA_CRUZADA_2.md",
             "resultados/calibracao_worker_cruzada2.json", "resultados/experiencia_cruzada_v2_recursos.json"]
    for game in ("breakout", "pong"):
        if manifest_path(game).exists() or base(game, 137).with_suffix(".npz").exists() or result_path(game).exists():
            raise RuntimeError("retomada só vale antes de ajuste/teste")
        files += [base(game, TRAIN_SEED).with_suffix(s).relative_to(HERE).as_posix() for s in (".npz", ".json", ".jsonl")]
    save_json(freeze_path, {"status": "frozen_before_remaining_stages", "proposal_v2_sha256": sha(PROPOSAL),
                           "source_hashes": {p: sha(HERE / p) for p in files}, "algorithm_and_gates_unchanged": True,
                           "test137_not_collected": True})
    monitor = PilotMonitor()
    old_v1 = json.loads((HERE / "resultados" / "experiencia_cruzada_recursos.json").read_text(encoding="utf-8"))
    earlier_seconds = json.loads((HERE / "resultados" / "experiencia_cruzada_retomada_recursos.json").read_text(encoding="utf-8"))["elapsed_wall_seconds_including_previous_attempts"]
    monitor.start -= earlier_seconds
    stages = [("avaliar_experiencia_cruzada_v2.py", [g, "--ajustar"], g + "_retomada2_ajuste131") for g in ("breakout", "pong")]
    stages += [("coletar_experiencia_cruzada_v2.py", [g, "137"], g + "_retomada2_coleta137") for g in ("breakout", "pong")]
    stages += [("avaliar_experiencia_cruzada_v2.py", [g, "--avaliar"], g + "_retomada2_avaliacao137") for g in ("breakout", "pong")]
    complete = True
    for script, arguments, label in stages:
        stage = monitor.run([str(HERE / "worker_cruzada_sem_console_v2.py"), str(HERE / script), *arguments], label)
        record = {"status": "running_not_final", "stages": monitor.stages, "completed_prior_collections": prior["stages"][:2],
                  "earlier_pilot_wall_seconds": earlier_seconds, "elapsed_wall_seconds_including_previous_attempts": time.monotonic() - monitor.start,
                  "peak_sampled_aggregate_rss_bytes_this_resume": monitor.peak_rss, "maximum_processes_this_resume": monitor.maximum_processes,
                  "previous_failed_attempt_maximum_processes": prior["maximum_processes"], "sample_interval_seconds": .1,
                  "limits": {"wall_seconds": 1800, "aggregate_rss_bytes": 6 * 1024 ** 3, "processes": 2},
                  "human_level_goal_achieved": False, "freeze_sha256": sha(freeze_path)}
        save_json(resources_path, record)
        if stage["status"] != "finished":
            complete = False
            break
    record["status"] = "remaining_stages_finished_not_agent" if complete else "retomada_incomplete_inconclusive"
    save_json(resources_path, record)
    print(json.dumps(record, indent=2))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
