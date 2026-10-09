"""Um trabalhador e um monitor; limite agregado deRSS e prazo do piloto."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent


class PilotMonitor:
    def __init__(self, seconds=1800, rss_bytes=6 * 1024 ** 3):
        self.start = time.monotonic()
        self.seconds, self.rss_bytes = seconds, rss_bytes
        self.peak_rss, self.maximum_processes, self.stages = 0, 1, []

    def run(self, arguments, label, directory=HERE):
        if time.monotonic() - self.start >= self.seconds:
            return {"label": label, "status": "inconclusive_wall_limit_before_launch", "exit_code": None}
        log_path = HERE / "resultados" / ("experiencia_cruzada_" + label + ".log")
        if log_path.exists():
            raise RuntimeError("log existente; preservar")
        environment = os.environ.copy()
        environment.update({"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        with log_path.open("w", encoding="utf-8") as output:
            child = subprocess.Popen([sys.executable, "-B", *arguments], cwd=directory, env=environment,
                                     stdout=output, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
            owned = psutil.Process(child.pid)
            creation = owned.create_time()
            stop_reason = None
            while child.poll() is None:
                try:
                    if owned.create_time() != creation:
                        raise RuntimeError("identidade do trabalhador mudou")
                    children = owned.children(recursive=True)
                    live = [psutil.Process(os.getpid()), owned, *children]
                    rss = sum(p.memory_info().rss for p in live if p.is_running())
                    self.peak_rss = max(self.peak_rss, rss)
                    self.maximum_processes = max(self.maximum_processes, len(live))
                    if len(live) > 2:
                        stop_reason = "inconclusive_process_limit"
                    elif rss > self.rss_bytes:
                        stop_reason = "inconclusive_memory_limit"
                    elif time.monotonic() - self.start > self.seconds:
                        stop_reason = "inconclusive_wall_limit"
                    if stop_reason:
                        # Only this Popen worker and its children; never enumerate other Python jobs.
                        for process in reversed(children):
                            process.terminate()
                        if owned.create_time() == creation:
                            child.terminate()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            if owned.create_time() == creation:
                                child.kill()
                            child.wait(timeout=5)
                        break
                except psutil.NoSuchProcess:
                    # It may have completed between poll and sampling; the handle remains authoritative.
                    pass
                time.sleep(.1)
            code = child.wait()
        record = {"label": label, "status": stop_reason or ("finished" if code == 0 else "inconclusive_worker_error"),
                  "exit_code": code, "log": log_path.relative_to(HERE).as_posix(),
                  "elapsed_wall_seconds": time.monotonic() - self.start}
        self.stages.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
        return record


def execute():
    from experiencia_cruzada import PROPOSAL, prepare, save_json
    path = HERE / "resultados" / "experiencia_cruzada_recursos.json"
    if path.exists() or PROPOSAL.exists():
        raise RuntimeError("piloto/proposta já existe; preservar")
    monitor = PilotMonitor()
    prepare()
    jobs = [("coletar_experiencia_cruzada.py", [game, "131"], game + "_coleta131") for game in ("breakout", "pong")]
    jobs += [("avaliar_experiencia_cruzada.py", [game, "--ajustar"], game + "_ajuste131") for game in ("breakout", "pong")]
    jobs += [("coletar_experiencia_cruzada.py", [game, "137"], game + "_coleta137") for game in ("breakout", "pong")]
    jobs += [("avaliar_experiencia_cruzada.py", [game, "--avaliar"], game + "_avaliacao137") for game in ("breakout", "pong")]
    completed = True
    for script, arguments, label in jobs:
        stage = monitor.run([str(HERE / script), *arguments], label)
        partial = {"status": "running_not_final" if stage["status"] == "finished" else stage["status"],
                   "stages": monitor.stages, "elapsed_wall_seconds": time.monotonic() - monitor.start,
                   "peak_sampled_aggregate_rss_bytes": monitor.peak_rss, "maximum_processes": monitor.maximum_processes,
                   "limits": {"wall_seconds": monitor.seconds, "aggregate_rss_bytes": monitor.rss_bytes, "processes": 2},
                   "sample_interval_seconds": .1, "rss_peak_is_sampled_not_os_hard_limit": True,
                   "human_level_goal_achieved": False}
        save_json(path, partial)
        if stage["status"] != "finished":
            completed = False
            break
    partial["status"] = "pilot_finished_not_agent" if completed else "pilot_incomplete_inconclusive"
    save_json(path, partial)
    print(json.dumps(partial, indent=2, allow_nan=False))
    return 0 if completed else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executar", action="store_true", required=True)
    parser.parse_args()
    raise SystemExit(execute())
