"""Diagnóstico posterior sem ajuste: leitores congelados em experiência 163.

Definido depois de ver o portão de planejamento. Não muda esse portão e não
é um novo teste confirmatório de capacidade geral. Compara todos os leitores.
"""
import os
import platform
import sys
if sys.platform == "win32":
    platform.machine = platform._get_machine_win32
    platform.system = lambda: "Windows"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
import argparse
import json
import numpy as np
from piloto_leitor_eventos import (HERE, STEM, BODIES, RESULT, BASES, READERS, EventReader, GeneratedTransport,
    load_bank, load_fresh, transitions, verify_frozen, require, save_new, sha, local_metrics)
from auditar_leitor_eventos import independent_design, local_scores
from auditar_semantica_prefixo import equal_report

MANIFEST = STEM.with_name(STEM.name + "_diagnostico_fresco_manifesto.json")
OUTPUT = STEM.with_name(STEM.name + "_diagnostico_fresco.json")


def prepare():
    verify_frozen()
    require(not MANIFEST.exists() and not OUTPUT.exists(), "diagnóstico existe; preservar")
    sources = ("diagnosticar_eventos_frescos.py", "auditar_leitor_eventos.py", BODIES.relative_to(HERE).as_posix(),
        RESULT.relative_to(HERE).as_posix(), "resultados/eventos_auditoria.json")
    save_new(MANIFEST, {"status": "posthoc_diagnostic_frozen_before_scoring", "source_hashes": {p:sha(HERE/p) for p in sources},
        "posthoc_after_planning_results": True, "fit_calls": 0, "ale_calls": 0, "changes_confirmatory_gate": False})
    return {"status": "posthoc_event_diagnostic_frozen", "sha256":sha(MANIFEST)}


def run():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path, expected in manifest["source_hashes"].items():
        require(sha(HERE/path) == expected, "fonte/resultado mudou")
    require(not OUTPUT.exists(), "diagnóstico existe; preservar")
    frozen, reports = verify_frozen(), {}
    maximum, evaluated = 0., 0
    for game in ("breakout", "pong"):
        data = transitions(load_fresh(game))
        train = transitions(load_bank(game,149))
        graph = frozen["construction"][game]["observed_integer_graph"]
        reports[game] = {"training_graph": {"constraint_count":len(train["reward"]),
            "nodes":graph["observed_nodes"], "components":graph["connected_components"],
            "constraints_beyond_spanning_forest":len(train["reward"])-graph["observed_nodes"]+graph["connected_components"],
            "compatibility_not_generalization":True}, "bases":{}}
        for basis in BASES:
            files = frozen["bodies"][game][basis]
            body = GeneratedTransport.load(HERE/files["feature"])
            p, pn = body.features(data["source"]), body.features(data["successor"])
            rows = {}
            for arm in READERS:
                reader = EventReader.load(HERE/files["readers"][arm])
                predicted = reader.predict(p,pn,data["actions"])
                mode = "transition" if arm in ("shuffled_successor","shuffled_rewards") else arm
                x = independent_design(p,pn,mode)
                checked = np.array([row@reader.coefficient[a] for row,a in zip(x,data["actions"])])
                error = float(np.max(np.abs(predicted-checked)))
                maximum = max(maximum,error)
                require(error < 1e-8, "definição executável diverge")
                local = local_metrics(data["reward"],predicted)
                equal_report(local,local_scores(data["reward"],checked),"métricas locais independentes")
                rows[arm] = {"local":local,"per_episode": {str(e):local_metrics(data["reward"][data["episodes"]==e],predicted[data["episodes"]==e]) for e in range(8)}}
                evaluated += len(predicted)
            comparison = {}
            for reference in ("before","shuffled_successor","shuffled_rewards","action_prior","zero"):
                improvement = [rows[reference]["per_episode"][str(e)]["mse"]-rows["transition"]["per_episode"][str(e)]["mse"] for e in range(8)]
                comparison[reference] = {"pooled_mse_improvement":rows[reference]["local"]["mse"]-rows["transition"]["local"]["mse"],
                    "positive_episodes":sum(v>1e-12 for v in improvement),"episode_mse_improvements":improvement}
            reports[game]["bases"][basis] = {"readers":rows,"transition_comparisons_diagnostic_only":comparison}
    save_new(OUTPUT,{"status":"posthoc_event_recognition_diagnostic_complete","games":reports,
        "predictions_checked":evaluated,"maximum_independent_prediction_error":maximum,"ale_calls":0,"fit_calls":0,
        "manifest_sha256":sha(MANIFEST),"human_level_goal_achieved":False,"does_not_change_planning_verdict":True,
        "posthoc_after_planning_results":True,"recognition_uses_observed_successor_not_available_before_action":True})
    return {"status":"posthoc_event_recognition_diagnostic_complete","primary163": {g:{"readers":{a:r["local"] for a,r in v["bases"]["no_generated_functions"]["readers"].items()},
        "transition_comparisons":v["bases"]["no_generated_functions"]["transition_comparisons_diagnostic_only"],"training_graph":v["training_graph"]} for g,v in reports.items()},
        "predictions_checked":evaluated,"maximum_independent_prediction_error":maximum}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","run"))
    print(json.dumps({"prepare":prepare,"run":run}[parser.parse_args().mode](),ensure_ascii=False,indent=2))
