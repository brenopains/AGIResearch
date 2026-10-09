"""Auditoria adicional da retomada e auditoria independenteV2 dos resultados."""

import json

from auditar_experiencia_cruzada_v2 import audit, require
from experiencia_cruzada_v2 import PROPOSAL, save_json, verify_proposal
from piloto_planos_propostos import HERE, sha


verify_proposal()
freeze_path = HERE / "resultados" / "experiencia_cruzada_retomada_manifesto.json"
resource_path = HERE / "resultados" / "experiencia_cruzada_retomada_recursos.json"
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
for p, expected in freeze["source_hashes"].items():
    require(sha(HERE / p) == expected, "fonte/corpo anterior à retomada mudou")
require(freeze["proposal_v2_sha256"] == sha(PROPOSAL) and freeze["algorithm_and_gates_unchanged"] and freeze["test137_not_collected"], "contrato de retomada divergiu")
resources = json.loads(resource_path.read_text(encoding="utf-8"))
require(resources["status"] == "remaining_stages_finished_not_agent" and len(resources["stages"]) == 6
        and all(s["status"] == "finished" and s["exit_code"] == 0 for s in resources["stages"]), "retomada incompleta")
require(resources["peak_sampled_aggregate_rss_bytes_this_resume"] <= 6 * 1024 ** 3 and resources["maximum_processes_this_resume"] <= 2
        and resources["elapsed_wall_seconds_including_previous_attempts"] <= 1800 and resources["freeze_sha256"] == sha(freeze_path), "limite/manifesto de retomada divergiu")
result = {"status": "audit_passed", "games": [audit(game) for game in ("breakout", "pong")],
          "resource_monitor_resume_verified": True, "previous_failed_attempts_preserved": True,
          "resources_sha256": sha(resource_path), "human_level_goal_achieved": False}
output = HERE / "resultados" / "auditoria_experiencia_cruzada_retomada.json"
if output.exists():
    require(json.loads(output.read_text(encoding="utf-8")) == result, "auditoria existente diverge; preservar")
else:
    save_json(output, result)
print(json.dumps(result, indent=2))
