"""Perfil posterior: equivalência do ajuste eficiente; sem avaliar capacidade."""
import os
import platform
import sys
if sys.platform=="win32":
    platform.machine=platform._get_machine_win32
    platform.system=lambda:"Windows"
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable]="2"
import argparse
import json
import time
import numpy as np
from piloto_resolucao import HERE,STEM,PROPOSAL,FunctionTransport,ObservedFilters,raw_inputs,scaled,verify,require,save_new,sha
from dicionario_rapido import FastFunctionTransport
from coletar_prospectiva import rss_bytes

MANIFEST=STEM.with_name(STEM.name+"_perfil_manifesto.json")
OUTPUT=STEM.with_name(STEM.name+"_perfil.json")
BODY=STEM.with_name(STEM.name+"_breakout_learned_body.npz")
SCALE=BODY.with_name(BODY.stem+"_scale.npy")
FILTER=STEM.with_name(STEM.name+"_breakout_learned_filters.npz")


def prepare():
    verify()
    require(not MANIFEST.exists() and not OUTPUT.exists(),"perfil existe; preservar")
    sources=[HERE/"diagnosticar_resolucao_custo.py",HERE/"dicionario_rapido.py",HERE/"testes_dicionario_rapido.py",PROPOSAL,BODY,SCALE,FILTER]
    save_new(MANIFEST,{"status":"profile_frozen_after_incomplete_fit","maximum_wall_seconds":60,
        "scope":"performance_and_semantic_equivalence_only_not_capability", "prior_fit_incomplete_at_budget_checkpoint":True,
        "source_hashes":{p.relative_to(HERE).as_posix():sha(p) for p in sources}})
    return {"status":"resolution_profile_frozen","sha256":sha(MANIFEST)}


def run():
    verify()
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path,expected in manifest["source_hashes"].items():
        require(sha(HERE/path)==expected,"fonte/corpo mudou")
    require(not OUTPUT.exists(),"perfil existe")
    started=time.perf_counter()
    functions=ObservedFilters.load(FILTER)
    data=raw_inputs("breakout",149,"learned",functions)
    extraction=time.perf_counter()-started
    scale=np.load(SCALE,allow_pickle=False)
    source,successor=scaled(data["source"],scale),scaled(data["successor"],scale)
    begin=time.perf_counter()
    body=FastFunctionTransport().fit(source,successor,data["actions"],data["reward"],data["continuation"])
    fitting=time.perf_counter()-begin
    old=FunctionTransport.load(BODY)
    same_support=np.array_equal(old.support,body.support)
    coefficient_error=max(float(np.max(np.abs(old.reward-body.reward))),float(np.max(np.abs(old.transport-body.transport))))
    words=[[0,1,3,2]*4]*32
    prediction_error=float(np.max(np.abs(old.predict(source[:32],words)-body.predict(source[:32],words))))
    elapsed=time.perf_counter()-started
    result={"status":"equivalent_faster_dictionary_instance_verified" if same_support and old.scale==body.scale and coefficient_error<1e-8 and prediction_error<1e-8 else "dictionary_equivalence_failed",
        "extraction_wall_seconds":extraction,"fit_wall_seconds":fitting,"total_wall_seconds":elapsed,"observed_rss_bytes_at_end":rss_bytes(),
        "same_support_numeric_values":same_support,"same_scale":old.scale==body.scale,"maximum_coefficient_error":coefficient_error,
        "maximum_prediction_error":prediction_error,"physical_calls":0,"additional_fit_calls":1,"manifest_sha256":sha(MANIFEST),
        "human_level_goal_achieved":False,"main_pilot_still_incomplete":True,"profile_budget_satisfied":elapsed<60}
    save_new(OUTPUT,result)
    require(result["status"]=="equivalent_faster_dictionary_instance_verified" and elapsed<60,"perfil não certificou equivalência/orçamento")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","run"))
    print(json.dumps({"prepare":prepare,"run":run}[parser.parse_args().mode](),ensure_ascii=False,indent=2))
