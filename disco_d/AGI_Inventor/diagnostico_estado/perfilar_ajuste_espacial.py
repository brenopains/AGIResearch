"""Perfil da classe original, com verificação de equivalência e sem avaliação."""
import os
import platform
import sys
if sys.platform=="win32":
    platform.machine=platform._get_machine_win32
    platform.system=lambda:"Windows"
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable]="2"
import argparse
import cProfile
import json
import pstats
import time
import numpy as np
from piloto_resolucao import HERE,STEM,PROPOSAL,ObservedFilters,FunctionTransport,raw_inputs,scaled,verify,require,save_new,sha

MANIFEST=STEM.with_name(STEM.name+"_perfil_original_manifesto.json")
OUTPUT=STEM.with_name(STEM.name+"_perfil_original.json")


def prepare():
    verify()
    require(not MANIFEST.exists() and not OUTPUT.exists(),"perfil existente; preservar")
    names=("perfilar_ajuste_espacial.py","resultados/resolucao_breakout_learned_body.npz",
        "resultados/resolucao_breakout_learned_body_scale.npy","resultados/resolucao_breakout_learned_filters.npz")
    save_new(MANIFEST,{"source_hashes":{p:sha(HERE/p) for p in names},"proposal_sha256":sha(PROPOSAL),
        "scope":"posthoc_performance_profile_not_capability","maximum_wall_seconds":60})
    return {"status":"original_fit_profile_frozen"}


def run():
    verify()
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path,expected in manifest["source_hashes"].items():
        require(sha(HERE/path)==expected,"fonte/corpo mudou")
    require(not OUTPUT.exists(),"perfil existente")
    started=time.perf_counter()
    filters=ObservedFilters.load(HERE/"resultados/resolucao_breakout_learned_filters.npz")
    data=raw_inputs("breakout",149,"learned",filters)
    scale=np.load(HERE/"resultados/resolucao_breakout_learned_body_scale.npy",allow_pickle=False)
    before,after=scaled(data["source"],scale),scaled(data["successor"],scale)
    extraction=time.perf_counter()-started
    profiler=cProfile.Profile()
    begin=time.perf_counter()
    profiler.enable()
    model=FunctionTransport().fit(before,after,data["actions"],data["reward"],data["continuation"])
    profiler.disable()
    elapsed=time.perf_counter()-begin
    old=FunctionTransport.load(HERE/"resultados/resolucao_breakout_learned_body.npz")
    require(np.array_equal(model.support,old.support) and np.array_equal(model.reward,old.reward)
        and np.array_equal(model.transport,old.transport),"perfil mudou o problema")
    entries=[]
    for (filename,line,function),(primitive,total,own,cumulative,callers) in pstats.Stats(profiler).stats.items():
        entries.append({"file":filename,"line":line,"function":function,"calls":total,"own_seconds":own,"cumulative_seconds":cumulative})
    result={"status":"original_fit_profile_complete","extraction_wall_seconds":extraction,"profiled_fit_wall_seconds":elapsed,
        "top_cumulative":sorted(entries,key=lambda e:e["cumulative_seconds"],reverse=True)[:16],
        "top_own":sorted(entries,key=lambda e:e["own_seconds"],reverse=True)[:12],"total_wall_seconds":time.perf_counter()-started,
        "model_numerically_identical":True,"fit_calls":1,"physical_calls":0,"human_level_goal_achieved":False,"manifest_sha256":sha(MANIFEST)}
    save_new(OUTPUT,result)
    require(result["total_wall_seconds"]<60,"perfil excedeu o teto; resultado de custo preservado")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","run"))
    print(json.dumps({"prepare":prepare,"run":run}[parser.parse_args().mode](),ensure_ascii=False,indent=2))
