"""Comparação pareada de custo sem usar desempenho de avaliação."""
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
from piloto_resolucao import HERE,STEM,PROPOSAL,ObservedFilters,FunctionTransport,raw_inputs,scaled,verify,require,save_new,sha
from transporte_compacto import CompactTransport
from coletar_prospectiva import rss_bytes

MANIFEST=STEM.with_name(STEM.name+"_compacto_manifesto.json")
OUTPUT=STEM.with_name(STEM.name+"_compacto.json")
EVENTS=STEM.with_name(STEM.name+"_compacto.jsonl")


def prepare():
    verify()
    require(not any(p.exists() for p in (MANIFEST,OUTPUT,EVENTS)),"comparação existente; preservar")
    paths=[HERE/"comparar_transporte_compacto.py",HERE/"transporte_compacto.py",HERE/"testes_transporte_compacto.py",PROPOSAL]
    for game in ("breakout","pong"):
        prefix=STEM.with_name(STEM.name+f"_{game}_learned")
        paths.extend(prefix.with_name(prefix.name+s) for s in ("_body.npz","_body_scale.npy","_filters.npz"))
    save_new(MANIFEST,{"scope":"paired_compute_cost_and_numeric_equivalence_not_capability","maximum_wall_seconds":90,
        "orders":[["original","compact"],["compact","original"]],"source_hashes":{p.relative_to(HERE).as_posix():sha(p) for p in paths}})
    return {"status":"paired_compact_benchmark_frozen"}


def run():
    verify()
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path,expected in manifest["source_hashes"].items():
        require(sha(HERE/path)==expected,"fonte/corpo mudou")
    require(not OUTPUT.exists() and not EVENTS.exists(),"comparação existente")
    started=time.perf_counter()
    records,reports=[],{}
    with EVENTS.open("x",encoding="utf-8") as stream:
        for game in ("breakout","pong"):
            prefix=STEM.with_name(STEM.name+f"_{game}_learned")
            filters=ObservedFilters.load(prefix.with_name(prefix.name+"_filters.npz"))
            begin=time.perf_counter()
            data=raw_inputs(game,149,"learned",filters)
            scale=np.load(prefix.with_name(prefix.name+"_body_scale.npy"),allow_pickle=False)
            before,after=scaled(data["source"],scale),scaled(data["successor"],scale)
            extraction=time.perf_counter()-begin
            old=FunctionTransport.load(prefix.with_name(prefix.name+"_body.npz"))
            samples=[]
            for pair,order in enumerate(manifest["orders"]):
                times={}
                for name in order:
                    require(time.perf_counter()-started<90,"teto da comparação; preservar eventos parciais")
                    begin=time.perf_counter()
                    cls=FunctionTransport if name=="original" else CompactTransport
                    body=cls().fit(before,after,data["actions"],data["reward"],data["continuation"])
                    elapsed=time.perf_counter()-begin
                    support_equal=np.array_equal(body.support,old.support)
                    coefficient_error=max(float(np.max(np.abs(body.reward-old.reward))),float(np.max(np.abs(body.transport-old.transport))))
                    prediction_error=float(np.max(np.abs(body.predict(before[:32],[[0,1,3,2]*4]*32)-old.predict(before[:32],[[0,1,3,2]*4]*32))))
                    record={"event":"fit_cost_measured","game":game,"pair":pair,"implementation":name,"wall_seconds":elapsed,
                        "same_supports":support_equal,"same_scale":body.scale==old.scale,"maximum_coefficient_error":coefficient_error,
                        "maximum_prediction_error":prediction_error,"rss_bytes_at_end":rss_bytes(),
                        "subtiming":getattr(body,"fit_timing",None)}
                    records.append(record)
                    stream.write(json.dumps(record,allow_nan=False)+"\n")
                    stream.flush()
                    require(support_equal and body.scale==old.scale and coefficient_error<1e-8 and prediction_error<1e-8,"problema numérico foi alterado")
                    require(record["rss_bytes_at_end"]<6*1024**3,"teto de memória observado")
                    times[name]=elapsed
                samples.append({"pair":pair,"original_seconds":times["original"],"compact_seconds":times["compact"],
                    "speed_ratio":times["original"]/times["compact"]})
            reports[game]={"extraction_seconds":extraction,"pairs":samples,"all_pairs_compact_faster":all(p["compact_seconds"]<p["original_seconds"] for p in samples)}
    result={"status":"paired_compact_benchmark_complete","games":reports,"fit_calls":len(records),"physical_calls":0,
        "wall_seconds":time.perf_counter()-started,"events_sha256":sha(EVENTS),"manifest_sha256":sha(MANIFEST),
        "all_numeric_instances_preserved":True,"human_level_goal_achieved":False}
    save_new(OUTPUT,result)
    require(result["wall_seconds"]<90,"comparação excedeu orçamento; custos preservados")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","run"))
    print(json.dumps({"prepare":prepare,"run":run}[parser.parse_args().mode](),ensure_ascii=False,indent=2))
