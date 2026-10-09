"""Limites necessários das funções positivas e atribuição de erro, sem reajuste."""
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
from piloto_resolucao_compacto import (HERE,STEM as PARENT_STEM,RESULT as PARENT_RESULT,BODIES,PROPOSAL,EVENTS as PARENT_EVENTS,
    FRAME_IDS,FunctionTransport,ObservedFilters,raw_bank,root_chunk,load_raw,encode,scaled,verify_bodies,require,save_new,sha)
from decompor_erros_transporte import trajectory
from diagnosticar_semantica_prefixo import metrics
from coletar_prospectiva import rss_bytes

STEM=HERE/"resultados"/"resolucao_realizabilidade"
MANIFEST=STEM.with_name(STEM.name+"_manifesto.json")
RESULT,EVENTS=STEM.with_suffix(".json"),STEM.with_suffix(".jsonl")
SENSORS=("coarse","moments","learned","random","transferred")
ARMS=(*SENSORS,"shuffled_outcomes","no_context")
TOL=1e-8


def bounds(state,previous_mass=None):
    state=np.asarray(state,float)
    require(state.ndim==1 and len(state)>1 and np.isfinite(state).all(),"estado incompatível")
    mass=float(state[0])
    violations={"negative_mass":max(-mass,0.),"mass_above_one":max(mass-1.,0.),
        "negative_positive_feature":max(-float(state[1:].min()),0.),
        "feature_above_alive_mass":max(float(state[1:].max())-mass,0.),
        "reviving_absorbed_mass":max(mass-previous_mass,0.) if previous_mass is not None else 0.}
    return {"mass":mass,"maximum_necessary_bound_violation":max(violations.values()),
        "outside_necessary_bounds":any(v>TOL for v in violations.values()),"violations":violations}


def rollout(body,initial,word):
    state=np.asarray(initial,float).copy()
    if body.mode=="no_context":
        state=body.initial_prior.copy()
    out=[bounds(state)]
    for action in word:
        old=float(state[0])
        state=state@body.transport[action]
        out.append(bounds(state,old))
    return out


def full_features(game,sensor,entry,bank):
    body=FunctionTransport.load(HERE/entry["body"])
    filters=ObservedFilters.load(HERE/entry["filters"]) if entry["filters"] else None
    scale=np.load(HERE/entry["scale"],allow_pickle=False)
    result=np.empty((32,8,17,len(body.initial_prior)))
    for i,root in enumerate(bank["roots"]):
        with np.load(root_chunk(game,179,int(root)),allow_pickle=False) as chunk:
            for j in range(8):
                z=encode(chunk[f"frames_{j}"],sensor,filters)
                ends=chunk["endpoints"][j]
                history=np.stack([z[int(end)-16+FRAME_IDS] for end in ends])
                result[i,j]=body.features(scaled(history,scale))
    require(np.isfinite(result).all() and np.all(result[...,0]==1),"funções observadas inválidas")
    return body,result


def prepare():
    verify_bodies()
    require(not any(p.exists() for p in (MANIFEST,RESULT,EVENTS)),"diagnóstico existente; preservar")
    sources=["diagnosticar_realizabilidade.py","testes_realizabilidade.py","REALIZABILIDADE.md","decompor_erros_transporte.py",
        "resultados/resolucao_compacta_auditoria.json",BODIES.relative_to(HERE).as_posix(),PROPOSAL.relative_to(HERE).as_posix(),
        PARENT_RESULT.relative_to(HERE).as_posix(),PARENT_EVENTS.relative_to(HERE).as_posix()]
    for game in ("breakout","pong"):
        sources.append(raw_bank(game,179).with_suffix(".json").relative_to(HERE).as_posix())
    save_new(MANIFEST,{"status":"post_evaluation_attribution_and_necessary_bounds_frozen","arms":list(ARMS),"tolerance":TOL,
        "maximum_wall_seconds":180,"maximum_sampled_rss_bytes":6*1024**3,"fit_calls":0,"ale_calls":0,
        "source_hashes":{p:sha(HERE/p) for p in sources},"does_not_establish_unique_cause":True})
    return {"status":"diagnostic_frozen","sha256":sha(MANIFEST)}


def run():
    frozen=verify_bodies()
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name,expected in manifest["source_hashes"].items():
        require(sha(HERE/name)==expected,"fonte mudou: "+name)
    require(not RESULT.exists() and not EVENTS.exists(),"saídas existentes; preservar")
    prior=json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
    original=[json.loads(line) for line in PARENT_EVENTS.read_text(encoding="utf-8").splitlines()]
    original={(r["game"],r["partition"],r["horizon"],r["arm"],r["bank_row"],r["left"],r["right"]):r for r in original}
    require(len(original)==6912,"identidades originais duplicadas")
    started=time.perf_counter()
    reports,records,peak,maximum_error,checked={},[],0,0.,0
    for game in ("breakout","pong"):
        bank=load_raw(game,179)
        reports[game]={}
        for sensor in SENSORS:
            require(time.perf_counter()-started<180,"teto do diagnóstico; inconclusivo")
            body,phi=full_features(game,sensor,frozen["bodies"][game][sensor],bank)
            arms=[sensor]+(["shuffled_outcomes","no_context"] if sensor=="learned" else [])
            for arm in arms:
                model=body if arm==sensor else FunctionTransport.load(HERE/frozen["bodies"][game][arm]["body"])
                for partition,offset in (("known",0),("new",4)):
                    curves={}
                    planned={}
                    for i in range(32):
                        for side in range(offset,offset+4):
                            observations=phi[i,side]*bank["valid"][i,side,:,None]
                            for t,state in enumerate(observations):
                                control=bounds(state,float(observations[t-1,0]) if t else None)
                                require(not control["outside_necessary_bounds"],"controle real fisicamente impossível")
                            planned[i,side]=rollout(model,phi[i,side,0],bank["programs"][side].tolist())
                    for horizon in (4,8,16):
                        rows=[]
                        invalid,maxv,state_count=0,0.,0
                        for i in range(32):
                            for left in (offset,offset+2):
                                branches=[]
                                for side in (left,left+1):
                                    word=bank["programs"][side,:horizon]
                                    result=trajectory(model,phi[i,side,:horizon+1],word,
                                        bank["rewards"][i,side,:horizon],bank["valid"][i,side,:horizon+1])
                                    maximum_error=max(maximum_error,result["identity_error"])
                                    states=planned[i,side][1:horizon+1]
                                    invalid+=sum(s["outside_necessary_bounds"] for s in states)
                                    maxv=max(maxv,max(s["maximum_necessary_bound_violation"] for s in states))
                                    state_count+=len(states)
                                    branches.append(result)
                                delta={field:branches[0][field]-branches[1][field] for field in
                                    ("actual","predicted","reward_error","transport_error","boundary_error","real_readout")}
                                key=(game,partition,horizon,arm,i,left,left+1)
                                require(key in original and abs(delta["predicted"]-original[key]["predicted_delta"])<TOL
                                    and delta["actual"]==original[key]["actual_delta"],"previsão original diverge")
                                checked+=1
                                row={"event":"spatial_error_attribution","game":game,"partition":partition,"horizon":horizon,"arm":arm,
                                    "bank_row":i,"left":left,"right":left+1,**delta}
                                records.append(row);rows.append(row)
                        actual=np.array([r["actual"] for r in rows])
                        r,t,b=(np.array([v[name] for v in rows]) for name in ("reward_error","transport_error","boundary_error"))
                        queries={"composed":metrics(actual,[v["predicted"] for v in rows]),
                            "real_past_histories_before_each_block_oracle":metrics(actual,[v["real_readout"] for v in rows]),
                            "reward_error_removed_oracle":metrics(actual,[v["predicted"]-v["reward_error"] for v in rows])}
                        terms={"reward_squared":float(np.mean(r*r)),"transport_squared":float(np.mean(t*t)),"boundary_squared":float(np.mean(b*b)),
                            "reward_transport_cross":float(np.mean(2*r*t)),"reward_boundary_cross":float(np.mean(2*r*b)),"transport_boundary_cross":float(np.mean(2*t*b))}
                        require(abs(sum(terms.values())-queries["composed"]["mse"])<TOL,"parcelas não reconciliam")
                        curves[str(horizon)]={"queries":queries,"mse_decomposition":terms,"necessary_bounds":{
                            "states_checked":state_count,"outside_bounds":invalid,"maximum_violation":maxv,"real_observed_control_violations":0}}
                    reports[game].setdefault(arm,{})[partition]=curves
            peak=max(peak,rss_bytes())
            require(peak<6*1024**3,"teto de memória amostrada excedido")
            print(json.dumps({"event":"sensor_diagnostic_finished","game":game,"sensor":sensor,"wall_seconds":time.perf_counter()-started}),flush=True)
        del bank
    elapsed=time.perf_counter()-started
    require(elapsed<180 and checked==2*2*3*len(ARMS)*64,"orçamento/casos incompletos")
    with EVENTS.open("x",encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row,allow_nan=False)+"\n")
    save_new(RESULT,{"status":"spatial_realizability_and_attribution_complete_no_promotion","games":reports,
        "wall_seconds":elapsed,"maximum_sampled_rss_bytes":peak,"comparisons_reconciled":checked,"maximum_telescopic_error":maximum_error,
        "previous_gate_preserved":prior["status"],"fit_calls":0,"ale_calls":0,"manifest_sha256":sha(MANIFEST),"events_sha256":sha(EVENTS),
        "human_level_goal_achieved":False,"bounds_necessary_not_sufficient":True})
    return {"status":"diagnostic_complete","wall_seconds":elapsed,"comparisons":checked,"primary16":{
        g:{p:r["learned"][p]["16"] for p in ("known","new")} for g,r in reports.items()}}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","run"))
    mode=parser.parse_args().mode
    print(json.dumps(prepare() if mode=="prepare" else run(),ensure_ascii=False,indent=2))
