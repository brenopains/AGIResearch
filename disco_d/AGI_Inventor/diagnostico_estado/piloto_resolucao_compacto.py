"""Aprender filtros locais de experiência preservada e testar antecipação composta."""
import os
import platform
import sys
if sys.platform=="win32":
    platform.machine=platform._get_machine_win32
    platform.system=lambda:"Windows"
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable]="2"
import argparse
import copy
import json
import time
import numpy as np
from piloto_prospectivo import (HERE,PROPOSAL as PARENT_PROPOSAL,BODIES as PARENT_BODIES,
    GeneratedTransport,load_bank as load_prior,transitions,view,coverage,verify_bodies as verify_parent,require,save_new,sha)
from transporte_compacto import CompactTransport as FunctionTransport
from piloto_resolucao import raw_bank as legacy_raw_bank, load_raw as legacy_load_raw, raw_inputs as legacy_raw_inputs, verify as verify_legacy
from coletar_prospectiva import rss_bytes
from filtros_observados import ObservedFilters,patches
from sensor_resolucao import coarse,moments
from diagnosticar_semantica_prefixo import metrics

STEM=HERE/"resultados"/"resolucao_compacta"
PROPOSAL=STEM.with_name(STEM.name+"_proposta.json")
BODIES=STEM.with_name(STEM.name+"_corpos_manifesto.json")
RESULT,EVENTS=STEM.with_suffix(".json"),STEM.with_suffix(".jsonl")
SENSORS=("coarse","moments","learned","random","transferred")
ARMS=(*SENSORS,"shuffled_outcomes","no_context","legacy","zero")
FRAME_IDS=np.array([0,2,4,6,8,10,12,15])


def raw_bank(game,seed):
    if seed in (149,151):
        return legacy_raw_bank(game,seed)
    return STEM.with_name(STEM.name+f"_{game}_{seed}")


def prepare():
    verify_legacy()
    require(not any(p.exists() for p in (PROPOSAL,BODIES,RESULT,EVENTS)),"rodada existente; preservar")
    for game in ("breakout","pong"):
        for seed in (179,):
            require(not list((HERE/"resultados").glob(raw_bank(game,seed).name+"*")),"coleta existente; preservar")
    parent=json.loads(PARENT_PROPOSAL.read_text(encoding="utf-8"))
    sources=("piloto_resolucao_compacto.py","coletar_resolucao_compacto.py","auditar_resolucao_compacto.py","sensor_resolucao.py","filtros_observados.py",
        "testes_sensor_resolucao.py","testes_filtros_observados.py","testes_transporte_compacto.py","transporte_compacto.py",
        "RESOLUCAO_COMPACTA.md","piloto_resolucao.py","resultados/resolucao_proposta.json",
        "resultados/resolucao_ajuste_interrupcao.json","resultados/resolucao_compacto.json","coleta.py","transporte_funcoes.py","coletar_prospectiva.py",
        "auditar_semantica_prefixo.py",PARENT_BODIES.relative_to(HERE).as_posix(),"resultados/eventos_fechamento.json")
    save_new(PROPOSAL,{"status":"new_compute_round_frozen_before_fit_and_fresh_evaluation","programs":parent["programs"],"training_seed":149,
        "development_seed_previously_seen":151,"evaluation_seed":179,"sensors":list(SENSORS),"arms":list(ARMS),"primary":"learned",
        "feature_frames":FRAME_IDS.tolist(),"local_patch_shape":[10,10],"learned_filter_count":5,"filter_samples_per_root_program":64,
        "supports":32,"alpha":.05,"maximum_calls_per_bank":50000,"maximum_collection_wall_seconds":120,
        "maximum_fit_wall_seconds":120,"maximum_worker_rss_bytes":6*1024**3,"known_games_not_unseen_worlds":True,"training_raw_banks_reused_without_new_collection":True,
        "previous_fit_incomplete_and_costs_preserved":True,
        "gate":"lower_mse_and_fair_regret_than_coarse_moments_random_shuffled_no_context_legacy_zero_in_four_cells_at_16",
        "source_hashes":{p:sha(HERE/p) for p in sources},"human_level_goal_achieved":False})
    return {"status":"raw_resolution_protocol_frozen","sha256":sha(PROPOSAL)}


def verify():
    value=json.loads(PROPOSAL.read_text(encoding="utf-8"))
    for path,expected in value["source_hashes"].items():
        require(sha(HERE/path)==expected,"fonte mudou: "+path)
    verify_legacy()
    return value


def verify_bodies():
    verify()
    value=json.loads(BODIES.read_text(encoding="utf-8"))
    for path,expected in value["source_hashes"].items():
        require(sha(HERE/path)==expected,"corpo/dado mudou: "+path)
    return value


def load_raw(game,seed):
    verify()
    if seed in (149,151):
        return legacy_load_raw(game,seed)
    stem=raw_bank(game,seed)
    certificate=json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    require(certificate["status"]=="lossless_sensor_bank_complete" and certificate["proposal_sha256"]==sha(PROPOSAL)
        and certificate["bank_sha256"]==sha(stem.with_suffix(".npz")) and certificate["events_sha256"]==sha(stem.with_suffix(".jsonl")),"banco não certificado")
    for path,expected in certificate["chunks_sha256"].items():
        require(sha(HERE/path)==expected,"trajetória mudou")
    if seed==179:
        require(certificate["bodies_sha256_before_collection"]==sha(BODIES),"avaliação precedeu congelamento")
    with np.load(stem.with_suffix(".npz"),allow_pickle=False) as data:
        bank={k:data[k].copy() for k in data.files}
    p=8 if seed==179 else 4
    require(bank["histories"].shape==(32,p,17,16,336) and bank["rewards"].shape==(32,p,16) and bank["valid"].shape==(32,p,17),"dimensões inválidas")
    require(not np.any((~bank["valid"][...,:-1])&bank["valid"][...,1:]) and not np.any(bank["rewards"][~bank["valid"][...,:-1]]),"terminal revive")
    require(np.array_equal(bank["programs"],json.loads(PROPOSAL.read_text(encoding="utf-8"))["programs"][:p]),"programas divergem")
    return bank


def root_chunk(game,seed,root):
    stem=raw_bank(game,seed)
    return stem.with_name(stem.name+f"_root_{root:02d}.npz")


def innovation_samples(game):
    rng=np.random.default_rng(19173)
    rows=[]
    for root in [8*e+p for e in range(8) for p in (0,2,4,6)]:
        with np.load(root_chunk(game,149,root),allow_pickle=False) as data:
            for j in range(4):
                frames=data[f"frames_{j}"]
                x=patches(frames)
                for t in np.linspace(1,len(x)-1,8,dtype=int):
                    cells=rng.integers(0,336,8)
                    rows.extend(x[t,cells]-x[t-1,cells])
    return np.asarray(rows)


def encode(frames,sensor,filters=None):
    if sensor=="coarse":
        return np.repeat(coarse(frames)[:,None,:],6,axis=1)
    if sensor=="moments":
        return moments(frames)
    return filters.encode(frames)


def raw_inputs(game,seed,sensor,filters=None,initial_only=False):
    if seed in (149,151):
        return legacy_raw_inputs(game,seed,sensor,filters,initial_only)
    bank=load_raw(game,seed)
    source,successor,initial=[],[],[]
    for i,root in enumerate(bank["roots"]):
        with np.load(root_chunk(game,seed,int(root)),allow_pickle=False) as chunk:
            for j in range(len(bank["programs"])):
                if initial_only and j>0:
                    continue
                frames=chunk[f"frames_{j}"]
                z=encode(frames[:16] if initial_only else frames,sensor,filters)
                ends=chunk["endpoints"][j]
                if j==0:
                    initial.append(z[FRAME_IDS])
                if not initial_only:
                    h=np.stack([z[int(end)-16+FRAME_IDS] for end in ends])
                    for t in np.flatnonzero(bank["valid"][i,j,:-1]):
                        source.append(h[t]);successor.append(h[t+1])
    if initial_only:
        return np.stack(initial)
    data=transitions(bank)
    data["source"],data["successor"]=np.stack(source),np.stack(successor)
    return data


def scaled(x,scale):
    return np.asarray(x/scale[None,None,:,None],np.float32)


def fit():
    proposal=verify()
    require(not BODIES.exists() and not any(raw_bank(g,179).with_suffix(".json").exists() for g in ("breakout","pong")),"ajuste existe ou avaliação precedeu ajuste")
    started=time.perf_counter()
    paths=[PROPOSAL]
    records=[]
    checkpoint=STEM.with_name(STEM.name+"_ajuste_checkpoint.json")
    checkpoint.write_text(json.dumps({"status":"partial_fitting_not_eligible","completed_sensor_models":0,
        "elapsed_wall_seconds":0.,"records":[]},allow_nan=False),encoding="utf-8")
    filters={}
    for game in ("breakout","pong"):
        load_raw(game,149)
        samples=innovation_samples(game)
        require(len(samples)==8192,"amostragem mudou")
        filters[game]={"learned":ObservedFilters.fit(samples),"random":ObservedFilters.random()}
        for kind,body in filters[game].items():
            path=STEM.with_name(STEM.name+f"_{game}_{kind}_filters.npz")
            body.save(path,{"training_seed":149,"construction_game":game,"samples":8192,"no_reward_used":True})
            paths.extend((path,path.with_suffix(".json")))
        certificate=raw_bank(game,149).with_suffix(".json")
        paths.extend((certificate,raw_bank(game,149).with_suffix(".npz")))
    models,reports={},{}
    for game in ("breakout","pong"):
        models[game],reports[game]={},{}
        for sensor in SENSORS:
            require(time.perf_counter()-started<120,"teto de ajuste; inconclusivo")
            owner=("pong" if game=="breakout" else "breakout") if sensor=="transferred" else game
            filter_kind="random" if sensor=="random" else "learned"
            functions=filters[owner][filter_kind] if sensor not in ("coarse","moments") else None
            input_started=time.perf_counter()
            data=raw_inputs(game,149,sensor,functions)
            input_seconds=time.perf_counter()-input_started
            scale=np.maximum(data["source"].std(axis=(0,1,3)),.001)
            before,after=scaled(data["source"],scale),scaled(data["successor"],scale)
            model=FunctionTransport().fit(before,after,data["actions"],data["reward"],data["continuation"])
            path=STEM.with_name(STEM.name+f"_{game}_{sensor}_body.npz")
            model.save(path)
            scale_path=path.with_name(path.stem+"_scale.npy")
            require(not scale_path.exists(),"escala existente")
            np.save(scale_path,scale,allow_pickle=False)
            entry={"body":path.relative_to(HERE).as_posix(),"scale":scale_path.relative_to(HERE).as_posix(),"sensor":sensor,
                "filters":STEM.with_name(STEM.name+f"_{owner}_{filter_kind}_filters.npz").relative_to(HERE).as_posix() if functions else None}
            models[game][sensor]=entry
            paths.extend((path,path.with_suffix(".json"),scale_path))
            p,pn=model.fit_phi,model.fit_next_phi
            predicted=np.einsum("ij,ij->i",p,model.reward[data["actions"]])
            reports[game][sensor]={"training_transitions":len(p),"support_count":len(model.support),"channel_scale":scale.tolist(),
                "training_reward_mse":float(np.mean((predicted-data["reward"])**2)),"filters_origin_game":owner if functions else None,
                "input_wall_seconds":input_seconds,"fit_timing":model.fit_timing,"rss_bytes_at_export":rss_bytes()}
            require(rss_bytes()<6*1024**3,"limite amostrado de memória")
            records.append({"event":"spatial_sensor_model_fitted","game":game,"sensor":sensor,"parameters":entry["body"],
                "input_wall_seconds":input_seconds,"fit_timing":model.fit_timing,"wall_seconds_since_fit_start":time.perf_counter()-started})
            checkpoint.write_text(json.dumps({"status":"partial_fitting_not_eligible","completed_sensor_models":len(records),
                "elapsed_wall_seconds":time.perf_counter()-started,"records":records},allow_nan=False),encoding="utf-8")
            if sensor=="learned":
                shuffled=copy.deepcopy(model)
                rng=np.random.default_rng(9861)
                for action in range(4):
                    ids=np.flatnonzero(data["actions"]==action)
                    x=p[ids]
                    y=np.column_stack((data["reward"][ids],data["continuation"][ids,None]*pn[ids]))
                    y=y[rng.permutation(len(ids))]
                    coefficient=np.linalg.solve(x.T@x+.05*np.eye(x.shape[1]),x.T@y)
                    shuffled.reward[action],shuffled.transport[action]=coefficient[:,0],coefficient[:,1:]
                    shuffled.local_residual_mse[action]=float(np.mean((x@coefficient-y)**2))
                for arm,body in (("shuffled_outcomes",shuffled),("no_context",copy.deepcopy(model))):
                    if arm=="no_context":
                        body.mode="no_context"
                    pp=STEM.with_name(STEM.name+f"_{game}_{arm}_body.npz")
                    body.save(pp)
                    models[game][arm]={**entry,"body":pp.relative_to(HERE).as_posix()}
                    paths.extend((pp,pp.with_suffix(".json")))
        require(time.perf_counter()-started<120,"teto de ajuste; inconclusivo")
    fit_events=STEM.with_name(STEM.name+"_ajuste.jsonl")
    with fit_events.open("x",encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record,allow_nan=False)+"\n")
    paths.append(fit_events)
    save_new(BODIES,{"status":"spatial_bodies_frozen_before_new_evaluation","bodies":models,"construction":reports,
        "wall_seconds":time.perf_counter()-started,"sensor_fit_calls":len(records),"source_hashes":{p.relative_to(HERE).as_posix():sha(p) for p in paths}})
    return {"status":"spatial_bodies_frozen","wall_seconds":time.perf_counter()-started,"construction":reports}


def evaluate(seed):
    frozen=verify_bodies()
    output=RESULT if seed==179 else STEM.with_name(STEM.name+"_desenvolvimento.json")
    eventpath=EVENTS if seed==179 else STEM.with_name(STEM.name+"_desenvolvimento.jsonl")
    require(not output.exists() and not eventpath.exists(),"avaliação existente; preservar")
    parent=json.loads(PARENT_BODIES.read_text(encoding="utf-8"))
    reports,records,statuses={},[],[]
    for game in ("breakout","pong"):
        bank=load_raw(game,seed)
        models={}
        for arm,entry in frozen["bodies"][game].items():
            functions=ObservedFilters.load(HERE/entry["filters"]) if entry["filters"] else None
            initial=raw_inputs(game,seed,entry["sensor"],functions,True)
            scale=np.load(HERE/entry["scale"],allow_pickle=False)
            model=FunctionTransport.load(HERE/entry["body"])
            models[arm]=(model,scaled(initial,scale))
        legacy=GeneratedTransport.load(HERE/parent["bodies"][game]["no_generated_functions"])
        reports[game]={}
        for partition,offset in (("known",0),("new",4)) if seed==179 else (("known",0),):
            curves={}
            for horizon in (4,8,16):
                cases=view(bank,offset,horizon)
                curves[str(horizon)]={}
                for arm in ARMS:
                    if arm=="zero":
                        prediction=np.zeros(64)
                    elif arm=="legacy":
                        prediction=legacy.comparison(cases["history"],cases["left"],cases["right"])
                    else:
                        model,initial=models[arm]
                        prediction=model.comparison(np.repeat(initial,2,axis=0),cases["left"],cases["right"])
                    require(np.isfinite(prediction).all(),"previsão não finita")
                    curves[str(horizon)][arm]=metrics(cases["actual"],prediction)
                    for i,(row,a,b) in enumerate(cases["bank_cases"]):
                        records.append({"event":"spatial_initial_history_decision","game":game,"partition":partition,"horizon":horizon,"arm":arm,
                            "bank_row":row,"left":a,"right":b,"actual_delta":float(cases["actual"][i]),"predicted_delta":float(prediction[i])})
            cov=coverage(cases["actual"],cases["episodes"])
            checks={"coverage":cov["sufficient"]}
            for arm in ("coarse","moments","random","shuffled_outcomes","no_context","legacy","zero"):
                for metric in ("mse","fair_tie_regret"):
                    checks[f"{metric}_vs_{arm}"]=bool(curves["16"]["learned"][metric]<curves["16"][arm][metric]-1e-12)
            status="inconclusive_coverage" if not cov["sufficient"] else "passed" if all(checks.values()) else "failed"
            reports[game][partition]={"coverage":cov,"curves":curves,"gate_checks":checks,"status":status}
            statuses.append(status)
    with eventpath.open("x",encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row,allow_nan=False)+"\n")
    status="spatial_capability_gate_passed_not_agi" if all(s=="passed" for s in statuses) else "spatial_capability_inconclusive_coverage" if "inconclusive_coverage" in statuses else "spatial_capability_gate_failed"
    save_new(output,{"status":status,"seed":seed,"development_only":seed!=179,"games":reports,"events_sha256":sha(eventpath),
        "event_count":len(records),"bodies_sha256":sha(BODIES),"proposal_sha256":sha(PROPOSAL),"fit_calls_during_evaluation":0,
        "human_level_goal_achieved":False,"unknown_world_transfer_demonstrated":False,"same_filter_body_reused_across_known_games":True})
    return {"status":status,"seed":seed,"primary16":{g:{p:r["curves"]["16"] for p,r in v.items()} for g,v in reports.items()}}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","fit","development","evaluate"))
    mode=parser.parse_args().mode
    command_started=time.perf_counter()
    try:
        outcome={"prepare":prepare,"fit":fit,"development":lambda:evaluate(151),"evaluate":lambda:evaluate(179)}[mode]()
    except Exception as error:
        if mode=="fit":
            checkpoint=STEM.with_name(STEM.name+"_ajuste_checkpoint.json")
            state=json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
            partial=sorted((HERE/"resultados").glob(STEM.name+"_*_body.npz"))
            failure={"status":"inconclusive_fitting_interrupted","error":type(error).__name__+": "+str(error),
                "command_wall_seconds_including_verification":time.perf_counter()-command_started,
                "last_fit_checkpoint":state,"partial_body_hashes":{p.relative_to(HERE).as_posix():sha(p) for p in partial},
                "body_manifest_exists":BODIES.exists(),"capability_gate_not_evaluated":True,"human_level_goal_achieved":False}
            save_new(STEM.with_name(STEM.name+"_ajuste_interrupcao.json"),failure)
        raise
    print(json.dumps(outcome,ensure_ascii=False,indent=2))
