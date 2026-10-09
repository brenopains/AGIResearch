"""Auditoria física, origem de funções e previsões por composição adiante."""
import os
import platform
import sys
if sys.platform=="win32":
    platform.machine=platform._get_machine_win32
    platform.system=lambda:"Windows"
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable]="2"
import hashlib
import json
import numpy as np
from piloto_resolucao_compacto import (HERE,STEM,PROPOSAL,BODIES,RESULT,EVENTS,SENSORS,ARMS,FunctionTransport,GeneratedTransport,
    PARENT_BODIES,ObservedFilters,raw_bank,root_chunk,load_raw,raw_inputs,load_prior,scaled,innovation_samples,
    view,verify_bodies,require,save_new,sha)
from sensor_resolucao import encoded_windows
from auditar_semantica_prefixo import scores,equal_report
from auditar_funcoes_contraexemplos import forward


def coefficients(x,y):
    l=np.linalg.cholesky(x.T@x+.05*np.eye(x.shape[1]))
    return np.linalg.solve(l.T,np.linalg.solve(l,x.T@y))


def audit_bank(game,seed):
    bank=load_raw(game,seed)
    stem=raw_bank(game,seed)
    cert=json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    events=[json.loads(line) for line in stem.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    require(sum(e["ale_calls"] for e in events)==cert["ale_calls"]<=50000,"chamadas físicas divergem")
    require(cert["wall_seconds"]<120 and cert["maximum_sampled_rss_bytes"]<6*1024**3,"teto da coleta excedido")
    require(sum(e["event"]=="walk_checkpoint" for e in events)==64 and sum(e["event"]=="raw_replay_certified" for e in events)==8==cert["certified_raw_replays"],"caminhada/replays incompletos")
    rows=[e for e in events if e["event"]=="raw_program_measured"]
    require(len(rows)==32*len(bank["programs"]),"ramos ausentes")
    for i,root in enumerate(bank["roots"]):
        group=[r for r in rows if r["root"]==int(root)]
        require([r["program"] for r in group]==list(range(len(bank["programs"]))),"ramos duplicados/omitidos")
        with np.load(root_chunk(game,seed,int(root)),allow_pickle=False) as chunk:
            for j,row in enumerate(group):
                frames=chunk[f"frames_{j}"]
                require(frames.dtype==np.uint8 and frames.shape[1:]==(210,160) and len(frames)==16+row["ale_calls"],"trajetória inválida")
                ends=chunk["endpoints"][j]
                require(np.array_equal(ends,row["endpoints"]) and np.array_equal(chunk["programs"][j],row["word"]),"origem da janela diverge")
                reconstructed=encoded_windows(frames,ends,"coarse")
                require(np.array_equal(reconstructed,bank["histories"][i,j]),"história não corresponde à trajetória")
                require(np.array_equal(row["rewards"],bank["rewards"][i,j]) and np.array_equal(row["valid"],bank["valid"][i,j]),"evento/banco diverge")
    if seed!=179:
        prior=load_prior(game,seed)
        for key in ("histories","rewards","valid","roots","episode","programs"):
            require(np.array_equal(bank[key],prior[key]),"banco anterior não reproduzido")
    return cert["ale_calls"]


def audit():
    frozen=verify_bodies()
    require(frozen["wall_seconds"]<120 and frozen["sensor_fit_calls"]==10,"ajuste incompleto ou orçamento excedido")
    fitpath=STEM.with_name(STEM.name+"_ajuste.jsonl")
    fits=[json.loads(line) for line in fitpath.read_text(encoding="utf-8").splitlines()]
    require(len(fits)==10 and {(r["game"],r["sensor"]) for r in fits}=={(g,s) for g in ("breakout","pong") for s in SENSORS},"eventos de ajuste ausentes")
    for construction in frozen["construction"].values():
        require(all(r["rss_bytes_at_export"]<6*1024**3 for r in construction.values()),"memória amostrada excedida")
    result=json.loads(RESULT.read_text(encoding="utf-8"))
    events=[json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines()]
    require(result["events_sha256"]==sha(EVENTS) and result["bodies_sha256"]==sha(BODIES) and result["proposal_sha256"]==sha(PROPOSAL),"hash do resultado diverge")
    require(len(events)==result["event_count"]==2*2*3*len(ARMS)*64,"comparações incompletas")
    require(not result["human_level_goal_achieved"] and not result["unknown_world_transfer_demonstrated"] and result["fit_calls_during_evaluation"]==0,"escopo exagerado")
    parent=json.loads(PARENT_BODIES.read_text(encoding="utf-8"))
    physical,checked,maximum,parameter_error,statuses=0,0,0.,0.,[]
    for game in ("breakout","pong"):
        for seed in (149,151,179):
            physical+=audit_bank(game,seed)
        # Independent covariance eigen-equation/origin check; no reward enters.
        samples=innovation_samples(game).astype(float)
        samples-=samples.mean(1,keepdims=True)
        covariance=samples.T@samples/len(samples)
        filters=ObservedFilters.load(STEM.with_name(STEM.name+f"_{game}_learned_filters.npz"))
        for weight,eigenvalue in zip(filters.weights,filters.eigenvalues):
            require(np.max(np.abs(covariance@weight-eigenvalue*weight))<1e-8,"função não deriva da inovação observada")
        for sensor in SENSORS:
            entry=frozen["bodies"][game][sensor]
            filters=ObservedFilters.load(HERE/entry["filters"]) if entry["filters"] else None
            data=raw_inputs(game,149,sensor,filters)
            scale=np.load(HERE/entry["scale"],allow_pickle=False)
            require(np.array_equal(scale,np.maximum(data["source"].std(axis=(0,1,3)),.001)),"escala usou outros dados")
            before,after=scaled(data["source"],scale),scaled(data["successor"],scale)
            model=FunctionTransport.load(HERE/entry["body"])
            observed={hashlib.sha256(np.asarray(row,float).tobytes()).digest() for array in (before,after) for row in array.reshape(len(array),-1)}
            require(all(hashlib.sha256(row.tobytes()).digest() in observed for row in model.support),"suporte não deriva das observações")
            p,pn=model.features(before),model.features(after)
            for action in range(4):
                ids=data["actions"]==action
                target=np.column_stack((data["reward"][ids],data["continuation"][ids,None]*pn[ids]))
                coef=coefficients(p[ids],target)
                error=max(float(np.max(np.abs(coef[:,0]-model.reward[action]))),float(np.max(np.abs(coef[:,1:]-model.transport[action]))))
                parameter_error=max(parameter_error,error)
                require(error<1e-8,"corpo não reproduz equações normais")
            if sensor=="learned":
                shuffled=FunctionTransport.load(HERE/frozen["bodies"][game]["shuffled_outcomes"]["body"])
                rng=np.random.default_rng(9861)
                for action in range(4):
                    ids=np.flatnonzero(data["actions"]==action)
                    target=np.column_stack((data["reward"][ids],data["continuation"][ids,None]*pn[ids]))
                    target=target[rng.permutation(len(ids))]
                    coef=coefficients(p[ids],target)
                    require(np.allclose(coef[:,0],shuffled.reward[action],atol=1e-8) and np.allclose(coef[:,1:],shuffled.transport[action],atol=1e-8),"placebo diverge")
                context=FunctionTransport.load(HERE/frozen["bodies"][game]["no_context"]["body"])
                require(context.mode=="no_context" and np.array_equal(context.reward,model.reward) and np.array_equal(context.transport,model.transport),"ablação alterou corpo")
        bank=load_raw(game,179)
        models={}
        for arm,entry in frozen["bodies"][game].items():
            filters=ObservedFilters.load(HERE/entry["filters"]) if entry["filters"] else None
            initial=raw_inputs(game,179,entry["sensor"],filters,True)
            model=FunctionTransport.load(HERE/entry["body"])
            initial=scaled(initial,np.load(HERE/entry["scale"],allow_pickle=False))
            models[arm]=(model,initial)
        legacy=GeneratedTransport.load(HERE/parent["bodies"][game]["no_generated_functions"])
        for partition,offset in (("known",0),("new",4)):
            cell=result["games"][game][partition]
            for horizon in (4,8,16):
                cases=view(bank,offset,horizon)
                for arm in ARMS:
                    if arm=="zero":
                        prediction=np.zeros(64)
                    elif arm=="legacy":
                        phi=legacy.features(cases["history"])
                        prediction=forward(phi,legacy.reward,legacy.transport,cases["left"],cases["right"])
                    else:
                        model,initial=models[arm]
                        phi=model.features(np.repeat(initial,2,axis=0))
                        if arm=="no_context":
                            phi=np.broadcast_to(model.initial_prior,phi.shape)
                        prediction=forward(phi,model.reward,model.transport,cases["left"],cases["right"])
                    rows=[r for r in events if r["game"]==game and r["partition"]==partition and r["horizon"]==horizon and r["arm"]==arm]
                    require(len(rows)==64,"eventos duplicados/omitidos")
                    for i,(root,a,b) in enumerate(cases["bank_cases"]):
                        require((rows[i]["bank_row"],rows[i]["left"],rows[i]["right"])==(root,a,b) and rows[i]["actual_delta"]==float(cases["actual"][i]),"caso diverge")
                    error=float(np.max(np.abs(prediction-np.array([r["predicted_delta"] for r in rows]))))
                    maximum=max(maximum,error)
                    require(error<1e-8,"reexecução diverge")
                    equal_report(scores(cases["actual"],prediction),cell["curves"][str(horizon)][arm],"composição")
                    checked+=len(rows)
            values=cases["actual"].reshape(32,2)
            active=sorted(map(int,set(cases["episodes"][np.abs(cases["actual"])>1e-12])))
            opposite=[bool(np.any(values[:,i]<-1e-12) and np.any(values[:,i]>1e-12)) for i in range(2)]
            cov={"opposite_preferences_per_pair":opposite,"episodes_with_contrast":active,
                "non_tied_comparisons":int(np.count_nonzero(np.abs(cases["actual"])>1e-12)),"sufficient":bool(all(opposite) and len(active)>=3)}
            require(cov==cell["coverage"],"cobertura diverge")
            checks={"coverage":cov["sufficient"]}
            for arm in ("coarse","moments","random","shuffled_outcomes","no_context","legacy","zero"):
                for metric in ("mse","fair_tie_regret"):
                    checks[f"{metric}_vs_{arm}"]=bool(cell["curves"]["16"]["learned"][metric]<cell["curves"]["16"][arm][metric]-1e-12)
            require(checks==cell["gate_checks"],"portão diverge")
            status="inconclusive_coverage" if not cov["sufficient"] else "passed" if all(checks.values()) else "failed"
            require(status==cell["status"],"estado da célula diverge")
            statuses.append(status)
    verdict="spatial_capability_gate_passed_not_agi" if all(s=="passed" for s in statuses) else "spatial_capability_inconclusive_coverage" if "inconclusive_coverage" in statuses else "spatial_capability_gate_failed"
    require(verdict==result["status"],"veredito agregado diverge")
    output={"status":"spatial_function_audit_passed","physical_observations_including_previous_replays":physical,
        "certified_raw_replays":48,"held_comparisons_checked":checked,"maximum_prediction_error":maximum,
        "maximum_parameter_error":parameter_error,"result_sha256":sha(RESULT),"human_level_goal_achieved":False}
    save_new(STEM.with_name(STEM.name+"_auditoria.json"),output)
    return output


if __name__=="__main__":
    print(json.dumps(audit(),ensure_ascii=False,indent=2))
