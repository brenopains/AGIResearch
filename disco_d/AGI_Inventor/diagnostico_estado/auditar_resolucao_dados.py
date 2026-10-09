"""Auditar dados e colisões observadas antes de qualquer veredito de capacidade."""
import os
import platform
import sys
if sys.platform=="win32":
    platform.machine=platform._get_machine_win32
    platform.system=lambda:"Windows"
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable]="2"
import argparse
import hashlib
import json
import time
import numpy as np
from piloto_resolucao import HERE,STEM,PROPOSAL,BODIES,raw_bank,root_chunk,load_raw,verify,require,save_new,sha
from auditar_resolucao import audit_bank
from coletar_prospectiva import rss_bytes

MANIFEST=STEM.with_name(STEM.name+"_dados_manifesto.json")
OUTPUT=STEM.with_name(STEM.name+"_dados_auditoria.json")


def prepare():
    verify()
    require(not MANIFEST.exists() and not OUTPUT.exists(),"auditoria existente")
    sources=[HERE/"auditar_resolucao_dados.py",HERE/"auditar_resolucao.py",PROPOSAL]
    for game in ("breakout","pong"):
        for seed in (149,151):
            sources.extend(raw_bank(game,seed).with_suffix(s) for s in (".json",".npz",".jsonl"))
    save_new(MANIFEST,{"status":"posthoc_data_audit_frozen_before_collision_search","source_hashes":{p.relative_to(HERE).as_posix():sha(p) for p in sources},
        "scope":"observed_finite_data_not_general_state_sufficiency","fit_calls":0,"physical_calls":0,"main_fit_incomplete":not BODIES.exists()})
    return {"status":"raw_data_audit_frozen","sha256":sha(MANIFEST)}


def collisions(game,seed):
    bank=load_raw(game,seed)
    groups={}
    count=0
    for i,root in enumerate(bank["roots"]):
        with np.load(root_chunk(game,seed,int(root)),allow_pickle=False) as chunk:
            for j,word in enumerate(bank["programs"]):
                frames=chunk[f"frames_{j}"]
                for t in np.flatnonzero(bank["valid"][i,j,:-1]):
                    count+=1
                    reduced=bank["histories"][i,j,t]
                    key=(int(word[t]),hashlib.sha256(reduced.tobytes()).digest())
                    end=int(chunk["endpoints"][j,t])
                    full=frames[end-16:end]
                    rawkey=hashlib.sha256(full.tobytes()).digest()
                    reward=float(bank["rewards"][i,j,t])
                    if key not in groups:
                        groups[key]={"reduced":reduced.copy(),"raw":{},"cases":[],"reward":[]}
                    group=groups[key]
                    require(np.array_equal(group["reduced"],reduced),"hash igual com dados reduzidos distintos")
                    if rawkey in group["raw"]:
                        require(np.array_equal(group["raw"][rawkey]["pixels"],full),"hash raw igual com dados distintos")
                    else:
                        group["raw"][rawkey]={"pixels":full.copy(),"reward":[]}
                    group["raw"][rawkey]["reward"].append(reward)
                    group["reward"].append(reward)
                    group["cases"].append({"root":int(root),"program":j,"block":int(t),"reward":reward})
    reduced_ss,raw_ss=0.,0.
    aliases,conflicts,raw_conflicts,witnesses=0,0,0,[]
    for group in groups.values():
        y=np.asarray(group["reward"])
        reduced_ss+=float(((y-y.mean())**2).sum())
        aliases+=len(group["raw"])>1
        for raw in group["raw"].values():
            yy=np.asarray(raw["reward"])
            raw_ss+=float(((yy-yy.mean())**2).sum())
            raw_conflicts+=len(set(raw["reward"]))>1
        if len(set(group["reward"]))>1:
            conflicts+=1
            if len(witnesses)<8:
                witnesses.append({"raw_history_variants":len(group["raw"]),"cases":group["cases"][:12]})
    return {"valid_transitions":count,"coarse_history_action_groups":len(groups),"groups_with_distinct_raw_histories":aliases,
        "coarse_groups_with_conflicting_rewards":conflicts,"raw_groups_with_conflicting_rewards":raw_conflicts,
        "finite_empirical_mse_lower_bound_coarse_key":reduced_ss/count,"finite_empirical_mse_lower_bound_raw_key":raw_ss/count,
        "first_reward_conflicts":witnesses,"absence_of_conflict_does_not_prove_state_sufficient":True,
        "observed_rss_bytes_at_end":rss_bytes()}


def run():
    verify()
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path,expected in manifest["source_hashes"].items():
        require(sha(HERE/path)==expected,"fonte/banco mudou")
    require(not OUTPUT.exists(),"auditoria existente")
    reports,physical={},0
    started=time.perf_counter()
    for game in ("breakout","pong"):
        for seed in (149,151):
            physical+=audit_bank(game,seed)
            reports[f"{game}_{seed}"]=collisions(game,seed)
    result={"status":"raw_sensor_data_audit_complete","banks":reports,"accounted_replay_collection_ale_calls":physical,
        "certified_raw_replays":32,"physical_calls_during_audit":0,"fit_calls_during_audit":0,"wall_seconds":time.perf_counter()-started,
        "human_level_goal_achieved":False,"main_pilot_incomplete":not BODIES.exists(),"manifest_sha256":sha(MANIFEST)}
    save_new(OUTPUT,result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","run"))
    print(json.dumps({"prepare":prepare,"run":run}[parser.parse_args().mode](),ensure_ascii=False,indent=2))
