"""Preservar trajetórias lossless e verificar replay dos bancos reduzidos."""
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
from collections import deque
import numpy as np
from coleta import gray,make_ale
from coletar_prospectiva import Budget
from sensor_resolucao import encoded_windows
from piloto_resolucao_compacto import HERE,PROPOSAL,BODIES,raw_bank,verify,verify_bodies,require,sha,save_new,load_prior


def play_raw(ale,actions,word,history,budget):
    frames=[f.copy() for f in history]
    ends,rewards,valid=[16],[],[True]
    stopped,used=False,0
    for symbol in word:
        reward=0.
        for _ in range(8):
            if stopped:
                break
            reward+=budget.act(ale,actions[int(symbol)])
            used+=1
            frames.append(gray(ale))
            stopped=bool(ale.game_over())
        ends.append(len(frames))
        rewards.append(reward)
        valid.append(not stopped)
    return np.stack(frames),np.array(ends),np.array(rewards),np.array(valid),used


def collect(game,seed):
    proposal=verify()
    require(seed in (149,151,179),"semente fora do protocolo")
    output=raw_bank(game,seed)
    require(not list((HERE/"resultados").glob(output.name+"*")),"coleta já iniciada; preservar")
    frozen=sha(BODIES) if seed==179 else None
    if seed==179:
        verify_bodies()
    reference=load_prior(game,seed) if seed!=179 else None
    programs=np.asarray(proposal["programs"][:4 if seed!=179 else 8],int)
    budget=Budget(proposal)
    ale,rng=make_ale(game,seed),np.random.default_rng(seed)
    actions=list(ale.getMinimalActionSet()[:4])
    require(len(actions)==4,"alfabeto físico insuficiente")
    histories,rewards,valids,roots,episodes,records,chunks=[],[],[],[],[],[],{}
    eventpath=output.with_suffix(".jsonl")
    stream=eventpath.open("x",encoding="utf-8")
    certified=0
    def emit(record):
        records.append(record)
        stream.write(json.dumps(record,allow_nan=False)+"\n")
        stream.flush()
    try:
        for episode in range(8):
            ale.reset_game()
            first=gray(ale)
            history=deque([first.copy() for _ in range(16)],maxlen=16)
            for point in range(8):
                for _ in range(128):
                    budget.act(ale,actions[int(rng.integers(0,4))])
                    if ale.game_over():
                        ale.reset_game()
                        first=gray(ale)
                        history=deque([first.copy() for _ in range(16)],maxlen=16)
                    else:
                        history.append(gray(ale))
                root=8*episode+point
                emit({"event":"walk_checkpoint","root":root,"episode":episode,"ale_calls":128})
                if point not in (0,2,4,6):
                    continue
                state=ale.cloneSystemState()
                raw,reduced,rr,vv,endpoints={},[],[],[],[]
                for j,word in enumerate(programs):
                    ale.restoreSystemState(state)
                    require(ale.cloneSystemState().equals(state),"restauração diverge")
                    frames,ends,reward,valid,used=play_raw(ale,actions,word,history,budget)
                    encoded=encoded_windows(frames,ends,"coarse")
                    raw[f"frames_{j}"]=frames
                    reduced.append(encoded)
                    rr.append(reward)
                    vv.append(valid)
                    endpoints.append(ends)
                    if reference is not None:
                        i=len(roots)
                        require(np.array_equal(encoded,reference["histories"][i,j]) and np.array_equal(reward,reference["rewards"][i,j])
                            and np.array_equal(valid,reference["valid"][i,j]),"replay reduzido não coincide; não usar dados")
                    emit({"event":"raw_program_measured","root":root,"episode":episode,"program":j,"word":word.tolist(),
                        "rewards":reward.tolist(),"valid":valid.tolist(),"endpoints":ends.tolist(),"ale_calls":used,"full_frames":len(frames)})
                    if j==0 and point==0:
                        ale.restoreSystemState(state)
                        replay=play_raw(ale,actions,word,history,budget)
                        require(all(np.array_equal(a,b) for a,b in zip((frames,ends,reward,valid,used),replay)),"replay lossless diverge")
                        certified+=1
                        emit({"event":"raw_replay_certified","root":root,"ale_calls":replay[-1]})
                chunk=output.with_name(output.name+f"_root_{root:02d}.npz")
                np.savez_compressed(chunk,**raw,endpoints=np.stack(endpoints),rewards=np.stack(rr),valid=np.stack(vv),
                    episode=episode,root=root,programs=programs)
                chunks[chunk.relative_to(HERE).as_posix()]=sha(chunk)
                histories.append(np.stack(reduced))
                rewards.append(np.stack(rr))
                valids.append(np.stack(vv))
                roots.append(root)
                episodes.append(episode)
                ale.restoreSystemState(state)
                checkpoint=output.with_name(output.name+"_checkpoint.json")
                checkpoint.write_text(json.dumps({"status":"partial_not_evaluable","roots":len(roots),"ale_calls":budget.used,
                    "can_resume_exact_ale_state":False}),encoding="utf-8")
            print(json.dumps({"game":game,"seed":seed,"episode":episode,"roots":len(roots),"ale_calls":budget.used}),flush=True)
        budget.check()
        require(len(roots)==32 and certified==8 and budget.used==sum(r["ale_calls"] for r in records),"contabilidade incompleta")
        np.savez_compressed(output.with_suffix(".npz"),histories=np.stack(histories),rewards=np.stack(rewards),valid=np.stack(valids),
            roots=np.array(roots),episode=np.array(episodes),programs=programs)
        stream.close()
        save_new(output.with_suffix(".json"),{"status":"lossless_sensor_bank_complete","game":game,"seed":seed,"ale_calls":budget.used,
            "wall_seconds":time.perf_counter()-budget.started,"maximum_sampled_rss_bytes":budget.peak,"certified_raw_replays":certified,
            "proposal_sha256":sha(PROPOSAL),"bodies_sha256_before_collection":frozen,"chunks_sha256":chunks,
            "bank_sha256":sha(output.with_suffix(".npz")),"events_sha256":sha(eventpath),"old_reduced_bank_reproduced":reference is not None,
            "replayed_old_experience_counts_as_new_physical_calls":True,"human_level_goal_achieved":False})
        return {"status":"lossless_sensor_bank_complete","game":game,"seed":seed,"ale_calls":budget.used,
            "wall_seconds":time.perf_counter()-budget.started,"raw_replays":certified,"chunks":len(chunks)}
    except BaseException as error:
        stream.close()
        save_new(output.with_name(output.name+"_interrupcao.json"),{"status":"inconclusive","error":repr(error),"ale_calls":budget.used,
            "do_not_restart_ignoring_previous_calls":True})
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("game",choices=("breakout","pong"))
    parser.add_argument("seed",type=int,choices=(149,151,179))
    args=parser.parse_args()
    print(json.dumps(collect(args.game,args.seed),ensure_ascii=False,indent=2))
