"""
H56-E — Ensemble induction+transduction: UNIAO de todos os solvers
==================================================================
Une os conjuntos de tarefas resolvidas (no teste held-out, casamento exato):
  - ABDUCE (H55, simbolico abdutivo objeto-centrico)
  - PULSE  (H54, busca por energia/ritmo)
  - TRANSDUTOR neural por-puzzle (H56-T)  [se houver tmp/h56_transducer_solved.json]
  - LLM transdutivo pequeno (H56-L)       [se houver tmp/h56_llm_solved.json]
Reporta contribuicao SOLO e MARGINAL de cada um + a uniao final /400.
"""
import os, json, glob, numpy as np
import importlib.util
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")

def _load(modfile):
    p=os.path.join(os.path.dirname(__file__),modfile)
    spec=importlib.util.spec_from_file_location(modfile[:-3],p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def _read_ids(fn):
    p=os.path.join(os.path.dirname(__file__),fn)
    if os.path.exists(p):
        try: return set(json.load(open(p)))
        except Exception: return set()
    return set()

def main():
    AB=_load("exp_h55_abduce.py"); PU=_load("exp_h54_pulse.py")
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    ab_set=set(); pu_set=set()
    for fp in files:
        name=os.path.basename(fp)
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        # ABDUCE
        r,_,_=AB.abduce(train)
        if r is not None and all((lambda o,yo:o is not None and isinstance(o,np.ndarray) and o.shape==yo.shape and np.array_equal(o,yo))(r(xi),yo) for xi,yo in tests):
            ab_set.add(name)
        # PULSE
        pool=dict(PU.FIXED)
        if train[0][0].shape[0]==train[0][0].shape[1]: pool.update(PU.square_ops())
        prog,_,_=PU.pulse_search(pool,train)
        if prog is not None and all((lambda o,yo:o is not None and o.shape==yo.shape and np.array_equal(o,yo))(prog(xi),yo) for xi,yo in tests):
            pu_set.add(name)
    tr_set=_read_ids("h56_transducer_solved.json")
    llm_set=_read_ids("h56_llm_solved.json")
    symbolic=ab_set|pu_set
    union=symbolic|tr_set|llm_set
    print("="*64)
    print("H56-E Ensemble (induction + transduction) — uniao de solvers")
    print(f"  ABDUCE  solo: {len(ab_set)}/400")
    print(f"  PULSE   solo: {len(pu_set)}/400")
    print(f"  simbolico (ABDUCE u PULSE): {len(symbolic)}/400 = {100*len(symbolic)/400:.1f}%")
    print(f"  TRANSDUTOR neural solo: {len(tr_set)} | NOVAS vs simbolico: {len(tr_set-symbolic)}")
    print(f"  LLM transdutivo solo:   {len(llm_set)} | NOVAS vs simbolico: {len(llm_set-symbolic)}")
    print(f"  UNIAO FINAL: {len(union)}/400 = {100*len(union)/400:.1f}%   (baseline simbolico 12.5%)")

if __name__=="__main__": main()
