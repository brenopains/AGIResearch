"""Compara 3 variantes da Ponte p/ isolar o que faz a transferencia (lib-solve sem LLM) emergir:
  (1) AGRUPADO  — tasks de mesmo tipo adjacentes (V3_GROUPED=1)
  (2) GENERAL   — primitivas parametrizadas/gerais (V3_GENERALIZE=1)
  (3) AMBOS     — agrupado + generalizado
Mesmo conjunto de tasks nas 3. Mede lib_rate_thirds (cresceu?) e by_library em cada."""
import sys, os, json, time, random, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from v3_bridge import run, grouped_stream
from solver import ARC_DIR, OUT_DIR

def main(n=18, seed=11, gens=2, refine=1):
    allt=sorted(p.name for p in ARC_DIR.glob("*.json")); random.Random(seed).shuffle(allt)
    task_ids=allt[:n]
    print(f"### COMPARA 3 VARIANTES DA PONTE | {n} tasks | seed {seed}", flush=True)
    res={}
    conds=[("AGRUPADO",{"V3_GROUPED":"1","V3_GENERALIZE":"0"}),
           ("GENERAL", {"V3_GROUPED":"0","V3_GENERALIZE":"1"}),
           ("AMBOS",   {"V3_GROUPED":"1","V3_GENERALIZE":"1"})]
    t0=time.time()
    for name,env in conds:
        for k,v in env.items(): os.environ[k]=v
        print(f"\n{'#'*60}\n# CONDICAO: {name}\n{'#'*60}", flush=True)
        p=run(n_tasks=n, seed=seed, gens=gens, refine=refine, fresh_lib=True,
              tag=f"v3cmp_{name}", task_ids=list(task_ids))
        res[name]=p["final"]
    # zera env
    os.environ["V3_GROUPED"]="0"; os.environ["V3_GENERALIZE"]="0"
    summ={"n":n,"seed":seed,"duration_s":time.time()-t0,"conditions":res}
    json.dump(summ, open(OUT_DIR/f"v3cmp_summary_{int(time.time())}.json","w"), indent=2)
    print(f"\n{'#'*60}\n# RESULTADO — qual variante fez a Ponte transferir?\n{'#'*60}", flush=True)
    for name in res:
        f=res[name]; lr=f["lib_rate_thirds"]; cr=f["call_rate_thirds"]
        print(f"# {name:9s}: TEST {f['test']}/{n}={f['rate']:.0%} | SEM-LLM={f['by_library']} | "
              f"lib_rate {lr[0]:.0%}->{lr[1]:.0%}->{lr[2]:.0%} | LLM/task {cr[0]:.1f}->{cr[2]:.1f} | prims={f['n_primitives']}", flush=True)
    print("# (lib_rate crescente + by_library>0 = a Ponte transferiu = tese confirmada)", flush=True)

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=18); ap.add_argument("--seed",type=int,default=11)
    ap.add_argument("--gens",type=int,default=2); ap.add_argument("--refine",type=int,default=1)
    a=ap.parse_args(); main(a.n,a.seed,a.gens,a.refine)
