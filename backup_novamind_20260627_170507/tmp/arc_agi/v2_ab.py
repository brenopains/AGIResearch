"""A/B v2: mesmas tasks, QUENTE (biblioteca de abstracoes ON + drives) vs FRIO (OFF).
Mede 4 sinais de AGI: taxa absoluta (vs 25.2%), transferencia (quente-frio),
calibracao do consenso (concordancia->acerto), e solves unicos por refinamento/composicao."""
import sys, json, time, random, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from v2_run import run
from v2_solver import ARC_DIR, OUT_DIR

def main(n=30, gens=5, refine=2, seed=7):
    allt = sorted(p.name for p in ARC_DIR.glob("*.json"))
    random.Random(seed).shuffle(allt); task_ids = allt[:n]
    cfg={"gens":gens,"refine":refine}
    print(f"### A/B v2 | {n} tasks | gens={gens} refine={refine} | seed {seed}", flush=True)
    t0=time.time()
    print("\n########## QUENTE (biblioteca+drives ON) ##########", flush=True)
    hot = run(n_tasks=n, hot=True, seed=seed, tag="v2AB_hot", cfg=cfg, task_ids=list(task_ids))
    print("\n########## FRIO (zera por task) ##########", flush=True)
    cold = run(n_tasks=n, hot=False, seed=seed, tag="v2AB_cold", cfg=cfg, task_ids=list(task_ids))
    hs,cs=hot["final"],cold["final"]
    summ={"n":n,"gens":gens,"refine":refine,"seed":seed,"duration_s":time.time()-t0,
          "hot":hs,"cold":cs,
          "transfer_gain": hs["test"]-cs["test"],
          "hot_2nd_minus_1st": hs["second_half"]-hs["first_half"],
          "baseline":"GrammarForge 25.2%"}
    json.dump(summ, open(OUT_DIR/f"v2AB_summary_{int(time.time())}.json","w"), indent=2)
    print("\n"+"#"*64, flush=True)
    print("# RESULTADO A/B v2", flush=True)
    print(f"# QUENTE: {hs['test']}/{n} = {hs['rate']:.1%} | 1a {hs['first_half']} vs 2a {hs['second_half']} | refine {hs['refine_solves']} compose {hs['compose_solves']}", flush=True)
    print(f"# FRIO:   {cs['test']}/{n} = {cs['rate']:.1%} | 1a {cs['first_half']} vs 2a {cs['second_half']}", flush=True)
    print(f"# GANHO TRANSFERENCIA (quente-frio): {summ['transfer_gain']:+d} tasks", flush=True)
    print(f"# baseline GrammarForge: 25.2%", flush=True)
    print("#"*64, flush=True)

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=30); ap.add_argument("--gens",type=int,default=5)
    ap.add_argument("--refine",type=int,default=2); ap.add_argument("--seed",type=int,default=7)
    a=ap.parse_args(); main(a.n,a.gens,a.refine,a.seed)
