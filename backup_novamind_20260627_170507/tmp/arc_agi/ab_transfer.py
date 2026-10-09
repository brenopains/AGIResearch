"""A/B de TRANSFERENCIA: roda o MESMO conjunto de tasks em modo QUENTE (acumula drives+biblioteca)
e FRIO (zera por task). Se QUENTE > FRIO, ha aquisicao de breadth por experiencia (transferencia real)."""
import sys, json, time, random, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from run import run
from solver import ARC_DIR, OUT_DIR

def main(n=40, attempts=4, seed=7):
    all_tasks = sorted(p.name for p in ARC_DIR.glob("*.json"))
    random.Random(seed).shuffle(all_tasks)
    task_ids = all_tasks[:n]
    print(f"### A/B TRANSFERENCIA | {n} tasks | seed {seed} | attempts {attempts}", flush=True)
    print(f"### tasks: {task_ids[:5]}...{task_ids[-2:]}", flush=True)
    t0=time.time()
    print("\n########## CONDICAO QUENTE (acumula) ##########", flush=True)
    hot = run(n_tasks=n, attempts=attempts, hot=True, seed=seed, tag="AB_hot", task_ids=list(task_ids))
    print("\n########## CONDICAO FRIA (zera por task) ##########", flush=True)
    cold = run(n_tasks=n, attempts=attempts, hot=False, seed=seed, tag="AB_cold", task_ids=list(task_ids))
    # comparacao
    hs, cs = hot["final"], cold["final"]
    summary = {
        "n":n,"attempts":attempts,"seed":seed,"duration_s":time.time()-t0,
        "hot": hs, "cold": cs,
        "transfer_gain_test": hs["solved_test"] - cs["solved_test"],
        "hot_2nd_minus_1st": hs["second_half"] - hs["first_half"],
        "cold_2nd_minus_1st": cs["second_half"] - cs["first_half"],
        "baseline_grammarforge": "101/400 = 25.2%",
    }
    json.dump(summary, open(OUT_DIR/f"AB_summary_{int(time.time())}.json","w"), indent=2)
    print("\n"+"#"*64, flush=True)
    print("# RESULTADO A/B TRANSFERENCIA", flush=True)
    print(f"# QUENTE: TEST {hs['solved_test']}/{n} = {hs['rate_test']:.1%} | 1a metade {hs['first_half']} vs 2a {hs['second_half']}", flush=True)
    print(f"# FRIO:   TEST {cs['solved_test']}/{n} = {cs['rate_test']:.1%} | 1a metade {cs['first_half']} vs 2a {cs['second_half']}", flush=True)
    print(f"# GANHO DE TRANSFERENCIA (quente-frio): {summary['transfer_gain_test']:+d} tasks", flush=True)
    print(f"# baseline GrammarForge (sem LLM): 25.2%", flush=True)
    print("#"*64, flush=True)

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=40); ap.add_argument("--attempts",type=int,default=4); ap.add_argument("--seed",type=int,default=7)
    a=ap.parse_args(); main(a.n,a.attempts,a.seed)
