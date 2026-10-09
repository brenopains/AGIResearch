"""Orquestrador Co-Scientist local. Loop: Generate -> Reflect -> Rank -> Evolve -> Proximity -> MetaReview.
Aceita callback `emit(event_type, payload)` p/ telemetria em tempo real (UI/CLI).
"""
import sys, time, pathlib, argparse, json
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from shared.agent_base import save_run
from co_scientist.generation_agent import generate
from co_scientist.reflection_agent import apply as reflect_apply
from co_scientist.ranking_agent import tournament
from co_scientist.evolution_agent import refine, cross
from co_scientist.proximity_meta import cluster, dedupe_keep_best, meta_review

def run(disease: str, n_initial=6, rounds=2, tourn_rounds=2, keep_top=5, emit=None):
    if emit is None:
        emit = lambda et, p=None: print(f"[{et}] {json.dumps(p,ensure_ascii=False)[:200] if p else ''}")
    log = []
    def L(s):
        print(s); log.append(s); emit("log", {"line": s})
    t0 = time.time()
    emit("start", {"system":"co_scientist","disease":disease,"n_initial":n_initial,"rounds":rounds})
    run_name = f"coscientist_{disease.replace(' ','_').lower()[:40]}_{int(t0)}"
    def snapshot(stage:str, partial:dict):
        try: partial["__partial__"]=stage; partial["__updated_at__"]=time.time(); save_run(run_name, partial)
        except Exception as e: emit("warn",{"msg":f"snapshot fail: {e}"})
    L(f"=== CO-SCIENTIST LOCAL ===\ndoenca: {disease}\ninicial: {n_initial} hipoteses, {rounds} rounds de evolucao, {tourn_rounds} rounds de torneio por ciclo")
    L("[1] Generation: buscando literatura + propondo hipoteses iniciais...")
    emit("phase", {"name":"generation","cycle":0})
    hyps = generate(disease, n=n_initial)
    L(f"  -> {len(hyps)} hipoteses geradas.")
    for h in hyps: emit("hypothesis_new", h.to_dict())
    snapshot("post_generation", {"system":"co_scientist","disease":disease,"duration_s":time.time()-t0,"hypotheses":[h.to_dict() for h in hyps],"log":log})
    for cycle in range(rounds+1):
        L(f"\n--- ciclo {cycle} ---")
        emit("cycle", {"index":cycle,"total":rounds+1})
        L(f"[2] Reflection: criticando cada hipotese...")
        emit("phase", {"name":"reflection","cycle":cycle})
        for h in hyps:
            try:
                reflect_apply(h); emit("hypothesis_update", h.to_dict())
            except Exception as e: L(f"  [warn] reflect falhou em {h.id}: {e}")
        hyps.sort(key=lambda h:-h.overall)
        for h in hyps[:5]: L(f"  {h.id} ov={h.overall:.2f} elo={h.elo:.0f} | {h.title[:80]}")
        L(f"[3] Ranking: torneio Elo ({tourn_rounds} rounds)...")
        emit("phase", {"name":"ranking","cycle":cycle})
        hyps = tournament(disease, hyps, rounds=tourn_rounds)
        for h in hyps: emit("hypothesis_update", h.to_dict())
        L(f"  -> top apos torneio:")
        for h in hyps[:5]: L(f"  {h.id} elo={h.elo:.0f} ov={h.overall:.2f} | {h.title[:80]}")
        L(f"[4] Proximity: agrupando similares...")
        emit("phase", {"name":"proximity","cycle":cycle})
        groups = cluster(hyps); dedup = dedupe_keep_best(hyps, groups)
        L(f"  -> {len(hyps)} -> {len(dedup)} apos dedup")
        hyps = dedup
        emit("population", {"size":len(hyps), "top":[h.to_dict() for h in hyps[:5]]})
        snapshot(f"cycle_{cycle}", {"system":"co_scientist","disease":disease,"duration_s":time.time()-t0,"hypotheses":[h.to_dict() for h in sorted(hyps, key=lambda h:-h.elo)],"log":log})
        if cycle < rounds:
            L(f"[5] Evolution: refinando top + cross dos 2 melhores...")
            emit("phase", {"name":"evolution","cycle":cycle})
            top = sorted(hyps, key=lambda h:-h.elo)[:max(3, keep_top)]
            children = []
            for h in top[:3]:
                try:
                    r = refine(h)
                    if r:
                        children.append(r); emit("hypothesis_new", r.to_dict())
                except Exception as e: L(f"  [warn] refine falhou: {e}")
            if len(top) >= 2:
                try:
                    c = cross(top[0], top[1])
                    if c:
                        children.append(c); emit("hypothesis_new", c.to_dict())
                except Exception as e: L(f"  [warn] cross falhou: {e}")
            hyps = top + children
            L(f"  -> {len(children)} filhos. populacao = {len(hyps)}")
    L(f"\n[6] Meta-review final...")
    emit("phase", {"name":"meta_review","cycle":rounds+1})
    top = sorted(hyps, key=lambda h:-h.elo)[:8]
    review = meta_review(disease, top)
    L("\n========= META-REVIEW =========\n" + review)
    emit("meta_review", {"markdown": review})
    L(f"\nTempo total: {time.time()-t0:.0f}s")
    payload = {
        "system":"co_scientist","disease":disease, "duration_s":time.time()-t0,
        "hypotheses":[h.to_dict() for h in sorted(hyps, key=lambda h:-h.elo)],
        "meta_review": review, "log": log,
    }
    payload["__partial__"]=None
    p = save_run(run_name, payload)
    print(f"\nSalvo em: {p}")
    emit("saved", {"path": str(p), "name": run_name})
    return payload

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("disease")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--tourn", type=int, default=2)
    a = ap.parse_args()
    run(a.disease, n_initial=a.n, rounds=a.rounds, tourn_rounds=a.tourn)
