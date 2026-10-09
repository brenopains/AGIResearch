"""Orquestrador Robin: ITERA gerando+aprofundando+propondo analise. Diferente de Co-Scientist (que e torneio).
Loop: Crow gera -> Falcon mergulha -> Owl plano de dados -> DataAnalyst gera codigo -> repetir N vezes com hipoteses distintas.
"""
import sys, time, pathlib, argparse, json
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from shared.agent_base import save_run
from shared.pubmed import search_and_fetch, format_for_llm
from robin.agents import crow, falcon, finch, data_analyst

def run(disease: str, n_hypotheses=4, lit_retmax=30, emit=None):
    if emit is None:
        import json as _j
        emit = lambda et, p=None: print(f"[{et}] {_j.dumps(p,ensure_ascii=False)[:200] if p else ''}")
    log = []
    def L(s):
        print(s); log.append(s); emit("log", {"line": s})
    t0=time.time()
    emit("start", {"system":"robin","disease":disease,"n_hypotheses":n_hypotheses})
    run_name = f"robin_{disease.replace(' ','_').lower()[:40]}_{int(t0)}"
    def snapshot(stage:str, partial:dict):
        try: partial["__partial__"]=stage; partial["__updated_at__"]=time.time(); save_run(run_name, partial)
        except Exception as e: emit("warn",{"msg":f"snapshot fail: {e}"})
    L(f"=== ROBIN LOCAL ===\ndoenca: {disease}\nhipoteses: {n_hypotheses}\n")
    L("[lit] buscando literatura ampla...")
    papers = search_and_fetch(f"{disease} mechanism pathogenesis treatment", retmax=lit_retmax)
    lit = format_for_llm(papers, max_chars_each=1100)
    hyps=[]; results=[]
    prior=[]
    for k in range(n_hypotheses):
        L(f"\n--- HIPOTESE {k+1} ---")
        emit("cycle", {"index":k+1,"total":n_hypotheses})
        L("[Crow] gerando...")
        emit("phase", {"name":"crow","cycle":k+1})
        h = crow(disease, prior, lit)
        if not h:
            L("  falhou."); emit("warn",{"msg":"crow falhou"}); continue
        L(f"  -> {h.title}")
        hyps.append(h); prior.append(h.title)
        emit("hypothesis_new", h.to_dict())
        L("[Falcon] mergulho na literatura especifica...")
        emit("phase", {"name":"falcon","cycle":k+1})
        falcon_report = falcon(h, disease)
        L(f"  -> {len(falcon_report)} chars")
        emit("falcon_report", {"hypothesis_id":h.id, "markdown": falcon_report})
        L("[Finch] plano de analise de dados...")
        emit("phase", {"name":"finch","cycle":k+1})
        finch_plan = finch(h)
        L(f"  -> {len(finch_plan)} chars")
        emit("finch_plan", {"hypothesis_id":h.id, "markdown": finch_plan})
        L("[Finch-code] gerando codigo Python...")
        emit("phase", {"name":"data_analyst","cycle":k+1})
        code = data_analyst(finch_plan)
        L(f"  -> {len(code)} chars")
        emit("analysis_code", {"hypothesis_id":h.id, "code": code[:200]+("..." if len(code)>200 else "")})
        results.append({
            "hypothesis": h.to_dict(),
            "falcon": falcon_report,
            "finch": finch_plan,
            "data_analyst_code": code,
        })
        snapshot(f"after_{k+1}", {"system":"robin","disease":disease,"duration_s":time.time()-t0,
            "results":results,"hypotheses":[h.to_dict() for h in hyps],"log":log})
    L(f"\nTempo: {time.time()-t0:.0f}s")
    payload = {"disease":disease,"duration_s":time.time()-t0,"results":results,"hypotheses":[h.to_dict() for h in hyps],"log":log}
    payload["__partial__"]=None
    p = save_run(run_name, payload); print(f"\nSalvo: {p}")
    emit("saved", {"path": str(p), "name": run_name})
    return payload

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("disease")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    run(a.disease, n_hypotheses=a.n)
