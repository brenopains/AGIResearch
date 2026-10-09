"""Orquestrador NOVAMIND-SCI — fusao com nossa AGI. Aceita callback `emit` p/ telemetria realtime."""
import sys, time, pathlib, argparse, json
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from shared.agent_base import save_run
from shared.pubmed import search_and_fetch, format_for_llm
from novamind_sci.agents import (PERSONAS, DRIVES_INIT, generate_by_persona, evaluate_vec,
                                 compose_ordered, sample_persona, update_drives,
                                 persona_score, vec_score)

def run(disease: str, cycles=15, pop=8, compose_every=4, emit=None):
    if emit is None:
        emit = lambda et, p=None: print(f"[{et}] {json.dumps(p,ensure_ascii=False)[:200] if p else ''}")
    log=[]
    def L(s):
        print(s); log.append(s); emit("log", {"line": s})
    t0=time.time()
    emit("start", {"system":"novamind_sci","disease":disease,"cycles":cycles,"pop":pop})
    run_name = f"novamindsci_{disease.replace(' ','_').lower()[:40]}_{int(t0)}"
    def snapshot(stage:str, partial:dict):
        try: partial["__partial__"]=stage; partial["__updated_at__"]=time.time(); save_run(run_name, partial)
        except Exception as e: emit("warn",{"msg":f"snapshot fail: {e}"})
    L(f"=== NOVAMIND-SCI ===\ndoenca: {disease}\nciclos: {cycles}, pop: {pop}, composicao a cada {compose_every}")
    L("[lit] literatura ampla...")
    emit("phase", {"name":"literature","cycle":0})
    papers = search_and_fetch(f"{disease} mechanism pathogenesis treatment review", retmax=30)
    lit = format_for_llm(papers, max_chars_each=1100)
    emit("literature", {"n_papers": len(papers), "pmids": [p["pmid"] for p in papers[:10]]})
    drives = dict(DRIVES_INIT)
    emit("drives", {"drives": {k: round(v,3) for k,v in drives.items()}})
    pop_h=[]; pop_v={}; pop_persona={}
    for c in range(1, cycles+1):
        persona = sample_persona(drives)
        emit("cycle", {"index":c,"total":cycles,"persona":persona,"persona_focus":PERSONAS[persona]["focus"]})
        L(f"\n[ciclo {c}] persona={persona} (drives={ {k:round(v,2) for k,v in drives.items()} })")
        h = generate_by_persona(disease, persona, lit, [x.title for x in pop_h])
        if not h:
            L("  geracao falhou."); emit("warn",{"msg":"geracao falhou"}); continue
        L(f"  GERADA: {h.title[:80]}")
        v = evaluate_vec(h)
        pop_v[h.id]=v; pop_persona[h.id]=persona
        sc = persona_score(v, persona)
        L(f"  vetor={ {k:round(v[k],2) for k in v} }")
        L(f"  score(default)={vec_score(v):.2f}  score(persona)={sc:.2f}")
        update_drives(drives, persona, v)
        h.overall = vec_score(v); h.novelty=v["novelty"]; h.plausibility=v["plausibility"]
        h.tractability=v["tractability"]; h.safety=v["safety"]; h.evidence_strength=v["evidence_strength"]
        pop_h.append(h)
        emit("hypothesis_new", {**h.to_dict(), "persona": persona, "vector": v, "default_score": vec_score(v), "persona_score": sc})
        emit("drives", {"drives": {k: round(v_,3) for k,v_ in drives.items()}})
        snapshot(f"gen_{c}", {"system":"novamind_sci","disease":disease,"duration_s":time.time()-t0,
            "hypotheses":[h.to_dict() for h in pop_h],"vectors":pop_v,"personas":pop_persona,
            "drives_final":drives,"log":log})
        # composicao ordenada periodica (proto-sintaxe nao-comutativa)
        if c % compose_every == 0 and len(pop_h)>=2:
            top = sorted(pop_h, key=lambda x: -vec_score(pop_v.get(x.id, {a:0 for a in v})))[:3]
            a, b = top[0], top[1]
            emit("compose", {"order":"A>>B", "a_title":a.title, "b_title":b.title})
            L(f"  [composicao ORDENADA] {a.title[:50]} >> {b.title[:50]}")
            child = compose_ordered(a, b)
            if child:
                cv = evaluate_vec(child)
                pop_v[child.id]=cv; pop_persona[child.id]="composto"
                child.overall=vec_score(cv); child.novelty=cv["novelty"]; child.plausibility=cv["plausibility"]
                child.tractability=cv["tractability"]; child.safety=cv["safety"]; child.evidence_strength=cv["evidence_strength"]
                pop_h.append(child)
                L(f"    filho: {child.title[:80]}  score={child.overall:.2f}")
                emit("hypothesis_new", {**child.to_dict(), "persona": "composto", "vector": cv, "default_score": vec_score(cv)})
                rev = compose_ordered(b, a)
                if rev:
                    rv = evaluate_vec(rev); pop_v[rev.id]=rv; pop_persona[rev.id]="composto_rev"
                    rev.overall=vec_score(rv); pop_h.append(rev)
                    asym = abs(child.overall-rev.overall)
                    L(f"    inverso: {rev.title[:80]} score={rev.overall:.2f} (assimetria de ordem: {asym:.2f})")
                    emit("hypothesis_new", {**rev.to_dict(), "persona": "composto_rev", "vector": rv, "default_score": vec_score(rv)})
                    emit("asymmetry", {"a_to_b": child.overall, "b_to_a": rev.overall, "delta": asym})
        # poda
        pop_h.sort(key=lambda x: -vec_score(pop_v.get(x.id, {a:0 for a in v})))
        pop_h = pop_h[:pop*2]
    L(f"\n=== TOP-{pop} FINAL ===")
    pop_h.sort(key=lambda x: -vec_score(pop_v.get(x.id, {a:0 for a in v})))
    for i,h in enumerate(pop_h[:pop],1):
        vv = pop_v.get(h.id, {})
        L(f"  #{i} [{pop_persona.get(h.id,'?')}] score={vec_score(vv):.2f} | {h.title[:80]}")
    L(f"\ndrives finais: { {k:round(v,2) for k,v in drives.items()} }")
    L(f"tempo: {time.time()-t0:.0f}s")
    emit("population", {"size":len(pop_h),"top":[{**h.to_dict(),"persona":pop_persona.get(h.id),"vector":pop_v.get(h.id,{})} for h in pop_h[:pop]]})
    payload = {"system":"novamind_sci","disease":disease,"duration_s":time.time()-t0,
               "hypotheses":[h.to_dict() for h in pop_h],
               "vectors":{k:pop_v[k] for k in pop_v},
               "personas":pop_persona,"drives_final":drives,"log":log}
    payload["__partial__"]=None
    p = save_run(run_name, payload); print(f"\nSalvo: {p}")
    emit("saved", {"path": str(p), "name": run_name})
    return payload

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("disease")
    ap.add_argument("--cycles", type=int, default=15)
    ap.add_argument("--pop", type=int, default=8)
    a = ap.parse_args()
    run(a.disease, cycles=a.cycles, pop=a.pop)
