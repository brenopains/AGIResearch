"""
UNIFIED — os 3 sistemas trabalhando juntos:
  • GERADORES = NovaMind-Sci AGI (5 personas com drives evolutivos + world-model multi-atributo)
  • CRITICOS + TORNEIO = Co-Scientist (Reflection + Ranking Elo + Proximity dedupe + Evolution)
  • DEEP-DIVE no top-3 = Robin (Falcon literatura profunda + Finch plano de dados + DataAnalyst codigo)
  • NEUROMODULADORES GLOBAIS = analogo da colonia (da=avg recente score, ne=desacordo entre criticas, se=convergencia)
  • COMPOSICAO NAO-COMUTATIVA = NovaMind-Sci (a cada K ciclos, top2: A>>B e B>>A com analise de assimetria)
  • PROMISE SCORE no fim = traduz tudo p/ leigo (semaforo + sinais interpretaveis)
"""
import sys, time, pathlib, argparse, json, random
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from shared.agent_base import save_run, Hypothesis
from shared.pubmed import search_and_fetch, format_for_llm

# --- NovaMind-Sci AGI ---
from novamind_sci.agents import (PERSONAS, DRIVES_INIT, generate_by_persona, evaluate_vec,
                                 compose_ordered, sample_persona, update_drives,
                                 persona_score, vec_score, ATTRS)
# --- Co-Scientist machinery ---
from co_scientist.reflection_agent import apply as reflect_apply
from co_scientist.ranking_agent import tournament
from co_scientist.proximity_meta import cluster, dedupe_keep_best, meta_review
from co_scientist.evolution_agent import refine
# --- Robin deep-dive ---
from robin.agents import falcon, finch, data_analyst
# --- Promise scoring ---
from shared.promise import evaluate_promise

def clamp01(x): return max(0.0, min(1.0, x))

def run(disease: str, generation_cycles=10, top_for_tournament=8, tourn_rounds=3,
        compose_every=4, deep_dive_top=3, emit=None):
    if emit is None:
        emit = lambda et,p=None: print(f"[{et}] {json.dumps(p,ensure_ascii=False)[:200] if p else ''}")
    log=[]
    def L(s): print(s); log.append(s); emit("log", {"line": s})
    t0=time.time()
    # ==== AUTOSAVE INCREMENTAL ==== persiste a cada hipotese: se o processo morrer mid-run, NADA se perde.
    run_name = f"unified_{disease.replace(' ','_').lower()[:40]}_{int(t0)}"
    def snapshot(stage:str, partial:dict):
        try:
            partial["__partial__"] = stage
            partial["__updated_at__"] = time.time()
            save_run(run_name, partial)
        except Exception as e: emit("warn",{"msg":f"snapshot fail: {e}"})
    emit("start", {"system":"unified","disease":disease,
                   "generation_cycles":generation_cycles,"top_for_tournament":top_for_tournament,
                   "tourn_rounds":tourn_rounds,"deep_dive_top":deep_dive_top})
    L(f"=== UNIFIED — AGI(NovaMind) + Torneio(Co-Sci) + DeepDive(Robin) ===")
    L(f"doenca: {disease}\nciclos de geracao: {generation_cycles} | top torneio: {top_for_tournament} | deep-dive: {deep_dive_top}")

    L("\n[lit] literatura ampla...")
    emit("phase", {"name":"literature","stage":1})
    papers = search_and_fetch(f"{disease} mechanism pathogenesis treatment review", retmax=30)
    lit = format_for_llm(papers, max_chars_each=1100)
    emit("literature", {"n_papers": len(papers), "pmids":[p["pmid"] for p in papers[:10]]})

    # ===== ESTADO COLONIA: drives + neuromoduladores globais =====
    drives = dict(DRIVES_INIT)
    nm = {"da":0.5, "ne":0.3, "se":0.5}
    emit("drives", {"drives":{k:round(v,3) for k,v in drives.items()}})
    emit("neuromod", {"da":round(nm["da"],2),"ne":round(nm["ne"],2),"se":round(nm["se"],2)})

    pop_h=[]; pop_v={}; pop_persona={}; comp_asymmetries=[]

    # ============================================================
    # FASE 1: GERACAO POR PERSONAS (NovaMind-Sci AGI)
    # ============================================================
    L("\n=== FASE 1: GERACAO POR PERSONAS (AGI) ===")
    emit("phase", {"name":"generation_agi","stage":2})
    for c in range(1, generation_cycles+1):
        persona = sample_persona(drives)
        emit("cycle", {"index":c,"total":generation_cycles,"persona":persona,"persona_focus":PERSONAS[persona]["focus"]})
        L(f"[ciclo {c}] persona={persona}")
        h = generate_by_persona(disease, persona, lit, [x.title for x in pop_h])
        if not h:
            L("  geracao falhou."); emit("warn",{"msg":"gen fail"}); continue
        L(f"  GERADA: {h.title[:80]}")
        v = evaluate_vec(h)
        pop_v[h.id]=v; pop_persona[h.id]=persona
        h.overall=vec_score(v); h.novelty=v["novelty"]; h.plausibility=v["plausibility"]
        h.tractability=v["tractability"]; h.safety=v["safety"]; h.evidence_strength=v["evidence_strength"]
        pop_h.append(h)
        update_drives(drives, persona, v)
        # neuromoduladores globais: da sobe se score alto; ne sobe se vetor desbalanceado; se sobe se replica padroes anteriores
        nm["da"] = clamp01(nm["da"]*0.85 + h.overall*0.15)
        var = max(v.values())-min(v.values())
        nm["ne"] = clamp01(nm["ne"]*0.85 + var*0.15)
        nm["se"] = clamp01(nm["se"]*0.90 + (h.plausibility)*0.10)
        emit("hypothesis_new", {**h.to_dict(),"persona":persona,"vector":v,"default_score":vec_score(v),"persona_score":persona_score(v,persona)})
        emit("drives", {"drives":{k:round(v_,3) for k,v_ in drives.items()}})
        emit("neuromod", {"da":round(nm["da"],2),"ne":round(nm["ne"],2),"se":round(nm["se"],2)})
        # AUTOSAVE: a cada hipotese o estado vai pro disco
        snapshot(f"gen_{c}", {"system":"unified","disease":disease,"duration_s":time.time()-t0,
            "hypotheses":[h.to_dict() for h in pop_h], "vectors":pop_v, "personas":pop_persona,
            "drives_final":drives, "neuromod_final":nm, "asymmetries":comp_asymmetries, "log":log})

        # composicao nao-comutativa periodica
        if c % compose_every == 0 and len(pop_h) >= 2:
            top2 = sorted(pop_h, key=lambda x: -vec_score(pop_v.get(x.id,{}))) [:2]
            a, b = top2[0], top2[1]
            L(f"  [composicao] {a.title[:40]} >> {b.title[:40]}")
            emit("compose", {"order":"A>>B","a_title":a.title,"b_title":b.title})
            child = compose_ordered(a,b)
            if child:
                cv=evaluate_vec(child); pop_v[child.id]=cv; pop_persona[child.id]="composto"
                child.overall=vec_score(cv); child.novelty=cv["novelty"]; child.plausibility=cv["plausibility"]
                child.tractability=cv["tractability"]; child.safety=cv["safety"]; child.evidence_strength=cv["evidence_strength"]
                pop_h.append(child)
                emit("hypothesis_new",{**child.to_dict(),"persona":"composto","vector":cv,"default_score":vec_score(cv)})
                rev = compose_ordered(b,a)
                if rev:
                    rv=evaluate_vec(rev); pop_v[rev.id]=rv; pop_persona[rev.id]="composto_rev"
                    rev.overall=vec_score(rv); pop_h.append(rev)
                    asym = abs(child.overall-rev.overall)
                    comp_asymmetries.append(asym)
                    L(f"    inverso: assimetria={asym:.2f}")
                    emit("hypothesis_new",{**rev.to_dict(),"persona":"composto_rev","vector":rv,"default_score":vec_score(rv)})
                    emit("asymmetry",{"a_to_b":child.overall,"b_to_a":rev.overall,"delta":asym})
                    snapshot(f"compose_{c}", {"system":"unified","disease":disease,"duration_s":time.time()-t0,
                        "hypotheses":[h.to_dict() for h in pop_h], "vectors":pop_v, "personas":pop_persona,
                        "drives_final":drives, "neuromod_final":nm, "asymmetries":comp_asymmetries, "log":log})

    L(f"\nFase 1 ok. populacao={len(pop_h)}")
    # poda inicial p/ levar ao torneio so os top
    pop_h.sort(key=lambda x: -vec_score(pop_v.get(x.id, {})))
    contestants = pop_h[:top_for_tournament]
    L(f"\n=== FASE 2: CRITICA + TORNEIO Elo (Co-Scientist) com top-{len(contestants)} ===")
    emit("phase", {"name":"reflection_tournament","stage":3})

    # ============================================================
    # FASE 2: REFLECTION + TOURNAMENT (Co-Scientist)
    # ============================================================
    for h in contestants:
        try: reflect_apply(h); emit("hypothesis_update", h.to_dict())
        except Exception as e: L(f"  [warn] reflect: {e}")
    contestants = tournament(disease, contestants, rounds=tourn_rounds)
    for h in contestants: emit("hypothesis_update", h.to_dict())
    L("Top apos torneio:")
    for h in contestants[:5]: L(f"  elo={h.elo:.0f} ov={h.overall:.2f} | {h.title[:70]}")
    # Proximity dedupe
    groups = cluster(contestants); dedup = dedupe_keep_best(contestants, groups)
    L(f"Dedupe: {len(contestants)} -> {len(dedup)}")
    contestants = dedup
    emit("population",{"size":len(contestants),"top":[h.to_dict() for h in contestants[:5]]})
    snapshot("post_tournament", {"system":"unified","disease":disease,"duration_s":time.time()-t0,
        "hypotheses":[h.to_dict() for h in pop_h], "vectors":pop_v, "personas":pop_persona,
        "drives_final":drives, "neuromod_final":nm, "asymmetries":comp_asymmetries,
        "contestants_after_tournament":[h.to_dict() for h in contestants], "log":log})

    # ============================================================
    # FASE 3: DEEP-DIVE Robin (Falcon + Finch + DataAnalyst) no top-N
    # ============================================================
    L(f"\n=== FASE 3: DEEP-DIVE Robin no top-{deep_dive_top} ===")
    emit("phase",{"name":"deep_dive","stage":4})
    deep_results = []
    for h in contestants[:deep_dive_top]:
        L(f"\n>>> {h.title[:70]}")
        emit("deep_dive_start", h.to_dict())
        try:
            L("  [Falcon] revisao profunda da literatura...")
            emit("phase",{"name":"falcon"})
            falcon_md = falcon(h, disease)
            emit("falcon_report",{"hypothesis_id":h.id,"markdown":falcon_md})
            L(f"    -> {len(falcon_md)} chars")
            L("  [Finch] plano de analise de dados...")
            emit("phase",{"name":"finch"})
            finch_md = finch(h)
            emit("finch_plan",{"hypothesis_id":h.id,"markdown":finch_md})
            L(f"    -> {len(finch_md)} chars")
            L("  [DataAnalyst] codigo Python...")
            emit("phase",{"name":"data_analyst"})
            code = data_analyst(finch_md)
            emit("analysis_code",{"hypothesis_id":h.id,"code":code[:200]+("..." if len(code)>200 else "")})
            L(f"    -> {len(code)} chars de codigo")
            deep_results.append({"hypothesis_id":h.id,"falcon":falcon_md,"finch":finch_md,"code":code})
            # AUTOSAVE apos cada deep-dive completo (Falcon + Finch + DataAnalyst sao caros, NAO perder)
            snapshot(f"deep_{len(deep_results)}", {"system":"unified","disease":disease,"duration_s":time.time()-t0,
                "hypotheses":[h.to_dict() for h in pop_h], "vectors":pop_v, "personas":pop_persona,
                "drives_final":drives, "neuromod_final":nm, "asymmetries":comp_asymmetries,
                "contestants_after_tournament":[h.to_dict() for h in contestants],
                "deep_dive":deep_results, "log":log})
        except Exception as e:
            L(f"  deep-dive falhou: {e}"); emit("warn",{"msg":f"deep_dive_fail {e}"})

    # ============================================================
    # FASE 4: META-REVIEW + PROMISE SCORE
    # ============================================================
    L("\n=== FASE 4: META-REVIEW + PROMISE SCORE ===")
    emit("phase",{"name":"meta_review","stage":5})
    review = meta_review(disease, contestants[:6])
    L("\n--- meta-review ---\n"+review)
    emit("meta_review",{"markdown": review})

    emit("phase",{"name":"promise","stage":6})
    L("\nCalculando indicador de promessa (p/ leigo)...")
    promise = evaluate_promise(disease, [h.to_dict() for h in contestants],
                              compositions_asymmetry=comp_asymmetries, top_k=6)
    L(f"\n{promise['summary']}")
    emit("promise", promise)

    L(f"\ntempo total: {time.time()-t0:.0f}s")
    payload = {
        "system":"unified","disease":disease,"duration_s":time.time()-t0,
        "hypotheses":[h.to_dict() for h in sorted(pop_h, key=lambda x: -vec_score(pop_v.get(x.id,{})))],
        "vectors":pop_v,"personas":pop_persona,"drives_final":drives,"neuromod_final":nm,
        "asymmetries":comp_asymmetries,"deep_dive":deep_results,
        "meta_review":review,"promise":promise,"log":log,
        "__partial__":None,  # marca como completo
    }
    p = save_run(run_name, payload); print(f"\nSalvo: {p}")
    emit("saved",{"path":str(p),"name":run_name})
    return payload

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("disease")
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--tourn", type=int, default=3)
    ap.add_argument("--deep", type=int, default=3)
    a = ap.parse_args()
    run(a.disease, generation_cycles=a.cycles, top_for_tournament=a.top, tourn_rounds=a.tourn, deep_dive_top=a.deep)
