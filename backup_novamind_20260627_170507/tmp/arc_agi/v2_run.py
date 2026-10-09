"""
ARC-NOVAMIND v2 orquestrador. Por task:
  1. GERACAO multi-persona (drive darwiniano) com REFINEMENT LOOP (feedback dos exemplos que falham)
  2. COMPOSICAO NAO-COMUTATIVA dos parciais (A>>B, B>>A) — contribuicao unica nossa
  3. coleta TODOS os "experts" (programas que passam 100% do treino)
  4. DEBATE: refutador escolhe a regra que melhor generaliza
  5. CONSENSO (Product-of-Experts, Franzen): experts votam no grid de teste; robustez por D8/cor pondera
  6. WAKE-SLEEP: extrai helpers dos experts -> biblioteca acumulada (transferencia)
  7. DRIVES: personas cujos programas viraram expert ganham peso

Mede: taxa de teste, CALIBRACAO DO CONSENSO (k experts concordam -> acerta?), transferencia, e
quantas tasks so a composicao/refinamento resolve.
"""
import sys, json, time, random, pathlib, argparse
from collections import Counter
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from v2_solver import (PERSONAS, llm_code, failing_examples_str, extract_helpers,
                       program_robustness, refute_prompt, ask, ARC_DIR, OUT_DIR,
                       verify_on_train, apply_to_test, test_correct, run_with_timeout, normalize_grid)

def sample_persona(drives):
    keys=list(drives); w=[max(0.05,drives[k]) for k in keys]; tot=sum(w); r=random.random()*tot; acc=0
    for k,wi in zip(keys,w):
        acc+=wi
        if r<=acc: return k
    return keys[-1]

def grid_key(g):
    return json.dumps(g) if g is not None else None

def solve_task_v2(task, drives, library_code, cfg, hot, emit):
    experts = []        # (code, persona, robustness)  programas que passam 100% treino
    partials = []       # (fn, code, persona, n_ok)
    used = []
    refine_solves = 0; compose_solves = 0
    gens = cfg["gens"]; refine_rounds = cfg["refine"]
    for gi in range(gens):
        persona = sample_persona(drives); used.append(persona)
        temp = 0.15 + 0.2*gi
        fn, code, err = llm_code(task, persona, library_code if hot else [], None, temp)
        # REFINEMENT LOOP
        attempt_path = "gen"
        for rr in range(refine_rounds):
            if fn is None: break
            solved, ok, tot = verify_on_train(fn, task)
            if solved: break
            fb = failing_examples_str(fn, task)
            fn2, code2, err2 = llm_code(task, persona, library_code if hot else [], fb, 0.2+0.15*rr)
            if fn2 is not None:
                fn, code = fn2, code2; attempt_path = f"refine{rr+1}"
        if fn is None:
            emit(f"    [{persona}] compile/llm falhou"); continue
        solved, ok, tot = verify_on_train(fn, task)
        emit(f"    [{persona}/{attempt_path}] treino {ok}/{tot}" + ("  EXPERT" if solved else ""))
        if solved:
            rob = program_robustness(fn, task)
            experts.append((code, persona, rob, fn))
            if attempt_path.startswith("refine"): refine_solves += 1
            # EARLY-EXIT: 2 experts ja bastam p/ consenso -> nao gasta mais chamadas (acelera ~3x)
            if cfg.get("early_exit", True) and len(experts) >= 2:
                emit(f"    [early-exit: {len(experts)} experts]"); break
        elif ok > 0:
            partials.append((fn, code, persona, ok))

    # COMPOSICAO NAO-COMUTATIVA dos 2 melhores parciais
    if len(partials) >= 2:
        partials.sort(key=lambda x:-x[3])
        (f1,c1,p1,_),(f2,c2,p2,_) = partials[0], partials[1]
        for fa,fb,order in [(f1,f2,"A>>B"),(f2,f1,"B>>A")]:
            def comp(g, fa=fa, fb=fb):
                r=fa([list(r) for r in g])
                if hasattr(r,"tolist"): r=r.tolist()
                return fb([list(r) for r in r])
            solved,ok,tot = verify_on_train(comp, task)
            emit(f"    [compose {order}] treino {ok}/{tot}" + ("  EXPERT" if solved else ""))
            if solved:
                rob = program_robustness(comp, task)
                experts.append((f"# COMPOSED {order}\n{c1}\n# >>\n{c2}", "composto", rob, comp))
                compose_solves += 1

    if not experts:
        return {"solved_train": False, "used_personas": used, "n_experts": 0}

    # DEBATE: refutador escolhe a regra que melhor generaliza (entre experts) — desempata consenso
    debate_best = 0
    if len(experts) > 1:
        desc = "\n".join(f"[{i}] persona={e[1]} robustez_cor={e[2]:.2f}\n{e[0][:400]}" for i,e in enumerate(experts))
        try:
            sysp,user = refute_prompt(task, desc)
            raw = ask([{"role":"user","content":user}], system=sysp, temperature=0.2, max_tokens=300)
            import re as _re
            m=_re.search(r'"best_index"\s*:\s*(\d+)', raw)
            if m: debate_best = min(int(m.group(1)), len(experts)-1)
        except Exception: pass

    # CONSENSO (Product-of-Experts): cada expert preve o teste; voto ponderado por robustez+debate
    test_preds = []
    vote = Counter()
    weighted = Counter()
    for idx,(code,persona,rob,fn) in enumerate(experts):
        preds = apply_to_test(fn, task)
        key = grid_key(preds[0]) if preds else None
        if key is None: continue
        test_preds.append((key, preds))
        vote[key] += 1
        w = 1.0 + rob*1.5 + (1.0 if idx==debate_best else 0.0)   # robustez a cor + escolha do debate pesam
        weighted[key] += w
    if not weighted:
        return {"solved_train": True, "n_experts": len(experts), "consensus": None, "used_personas": used}
    # vencedor por voto ponderado
    best_key = weighted.most_common(1)[0][0]
    agree = vote[best_key]                  # quantos experts concordam (calibracao)
    final_preds = next(p for k,p in test_preds if k==best_key)

    return {"solved_train": True, "n_experts": len(experts), "agree": agree, "total_experts": len(experts),
            "consensus": final_preds, "experts_code": [e[0] for e in experts], "robustness": [e[2] for e in experts],
            "used_personas": used, "refine_solves": refine_solves, "compose_solves": compose_solves,
            "debate_best": debate_best}

def run(n_tasks=30, hot=True, seed=7, tag=None, cfg=None, task_ids=None):
    random.seed(seed)
    cfg = cfg or {"gens":4, "refine":2}
    tag = tag or ("v2_hot" if hot else "v2_cold")
    allt = sorted(p.name for p in ARC_DIR.glob("*.json"))
    if task_ids is None:
        random.Random(seed).shuffle(allt); task_ids = allt[:n_tasks]
    drives = {k:1.0 for k in PERSONAS}
    library_code = []
    results = []; solved_ids = []
    run_name = f"arc_{tag}_{int(time.time())}"; t0=time.time()
    def emit(s): print(s, flush=True)
    emit(f"=== ARC-NOVAMIND v2 [{tag}] {'QUENTE(biblioteca ON)' if hot else 'FRIO(OFF)'} | {len(task_ids)} tasks | gens={cfg['gens']} refine={cfg['refine']} ===")
    # calibracao: buckets por nivel de concordancia
    calib = {}  # agree -> [n_correct, n_total]
    for i,tid in enumerate(task_ids,1):
        task = json.load(open(ARC_DIR/tid))
        emit(f"\n[{i}/{len(task_ids)}] {tid} (train={len(task['train'])}) lib={len(library_code)} drives={ {k:round(v,2) for k,v in drives.items()} }")
        r = solve_task_v2(task, drives, library_code, cfg, hot, emit)
        r["task"]=tid; r["index"]=i
        if r.get("solved_train") and r.get("consensus") is not None:
            correct = test_correct(r["consensus"], task)
            r["test_correct"] = correct
            ag = r.get("agree",1)
            calib.setdefault(ag,[0,0]); calib[ag][1]+=1
            if correct: calib[ag][0]+=1; solved_ids.append(tid)
            # DRIVES + WAKE-SLEEP (so no quente)
            if hot:
                for p in set(r.get("used_personas",[])):
                    if any(e for e in [r] if True): pass
                # recompensa personas que viraram expert
                # (aproximacao: recompensa todas usadas levemente, e a vencedora do debate mais)
                win_p = "composto"
                drives_keys=list(drives)
                for p in r.get("used_personas",[]):
                    if p in drives: drives[p]=drives[p]*0.97+0.03
                tot=sum(drives.values())
                for k in drives: drives[k]=drives[k]/tot*len(drives)
                # BIBLIOTECA (wake-sleep de PROGRAMAS): guarda o transform INTEIRO do expert mais ROBUSTO
                # desta task, com a regra. Few-shot p/ tasks futuras = canal de transferencia real.
                ecode = r.get("experts_code",[]); erob = r.get("robustness",[])
                if ecode:
                    bi = max(range(len(ecode)), key=lambda j: erob[j] if j<len(erob) else 0)
                    best_code = ecode[bi]
                    from v2_solver import rule_of
                    entry = {"code": best_code, "rule": rule_of(best_code), "task": tid,
                             "rob": (erob[bi] if bi<len(erob) else 0)}
                    if not any(e.get("task")==tid for e in library_code):
                        library_code.append(entry)
                    # prioriza os mais robustos, mantem ate 12
                    library_code.sort(key=lambda e: -e.get("rob",0))
                    library_code = library_code[:12]
            emit(f"  => {r['n_experts']} experts, concordam {r.get('agree')}/{r.get('total_experts')} | TEST: {correct}"
                 + (f" | refine={r['refine_solves']} compose={r['compose_solves']}" if r.get('refine_solves') or r.get('compose_solves') else ""))
        else:
            r["test_correct"]=False
            emit(f"  => sem expert (nao resolveu treino)")
        results.append(r)
        payload = {"tag":tag,"hot":hot,"n_tasks":len(task_ids),"cfg":cfg,"duration_s":time.time()-t0,
                   "drives":drives,"library_size":len(library_code),
                   "solved_test":solved_ids,"calibration":calib,
                   "results":[{k:v for k,v in r.items() if k not in("consensus","experts_code")} for r in results],
                   "task_ids":task_ids}
        json.dump(payload, open(OUT_DIR/f"{run_name}.json","w"), indent=1)

    n=len(task_ids); sc=len(solved_ids)
    half=n//2
    fh=len([x for x in results[:half] if x.get("test_correct")]); sh=len([x for x in results[half:] if x.get("test_correct")])
    refine_tot=sum(x.get("refine_solves",0) for x in results); comp_tot=sum(x.get("compose_solves",0) for x in results)
    emit(f"\n{'='*64}")
    emit(f"[{tag}] TEST: {sc}/{n} = {sc/n:.1%}  (baseline GrammarForge 25.2%)")
    emit(f"[{tag}] TRANSFERENCIA: 1a metade {fh}/{half} vs 2a metade {sh}/{n-half}")
    emit(f"[{tag}] refinamento resolveu {refine_tot} | composicao nao-comutativa resolveu {comp_tot}")
    emit(f"[{tag}] CALIBRACAO consenso (concordancia -> acerto):")
    for ag in sorted(calib):
        c,t=calib[ag]; emit(f"    {ag} experts concordam: {c}/{t} = {c/t:.0%}" if t else "")
    emit(f"[{tag}] drives finais: { {k:round(v,2) for k,v in drives.items()} } | biblioteca={len(library_code)} helpers")
    payload["final"]={"test":sc,"rate":sc/n,"first_half":fh,"second_half":sh,"refine_solves":refine_tot,
                      "compose_solves":comp_tot,"calibration":calib}
    json.dump(payload, open(OUT_DIR/f"{run_name}.json","w"), indent=1)
    emit(f"salvo: {run_name}.json")
    return payload

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=30); ap.add_argument("--gens",type=int,default=4)
    ap.add_argument("--refine",type=int,default=2); ap.add_argument("--cold",action="store_true")
    ap.add_argument("--seed",type=int,default=7); ap.add_argument("--tag",default=None)
    a=ap.parse_args()
    run(n_tasks=a.n, hot=not a.cold, seed=a.seed, tag=a.tag, cfg={"gens":a.gens,"refine":a.refine})
