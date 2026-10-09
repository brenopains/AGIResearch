"""
ARC-NOVAMIND v3 — A PONTE (ideia do usuario no quadro branco).

Tese: o LLM e o grounding/world-model (a "sacola que pensa"). O crescimento real vem do LOOP:
  AGI testa -> erra -> manda o TRACO pro LLM -> LLM analisa memoria versionada + sintetiza uma
  PRIMITIVA NOVA (codigo executavel categorizado) -> a AGI passa a usar essa primitiva SOZINHA,
  sem LLM. Cada primitiva = "neuronio novo" que se FORTALECE com reuso. O LLM so dispara no
  FRACASSO. Conforme a biblioteca cresce, o LLM e chamado cada vez MENOS = economia = generatividade.

SINAL DE AGI (medido): a fracao de tasks resolvida SO pela biblioteca (sem LLM) CRESCE ao longo
do stream? As chamadas de LLM por task CAEM? Se sim -> aquisicao aberta de habilidade transferivel.

Tudo verificado por execucao. Biblioteca PERSISTENTE em disco (memoria da AGI que sobrevive).
"""
import sys, os, json, time, re, random, pathlib, hashlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "sci"))
from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent.parent / "sci" / ".env")
from solver import (PERSONAS, SAFE_BUILTINS, grid_to_str, extract_code, compile_transform,
                    run_with_timeout, normalize_grid, verify_on_train, apply_to_test,
                    test_correct, ARC_DIR, OUT_DIR)
from v2_solver import llm_code, failing_examples_str, gen_prompt, program_robustness, rule_of
from shared.llm_client import ask
import numpy as np

LIB_PATH = OUT_DIR / "agi_library.json"   # memoria persistente da AGI

# ===== MEMORIA ASSOCIATIVA: FINGERPRINT PERCEPTUAL (visual/espacial/estrutural) =====
# Mesmo principio do world-model multidimensional da colonia: cada primitiva guarda o vetor
# perceptual da situacao onde "disparou"; busca por SIMILARIDADE (memoria endereçavel por conteudo).
_FP_SCALE = np.array([30,30,30,30, 5,5,5, 1,1, 10,10, 1,1,1,1,1, 12, 6], dtype=float)
def fingerprint(task):
    feats=[]
    for ex in task["train"][:4]:
        gi=np.array(ex["input"]); go=np.array(ex["output"])
        hi,wi=gi.shape; ho,wo=go.shape
        ci=set(gi.flatten().tolist()); co=set(go.flatten().tolist())
        nobj=0
        try:  # nº de objetos (componentes nao-zero) — barato, sem scipy
            seen=np.zeros_like(gi,dtype=bool)
            for r in range(hi):
                for c in range(wi):
                    if gi[r,c]!=0 and not seen[r,c]:
                        nobj+=1; stack=[(r,c)]; seen[r,c]=True
                        while stack:
                            y,x=stack.pop()
                            for dy,dx in((1,0),(-1,0),(0,1),(0,-1)):
                                ny,nx=y+dy,x+dx
                                if 0<=ny<hi and 0<=nx<wi and gi[ny,nx]!=0 and not seen[ny,nx]:
                                    seen[ny,nx]=True; stack.append((ny,nx))
        except Exception: nobj=0
        f=[hi,wi,ho,wo,
           (ho*wo)/(hi*wi+1e-9), wi/(hi+1e-9), wo/(ho+1e-9),
           float((gi!=0).mean()), float((go!=0).mean()),
           len(ci), len(co),
           1.0 if ci==co else 0.0, 1.0 if (co-ci) else 0.0,
           1.0 if np.array_equal(gi,np.fliplr(gi)) else 0.0,
           1.0 if np.array_equal(gi,np.flipud(gi)) else 0.0,
           1.0 if (hi==wi and np.array_equal(gi,np.rot90(gi))) else 0.0,
           float(len(np.unique(gi))), float(min(nobj,6))]
        feats.append(f)
    return (np.mean(feats,axis=0)).tolist()

def fp_similarity(fa, fb):
    a=np.array(fa)/_FP_SCALE; b=np.array(fb)/_FP_SCALE
    d=np.linalg.norm(a-b)
    return 1.0/(1.0+d)   # 1=identico, ->0 distante

def associative_order(task, lib):
    """recupera primitivas por SIMILARIDADE perceptual a task (memoria associativa), nao por forca cega."""
    fq=fingerprint(task)
    scored=[]
    for p in lib.prims:
        fp=p.get("fingerprint")
        sim = fp_similarity(fq,fp) if fp else 0.3
        score = sim*(1.0+0.15*p.get("strength",1)) + 0.05*p.get("rob",0)   # similaridade + reforço
        scored.append((score, sim, p))
    scored.sort(key=lambda x:-x[0])
    return scored   # [(score, sim, prim), ...]

# ===== ASSINATURA DE TIPO (p/ agrupar tasks similares no stream, SEM espiar solucao) =====
def task_signature(task):
    """assinatura barata do TIPO de transformacao a partir dos pares de treino (in->out)."""
    ex = task["train"][0]; gi, go = ex["input"], ex["output"]
    hi, wi = len(gi), len(gi[0]); ho, wo = len(go), len(go[0])
    # relacao de tamanho
    if (hi,wi)==(ho,wo): size="same"
    elif ho%hi==0 and wo%wi==0: size=f"up{ho//hi}x{wo//wi}"
    elif hi%ho==0 and wi%wo==0: size=f"down{hi//ho}x{wi//wo}"
    else: size="diff"
    ci = set(c for r in gi for c in r); co = set(c for r in go for c in r)
    colors = "samecolor" if ci==co else ("newcolor" if co-ci else "fewercolor")
    # simetria do output igual input?
    import numpy as _np
    a=_np.array(gi)
    sym = "sym" if (gi==go and False) else ("rot180" if (ho,wo)==(hi,wi) and _np.array_equal(_np.rot90(a,2), _np.array(go)) else "")
    return f"{size}|{colors}|{len(ci)}->{len(co)}c|{sym}"

def grouped_stream(task_ids):
    """ordena o stream agrupando tasks de mesma assinatura adjacentes (transferencia provavel)."""
    sigs = {}
    for tid in task_ids:
        try:
            t=json.load(open(ARC_DIR/tid)); sigs[tid]=task_signature(t)
        except Exception: sigs[tid]="?"
    # agrupa por assinatura, grupos maiores primeiro
    from collections import defaultdict
    groups=defaultdict(list)
    for tid in task_ids: groups[sigs[tid]].append(tid)
    ordered=[]
    for sig,ts in sorted(groups.items(), key=lambda kv:-len(kv[1])):
        ordered.extend(ts)
    return ordered, sigs

# ================= BIBLIOTECA DE PRIMITIVAS (memoria da AGI) =================
class Library:
    def __init__(self):
        self.prims = []   # cada: {name, category, code, strength, version, source_task, rob}
        self._compiled = {}

    def load(self):
        if LIB_PATH.exists():
            self.prims = json.load(open(LIB_PATH))
        return self

    def save(self):
        json.dump(self.prims, open(LIB_PATH,"w"), indent=1)

    def reset(self):
        self.prims = []; self._compiled = {}

    def seed(self):
        """semeia a DSL GERAL (instintos motores da AGI): operacoes genericas por construcao.
        fingerprint=None (neutro) — a primitiva aprende sua 'assinatura de uso' ao resolver."""
        try:
            from seed_primitives import SEED
        except Exception:
            return 0
        for cat, code in SEED:
            self.prims.append({"name":f"{cat}_seed", "category":cat, "code":code,
                               "strength":1, "version":0.0, "source_task":"SEED", "rob":0.0,
                               "fingerprint":None, "seed":True})
        return len(SEED)

    def fn_of(self, prim):
        h = hashlib.md5(prim["code"].encode()).hexdigest()
        if h not in self._compiled:
            fn, err = compile_transform(prim["code"])
            self._compiled[h] = fn
        return self._compiled[h]

    def add(self, code, category, source_task, rob=0.0, fp=None):
        # versionamento: se ja existe primitiva da mesma CATEGORIA, nova versao
        same = [p for p in self.prims if p["category"] == category]
        version = (max((p["version"] for p in same), default=0.0) + 1.0) if same else 1.0
        name = f"{category}_v{version:.0f}"
        # dedup exato de codigo
        if any(p["code"].strip()==code.strip() for p in self.prims): return None
        prim = {"name":name, "category":category, "code":code, "strength":1,
                "version":version, "source_task":source_task, "rob":rob, "fingerprint":fp}
        self.prims.append(prim)
        return prim

    def strengthen(self, prim):
        prim["strength"] = prim.get("strength",1) + 1

    def by_strength(self):
        return sorted(self.prims, key=lambda p: -(p.get("strength",1) + p.get("rob",0)))

# ================= MOTOR DA AGI (resolve SEM LLM, compondo primitivas) =================
def agi_engine_solve(task, lib, max_pairs=60):
    """Resolve SO com a biblioteca (sem LLM) usando MEMORIA ASSOCIATIVA: recupera primitivas por
    SIMILARIDADE perceptual a task (nao por forca cega) -> tenta as mais associadas solo + composicao
    nao-comutativa. Verificado por execucao. Retorna (preds, used_prims, how)."""
    if not lib.prims: return None, [], None
    ranked = associative_order(task, lib)   # [(score, sim, prim)] ordenado por similaridade perceptual
    prims = [p for _,_,p in ranked]
    # 1) primitivas SOLO (na ordem associativa — as mais parecidas primeiro)
    for p in prims:
        fn = lib.fn_of(p)
        if fn is None: continue
        solved, ok, tot = verify_on_train(fn, task)
        if solved:
            return apply_to_test(fn, task), [p], "solo"
    # 2) COMPOSICAO NAO-COMUTATIVA dos top-K associativos (ordem importa)
    top = prims[:12]
    tried = 0
    for pa in top:
        fa = lib.fn_of(pa)
        if fa is None: continue
        for pb in top:
            if pb is pa: continue
            fb = lib.fn_of(pb)
            if fb is None: continue
            def comp(g, fa=fa, fb=fb):
                r = fa([list(r) for r in g])
                if hasattr(r,"tolist"): r=r.tolist()
                return fb([list(r) for r in r])
            solved, ok, tot = verify_on_train(comp, task)
            tried += 1
            if solved:
                return apply_to_test(comp, task), [pa,pb], "compose"
            if tried >= max_pairs: break
        if tried >= max_pairs: break
    return None, [], None

# ================= CATEGORIZADOR (LLM nomeia a primitiva nova = "neuronio") =================
CAT_SYSTEM = """Voce CATEGORIZA uma transformacao de grid ARC numa palavra-chave curta (snake_case, 1-3 palavras)
que capture a OPERACAO GERAL (ex: rotate_180, recolor_by_count, extract_largest_object, tile_grid,
gravity_down, mirror_complete, flood_fill_holes). Responda APENAS a categoria, nada mais."""
def _heuristic_category(code):
    """deriva categoria do codigo+regra sem LLM (robusto a thinking vazio)."""
    t = (code + " " + rule_of(code)).lower()
    rules = [
        ("rotate", ["rot90","rotate","np.rot","transpose",".t)"]),
        ("flip_mirror", ["fliplr","flipud","flip","mirror","espelh","[::-1]"]),
        ("recolor", ["recolor","replace","color map","substitu","palette"]),
        ("tile_scale", ["tile","repeat","kron","scal","resize","amplia","zoom"]),
        ("crop_extract", ["crop","extract","slice","bbox","subgrid","recort","largest","maior"]),
        ("fill", ["fill","flood","preench","holes","buraco"]),
        ("objects", ["label","connected","component","object","objeto","region"]),
        ("gravity_move", ["gravity","shift","roll","move","queda","desloc"]),
        ("count", ["count","frequen","contagem","bincount","conta"]),
        ("symmetry", ["symmet","simetr","complete","reflect"]),
    ]
    for cat, keys in rules:
        if any(k in t for k in keys): return cat   # substring literal (sem regex)
    return "transform"

def categorize(code):
    rule = rule_of(code)
    try:
        raw = ask([{"role":"user","content":f"Codigo:\n{code[:600]}\nRegra: {rule}\n\nCategoria (snake_case):"}],
                  system=CAT_SYSTEM, temperature=0.0, max_tokens=120, cache=True)
        w = re.sub(r"[^a-z0-9_]","", raw.strip().lower().split()[0])[:30] if raw.strip() else ""
        if w and w not in ("python","def","the","a","categoria"): return w
    except Exception:
        pass
    return _heuristic_category(code)   # fallback robusto sem LLM

# ================= LLM-GENERALIZACAO (reescreve solucao especifica -> primitiva GERAL) =================
GEN_SYSTEM = """Voce GENERALIZA uma solucao ARC que funciona. Recebe um transform(g) que resolve uma task
especifica e os pares de treino. Reescreva-o na forma MAIS GERAL possivel: DETECTE os parametros a partir
do proprio grid (qual rotacao/cor/escala/objeto), NAO use constantes decoradas da task. A versao geral deve
funcionar tambem em OUTRAS tasks da mesma familia. DEVE continuar reproduzindo EXATAMENTE os mesmos OUTPUTs.
Responda APENAS o bloco ```python com a funcao transform(g) generalizada."""

def generalize_primitive(code, task, emit):
    """pede ao LLM a versao geral; so adota se AINDA passar no treino (senao mantem a especifica)."""
    if os.environ.get("V3_LLM_GENERALIZE") != "1":
        return code
    pairs = ""
    for ex in task["train"][:3]:
        pairs += f"\nIN:\n{grid_to_str(ex['input'])}\nOUT:\n{grid_to_str(ex['output'])}\n"
    user = f"SOLUCAO ESPECIFICA QUE FUNCIONA:\n```python\n{code}\n```\n\nPARES DE TREINO:{pairs}\n\nReescreva GERAL (detecte params do grid). Mesmos OUTPUTs. APENAS bloco python."
    try:
        raw = ask([{"role":"user","content":user}], system=GEN_SYSTEM, temperature=0.2, max_tokens=1200)
        gcode = extract_code(raw)
        gfn, err = compile_transform(gcode)
        if gfn is None: return code
        solved, ok, tot = verify_on_train(gfn, task)
        if solved:
            emit(f"    [generalizada: passou no treino, versao geral adotada]")
            return gcode
    except Exception: pass
    return code

# ================= LOOP DO LLM (so no fracasso: minta primitiva nova) =================
def llm_growth(task, lib, gens, refine, emit):
    """LLM analisa e sintetiza programa novo. Se resolve treino -> vira primitiva categorizada.
    Retorna (preds, n_llm_calls, new_prim) ou (None, calls, None)."""
    calls = 0
    # contexto: oferece primitivas existentes (a AGI ja 'sabe' isso) p/ o LLM reusar/compor
    lib_examples = [{"code":p["code"], "rule":p.get("category","")} for p in lib.by_strength()[:3]]
    best_partial = None
    for gi in range(gens):
        persona = random.choice(list(PERSONAS))
        fn, code, err = llm_code(task, persona, lib_examples, None, 0.2+0.2*gi); calls += 1
        for rr in range(refine):
            if fn is None: break
            solved, ok, tot = verify_on_train(fn, task)
            if solved: break
            fb = failing_examples_str(fn, task)
            fn2, code2, err2 = llm_code(task, persona, lib_examples, fb, 0.25); calls += 1
            if fn2: fn, code = fn2, code2
        if fn is None: continue
        solved, ok, tot = verify_on_train(fn, task)
        emit(f"    [LLM/{persona}] treino {ok}/{tot}" + ("  NOVA PRIMITIVA" if solved else ""))
        if solved:
            preds = apply_to_test(fn, task)   # captura predicao da versao que comprovadamente resolve
            gcode = generalize_primitive(code, task, emit); calls += 1   # LLM-generalizacao
            rob = program_robustness(fn, task)
            cat = categorize(gcode); calls += 1
            fp = fingerprint(task)   # memoria sensorial: onde a primitiva nasceu
            prim = lib.add(gcode, cat, task.get("_id","?"), rob, fp=fp)
            return preds, calls, prim
    return None, calls, None

# ================= ORQUESTRADOR =================
def run(n_tasks=24, seed=11, gens=3, refine=1, fresh_lib=True, tag="v3", task_ids=None, emit=None):
    if emit is None:
        emit = lambda s: print(s, flush=True)
    random.seed(seed)
    allt = sorted(p.name for p in ARC_DIR.glob("*.json"))
    if task_ids is None:
        random.Random(seed).shuffle(allt); task_ids = allt[:n_tasks]
    # AGRUPAMENTO opcional: tasks de mesmo tipo adjacentes (transferencia mais provavel)
    if os.environ.get("V3_GROUPED") == "1":
        task_ids, _sigs = grouped_stream(task_ids)
        emit(f"[stream AGRUPADO por assinatura de tipo]")
    lib = Library()
    if not fresh_lib: lib.load()
    else:
        lib.reset()
        if os.environ.get("V3_SEED","1")=="1":
            ns=lib.seed(); emit(f"[DSL semente: {ns} primitivas gerais carregadas]")
    results = []; t0=time.time(); run_name=f"arc_{tag}_{int(t0)}"
    by_lib=0; by_llm=0; none_=0; llm_calls_total=0
    emit(f"=== ARC-NOVAMIND v3 (A PONTE) | {len(task_ids)} tasks | lib inicial={len(lib.prims)} ===")
    for i,tid in enumerate(task_ids,1):
        task = json.load(open(ARC_DIR/tid)); task["_id"]=tid
        emit(f"\n[{i}/{len(task_ids)}] {tid} | biblioteca={len(lib.prims)} primitivas")
        # FASE 1: motor da AGI (sem LLM)
        preds, used, how = agi_engine_solve(task, lib)
        if preds is not None:
            correct = test_correct(preds, task)
            fpq = fingerprint(task)
            for p in used:
                lib.strengthen(p)
                # primitiva-semente aprende sua ASSINATURA DE USO (memoria associativa) ao resolver
                if p.get("fingerprint") is None: p["fingerprint"] = fpq
            by_lib += 1
            emit(f"  >> AGI resolveu SEM LLM ({how}, prims: {[p['name'] for p in used]}) | TEST: {correct}")
            results.append({"task":tid,"solved_by":"library","how":how,"prims":[p['name'] for p in used],
                            "test_correct":correct,"llm_calls":0,"lib_size":len(lib.prims)})
        else:
            # FASE 2: LLM cresce (so no fracasso)
            preds, calls, prim = llm_growth(task, lib, gens, refine, emit)
            llm_calls_total += calls
            if preds is not None:
                correct = test_correct(preds, task); by_llm += 1
                emit(f"  >> LLM mintou primitiva '{prim['name'] if prim else '?'}' ({calls} chamadas) | TEST: {correct}")
                results.append({"task":tid,"solved_by":"llm","new_prim":prim['name'] if prim else None,
                                "test_correct":correct,"llm_calls":calls,"lib_size":len(lib.prims)})
            else:
                none_ += 1
                emit(f"  >> nao resolvido ({calls} chamadas LLM)")
                results.append({"task":tid,"solved_by":"none","test_correct":False,"llm_calls":calls,"lib_size":len(lib.prims)})
        lib.save()
        # autosave de metricas
        n=len(results)
        payload={"tag":tag,"n_tasks":len(task_ids),"duration_s":time.time()-t0,
                 "by_library":by_lib,"by_llm":by_llm,"unsolved":none_,
                 "llm_calls_total":llm_calls_total,"library":lib.prims,"results":results,"task_ids":task_ids}
        json.dump(payload, open(OUT_DIR/f"{run_name}.json","w"), indent=1)

    # ===== METRICAS DE AGI =====
    n=len(task_ids); sc=sum(1 for r in results if r.get("test_correct"))
    third=max(1,n//3)
    def librate(rs): return sum(1 for r in rs if r["solved_by"]=="library")/max(1,len(rs))
    def callrate(rs): return sum(r["llm_calls"] for r in rs)/max(1,len(rs))
    t1,t2,t3 = results[:third], results[third:2*third], results[2*third:]
    emit(f"\n{'='*64}")
    emit(f"[{tag}] TEST correto: {sc}/{n} = {sc/n:.1%} (baseline 25.2%)")
    emit(f"[{tag}] resolvido por: BIBLIOTECA(sem LLM)={by_lib} | LLM={by_llm} | nao={none_}")
    emit(f"[{tag}] >>> SINAL AGI — fracao resolvida SEM LLM por terco do stream:")
    emit(f"      inicio {librate(t1):.0%}  meio {librate(t2):.0%}  fim {librate(t3):.0%}  (CRESCEU? = aquisicao de habilidade)")
    emit(f"[{tag}] >>> ECONOMIA — chamadas LLM/task por terco:")
    emit(f"      inicio {callrate(t1):.1f}  meio {callrate(t2):.1f}  fim {callrate(t3):.1f}  (CAIU? = generatividade)")
    emit(f"[{tag}] biblioteca final: {len(lib.prims)} primitivas | categorias: {sorted(set(p['category'] for p in lib.prims))[:12]}")
    payload["final"]={"test":sc,"rate":sc/n,"by_library":by_lib,"by_llm":by_llm,"unsolved":none_,
                      "lib_rate_thirds":[librate(t1),librate(t2),librate(t3)],
                      "call_rate_thirds":[callrate(t1),callrate(t2),callrate(t3)],
                      "n_primitives":len(lib.prims)}
    json.dump(payload, open(OUT_DIR/f"{run_name}.json","w"), indent=1)
    emit(f"salvo: {run_name}.json | biblioteca persistida em agi_library.json")
    return payload

if __name__ == "__main__":
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=24); ap.add_argument("--seed",type=int,default=11)
    ap.add_argument("--gens",type=int,default=3); ap.add_argument("--refine",type=int,default=1)
    ap.add_argument("--keep-lib",action="store_true",help="continua biblioteca existente (memoria persistente)")
    ap.add_argument("--tag",default="v3")
    a=ap.parse_args()
    run(n_tasks=a.n, seed=a.seed, gens=a.gens, refine=a.refine, fresh_lib=not a.keep_lib, tag=a.tag)
