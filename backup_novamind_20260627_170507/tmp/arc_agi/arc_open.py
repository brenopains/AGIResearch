"""
arc_open.py — MOTOR ABERTO no ARC (PowerPlay/Bridge): a biblioteca de FAMILIAS cresce sozinha.
Stream de tasks. Nossa arquitetura tenta resolver com as familias existentes SEM LLM (reuso=compounding).
No FRACASSO, o LLM minta uma FAMILIA NOVA (um inducer parametrizado em Python: le os train pairs e devolve
um transform), verificada por execucao (reproduz train) e generalizada (gate no teste). A familia entra na
biblioteca e e REUSADA nas tasks seguintes -> by_library CRESCE, call_rate CAI = compounding aberto no ARC.
Honesto: bounded por cobertura (Solomonoff) — nao "passa", mas a curva de aquisicao aberta sobe sozinha.
"""
import sys, os, json, time, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent/"sci"))
from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent.parent/"sci"/".env")
import tem_arc2 as T
from collections import Counter

LIBP = pathlib.Path(__file__).parent/"runs"/"arc_open_lib.json"; LIBP.parent.mkdir(exist_ok=True)

# familias-semente (reutilizaveis, content-factored) + as mintadas pelo LLM (codigo persistido)
SEED = list(T.TEMPLATES)
MINTED = []   # [{name, code}] inducers em texto, compilados on-the-fly

SAFE = {"range":range,"len":len,"min":min,"max":max,"sorted":sorted,"set":set,"list":list,"dict":dict,
        "abs":abs,"sum":sum,"enumerate":enumerate,"zip":zip,"any":any,"all":all,"frozenset":frozenset,
        "Counter":Counter,"tuple":tuple,"int":int,"map":map,"filter":filter,"reversed":reversed}
def compile_inducer(code):
    # o codigo define `def inducer(task):` usando utilitarios de T (dims,bg_color,components,...)
    g={"__builtins__":SAFE, "dims":T.dims,"bg_color":T.bg_color,"components":T.components,
       "shape_sig":T.shape_sig,"gcopy":T.gcopy,"holes":T.holes}
    try:
        exec(code, g)
        return g.get("inducer")
    except Exception:
        return None

def all_families():
    fams=list(SEED)
    for m in MINTED:
        fn=compile_inducer(m["code"])
        if fn: fams.append((m["name"], fn))
    return fams

def solve_no_llm(task):
    for nm,ind in all_families():
        try: fn=ind(task)
        except: fn=None
        if fn and T.vtrain(fn,task):
            if T.vtest(fn,task): return nm
    return None

# ===== LLM minta FAMILIA NOVA (inducer) no fracasso =====
def grid_str(g): return "\n".join("".join(str(v) for v in r) for r in g)
def llm_mint_family(task):
    try:
        from shared.llm_client import ask
    except Exception:
        return None
    ex=[]
    for p in task["train"][:3]:
        ex.append("IN:\n"+grid_str(p["input"])+"\nOUT:\n"+grid_str(p["output"]))
    sys_p=("Voce escreve um INDUCER Python para uma familia de regra ARC CONTENT-FACTORED (sobre ESTRUTURA, "
      "nao cores especificas). Disponiveis: dims(g),bg_color(g),components(g,bg),shape_sig(o),holes(g,o,bg),gcopy(g). "
      "Defina `def inducer(task):` que LE task['train'] (lista de {input,output}, grids = listas de listas de int 0-9), "
      "INDUZ os parametros, e RETORNA uma funcao fn(grid)->grid (ou None se nao se aplica). Sem imports, sem I/O. "
      "Responda APENAS o codigo Python.")
    usr="Train pairs:\n"+"\n\n".join(ex)+"\n\nEscreva o inducer (so codigo):"
    try:
        code=ask([{"role":"user","content":usr}], system=sys_p, model=os.environ.get("ARC_MODEL","qwen3-coder-480b-a35b-instruct-turbo"), temperature=0.3, max_tokens=1100)
    except Exception:
        return None
    code=re.sub(r"```(?:python)?","",code).replace("```","").strip()
    if "def inducer" not in code: return None
    fn_ind=compile_inducer(code)
    if not fn_ind: return None
    # valida: reproduz train E generaliza no teste? (gate exato)
    try:
        fn=fn_ind(task)
        if fn and T.vtrain(fn,task):
            name=f"mint_{len(MINTED)}"; MINTED.append({"name":name,"code":code})
            return name, T.vtest(fn,task)
    except Exception:
        pass
    return None

def run(limit=40, use_llm=True):
    tasks=T.load_tasks(limit); n=len(tasks)
    by_lib=0; by_llm=0; calls=0; per=[]; calls_per=[]; fam_use=Counter()
    t0=time.time()
    for i,t in enumerate(tasks):
        nm=solve_no_llm(t)                 # NOSSA ARQUITETURA (reuso de familias, SEM LLM)
        if nm:
            by_lib+=1; fam_use[nm.split('>>')[0]]+=1; per.append(("lib",0)); calls_per.append(0)
            print(f"[{i+1}/{n}] {t['_id']} -> by_library: {nm}")
            continue
        if use_llm:                        # FRACASSO -> LLM minta familia nova
            calls+=1
            r=llm_mint_family(t)
            if r and r[1]:
                by_llm+=1; per.append(("llm",1)); calls_per.append(1)
                print(f"[{i+1}/{n}] {t['_id']} -> LLM mintou familia {r[0]} | TEST ok (lib={len(MINTED)} mintadas)")
            else:
                per.append((None,1)); calls_per.append(1)
                print(f"[{i+1}/{n}] {t['_id']} -> nao resolvido (LLM tentou)")
        else:
            per.append((None,0)); calls_per.append(0)
    th=max(1,n//3)
    libr=lambda a,b: sum(1 for k,_ in per[a:b] if k=='lib')/max(1,len(per[a:b]))
    callr=lambda a,b: sum(calls_per[a:b])/max(1,len(per[a:b]))
    sc=by_lib+by_llm
    print(f"\n{'='*64}")
    print(f"ARC-OPEN: {sc}/{n} = {100*sc/n:.1f}% | by_library(SEM LLM)={by_lib} by_llm={by_llm} | familias mintadas={len(MINTED)}")
    print(f">>> COMPOUNDING (by_library por terco): inicio {libr(0,th):.0%} meio {libr(th,2*th):.0%} fim {libr(2*th,n):.0%}  (CRESCEU?)")
    print(f">>> ECONOMIA (LLM calls/task por terco): inicio {callr(0,th):.1f} meio {callr(th,2*th):.1f} fim {callr(2*th,n):.1f}  (CAIU?)")
    print(f">>> familias REUSADAS (>1 task): {sum(1 for c in fam_use.values() if c>1)} | {dict(fam_use.most_common(6))}")
    print(f"elapsed {time.time()-t0:.0f}s")
    json.dump({"minted":MINTED}, open(LIBP,"w"), indent=1)

if __name__=="__main__":
    limit=int(sys.argv[1]) if len(sys.argv)>1 else 40
    run(limit, use_llm=("nollm" not in sys.argv))
