"""
H114 — Receita LLM COMPLETA: Qwen2.5-0.5B-Instruct escreve codigo, eu executo e verifico (exato)
=================================================================================================
A unica fonte de sinal GENUINAMENTE EXTERNO ao nosso fecho disponivel localmente: um LLM real
(Qwen2.5-0.5B-Instruct, no cache D:) que absorveu conhecimento humano de codigo/raciocinio.
Pipeline (estado-da-arte de ARC+LLM): para cada tarefa do residuo, monta prompt com os exemplos
input->output; pede uma funcao Python transform(g) (numpy); amostra k completacoes; executa em
namespace restrito; GATE EXATO nos demos; se passa, aplica no teste e verifica. NOVAS vs teto.

Honesto: 0.5B e o MENOR LLM "de verdade" — fraco p/ ARC (fronteira usa modelos enormes), entao
espere modesto. Mas e o sinal EXTERNO real (fora do nosso fecho simbolico) — o lever que de fato
tem teto maior. Mede com dado.
"""
import os, sys, json, glob, time, re, numpy as np
HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,"arc_data","data","training")
os.environ.setdefault("HF_HOME","D:\\hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch
MAXIN=1100   # teto de tokens do prompt: tarefas maiores nao cabem na 3050/0.5B (pulamos, honesto)
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DEV="cuda" if torch.cuda.is_available() else "cpu"

def load():
    tok=AutoTokenizer.from_pretrained(MODEL)
    m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16 if DEV=="cuda" else torch.float32).to(DEV).eval()
    return tok,m

def grid_str(g): return "\n".join(" ".join(str(int(v)) for v in row) for row in g)

def make_prompt(train):
    ex="\n\n".join([f"Example {i+1} input:\n{grid_str(a)}\noutput:\n{grid_str(b)}" for i,(a,b) in enumerate(train)])
    return ("You are an expert solver of ARC grid puzzles. Each grid is a 2D array of integers 0-9.\n"
            "Find the transformation rule from input to output, then write ONE Python function:\n"
            "def transform(g):  # g is a numpy int array, return the output numpy int array\n"
            "Use numpy as np. Output ONLY a ```python code block with the function, no prose.\n\n"+ex)

def extract_code(text):
    m=re.search(r"```(?:python)?\s*(.*?)```",text,re.S)
    code=m.group(1) if m else text
    if "def transform" not in code: return None
    return code

SAFE={"np":np,"numpy":np,"abs":abs,"min":min,"max":max,"range":range,"len":len,"sorted":sorted,
      "list":list,"set":set,"dict":dict,"sum":sum,"enumerate":enumerate,"zip":zip,"int":int,"float":float}
def run_code(code, g):
    ns={"__builtins__":{k:v for k,v in __builtins__.__dict__.items() if k in {'range','len','min','max','abs','enumerate','zip','sorted','list','set','dict','sum','int','float','tuple','map','filter','bool','all','any','print','__import__'}}}
    ns.update(SAFE)
    try:
        exec(code,ns)
        fn=ns.get("transform")
        if fn is None: return None
        r=fn(np.array(g))
        r=np.array(r)
        if r.ndim!=2: return None
        return r.astype(np.int64)
    except Exception: return None

def solve(tok,m,train,tests,k,maxtok):
    prompt=make_prompt(train)
    msgs=[{"role":"user","content":prompt}]
    text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    ids=tok(text,return_tensors="pt").to(DEV)
    if ids["input_ids"].shape[1]>MAXIN: return "SKIP"   # grade grande demais p/ 4GB
    for _ in range(k):
        try:
            with torch.no_grad():
                out=m.generate(**ids,max_new_tokens=maxtok,do_sample=True,temperature=0.7,top_p=0.9,pad_token_id=tok.eos_token_id)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); return "SKIP"
        gen=tok.decode(out[0][ids["input_ids"].shape[1]:],skip_special_tokens=True)
        code=extract_code(gen)
        if code is None: continue
        ok=True
        for a,b in train:
            r=run_code(code,a)
            if r is None or r.shape!=b.shape or not np.array_equal(r,b): ok=False; break
        if ok:
            outs=[run_code(code,xi) for xi,_ in tests]
            if all(r is not None and r.shape==yo.shape and np.array_equal(r,yo) for r,(_,yo) in zip(outs,tests)):
                return code
    return None

def main():
    t0=time.time()
    limit=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--limit=')),'0'))
    k=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--k=')),'3'))
    maxtok=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--maxtok=')),'320'))
    ceiling=set(json.load(open(os.path.join(HERE,"arc_unified_final.json"))))
    print(f"H114 carregando {MODEL} em {DEV}..."); sys.stdout.flush()
    tok,m=load()
    vram=torch.cuda.max_memory_allocated()/1e6 if DEV=='cuda' else 0
    print(f"  carregado | VRAM {vram:.0f}MB | teto={len(ceiling)} | k={k}"); sys.stdout.flush()
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    residue=[f for f in files if os.path.basename(f) not in ceiling]
    if limit: residue=residue[:limit]
    novas={}; att=0; skipped=0; solpath=os.path.join(HERE,"h114_llm_solutions.json")
    for fp in residue:
        name=os.path.basename(fp); t=json.load(open(fp))
        train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        att+=1
        code=solve(tok,m,train,tests,k,maxtok)
        if code=="SKIP": skipped+=1; code=None
        if code is not None:
            novas[name]=code; print(f"  +SOLVE {name}"); sys.stdout.flush()
            json.dump(novas,open(solpath,"w"))   # salva INCREMENTAL (nao perde em crash)
            alln=ceiling|set(novas); json.dump(sorted(alln),open(os.path.join(HERE,"arc_unified_final.json"),"w"))
        if DEV=="cuda": torch.cuda.empty_cache()
        if att%20==0: print(f"  ... {att}/{len(residue)} (NOVAS={len(novas)}, skip={skipped}) {time.time()-t0:.0f}s"); sys.stdout.flush()
    alln=ceiling|set(novas)
    if novas: json.dump(sorted(alln),open(os.path.join(HERE,"arc_unified_final.json"),"w"))
    json.dump({n:c for n,c in novas.items()},open(os.path.join(HERE,"h114_llm_solutions.json"),"w"))
    print("="*74)
    print(f"H114 LLM codegen (Qwen2.5-0.5B) | tentou {att} do residuo | k={k}")
    print(f"  NOVAS: {len(novas)} {sorted(novas)[:14]}")
    print(f"  novo TETO: {len(alln)}/400 = {100*len(alln)/400:.1f}%")
    print(f"  tempo {time.time()-t0:.0f}s | VRAM {torch.cuda.max_memory_allocated()/1e6 if DEV=='cuda' else 0:.0f}MB")
    print("  Honesto: sinal EXTERNO real (conhecimento humano no LLM, fora do nosso fecho). 0.5B e fraco; teto maior viria de modelo maior.")

if __name__=="__main__": main()
