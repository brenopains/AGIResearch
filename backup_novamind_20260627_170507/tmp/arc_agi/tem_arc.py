"""
tem_arc.py — Ponte ARC fundamentada na Tolman-Eichenbaum Machine (Whittington 2020, Cell).
Principio: FATORAR estrutura (papel relacional dos objetos) de conteudo (cor/posicao). Uma regra induzida
sobre a ESTRUTURA (rank de tamanho, forma, relacao) generaliza para o CONTEUDO novo do teste (o "remapping" da TEM).
Ataca a parede de LARGURA por outro angulo: uma regra content-factored cobre uma FAMILIA inteira, nao 1 instancia.
Honesto: versao BOUNDED (object-centric + templates de regra estrutural), nao a TEM plena (modelo generativo treinado).
Roda nas 400 tasks de treino reais; mede solves (gate exato no teste held-out). Sem LLM.
"""
import json, glob, os, sys, pathlib
from collections import Counter, defaultdict

DATA = None
for cand in [pathlib.Path(__file__).parent.parent/"arc_data"/"data"/"training",
             pathlib.Path(__file__).parent.parent/"arc_data"/"data"/"evaluation"]:
    if cand.exists(): DATA = cand; break

def load_tasks(limit=None):
    files = sorted(glob.glob(str(DATA/"*.json")))
    if limit: files = files[:limit]
    tasks=[]
    for f in files:
        try:
            with open(f) as fh: t=json.load(fh)
            t["_id"]=os.path.basename(f)[:-5]; tasks.append(t)
        except: pass
    return tasks

# ====== utilidades de grid ======
def dims(g): return len(g), len(g[0]) if g else 0
def bg_color(g):  # cor de fundo = mais frequente (geralmente 0)
    c=Counter(v for row in g for v in row); return c.most_common(1)[0][0] if c else 0

def components(g, bg):
    """componentes conexos 4-conn de celulas != bg (cada objeto = celulas da MESMA cor conexas)."""
    H,W=dims(g); seen=[[False]*W for _ in range(H)]; objs=[]
    for i in range(H):
        for j in range(W):
            if g[i][j]==bg or seen[i][j]: continue
            col=g[i][j]; stack=[(i,j)]; cells=[]
            seen[i][j]=True
            while stack:
                y,x=stack.pop(); cells.append((y,x))
                for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny,nx=y+dy,x+dx
                    if 0<=ny<H and 0<=nx<W and not seen[ny][nx] and g[ny][nx]==col:
                        seen[ny][nx]=True; stack.append((ny,nx))
            ys=[c[0] for c in cells]; xs=[c[1] for c in cells]
            objs.append({"cells":cells,"color":col,"size":len(cells),
                         "top":min(ys),"left":min(xs),"bot":max(ys),"right":max(xs),
                         "cy":sum(ys)/len(ys),"cx":sum(xs)/len(xs)})
    return objs

def shape_sig(o):
    """assinatura de FORMA normalizada (independente de posicao e cor) -> papel estrutural."""
    cells=o["cells"]; t=o["top"]; l=o["left"]
    return frozenset((y-t,x-l) for y,x in cells)

def grid_copy(g): return [list(r) for r in g]

# ====== TEMPLATES DE REGRA CONTENT-FACTORED (indexados por ESTRUTURA) ======
# cada template: induz parametros dos train pairs; retorna fn(grid)->grid ou None se nao se aplica consistentemente.

def induce_recolor_by_size_rank(task):
    """cada objeto e recolorido por uma funcao do seu RANK de tamanho (estrutura), nao da cor original."""
    mapping={}  # rank -> nova cor
    for pair in task["train"]:
        gi,go=pair["input"],pair["output"]
        if dims(gi)!=dims(go): return None
        bg=bg_color(gi); oi=components(gi,bg)
        if not oi: return None
        ranks=sorted(range(len(oi)), key=lambda k:-oi[k]["size"])
        for rk,idx in enumerate(ranks):
            o=oi[idx]
            outcols={go[y][x] for y,x in o["cells"]}
            if len(outcols)!=1: return None
            oc=next(iter(outcols))
            if rk in mapping and mapping[rk]!=oc: return None
            mapping[rk]=oc
    def fn(g):
        bg=bg_color(g); o=components(g,bg); out=grid_copy(g)
        ranks=sorted(range(len(o)), key=lambda k:-o[k]["size"])
        for rk,idx in enumerate(ranks):
            nc=mapping.get(rk)
            if nc is None: continue
            for y,x in o[idx]["cells"]: out[y][x]=nc
        return out
    return fn

def induce_recolor_by_shape(task):
    """objetos com a MESMA forma recebem a MESMA cor (regra sobre estrutura-forma)."""
    mapping={}
    for pair in task["train"]:
        gi,go=pair["input"],pair["output"]
        if dims(gi)!=dims(go): return None
        bg=bg_color(gi); oi=components(gi,bg)
        for o in oi:
            sig=shape_sig(o)
            outcols={go[y][x] for y,x in o["cells"]}
            if len(outcols)!=1: return None
            oc=next(iter(outcols))
            if sig in mapping and mapping[sig]!=oc: return None
            mapping[sig]=oc
    def fn(g):
        bg=bg_color(g); o=components(g,bg); out=grid_copy(g)
        for ob in o:
            nc=mapping.get(shape_sig(ob))
            if nc is None: continue
            for y,x in ob["cells"]: out[y][x]=nc
        return out
    return fn

def induce_keep_extreme(task, which):
    """mantem so o objeto de tamanho EXTREMO (maior/menor) — selecao estrutural; zera o resto."""
    for pair in task["train"]:
        gi,go=pair["input"],pair["output"]
        if dims(gi)!=dims(go): return None
    def fn(g):
        bg=bg_color(g); o=components(g,bg); out=[[bg]*dims(g)[1] for _ in range(dims(g)[0])]
        if not o: return grid_copy(g)
        tgt = max(o,key=lambda z:z["size"]) if which=="largest" else min(o,key=lambda z:z["size"])
        for y,x in tgt["cells"]: out[y][x]=tgt["color"]
        return out
    return fn

def induce_move_by_vector(task):
    """todos os objetos transladados por um VETOR constante (estrutura preservada, posicao = conteudo)."""
    vec=None
    for pair in task["train"]:
        gi,go=pair["input"],pair["output"]
        if dims(gi)!=dims(go): return None
        bg=bg_color(gi); oi=components(gi,bg); oo=components(go,bg)
        if len(oi)!=len(oo) or not oi: return None
        # parear por forma+cor
        oi_s=sorted(oi,key=lambda z:(z["size"],z["top"],z["left"]))
        oo_s=sorted(oo,key=lambda z:(z["size"],z["top"],z["left"]))
        for a,b in zip(oi_s,oo_s):
            if shape_sig(a)!=shape_sig(b) or a["color"]!=b["color"]: return None
            v=(b["top"]-a["top"], b["left"]-a["left"])
            if vec is None: vec=v
            elif vec!=v: return None
    if vec is None or vec==(0,0): return None
    def fn(g):
        bg=bg_color(g); o=components(g,bg); H,W=dims(g); out=[[bg]*W for _ in range(H)]
        for ob in o:
            for y,x in ob["cells"]:
                ny,nx=y+vec[0],x+vec[1]
                if 0<=ny<H and 0<=nx<W: out[ny][nx]=ob["color"]
        return out
    return fn

TEMPLATES = [
    ("recolor_by_size_rank", induce_recolor_by_size_rank),
    ("recolor_by_shape", induce_recolor_by_shape),
    ("keep_largest", lambda t: induce_keep_extreme(t,"largest")),
    ("keep_smallest", lambda t: induce_keep_extreme(t,"smallest")),
    ("move_by_vector", induce_move_by_vector),
]

def verify_train(fn, task):
    for pair in task["train"]:
        try:
            if fn(pair["input"]) != pair["output"]: return False
        except Exception: return False
    return True

def test_correct(fn, task):
    for pair in task.get("test",[]):
        try:
            if fn(pair["input"]) != pair["output"]: return False
        except Exception: return False
    return len(task.get("test",[]))>0

def tem_solve(task):
    for name, inducer in TEMPLATES:
        try: fn = inducer(task)
        except Exception: fn=None
        if fn is None: continue
        if verify_train(fn, task):           # reproduz TODOS os train pairs (content-factored)
            ok = test_correct(fn, task)      # gate EXATO no teste held-out
            return name, ok
    return None, False

if __name__=="__main__":
    limit = int(sys.argv[1]) if len(sys.argv)>1 else None
    tasks = load_tasks(limit)
    print(f"TEM-ARC: {len(tasks)} tasks de {DATA}")
    solved=[]; train_only=[]
    by_template=Counter()
    for t in tasks:
        name, ok = tem_solve(t)
        if name and ok: solved.append((t["_id"],name)); by_template[name]+=1
        elif name: train_only.append((t["_id"],name))
    print(f"\nRESOLVIDAS (gate exato no teste): {len(solved)}/{len(tasks)} = {100*len(solved)/max(1,len(tasks)):.1f}%")
    print(f"(reproduziram train mas erraram teste: {len(train_only)} — overfit honesto)")
    print(f"por template: {dict(by_template)}")
    print(f"exemplos: {solved[:12]}")
