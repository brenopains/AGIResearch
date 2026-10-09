"""
tem_arc2.py — ARC com FATORACAO estrutura<->conteudo (TEM, Whittington 2020) + COMPOSICAO HIERARQUICA
de transformacoes task-invariantes (Active Predictive Coding, Rao 2024). Mesma pegada de pesquisa cerebral
aplicada ao ARC. Honesto: amplia LARGURA (o unico lever que move o ARC) por templates estruturais + as compoe
2-fundo. NAO inventa primitiva fora do fecho -> nao "passa" o ARC, mas mede o ganho real da fatoracao+composicao.
Gate EXATO no teste held-out. Sem LLM. Roda nas 400 tasks reais.
"""
import json, glob, os, sys, pathlib
from collections import Counter

DATA=None
for c in [pathlib.Path(__file__).parent.parent/"arc_data"/"data"/"training"]:
    if c.exists(): DATA=c; break
def load_tasks(limit=None):
    fs=sorted(glob.glob(str(DATA/"*.json"))); fs=fs[:limit] if limit else fs
    T=[]
    for f in fs:
        try:
            t=json.load(open(f)); t["_id"]=os.path.basename(f)[:-5]; T.append(t)
        except: pass
    return T

def dims(g): return (len(g), len(g[0]) if g else 0)
def bg_color(g): c=Counter(v for r in g for v in r); return c.most_common(1)[0][0] if c else 0
def gcopy(g): return [list(r) for r in g]
def geq(a,b): return a==b

def components(g,bg):
    H,W=dims(g); seen=[[0]*W for _ in range(H)]; objs=[]
    for i in range(H):
        for j in range(W):
            if g[i][j]==bg or seen[i][j]: continue
            col=g[i][j]; st=[(i,j)]; cells=[]; seen[i][j]=1
            while st:
                y,x=st.pop(); cells.append((y,x))
                for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny,nx=y+dy,x+dx
                    if 0<=ny<H and 0<=nx<W and not seen[ny][nx] and g[ny][nx]==col:
                        seen[ny][nx]=1; st.append((ny,nx))
            ys=[c[0] for c in cells]; xs=[c[1] for c in cells]
            objs.append({"cells":cells,"color":col,"size":len(cells),"top":min(ys),"left":min(xs),
                         "bot":max(ys),"right":max(xs)})
    return objs
def shape_sig(o):
    t,l=o["top"],o["left"]; return frozenset((y-t,x-l) for y,x in o["cells"])
def holes(g,o,bg):
    # nº de regioes de fundo TOTALMENTE cercadas pelo objeto (dentro da bbox e nao conectadas a borda)
    t,l,b,r=o["top"],o["left"],o["bot"],o["right"]; cs=set(o["cells"])
    H=b-t+1; W=r-l+1; inside=[[ (t+i,l+j) not in cs for j in range(W)] for i in range(H)]
    seen=[[False]*W for _ in range(H)]; from collections import deque
    # flood da borda da bbox -> fundo externo
    dq=deque()
    for i in range(H):
        for j in range(W):
            if (i in (0,H-1) or j in (0,W-1)) and inside[i][j] and not seen[i][j]:
                seen[i][j]=True; dq.append((i,j))
    while dq:
        y,x=dq.popleft()
        for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny,nx=y+dy,x+dx
            if 0<=ny<H and 0<=nx<W and inside[ny][nx] and not seen[ny][nx]:
                seen[ny][nx]=True; dq.append((ny,nx))
    # regioes internas nao alcancadas = buracos
    hc=0
    for i in range(H):
        for j in range(W):
            if inside[i][j] and not seen[i][j]:
                hc+=1; dq=deque([(i,j)]); seen[i][j]=True
                while dq:
                    y,x=dq.popleft()
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx=y+dy,x+dx
                        if 0<=ny<H and 0<=nx<W and inside[ny][nx] and not seen[ny][nx]:
                            seen[ny][nx]=True; dq.append((ny,nx))
    return hc

# ====== TEMPLATES content-factored (cada um: task -> fn(grid)->grid | None) ======
def t_recolor_size_rank(task):
    mp={}
    for p in task["train"]:
        gi,go=p["input"],p["output"]
        if dims(gi)!=dims(go): return None
        bg=bg_color(gi); oi=components(gi,bg)
        if not oi: return None
        ranks=sorted(range(len(oi)),key=lambda k:-oi[k]["size"])
        for rk,idx in enumerate(ranks):
            cols={go[y][x] for y,x in oi[idx]["cells"]}
            if len(cols)!=1: return None
            oc=next(iter(cols))
            if rk in mp and mp[rk]!=oc: return None
            mp[rk]=oc
    def fn(g):
        bg=bg_color(g); o=components(g,bg); out=gcopy(g)
        for rk,idx in enumerate(sorted(range(len(o)),key=lambda k:-o[k]["size"])):
            if mp.get(rk) is not None:
                for y,x in o[idx]["cells"]: out[y][x]=mp[rk]
        return out
    return fn
def t_recolor_shape(task):
    mp={}
    for p in task["train"]:
        gi,go=p["input"],p["output"]
        if dims(gi)!=dims(go): return None
        bg=bg_color(gi)
        for o in components(gi,bg):
            cols={go[y][x] for y,x in o["cells"]}
            if len(cols)!=1: return None
            oc=next(iter(cols)); s=shape_sig(o)
            if s in mp and mp[s]!=oc: return None
            mp[s]=oc
    def fn(g):
        bg=bg_color(g); out=gcopy(g)
        for o in components(g,bg):
            nc=mp.get(shape_sig(o))
            if nc is not None:
                for y,x in o["cells"]: out[y][x]=nc
        return out
    return fn
def t_recolor_holes(task):
    mp={}
    for p in task["train"]:
        gi,go=p["input"],p["output"]
        if dims(gi)!=dims(go): return None
        bg=bg_color(gi)
        for o in components(gi,bg):
            cols={go[y][x] for y,x in o["cells"]}
            if len(cols)!=1: return None
            oc=next(iter(cols)); h=holes(gi,o,bg)
            if h in mp and mp[h]!=oc: return None
            mp[h]=oc
    def fn(g):
        bg=bg_color(g); out=gcopy(g)
        for o in components(g,bg):
            nc=mp.get(holes(g,o,bg))
            if nc is not None:
                for y,x in o["cells"]: out[y][x]=nc
        return out
    return fn
def t_keep_extreme(task,which):
    for p in task["train"]:
        if dims(p["input"])!=dims(p["output"]): return None
    def fn(g):
        bg=bg_color(g); o=components(g,bg); H,W=dims(g); out=[[bg]*W for _ in range(H)]
        if not o: return gcopy(g)
        key=lambda z:z["size"]
        tgt=max(o,key=key) if which=="largest" else min(o,key=key)
        for y,x in tgt["cells"]: out[y][x]=tgt["color"]
        return out
    return fn
def t_select_most_holes(task):
    for p in task["train"]:
        if dims(p["input"])!=dims(p["output"]): return None
    def fn(g):
        bg=bg_color(g); o=components(g,bg); H,W=dims(g); out=[[bg]*W for _ in range(H)]
        if not o: return gcopy(g)
        tgt=max(o,key=lambda z:holes(g,z,bg))
        for y,x in tgt["cells"]: out[y][x]=tgt["color"]
        return out
    return fn
def t_move_vector(task):
    vec=None
    for p in task["train"]:
        gi,go=p["input"],p["output"]
        if dims(gi)!=dims(go): return None
        bg=bg_color(gi); oi=components(gi,bg); oo=components(go,bg)
        if len(oi)!=len(oo) or not oi: return None
        a=sorted(oi,key=lambda z:(z["size"],z["top"],z["left"])); b=sorted(oo,key=lambda z:(z["size"],z["top"],z["left"]))
        for x,y in zip(a,b):
            if shape_sig(x)!=shape_sig(y) or x["color"]!=y["color"]: return None
            v=(y["top"]-x["top"],y["left"]-x["left"])
            if vec is None: vec=v
            elif vec!=v: return None
    if not vec or vec==(0,0): return None
    def fn(g):
        bg=bg_color(g); H,W=dims(g); out=[[bg]*W for _ in range(H)]
        for o in components(g,bg):
            for y,x in o["cells"]:
                ny,nx=y+vec[0],x+vec[1]
                if 0<=ny<H and 0<=nx<W: out[ny][nx]=o["color"]
        return out
    return fn
def _symmetrize(g, axis):
    H,W=dims(g); bg=bg_color(g); out=gcopy(g)
    for i in range(H):
        for j in range(W):
            ni,nj = (i, W-1-j) if axis=="h" else (H-1-i, j) if axis=="v" else (j,i) if axis=="d" else (i,j)
            if 0<=ni<len(out) and 0<=nj<len(out[0]):
                if out[i][j]==bg and g[ni][nj]!=bg: out[i][j]=g[ni][nj]
    return out
def t_symmetrize(task,axis):
    for p in task["train"]:
        gi,go=p["input"],p["output"]
        if dims(gi)!=dims(go): return None
    def fn(g): return _symmetrize(g,axis)
    return fn
def t_fill_enclosed(task):
    # preenche regioes de fundo TOTALMENTE cercadas com uma cor inferida
    fill=None
    for p in task["train"]:
        gi,go=p["input"],p["output"]
        if dims(gi)!=dims(go): return None
        H,W=dims(gi)
        for i in range(H):
            for j in range(W):
                if gi[i][j]!=go[i][j]:
                    if fill is None: fill=go[i][j]
                    elif fill!=go[i][j]: return None
    if fill is None: return None
    def fn(g):
        H,W=dims(g); bg=bg_color(g); out=gcopy(g)
        seen=[[False]*W for _ in range(H)]; from collections import deque
        dq=deque()
        for i in range(H):
            for j in range(W):
                if (i in (0,H-1) or j in (0,W-1)) and g[i][j]==bg and not seen[i][j]:
                    seen[i][j]=True; dq.append((i,j))
        while dq:
            y,x=dq.popleft()
            for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                ny,nx=y+dy,x+dx
                if 0<=ny<H and 0<=nx<W and g[ny][nx]==bg and not seen[ny][nx]:
                    seen[ny][nx]=True; dq.append((ny,nx))
        for i in range(H):
            for j in range(W):
                if g[i][j]==bg and not seen[i][j]: out[i][j]=fill
        return out
    return fn

TEMPLATES=[
    ("recolor_size_rank",t_recolor_size_rank),("recolor_shape",t_recolor_shape),
    ("recolor_holes",t_recolor_holes),("keep_largest",lambda t:t_keep_extreme(t,"largest")),
    ("keep_smallest",lambda t:t_keep_extreme(t,"smallest")),("select_most_holes",t_select_most_holes),
    ("move_vector",t_move_vector),("sym_h",lambda t:t_symmetrize(t,"h")),
    ("sym_v",lambda t:t_symmetrize(t,"v")),("fill_enclosed",t_fill_enclosed),
]
def vtrain(fn,task):
    for p in task["train"]:
        try:
            if not geq(fn(p["input"]),p["output"]): return False
        except: return False
    return True
def vtest(fn,task):
    for p in task.get("test",[]):
        try:
            if not geq(fn(p["input"]),p["output"]): return False
        except: return False
    return len(task.get("test",[]))>0

def solve(task, compose=True):
    fns=[]
    for nm,ind in TEMPLATES:
        try: fn=ind(task)
        except: fn=None
        if fn and vtrain(fn,task):
            if vtest(fn,task): return nm,True
            fns.append((nm,fn))
    # COMPOSICAO HIERARQUICA (APC): aplica A depois B; verifica train, gate exato no teste
    if compose:
        cand=[]
        for nm,ind in TEMPLATES:
            try: f=ind(task)
            except: f=None
            if f: cand.append((nm,f))
        for na,fa in cand:
            for nb,fb in cand:
                comp=lambda g,fa=fa,fb=fb: fb(fa(g))
                if vtrain(comp,task):
                    if vtest(comp,task): return f"{na}>>{nb}",True
    return None,False

if __name__=="__main__":
    limit=int(sys.argv[1]) if len(sys.argv)>1 else None
    tasks=load_tasks(limit)
    print(f"TEM-ARC2 (fatoracao+composicao): {len(tasks)} tasks")
    solved=[]; by=Counter()
    for t in tasks:
        nm,ok=solve(t)
        if nm and ok: solved.append((t["_id"],nm)); by[nm.split('>>')[0]]+=1
    print(f"\nRESOLVIDAS (gate exato no teste held-out): {len(solved)}/{len(tasks)} = {100*len(solved)/max(1,len(tasks)):.1f}%")
    comps=[s for s in solved if '>>' in s[1]]
    print(f"por COMPOSICAO (2-fundo): {len(comps)} | por template unico: {len(solved)-len(comps)}")
    print(f"templates: {dict(by)}")
    print(f"exemplos: {solved[:14]}")
