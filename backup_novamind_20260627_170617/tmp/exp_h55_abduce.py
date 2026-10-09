"""
H55 — ABDUCE: ataca a parede de LARGURA por sintese ABDUTIVA de primitivas
===========================================================================
Em vez de ENUMERAR um pool fixo de operadores (H51/H54), CONSTROI a transformacao
lendo o DELTA (o que mudou input->output) de forma OBJETO-CENTRICA e PARAMETRICA.

Questiona a parede de Solomonoff em 4 fatores:
  (1) base finita (core-knowledge de Chollet) — nao infinita;
  (2) primitivas PARAMETRICAS (params lidos do dado) colapsam a largura;
  (3) ABDUCAO (le a regra do delta) no lugar de enumeracao forward;
  (4) representacao por OBJETOS, nao por pixels (ViTARC).

Hierarquia de regras (MDL, simples->complexo):
  Tier A — por-objeto simples (translacao uniforme; recolor uniforme);
  Tier B — global/condicional (recolor por rank de tamanho/forma; selecao do "diferente");
  Tier C — relacional (objeto se move em direcao a outro; maior absorve menor).
Aceita SO com casamento EXATO em todos os demos; senao FALLBACK -> Pulse (H54).
Treino-zero, CPU. Honesto: a abducao tem seu PROPRIO fecho (conjunto finito de schemas);
nao apaga a parede — converte "infinitas primitivas fixas" em "base parametrica finita".
"""
import os, json, glob, numpy as np
from collections import deque, Counter
import importlib.util

DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")

# ---------------- parser objeto-centrico (rico) ----------------
def _comps(g, conn, same_color):
    """rotula componentes. conn in {4,8}; same_color: separa por cor ou junta nao-zero."""
    H,W=g.shape; lab=np.zeros((H,W),int); cur=0
    nbrs=[(1,0),(-1,0),(0,1),(0,-1)]+([(1,1),(1,-1),(-1,1),(-1,-1)] if conn==8 else [])
    for r in range(H):
        for c in range(W):
            if g[r,c]!=0 and lab[r,c]==0:
                cur+=1; q=deque([(r,c)]); lab[r,c]=cur
                while q:
                    y,x=q.popleft()
                    for dy,dx in nbrs:
                        ny,nx=y+dy,x+dx
                        if 0<=ny<H and 0<=nx<W and lab[ny,nx]==0 and g[ny,nx]!=0 and (not same_color or g[ny,nx]==g[r,c]):
                            lab[ny,nx]=cur; q.append((ny,nx))
    return lab,cur

def _sym_scores(mask):
    h=float(np.array_equal(mask,mask[:,::-1])); v=float(np.array_equal(mask,mask[::-1,:]))
    d=float(mask.shape[0]==mask.shape[1] and np.array_equal(mask,mask.T)); return (h,v,d)

class Obj:
    __slots__=('cells','colors','color','mask','bbox','centroid','size','holes','aspect','sym')
    def __init__(s,g,cells):
        ys=[r for r,_ in cells]; xs=[c for _,c in cells]
        r0,r1,c0,c1=min(ys),max(ys),min(xs),max(xs)
        s.cells=frozenset(cells); s.bbox=(r0,c0,r1,c1); s.size=len(cells)
        s.centroid=(sum(ys)/len(ys),sum(xs)/len(xs))
        cols=[g[r,c] for r,c in cells]; cc=Counter(cols)
        s.colors=cc; s.color=cc.most_common(1)[0][0]
        m=np.zeros((r1-r0+1,c1-c0+1),int)
        for r,c in cells: m[r-r0,c-c0]=g[r,c]
        s.mask=m
        bm=(m!=0).astype(int); area=bm.sum()
        bb=bm.shape[0]*bm.shape[1]; s.holes=int(bb-area) if area<bb else 0
        s.aspect=round((r1-r0+1)/(c1-c0+1),3)
        s.sym=_sym_scores(bm)
    def shapesig(s):  # assinatura de forma exata (binaria, normalizada)
        return (s.mask!=0).astype(np.uint8).tobytes()+bytes(((s.mask!=0).shape[0],))
    def colorshapesig(s): return (s.color, s.shapesig())

def parse(g, conn=4, same_color=True):
    lab,n=_comps(g,conn,same_color); objs=[]
    for i in range(1,n+1):
        cells=[(r,c) for r,c in zip(*np.where(lab==i))]
        if cells: objs.append(Obj(g,cells))
    return objs

def blank_like(g): return np.zeros_like(g)
def stamp(canvas,obj,dr=0,dc=0):
    r0,c0,r1,c1=obj.bbox
    for (r,c) in obj.cells:
        nr,nc=r+dr,c+dc
        if 0<=nr<canvas.shape[0] and 0<=nc<canvas.shape[1]:
            # cor original da celula
            canvas[nr,nc]=obj_cellcolor(obj,r,c)
def obj_cellcolor(obj,r,c):
    r0,c0,_,_=obj.bbox; return obj.mask[r-r0,c-c0]

# ---------------- avaliacao exata ----------------
def exact(f, demos):
    try:
        for gi,go in demos:
            r=f(gi)
            if r is None or not isinstance(r,np.ndarray) or r.shape!=go.shape or not np.array_equal(r,go): return False
        return True
    except Exception:
        return False

# ============================================================
#  SCHEMAS DE REGRA (abducao). Cada um recebe demos e devolve (render, desc) ou None
#  Tentados em ordem MDL (simples -> complexo). conn/same_color variados.
# ============================================================
def _parses(demos,conn,sc): return [(parse(gi,conn,sc),parse(go,conn,sc),gi,go) for gi,go in demos]

# --- Tier A: translacao uniforme de todos os objetos (mesma forma+cor) ---
def schema_uniform_translate(demos,conn,sc):
    offs=set()
    for io,oo,gi,go in _parses(demos,conn,sc):
        if gi.shape!=go.shape or len(io)!=len(oo) or len(io)==0: return None
        # casa por (cor,forma); requer multiconjuntos iguais
        bi=Counter(o.colorshapesig() for o in io); bo=Counter(o.colorshapesig() for o in oo)
        if bi!=bo: return None
        # casa guloso por assinatura e centroides
        used=[False]*len(oo); pairs=[]
        for a in io:
            best=-1; bd=1e9
            for j,b in enumerate(oo):
                if used[j] or b.colorshapesig()!=a.colorshapesig(): continue
                d=abs(a.centroid[0]-b.centroid[0])+abs(a.centroid[1]-b.centroid[1])
                if d<bd: bd=d; best=j
            if best<0: return None
            used[best]=True; pairs.append((a,oo[best]))
        for a,b in pairs:
            dr=round(b.centroid[0]-a.centroid[0]); dc=round(b.centroid[1]-a.centroid[1])
            offs.add((dr,dc))
    if len(offs)!=1: return None
    dr,dc=offs.pop()
    if dr==0 and dc==0: return None
    def render(g):
        objs=parse(g,conn,sc); out=blank_like(g)
        for o in objs: stamp(out,o,dr,dc)
        return out
    return render,f"A:translate(dr={dr},dc={dc})"

# --- Tier B: recolor de cada objeto por uma PROPRIEDADE (rank de tamanho ou forma) ---
def _recolor_by_key(demos,conn,sc,keyfn,name):
    """keyfn(obj, allobjs)->chave. aprende chave->cor de saida; exige consistencia; mesma forma/posicao."""
    mapping={}
    for io,oo,gi,go in _parses(demos,conn,sc):
        if gi.shape!=go.shape or len(io)==0: return None
        # a saida deve ter os MESMOS objetos (mesmas celulas), so cor muda
        cellset_i={c for o in io for c in o.cells}; cellset_o={c for o in oo for c in o.cells}
        if cellset_i!=cellset_o: return None
        for o in io:
            k=keyfn(o,io)
            # cor de saida nessas celulas (deve ser uniforme)
            cols={go[r,c] for r,c in o.cells}
            if len(cols)!=1: return None
            col=next(iter(cols))
            if k in mapping and mapping[k]!=col: return None
            mapping[k]=col
    def render(g):
        objs=parse(g,conn,sc); out=g.copy()
        for o in objs:
            k=keyfn(o,objs)
            if k not in mapping: return None
            for r,c in o.cells: out[r,c]=mapping[k]
        return out
    return render,f"B:recolor_by_{name}"

def schema_recolor_size_rank(demos,conn,sc):
    def key(o,allo):
        ranks=sorted(allo,key=lambda x:(-x.size,x.centroid)); return ranks.index(o)
    return _recolor_by_key(demos,conn,sc,key,"size_rank")
def schema_recolor_shape(demos,conn,sc):
    return _recolor_by_key(demos,conn,sc,lambda o,allo:o.shapesig(),"shape")
def schema_recolor_size(demos,conn,sc):
    return _recolor_by_key(demos,conn,sc,lambda o,allo:o.size,"size")
def schema_recolor_holes(demos,conn,sc):
    return _recolor_by_key(demos,conn,sc,lambda o,allo:o.holes,"holes")

# --- Tier B: SELECIONAR um objeto (o "diferente") e devolver (cortado) ---
def _select_and_crop(demos,conn,sc,pickfn,name):
    for io,oo,gi,go in _parses(demos,conn,sc):
        if len(io)==0: return None
        sel=pickfn(io)
        if sel is None: return None
        r0,c0,r1,c1=sel.bbox; sub=gi[r0:r1+1,c0:c1+1]
        if sub.shape!=go.shape or not np.array_equal(sub,go): return None
    def render(g):
        objs=parse(g,conn,sc); sel=pickfn(objs)
        if sel is None: return None
        r0,c0,r1,c1=sel.bbox; return g[r0:r1+1,c0:c1+1].copy()
    return render,f"B:select_{name}"

def _odd_by(objs,attr):
    vals=[getattr(o,attr) if attr!='shapesig' else o.shapesig() for o in objs]
    cnt=Counter(vals); uniq=[o for o,v in zip(objs,vals) if cnt[v]==1]
    return uniq[0] if len(uniq)==1 else None
def schema_select_odd_shape(demos,conn,sc): return _select_and_crop(demos,conn,sc,lambda o:_odd_by(o,'shapesig'),"odd_shape")
def schema_select_odd_color(demos,conn,sc): return _select_and_crop(demos,conn,sc,lambda o:_odd_by(o,'color'),"odd_color")
def schema_select_largest(demos,conn,sc):
    def pick(o):
        if not o: return None
        return max(o,key=lambda x:x.size)
    return _select_and_crop(demos,conn,sc,pick,"largest")
def schema_select_smallest(demos,conn,sc):
    def pick(o):
        if not o: return None
        return min(o,key=lambda x:x.size)
    return _select_and_crop(demos,conn,sc,pick,"smallest")
def schema_select_most_holes(demos,conn,sc):
    def pick(o):
        if not o: return None
        return max(o,key=lambda x:x.holes)
    return _select_and_crop(demos,conn,sc,pick,"most_holes")

# --- Tier C: relacional — cada objeto se move (gravidade direcional) ate colidir/borda ---
def schema_per_object_gravity(demos,conn,sc):
    dirs={'d':(1,0),'u':(-1,0),'r':(0,1),'l':(0,-1)}
    good=None
    for dn,(dy,dx) in dirs.items():
        ok=True
        for io,oo,gi,go in _parses(demos,conn,sc):
            if gi.shape!=go.shape: ok=False; break
            if not np.array_equal(_grav_objects(gi,conn,sc,dy,dx),go): ok=False; break
        if ok: good=(dn,dy,dx); break
    if good is None: return None
    dn,dy,dx=good
    return (lambda g:_grav_objects(g,conn,sc,dy,dx)),f"C:per_object_gravity({dn})"

def _grav_objects(g,conn,sc,dy,dx):
    objs=parse(g,conn,sc); out=blank_like(g); H,W=g.shape
    # move objetos um a um na direcao ate colidir com borda ou celula ja ocupada
    order=sorted(objs,key=lambda o:-(o.centroid[0]*dy+o.centroid[1]*dx))
    for o in order:
        dr=dc=0
        while True:
            ndr,ndc=dr+dy,dc+dx; collide=False
            for (r,c) in o.cells:
                nr,nc=r+ndr,c+ndc
                if not(0<=nr<H and 0<=nc<W) or (out[nr,nc]!=0): collide=True; break
            if collide: break
            dr,dc=ndr,ndc
        for (r,c) in o.cells: out[r+dr,c+dc]=obj_cellcolor(o,r,c)
    return out

SCHEMAS=[  # ordem MDL
    ("A",schema_uniform_translate),
    ("B",schema_recolor_shape),("B",schema_recolor_size_rank),("B",schema_recolor_size),("B",schema_recolor_holes),
    ("B",schema_select_odd_shape),("B",schema_select_odd_color),
    ("B",schema_select_largest),("B",schema_select_smallest),("B",schema_select_most_holes),
    ("C",schema_per_object_gravity),
]
PARSE_SETTINGS=[(4,True),(8,True),(8,False)]

def abduce(demos):
    for tier,sch in SCHEMAS:
        for conn,sc in PARSE_SETTINGS:
            try: res=sch(demos,conn,sc)
            except Exception: res=None
            if res is None: continue
            render,desc=res
            if exact(render,demos):
                return render,tier,f"{desc}|parse({conn},{'sc' if sc else 'any'})"
    return None,None,None

# ---------------- Pulse para fallback/union ----------------
def _load_pulse():
    p=os.path.join(os.path.dirname(__file__),"exp_h54_pulse.py")
    spec=importlib.util.spec_from_file_location("pulse",p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
    pulse=_load_pulse()
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    ab_solved=0; pulse_solved=0; union=0; logs=[]; tier_ct=Counter()
    for fp in files:
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        # --- ABDUCE ---
        render,tier,desc=abduce(train)
        ab_ok=False
        if render is not None:
            ab_ok=all((lambda r,yo: r is not None and isinstance(r,np.ndarray) and r.shape==yo.shape and np.array_equal(r,yo))(render(xi),yo) for xi,yo in tests)
        # --- PULSE (fallback / para a uniao) ---
        pool=dict(pulse.FIXED)
        if train[0][0].shape[0]==train[0][0].shape[1]: pool.update(pulse.square_ops())
        prog,_,_=pulse.pulse_search(pool,train)
        pu_ok=False
        if prog is not None:
            pu_ok=all((lambda r,yo: r is not None and r.shape==yo.shape and np.array_equal(r,yo))(prog(xi),yo) for xi,yo in tests)
        if ab_ok: ab_solved+=1; tier_ct[tier]+=1
        if pu_ok: pulse_solved+=1
        if ab_ok or pu_ok: union+=1
        if ab_ok: logs.append((os.path.basename(fp),tier,desc, "(novo vs pulse)" if not pu_ok else ""))
    print("="*66)
    print("H55 ABDUCE — sintese abdutiva objeto-centrica (treino-zero) + uniao com Pulse")
    print(f"  Abduce sozinho : {ab_solved}/400 = {100*ab_solved/400:.1f}%   por tier: {dict(tier_ct)}")
    print(f"  Pulse  (H54)   : {pulse_solved}/400 = {100*pulse_solved/400:.1f}%")
    print(f"  UNIAO          : {union}/400 = {100*union/400:.1f}%   (baseline H54 = 11.2%)")
    new=[l for l in logs if l[3]]
    print(f"  Abduce resolveu {len(new)} tarefas que o Pulse NAO resolve:")
    for nm,tier,desc,_ in new[:40]: print(f"    [{tier}] {nm} -> {desc}")

if __name__=="__main__": main()
