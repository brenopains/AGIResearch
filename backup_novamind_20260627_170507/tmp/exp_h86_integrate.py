"""
H86 — Integracao entre-bases (cerebro entropico, feito CERTO; nao o caos do H66)
=================================================================================
Carhart-Harris 2014 (entropic brain): psicodelicos NAO sao "mais aleatoriedade" — sao
MODULARIDADE COMPROMETIDA + INTEGRACAO GLOBAL aumentada (DMN desacopla, modulos
normalmente segregados passam a conversar). H66 falhou porque so fez busca estocastica
mais funda DENTRO do fecho de operadores simbolicos = +0. Aqui colapsamos os SILOS:
passamos a REPRESENTACAO INTERMEDIARIA de uma base como feature EXTRA para outra.

Regras p/ escapar das paredes ja medidas:
  - one-directional, SEM objetivo/otimizador compartilhado (escapa parede de integracao H23-25);
  - feature NOVA (codigo MB esparso, estado RC, canais de objeto ABDUCE), nao remix de
    operadores simbolicos (escapa o +0 do H66).

Feeds:
  A) MB(codigo esparso 2048) + RC(estado 395) -> uma leitura ridge unica. (mais barato)
  B) ABDUCE(parse de objetos) -> canais EXTRA de entrada do NCA. (mais promissor: injeta
     pertinencia-a-objeto que o NCA nao enxerga)
Metrica honesta por feed: solved_fused - union (do H85), i.e., tarefas que NENHUMA base
isolada resolve. So tarefas mesmo-shape. Gate exato em demos antes de prever teste.
"""
import os, sys, json, glob, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

HERE=os.path.dirname(__file__)
DATA=os.path.join(HERE,"arc_data","data","training")
MB=_load("exp_h68_mb_arc.py")          # patches, mb_expand, Wproj, DEVICE
RC=_load("exp_h72_rc_ensemble.py")     # make_res, feats, ridge
NCA=_load("exp_h70_nca.py")            # to_state, to_target, SZ, CH, NCOL, HID, dihedral
AB=_load("exp_h55_abduce.py")          # parse, Obj
DEV=MB.DEVICE
torch.manual_seed(0)

def ok_outs(outs, tests):
    if outs is None: return False
    try: return all(p is not None and p.shape==yo.shape and np.array_equal(p,yo) for p,(_,yo) in zip(outs,tests))
    except Exception: return False

# ============ FEED A: MB-code + RC-state -> ridge ============
RES_A=[RC.make_res(0,0.95)]   # 1 reservatorio (395 feats) suficiente; barato

def fused_feats(g):
    mb=MB.mb_expand(MB.patches(g))                 # (HW,2048) DEV
    rcs=[RC.feats(g,Wr,Wi) for Wr,Wi in RES_A]     # cada (HW,395)
    return torch.cat([mb]+rcs,1)

def solve_feedA(train, tests, lam=1.0):
    if not all(i.shape==o.shape for i,o in train): return None
    Phi=[]; Y=[]
    for i,o in train:
        Phi.append(fused_feats(i))
        oh=torch.zeros(o.size,10,device=DEV); oh[torch.arange(o.size),torch.tensor(o.ravel(),device=DEV,dtype=torch.long)]=1; Y.append(oh)
    W=RC.ridge(torch.cat(Phi),torch.cat(Y),lam)
    def pred(g): return (fused_feats(g)@W).argmax(1).cpu().numpy().reshape(g.shape).astype(np.int64)
    for i,o in train:
        if not np.array_equal(pred(i),o): return None
    return [pred(xi) for xi,_ in tests]

# ============ FEED B: canais de objeto (ABDUCE) -> NCA ============
SZ=NCA.SZ; CHN=NCA.CH; NCOL=NCA.NCOL; KOBJ=3   # canais extra: tamanho-norm, rank-norm, is-maior

def obj_channels(g):
    """(KOBJ,SZ,SZ): por celula, atributos do objeto a que pertence (0 no fundo/pad)."""
    H,W=g.shape; ch=np.zeros((KOBJ,SZ,SZ),np.float32)
    try: objs=AB.parse(g,8,False)
    except Exception: objs=[]
    if objs:
        sizes=sorted([o.size for o in objs],reverse=True); mx=max(sizes);
        rankmap={s:i for i,s in enumerate(sorted(set(sizes),reverse=True))}
        nr=max(1,len(rankmap)-1)
        for o in objs:
            sz_n=o.size/mx; rk_n=rankmap[o.size]/nr; ismax=1.0 if o.size==mx else 0.0
            for (r,c) in o.cells:
                if r<SZ and c<SZ: ch[0,r,c]=sz_n; ch[1,r,c]=rk_n; ch[2,r,c]=ismax
    return torch.tensor(ch)

class NCAk(nn.Module):
    def __init__(s,hidden=96):
        super().__init__()
        s.perceive=nn.Conv2d(CHN+KOBJ,hidden,3,padding=1)   # ve estado(18) + objchannels(3)
        s.rule=nn.Conv2d(hidden,CHN,1); nn.init.zeros_(s.rule.weight); nn.init.zeros_(s.rule.bias)
    def forward(s,state,objc,T):
        for _ in range(T):
            dx=s.rule(F.relu(s.perceive(torch.cat([state,objc],1))))
            mask=(torch.rand_like(state[:, :1])<0.85).float()
            state=state+dx*mask
        return state

def solve_feedB(train, tests, T=12, steps=220):
    if not all(i.shape==o.shape for i,o in train) or len(train)==0: return None
    X=[]; OC=[]; Y=[]
    for i,o in train:
        for k in range(8):
            ik=np.ascontiguousarray(NCA.dihedral(i,k)); ok=np.ascontiguousarray(NCA.dihedral(o,k))
            X.append(NCA.to_state(ik)); OC.append(obj_channels(ik)); Y.append(NCA.to_target(ok))
    X=torch.stack(X).to(DEV); OC=torch.stack(OC).to(DEV); Y=torch.stack(Y).to(DEV)
    net=NCAk().to(DEV); opt=torch.optim.Adam(net.parameters(),2e-3); net.train()
    for _ in range(steps):
        out=net(X.clone(),OC,T); loss=F.cross_entropy(out[:, :NCOL],Y,ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    def predict(g):
        H,W=g.shape
        with torch.no_grad():
            out=net(NCA.to_state(g).unsqueeze(0).to(DEV), obj_channels(g).unsqueeze(0).to(DEV), T)
            return out[0,:NCOL,:H,:W].argmax(0).cpu().numpy().astype(np.int64)
    for i,o in train:
        if not np.array_equal(predict(i),o): return None
    return [predict(xi) for xi,_ in tests]

def main():
    t0=time.time()
    union=set(json.load(open(os.path.join(HERE,"h85_union.json")))) if os.path.exists(os.path.join(HERE,"h85_union.json")) else set()
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    only=[a for a in sys.argv[1:] if not a.startswith("--")]
    do_a = ("--b" not in sys.argv)   # default roda A; "--b" pula A
    do_b = ("--nob" not in sys.argv)
    A=set(); B=set(); att=0
    for fp in files:
        name=os.path.basename(fp)
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        if not all(i.shape==o.shape for i,o in train): continue
        att+=1
        if do_a and ok_outs(solve_feedA(train,tests),tests): A.add(name)
        if do_b and ok_outs(solve_feedB(train,tests),tests): B.add(name)
        if att%50==0: print(f"  ... {att} mesmo-shape avaliadas (A={len(A)} B={len(B)})"); sys.stdout.flush()
    fused=A|B; vram=torch.cuda.max_memory_allocated()/1e6 if DEV=='cuda' else 0
    print("="*72)
    print(f"H86 Integracao entre-bases (cerebro entropico) ({DEV})")
    print(f"  tarefas mesmo-shape: {att}/400 | union(H85)={len(union)}")
    print(f"  Feed A (MB+RC ridge): resolveu {len(A)} | NOVAS vs union: {len(A-union)} {sorted(A-union)[:8]}")
    print(f"  Feed B (ABDUCE->NCA): resolveu {len(B)} | NOVAS vs union: {len(B-union)} {sorted(B-union)[:8]}")
    print(f"  FUSED total NOVAS vs union: {len(fused-union)} {sorted(fused-union)[:12]}")
    print(f"  nova UNIAO: {len(union|fused)}/400 = {100*len(union|fused)/400:.1f}%  (era {100*len(union)/400:.1f}%)")
    print(f"  tempo {time.time()-t0:.0f}s | VRAM {vram:.0f}MB")
    json.dump(sorted(fused),open(os.path.join(HERE,"h86_fused_solved.json"),"w"))
    print("  Honesto: feature-augmentation one-directional; se NOVAS>0, integracao revelou regra")
    print("  que nenhuma base sozinha pegava. Se 0, os silos ja eram complementares no voto.")

if __name__=="__main__": main()
