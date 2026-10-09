"""
H87 — Base de CODIFICACAO PREDITIVA / inferencia ativa (Friston) como nova base complementar
=============================================================================================
Friston (Free Energy Principle): percepcao = inferir estados latentes que minimizam erro de
predicao PONDERADO POR PRECISAO; aprendizado = atualizar parametros do gerador. Aqui a
"regra" da tarefa e um latente z (codigo de regra) inferido por descida iterativa a partir
dos demos; o gerador g_theta(input, z) preve a saida. Precisao = peso por-cor aprendido no CE.

Familia de operador NOVA (distinta das outras bases mesmo-shape):
  - NCA aprende uma regra de atualizacao LOCAL iterativa no ESPACO;
  - RC usa dinamica FIXA + ridge;
  - aqui a COMPUTACAO e a INFERENCIA do latente de regra (E-step) alternada com aprendizado
    do gerador (M-step), ponderada por precisao. z e compartilhado entre os demos (a regra).

Inferencia no teste: z ja foi inferido dos demos; aplica g_theta(input_teste, z).
Gate exato em demos. So mesmo-shape. Honesto: alto risco de sobrepor o NCA; mede com dado.
"""
import os, sys, json, glob, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,"arc_data","data","training")
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
NCOL=10; PAD=10; NIN=11; CLAT=16

def onehot(g):
    H,W=g.shape; x=torch.zeros(NIN,H,W,device=DEV)
    for c in range(NCOL): x[c]=torch.tensor(g==c,dtype=torch.float32,device=DEV)
    return x   # canal 10 (pad) fica 0 dentro da grade

class Gen(nn.Module):
    def __init__(s,h=64):
        super().__init__()
        s.c1=nn.Conv2d(NIN+CLAT,h,3,padding=1); s.c2=nn.Conv2d(h,h,3,padding=1); s.c3=nn.Conv2d(h,NCOL,3,padding=1)
    def forward(s,x,z):
        H,W=x.shape[-2:]; zb=z.view(1,CLAT,1,1).expand(1,CLAT,H,W)
        h=F.relu(s.c1(torch.cat([x.unsqueeze(0),zb],1))); h=F.relu(s.c2(h)); return s.c3(h)[0]  # (NCOL,H,W)

def solve_pc(train, tests, n_outer=140, n_inner=4):
    if not all(i.shape==o.shape for i,o in train) or len(train)==0: return None
    ins=[onehot(i) for i,_ in train]
    tgt=[torch.tensor(o,device=DEV,dtype=torch.long) for _,o in train]
    net=Gen().to(DEV); z=torch.zeros(CLAT,device=DEV,requires_grad=True)
    logprec=torch.zeros(NCOL,device=DEV,requires_grad=True)   # precisao por cor (softplus)
    opt_t=torch.optim.Adam(list(net.parameters())+[logprec],3e-3)
    opt_z=torch.optim.Adam([z],5e-2)
    def free_energy():
        prec=F.softplus(logprec)            # (NCOL,)
        loss=0.0
        for x,y in zip(ins,tgt):
            logits=net(x,z)                 # (NCOL,H,W)
            ce=F.cross_entropy(logits.unsqueeze(0),y.unsqueeze(0),reduction='none')[0]  # (H,W)
            w=prec[y]                        # peso de precisao da cor verdadeira por celula
            loss=loss+(w*ce).mean()
        return loss/len(ins)
    for _ in range(n_outer):
        for _ in range(n_inner):            # E-step: infere z (estado) por descida
            opt_z.zero_grad(); fe=free_energy(); fe.backward(); opt_z.step()
        opt_t.zero_grad(); fe=free_energy(); fe.backward(); opt_t.step()  # M-step: aprende gerador+precisao
    net.eval()
    def predict(g):
        with torch.no_grad(): return net(onehot(g),z).argmax(0).cpu().numpy().astype(np.int64)
    for i,o in train:
        if not np.array_equal(predict(i),o): return None
    return [predict(xi) for xi,_ in tests]

def ok_outs(outs,tests):
    if outs is None: return False
    try: return all(p is not None and p.shape==yo.shape and np.array_equal(p,yo) for p,(_,yo) in zip(outs,tests))
    except Exception: return False

def main():
    t0=time.time()
    union=set(json.load(open(os.path.join(HERE,"h85_union.json")))) if os.path.exists(os.path.join(HERE,"h85_union.json")) else set()
    files=sorted(glob.glob(os.path.join(DATA,"*.json"))); pc=set(); att=0
    for fp in files:
        name=os.path.basename(fp); t=json.load(open(fp))
        train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        if not all(i.shape==o.shape for i,o in train): continue
        att+=1
        if ok_outs(solve_pc(train,tests),tests): pc.add(name)
        if att%50==0: print(f"  ... {att} mesmo-shape (pc={len(pc)})"); sys.stdout.flush()
    json.dump(sorted(pc),open(os.path.join(HERE,"h87_pc_solved.json"),"w"))
    vram=torch.cuda.max_memory_allocated()/1e6 if DEV=='cuda' else 0
    print("="*72)
    print(f"H87 Codificacao preditiva/inferencia ativa ({DEV})")
    print(f"  mesmo-shape: {att}/400 | union(H85)={len(union)}")
    print(f"  PC resolveu: {len(pc)} | NOVAS vs union: {len(pc-union)} {sorted(pc-union)[:10]}")
    print(f"  nova UNIAO: {len(union|pc)}/400 = {100*len(union|pc)/400:.1f}%  (era {100*len(union)/400:.1f}%)")
    print(f"  tempo {time.time()-t0:.0f}s | VRAM {vram:.0f}MB")
    print("  Honesto: latente de regra inferido (E) + gerador aprendido (M), precisao por cor.")
    print("  Se NOVAS>0, a inferencia ativa pegou regra que NCA/RC nao pegavam. Se 0, sobrepoe.")

if __name__=="__main__": main()
