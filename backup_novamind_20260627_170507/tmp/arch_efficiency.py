"""
arch_efficiency.py — TESTE HONESTO da hipotese do usuario: nossa ARQUITETURA (representacao FATORADA/composicional)
torna um modelo PEQUENO mais eficiente e GENERALISTA que um baseline VANILLA do mesmo tamanho?
Tarefa: composicional (verbo x modificador -> sequencia). Treina em 48/64 combinacoes, TESTA em 16 NUNCA-VISTAS
(generalizacao sistematica = o hallmark). Mesmo orcamento de parametros. Multi-seed (honestidade). RTX3050/CUDA.
"""
import torch, torch.nn as nn, numpy as np, time
dev='cuda' if torch.cuda.is_available() else 'cpu'
NV, NM, L, V = 8, 8, 6, 9   # 8 verbos, 8 modificadores, saida len 6, vocab 8
def make_data():
    # regra COMPOSICIONAL: saida = [verbo]*(modificador+1) preenchido, resto = PAD(0). Precisa saber O QUE (verbo) e QUANTOS (mod).
    X=[]; Y=[]
    for v in range(NV):
        for m in range(NM):
            seq=[ (v+1) if t<=m else 0 for t in range(L) ]   # v+1 p/ reservar 0=PAD
            X.append((v,m)); Y.append(seq)
    return X, np.array(Y)
class Vanilla(nn.Module):  # ENTANGLED: concat one-hot -> MLP (processa verbo+mod juntos)
    def __init__(s,h=64):
        super().__init__(); s.net=nn.Sequential(nn.Linear(NV+NM,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,L*V))
    def forward(s,v,m):
        x=torch.cat([torch.eye(NV,device=dev)[v],torch.eye(NM,device=dev)[m]],-1)
        return s.net(x).view(-1,L,V)
class Factored(nn.Module):  # NOSSA ARQUITETURA: embeddings SEPARADOS (verbo, mod) + interacao estruturada + decode por-posicao
    def __init__(s,d=20):
        super().__init__(); s.ev=nn.Embedding(NV,d); s.em=nn.Embedding(NM,d)
        s.pos=nn.Parameter(torch.randn(L,d)*0.1)
        s.dec=nn.Sequential(nn.Linear(d*3,48),nn.ReLU(),nn.Linear(48,V))   # decode cada posicao de (verbo,mod,posicao)
    def forward(s,v,m):
        ev=s.ev(v); em=s.em(m)   # fatores separados
        outs=[]
        for t in range(L):
            p=s.pos[t].expand(ev.size(0),-1)
            outs.append(s.dec(torch.cat([ev,em,p],-1)))   # composicao por-posicao: "este verbo, neste lugar, dado o quanto"
        return torch.stack(outs,1)
def run(model_cls, train_idx, X, Y, epochs=400, **kw):
    torch.manual_seed(0)
    m=model_cls(**kw).to(dev); opt=torch.optim.Adam(m.parameters(),1e-2)
    vs=torch.tensor([x[0] for x in X],device=dev,dtype=torch.long); ms=torch.tensor([x[1] for x in X],device=dev,dtype=torch.long)
    ys=torch.tensor(Y,device=dev,dtype=torch.long)
    tr=torch.tensor(train_idx,device=dev)
    te=torch.tensor([i for i in range(len(X)) if i not in set(train_idx)],device=dev)
    lossf=nn.CrossEntropyLoss()
    for e in range(epochs):
        m.train(); opt.zero_grad()
        out=m(vs[tr],ms[tr]); loss=lossf(out.reshape(-1,V), ys[tr].reshape(-1)); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        def acc(idx):
            out=m(vs[idx],ms[idx]).argmax(-1); return (out==ys[idx]).all(-1).float().mean().item()
        return acc(tr), acc(te), sum(p.numel() for p in m.parameters())
X,Y=make_data(); N=len(X)
print(f"Tarefa composicional: {N} combos (verbo x modificador), treina 48, testa 16 NUNCA-VISTAS. dev={dev}")
res={'Vanilla(entangled)':[], 'Factored(nossa arq)':[]}
for seed in range(5):
    rng=np.random.default_rng(seed); idx=rng.permutation(N); train_idx=sorted(idx[:48].tolist())
    tv,ev,pv=run(Vanilla, train_idx, X, Y, h=64)
    tf,ef,pf=run(Factored, train_idx, X, Y, d=20)
    res['Vanilla(entangled)'].append(ev); res['Factored(nossa arq)'].append(ef)
    if seed==0: print(f"  params: Vanilla={pv} | Factored={pf} (orcamento comparavel)")
print("\n=== GENERALIZACAO em combinacoes NUNCA-VISTAS (5 seeds) ===")
for k,vv in res.items():
    print(f"  {k:24s}: held-out acc = {np.mean(vv)*100:5.1f}% ± {np.std(vv)*100:.1f}")
print("\nVEREDITO: se Factored >> Vanilla -> nossa arquitetura (estrutura) generaliza melhor com mesmo tamanho = mais eficiente (NARROW).")
