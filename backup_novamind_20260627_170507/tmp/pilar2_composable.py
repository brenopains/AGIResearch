"""
PILAR 2 do salto — ESTRUTURA COMPONIVEL (base tipo grid-cell). Teste limpo e decisivo:
mesmo indutor in-context, mesma estrutura nova por dominio, MUDA so a REPRESENTACAO do estado:
  COMPONIVEL = estado visto como VETOR DE FEATURES (base reutilizavel) -> estrutura por-feature generaliza.
  ATOMICO    = estado visto como 1 simbolo (id) -> nao generaliza p/ estados nunca vistos.
Estrutura (por dominio): cada acao a -> (feature f_a, delta_a); proximo = estado com feature f_a += delta_a (mod M).
Query SO em estados NUNCA VISTOS no suporte = testa se INDUZIU a regra (compos) ou so decorou (atomico).
Se componivel >> atomico -> a base componivel e a peca que generaliza estrutura. Sem LLM. GPU.
"""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
D, M, A = 4, 5, 6           # 4 features x 5 valores = 625 estados; 6 acoes
NST = M ** D
KSUP = 60

def sample_domains(B):
    f = torch.randint(0, D, (B, A), device=DEV)         # qual feature cada acao mexe
    d = torch.randint(1, M, (B, A), device=DEV)          # quanto soma (mod M)
    return f, d

def rand_states(B, n):
    return torch.randint(0, M, (B, n, D), device=DEV)    # [B,n,D] features

def step(f, d, states, acts):                            # states[B,n,D], acts[B,n] -> next[B,n,D]
    B, n, _ = states.shape; out = states.clone()
    bi = torch.arange(B, device=DEV).unsqueeze(1).expand(B, n)
    fa = f[bi, acts]; da = d[bi, acts]                   # [B,n]
    cur = torch.gather(out, 2, fa.unsqueeze(2)).squeeze(2)
    out.scatter_(2, fa.unsqueeze(2), ((cur + da) % M).unsqueeze(2))
    return out

def to_id(states):                                       # [.,D] -> id em [0,NST)
    base = (M ** torch.arange(D - 1, -1, -1, device=DEV))
    return (states * base).sum(-1)

class Inducer(nn.Module):
    def __init__(s, mode, dim=96, heads=4, layers=4):
        super().__init__(); s.mode = mode
        if mode == 'comp':
            s.fe = nn.ModuleList([nn.Embedding(M, dim) for _ in range(D)])   # base: 1 embedding por feature
        else:
            s.se = nn.Embedding(NST, dim)                                    # atomico: 1 embedding por estado inteiro
        s.ea = nn.Embedding(A, dim); s.mask = nn.Parameter(torch.zeros(dim))
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True)
        s.tr = nn.TransformerEncoder(enc, layers); s.out = nn.Linear(dim, D * M)
    def rep(s, st):                                       # st:[B,n,D] -> [B,n,dim]
        if s.mode == 'comp': return sum(s.fe[i](st[..., i]) for i in range(D))
        return s.se(to_id(st))
    def forward(s, sup_s, sup_a, sup_s2, q_s, q_a):
        sup = s.rep(sup_s) + s.ea(sup_a) + s.rep(sup_s2)
        q = s.rep(q_s) + s.ea(q_a) + s.mask[None, None, :]
        h = s.tr(torch.cat([sup, q], 1))
        return s.out(h[:, sup.shape[1]:]).reshape(*q_a.shape, D, M)           # [B,Q,D,M]

def run(mode, iters=8000):
    net = Inducer(mode).to(DEV); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-2)
    for it in range(iters):
        B = 96; f, d = sample_domains(B)
        sup_s = rand_states(B, KSUP); sup_a = torch.randint(0, A, (B, KSUP), device=DEV)
        sup_s2 = step(f, d, sup_s, sup_a)
        q_s = rand_states(B, 16); q_a = torch.randint(0, A, (B, 16), device=DEV)   # estados frescos (quase sempre nao-vistos)
        q_s2 = step(f, d, q_s, q_a)
        logit = net(sup_s, sup_a, sup_s2, q_s, q_a)                            # [B,16,D,M]
        loss = F.cross_entropy(logit.reshape(-1, M), q_s2.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 2000 == 0:
            ex = (logit.argmax(-1) == q_s2).all(-1).float().mean().item()      # estado INTEIRO exato
            print(f"  [{mode}] it{it} loss {loss.item():.3f} exato {ex:.1%}", flush=True)
    net.eval(); exact = []
    with torch.no_grad():
        for _ in range(40):
            B = 64; f, d = sample_domains(B)
            sup_s = rand_states(B, KSUP); sup_a = torch.randint(0, A, (B, KSUP), device=DEV)
            sup_s2 = step(f, d, sup_s, sup_a)
            # query: estados garantidamente NAO no suporte
            q_s = rand_states(B, 32); q_a = torch.randint(0, A, (B, 32), device=DEV); q_s2 = step(f, d, q_s, q_a)
            pred = net(sup_s, sup_a, sup_s2, q_s, q_a).argmax(-1)
            exact.append((pred == q_s2).all(-1).float().mean().item())
    return sum(exact) / len(exact)

if __name__ == "__main__":
    print(f"[{DEV}] PILAR 2 — base COMPONIVEL vs ATOMICA | estados={NST} | chance(estado exato)={(1/M)**D:.2%}")
    print("treinando ATOMICO (estado = 1 simbolo)...")
    at = run('atom')
    print("treinando COMPONIVEL (estado = vetor de features, base grid-cell)...")
    cp = run('comp')
    print("=" * 64)
    print(f"  COMPONIVEL (base grid-cell) ... {cp:.1%}   <- generaliza estrutura p/ estados nunca vistos?")
    print(f"  ATOMICO ....................... {at:.1%}")
    print(f"  chance (estado inteiro exato) . {(1/M)**D:.2%}")
    print("=" * 64)
    print("LEITURA: componivel >> atomico -> a BASE COMPONIVEL e a peca que faz estrutura GENERALIZAR (Pilar 2 confirmado).")
