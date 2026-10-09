"""
A1 — O OPERADOR DE INDUCAO (a peca que faltou no H5). Transformer que INDUZ estrutura IN-CONTEXT:
dado K exemplos (s,a,s2) de uma estrutura NOVA + uma query (s,a) de um par NUNCA VISTO, infere s2.
Estrutura = AFIM s2=(k_a*s + b_a) mod N (regra inedita por dominio, compressivel -> da pra INDUZIR e generalizar,
diferente do aditivo do teste original). Query SO em pares NAO presentes no suporte = testa GENERALIZACAO (induziu
a regra), nao memoria. Compara: indutor in-context vs FactoredWorldModel (o do H5, viesado aditivo) vs chance.
Se o indutor generaliza pra pares nao-vistos >> chance e >> FactoredWorldModel -> a peca que falta e CONSTRUIVEL.
"""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N, A = 13, 6           # N primo -> qualquer k em 1..N-1 e bijetivo
KSUP = 46              # exemplos de suporte (de 78 pares possiveis)

def sample_domains(B):
    k = torch.randint(1, N, (B, A), device=DEV)        # multiplicador por acao (a parte que o aditivo NAO pega)
    b = torch.randint(0, N, (B, A), device=DEV)         # deslocamento por acao
    return k, b

def all_pairs():
    s = torch.arange(N, device=DEV).repeat_interleave(A); a = torch.arange(A, device=DEV).repeat(N)
    return s, a                                          # 78 pares (s,a)

def transition(k, b, s, a):                              # s2 = (k_a*s + b_a) mod N ; aceita s,a de shape [B] ou [B,M]
    sq = (s.dim() == 1)
    if sq: s = s.unsqueeze(1); a = a.unsqueeze(1)
    B = k.shape[0]; bi = torch.arange(B, device=DEV).unsqueeze(1).expand_as(s)
    r = (k[bi, a] * s + b[bi, a]) % N
    return r.squeeze(1) if sq else r

def split_support_query(B):
    s_all, a_all = all_pairs(); P = s_all.shape[0]
    perm = torch.argsort(torch.rand(B, P, device=DEV), dim=1)
    sup = perm[:, :KSUP]; qry = perm[:, KSUP:]           # query = pares NAO no suporte
    return s_all, a_all, sup, qry

# ---------- INDUTOR IN-CONTEXT (transformer) ----------
class Inducer(nn.Module):
    def __init__(s, d=96, heads=4, layers=4):
        super().__init__()
        s.es = nn.Embedding(N, d); s.ea = nn.Embedding(A, d); s.es2 = nn.Embedding(N + 1, d)  # N = token MASK
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True)
        s.tr = nn.TransformerEncoder(enc, layers); s.out = nn.Linear(d, N)
    def forward(s, sup_s, sup_a, sup_s2, q_s, q_a):
        sup = s.es(sup_s) + s.ea(sup_a) + s.es2(sup_s2)                       # [B,K,d]
        q = (s.es(q_s) + s.ea(q_a) + s.es2(torch.full_like(q_s, N))).unsqueeze(1)  # [B,1,d] s2=MASK
        h = s.tr(torch.cat([sup, q], 1))
        return s.out(h[:, -1])                                                # logits do s2 da query

def gather(t, idx):  # t:[B,P], idx:[B,M] -> [B,M]
    return torch.gather(t, 1, idx)

def train_inducer(iters=4000):
    net = Inducer().to(DEV); opt = torch.optim.Adam(net.parameters(), 3e-4)
    s_all, a_all = all_pairs()
    for it in range(iters):
        B = 128; k, b = sample_domains(B); _, _, sup, qry = split_support_query(B)
        S = s_all[sup]; Aa = a_all[sup]; S2 = transition(k, b, S, Aa)
        # 1 query por dominio (par nao-visto)
        qi = qry[:, torch.randint(0, qry.shape[1], (1,), device=DEV)].squeeze(1)
        qs = s_all[qi]; qa = a_all[qi]; qs2 = transition(k, b, qs, qa)
        logit = net(S, Aa, S2, qs, qa)
        loss = F.cross_entropy(logit, qs2); opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0: print(f"  it{it} loss {loss.item():.3f}", flush=True)
    return net

@torch.no_grad()
def eval_inducer(net, n=40):
    s_all, a_all = all_pairs(); acc = []
    for _ in range(n):
        B = 64; k, b = sample_domains(B); _, _, sup, qry = split_support_query(B)
        S = s_all[sup]; Aa = a_all[sup]; S2 = transition(k, b, S, Aa)
        for j in range(qry.shape[1]):
            qi = qry[:, j]; qs = s_all[qi]; qa = a_all[qi]; qs2 = transition(k, b, qs, qa)
            pred = net(S, Aa, S2, qs, qa).argmax(1)
            acc.append((pred == qs2).float().mean().item())
    return sum(acc) / len(acc)

# ---------- baseline: FactoredWorldModel (o do H5) no MESMO benchmark afim ----------
def eval_factored(iters=4000, n=40):
    from core.cross_domain import FactoredWorldModel
    E = torch.randn(N, 16, device=DEV); E = E / E.norm(dim=1, keepdim=True)
    fac = FactoredWorldModel(16, A).to(DEV); opt = torch.optim.Adam(fac.parameters(), 1e-3)
    s_all, a_all = all_pairs()
    for it in range(iters):
        B = 128; k, b = sample_domains(B); _, _, sup, qry = split_support_query(B)
        S = s_all[sup]; Aa = a_all[sup]; S2 = transition(k, b, S, Aa)
        sup_obs = (E[S], F.one_hot(Aa, A).float(), E[S2])
        c = fac.context(sup_obs)
        qi = qry[:, torch.randint(0, qry.shape[1], (1,), device=DEV)].squeeze(1)
        qs = s_all[qi]; qa = a_all[qi]; qs2 = transition(k, b, qs, qa)
        pred, _ = fac(E[qs], F.one_hot(qa, A).float(), c=c)
        loss = F.mse_loss(pred, E[qs2]); opt.zero_grad(); loss.backward(); opt.step()
    accs = []
    with torch.no_grad():
        for _ in range(n):
            B = 64; k, b = sample_domains(B); _, _, sup, qry = split_support_query(B)
            S = s_all[sup]; Aa = a_all[sup]; S2 = transition(k, b, S, Aa)
            c = fac.context((E[S], F.one_hot(Aa, A).float(), E[S2]))
            for j in range(qry.shape[1]):
                qi = qry[:, j]; qs = s_all[qi]; qa = a_all[qi]; qs2 = transition(k, b, qs, qa)
                pred, _ = fac(E[qs], F.one_hot(qa, A).float(), c=c)
                accs.append((torch.cdist(pred, E).argmin(1) == qs2).float().mean().item())
    return sum(accs) / len(accs)

if __name__ == "__main__":
    print(f"[{DEV}] A1 — induzir ESTRUTURA AFIM nova, generalizar p/ pares NUNCA vistos. chance={1/N:.1%}")
    print("treinando INDUTOR in-context (transformer)...")
    net = train_inducer()
    ind = eval_inducer(net)
    print("treinando/avaliando FactoredWorldModel (o do H5) no mesmo benchmark...")
    fac = eval_factored()
    print("=" * 70)
    print(f"  INDUTOR in-context (A1) .... {ind:.1%}   <- generalizou estrutura nova p/ pares nao-vistos?")
    print(f"  FactoredWorldModel (H5) .... {fac:.1%}")
    print(f"  chance ..................... {1/N:.1%}")
    print("=" * 70)
    print("LEITURA: indutor >> chance e >> factored -> a PECA QUE FALTA (inducao de estrutura) e CONSTRUIVEL.")
    print("         indutor ~ chance -> induzir estrutura nova few-shot e duro ate com transformer dedicado (honesto).")
