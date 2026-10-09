"""
H5 — TESTE DECISIVO: a fatoração transfere ESTRUTURA NOVA (nao so conteudo novo)?
O teste original (run_transfer.py, 85%) mantem a ESTRUTURA fixa (s2=(s+delta[a])%N, aditiva ciclica) e varia so
o CONTEUDO (qual delta por acao). Aqui troco por ESTRUTURA ARBITRARIA: cada dominio = um conjunto PROPRIO de A
permutacoes de estados (transicao qualquer). Meta-treino sobre estruturas aleatorias; teste few-shot em estruturas
HELD-OUT nunca vistas. Comparo com o caso aditivo (facil) lado a lado. Se a transferencia cair p/ ~isolado/chance
quando a estrutura e arbitraria -> a fatoracao transfere CONTEUDO, nao ESTRUTURA NOVA = exatamente o muro do ARC.
Honesto: e o teste que separa 'parece generalizar' de 'generaliza estrutura de verdade'.
"""
import sys, os, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.cross_domain import FactoredWorldModel, NaiveWorldModel
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N, A, OBS = 12, 6, 16
K, Q = 24, 48
E = torch.randn(N, OBS, device=DEV); E = E / E.norm(dim=1, keepdim=True)
DELTAS = torch.tensor([1, -1, 2, -2, 3, -3], device=DEV)

# ---- estruturas ----
def add_perms(B):                      # ESTRUTURA ADITIVA (a do teste original): so muda qual delta por acao
    idx = torch.argsort(torch.rand(B, A, device=DEV), dim=1); return ('add', DELTAS[idx])
def arb_perms(B):                      # ESTRUTURA ARBITRARIA: cada (dominio,acao) = uma permutacao QUALQUER de N estados
    return ('arb', torch.argsort(torch.rand(B, A, N, device=DEV), dim=2))   # [B,A,N] -> proximo estado

def step(struct, s, a):
    kind, P = struct
    if kind == 'add':
        return (s + torch.gather(P, 1, a)) % N
    else:                              # arbitraria
        B = s.shape[0]
        bi = torch.arange(B, device=DEV).unsqueeze(1).expand_as(s)
        return P[bi, a, s]

def gen(struct, M):
    B = struct[1].shape[0]
    s = torch.randint(0, N, (B, M), device=DEV); a = torch.randint(0, A, (B, M), device=DEV)
    s2 = step(struct, s, a)
    return E[s], F.one_hot(a, A).float(), E[s2], s2

def episode(struct):
    so, sa, so2, _ = gen(struct, K); qo, qa, qo2, qs2 = gen(struct, Q)
    return (so, sa, so2), (qo, qa), qs2

def decode_acc(pred, true_s): return (torch.cdist(pred, E).argmin(1) == true_s).float().mean().item()

def train_factored(gen_struct, iters=3000):
    fac = FactoredWorldModel(OBS, A).to(DEV); opt = torch.optim.Adam(fac.parameters(), 1e-3)
    for it in range(iters):
        struct = gen_struct(128); sup, (qo, qa), qs2 = episode(struct)
        c = fac.context(sup); B = qo.shape[0]
        cexp = c.unsqueeze(1).expand(-1, Q, -1).reshape(B * Q, -1)
        pred, _ = fac(qo.reshape(B * Q, OBS), qa.reshape(B * Q, A), c=cexp)
        loss = F.mse_loss(pred, E[qs2.reshape(B * Q)]); opt.zero_grad(); loss.backward(); opt.step()
    return fac

@torch.no_grad()
def eval_factored(fac, struct, n=30):
    accs = []
    for _ in range(n):
        sup, (qo, qa), qs2 = episode(struct); c = fac.context(sup); B = qo.shape[0]
        cexp = c.unsqueeze(1).expand(-1, Q, -1).reshape(B * Q, -1)
        pred, _ = fac(qo.reshape(B * Q, OBS), qa.reshape(B * Q, A), c=cexp)
        accs.append(decode_acc(pred, qs2.reshape(B * Q)))
    return sum(accs) / len(accs)

def eval_isolated(struct):             # baseline SEM transferencia: aprende do ZERO so com os K de suporte
    kind, P = struct; B = P.shape[0]; accs = []
    for b in range(min(B, 6)):
        sub = (kind, P[b:b+1]); so, sa, so2, ss2 = gen(sub, K)
        net = torch.nn.Sequential(torch.nn.Linear(OBS + A, 64), torch.nn.ReLU(), torch.nn.Linear(64, OBS)).to(DEV)
        opt = torch.optim.Adam(net.parameters(), 3e-3)
        x = torch.cat([so[0], sa[0]], -1)
        for _ in range(300):
            loss = F.mse_loss(net(x), so2[0]); opt.zero_grad(); loss.backward(); opt.step()
        qo, qa, qo2, qs2 = gen(sub, Q)
        with torch.no_grad(): accs.append(decode_acc(net(torch.cat([qo[0], qa[0]], -1)), qs2[0]))
    return sum(accs) / len(accs)

if __name__ == "__main__":
    print(f"[{DEV}] H5 — transferencia de ESTRUTURA (nao so conteudo). chance={1/N:.1%}")
    print("treinando FATORADO em estrutura ADITIVA (a do teste original)...")
    fac_add = train_factored(add_perms)
    print("treinando FATORADO em estrutura ARBITRARIA (estrutura nova por dominio)...")
    fac_arb = train_factored(arb_perms)
    HOLD_ADD = add_perms(6); HOLD_ARB = arb_perms(6)
    print("=" * 70)
    print("CASO A — ESTRUTURA ADITIVA fixa, conteudo novo (o teste 'facil', o original):")
    print(f"  fatorado, dominio novo (few-shot) ... {eval_factored(fac_add, HOLD_ADD):.1%}   <- transferencia de CONTEUDO")
    print(f"  isolado (do zero, K exemplos) ....... {eval_isolated(HOLD_ADD):.1%}")
    print("CASO B — ESTRUTURA ARBITRARIA nova por dominio (o teste DECISIVO, novo):")
    print(f"  fatorado, ESTRUTURA nova (few-shot) . {eval_factored(fac_arb, HOLD_ARB):.1%}   <- transferencia de ESTRUTURA")
    print(f"  isolado (do zero, K exemplos) ....... {eval_isolated(HOLD_ARB):.1%}")
    # cross: o modelo treinado em aditivo aguenta estrutura arbitraria? (generalizacao out-of-family)
    print(f"  fatorado-treino-ADITIVO testado em ARBITRARIA (out-of-family): {eval_factored(fac_add, HOLD_ARB):.1%}")
    print("=" * 70)
    print("LEITURA: se CASO A alto e CASO B ~ isolado/chance -> a fatoracao transfere CONTEUDO, NAO estrutura nova")
    print("         = exatamente o muro do ARC. Se CASO B tambem alto -> sinal forte de generalizacao de estrutura.")
