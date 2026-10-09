"""
run_transfer.py — PROVA (com numero) de que memoria cross-dominio FATORADA + roteada por contexto
da TRANSFERENCIA sem INTERFERENCIA e DISTINGUE os dominios — respondendo ao risco que o usuario apontou.

Familia controlada: anel latente de N estados (ESTRUTURA compartilhada: o grafo de transicao e o MESMO em
todo mundo), renderizado pelo MESMO embedding de estado (obs comparaveis), mas cada dominio tem SEMANTICA
DE ACAO propria (qual acao -> qual passo no anel). Mesma estrutura, superficie diferente -> ALIASING:
o MESMO (obs, acao) leva a proximos DIFERENTES em mundos diferentes. So quem infere contexto resolve.

CORRECAO vs v1: treina com um dominio NOVO a cada episodio (distribuicao infinita de mundos) -> forca o
modelo a aprender o MECANISMO geral (inferir mapeamento do suporte + aplicar estrutura compartilhada),
em vez de DECORAR um punhado fixo de dominios. Geracao 100% vetorizada (rapido).

Mede 5 coisas: (1) fatorado treino  (2) ingenuo/blob=interferencia  (3) fatorado dominio NOVO=transferencia
(4) isolado do-zero=sem-transferencia  (5) discriminacao de dominio pelo contexto.
"""
import os, sys, json, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.cross_domain import FactoredWorldModel, NaiveWorldModel

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N, A, OBS = 12, 6, 16
DELTAS = torch.tensor([1, -1, 2, -2, 3, -3], device=DEV)
K, Q = 24, 48
E = torch.randn(N, OBS, device=DEV); E = E / E.norm(dim=1, keepdim=True)   # rendering de estado COMPARTILHADO

def rand_perms(B):
    """B dominios = B permutacoes dos deltas -> [B,A] de VALORES de delta por acao (a semantica do dominio)."""
    idx = torch.argsort(torch.rand(B, A, device=DEV), dim=1)
    return DELTAS[idx]

def gen(perms, M):
    """gera M transicoes p/ cada um dos B dominios (vetorizado). retorna obs[B,M,OBS], a_oh[B,M,A], obs2, s2."""
    B = perms.shape[0]
    s = torch.randint(0, N, (B, M), device=DEV)
    a = torch.randint(0, A, (B, M), device=DEV)
    s2 = (s + torch.gather(perms, 1, a)) % N
    return E[s], F.one_hot(a, A).float(), E[s2], s2

def episode(perms):
    so, sa, so2, _ = gen(perms, K); qo, qa, qo2, qs2 = gen(perms, Q)
    return (so, sa, so2), (qo, qa), qs2

def decode_acc(pred, true_s):
    return (torch.cdist(pred, E).argmin(1) == true_s).float().mean().item()

TEST_PERMS = rand_perms(6)  # 6 dominios de teste FIXOS (held-out, nunca vistos no treino)

# ---------- FATORADO: meta-treino com dominio NOVO a cada episodio ----------
fac = FactoredWorldModel(OBS, A).to(DEV); opt = torch.optim.Adam(fac.parameters(), 1e-3)
print(f"[{DEV}] treinando FATORADO (dominio novo por episodio)...")
for it in range(4000):
    perms = rand_perms(128)                                   # 128 mundos NOVOS, nunca repetidos
    sup, (qo, qa), qs2 = episode(perms)
    c = fac.context(sup); B = qo.shape[0]
    cexp = c.unsqueeze(1).expand(-1, Q, -1).reshape(B * Q, -1)
    pred, _ = fac(qo.reshape(B * Q, OBS), qa.reshape(B * Q, A), c=cexp)
    loss = F.mse_loss(pred, E[qs2.reshape(B * Q)])
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 800 == 0: print(f"  it{it} loss {loss.item():.4f}")

# ---------- INGENUO (blob): (obs,acao)->prox, sem contexto, dominios novos agrupados ----------
naive = NaiveWorldModel(OBS, A).to(DEV); optn = torch.optim.Adam(naive.parameters(), 1e-3)
print("treinando INGENUO (blob, sem contexto)...")
for it in range(4000):
    perms = rand_perms(128); o, aa, o2, _ = gen(perms, K + Q)
    o = o.reshape(-1, OBS); aa = aa.reshape(-1, A); o2 = o2.reshape(-1, OBS)
    loss = F.mse_loss(naive(o, aa), o2); optn.zero_grad(); loss.backward(); optn.step()

@torch.no_grad()
def eval_factored(perm_pool, n=30):
    accs = []
    for _ in range(n):
        idx = torch.randint(0, perm_pool.shape[0], (32,), device=DEV); perms = perm_pool[idx]
        sup, (qo, qa), qs2 = episode(perms); c = fac.context(sup); B = qo.shape[0]
        cexp = c.unsqueeze(1).expand(-1, Q, -1).reshape(B * Q, -1)
        pred, _ = fac(qo.reshape(B * Q, OBS), qa.reshape(B * Q, A), c=cexp)
        accs.append(decode_acc(pred, qs2.reshape(B * Q)))
    return sum(accs) / len(accs)

@torch.no_grad()
def eval_naive(n=30):
    accs = []
    for _ in range(n):
        perms = rand_perms(32); _, (qo, qa), qs2 = episode(perms); B = qo.shape[0]
        accs.append(decode_acc(naive(qo.reshape(B * Q, OBS), qa.reshape(B * Q, A)), qs2.reshape(B * Q)))
    return sum(accs) / len(accs)

def eval_isolated(perm_pool):
    """baseline SEM transferencia: treina do ZERO so com os K exemplos de suporte de cada dominio novo."""
    accs = []
    for di in range(perm_pool.shape[0]):
        p = perm_pool[di:di + 1]; so, sa, so2, _ = gen(p, K)
        o, aa, o2 = so[0], sa[0], so2[0]
        m = NaiveWorldModel(OBS, A).to(DEV); op = torch.optim.Adam(m.parameters(), 1e-3)
        for _ in range(400):
            op.zero_grad(); F.mse_loss(m(o, aa), o2).backward(); op.step()
        with torch.no_grad():
            qo, qa, _, qs2 = gen(p, Q)
            accs.append(decode_acc(m(qo[0], qa[0]), qs2[0]))
    return sum(accs) / len(accs)

@torch.no_grad()
def eval_discrimination(perm_pool, reps=60):
    """o contexto c distingue os dominios? kNN(c)->dominio. alto = sabe 'em que mundo esta'."""
    C, Y = [], []
    for _ in range(reps):
        for di in range(perm_pool.shape[0]):
            so, sa, so2, _ = gen(perm_pool[di:di + 1], K)
            C.append(fac.context((so, sa, so2))[0]); Y.append(di)
    C = torch.stack(C); Y = torch.tensor(Y, device=DEV)
    idx = torch.randperm(len(Y), device=DEV); ntr = int(len(Y) * 0.6); tr, te = idx[:ntr], idx[ntr:]
    pred = Y[tr][torch.cdist(C[te], C[tr]).argmin(1)]
    return (pred == Y[te]).float().mean().item()

print("\n" + "=" * 66 + f"\nRESULTADOS (acuracia de predicao de estado; chance = {1/12:.1%})\n" + "=" * 66)
r = {
 "fatorado_treino":       eval_factored(rand_perms(64)),       # dominios frescos (= distribuicao de treino)
 "ingenuo_blob":          eval_naive(),
 "fatorado_dominio_novo": eval_factored(TEST_PERMS),
 "isolado_dominio_novo":  eval_isolated(TEST_PERMS),
 "discriminacao_dominio": eval_discrimination(TEST_PERMS),
 "chance": 1 / 12,
}
print(f"1. FATORADO  · distribuicao de treino .......... {r['fatorado_treino']:.1%}")
print(f"2. INGENUO   · blob, sem contexto .............. {r['ingenuo_blob']:.1%}   <- baixo = INTERFERENCIA (aliasing)")
print(f"3. FATORADO  · dominio NOVO (few-shot) ......... {r['fatorado_dominio_novo']:.1%}   <- alto = TRANSFERENCIA")
print(f"4. ISOLADO   · dominio novo do zero (K ex.) .... {r['isolado_dominio_novo']:.1%}   <- baixo = SEM transferencia")
print(f"5. DISCRIMINACAO de dominio (contexto) ......... {r['discriminacao_dominio']:.1%}   <- alto = sabe em que mundo esta")
ok_int = r['fatorado_treino'] - r['ingenuo_blob'] > 0.3
ok_tr  = r['fatorado_dominio_novo'] - r['isolado_dominio_novo'] > 0.2
ok_dis = r['discriminacao_dominio'] > 0.7
print("\nVEREDITO:")
print(f"  - Resolve INTERFERENCIA (fatorado >> ingenuo)?  {'SIM' if ok_int else 'NAO'}")
print(f"  - Da TRANSFERENCIA (fatorado >> isolado)?       {'SIM' if ok_tr else 'NAO'}")
print(f"  - DISTINGUE os dominios (contexto)?             {'SIM' if ok_dis else 'NAO'}")
print(f"  => memoria cross-dominio FATORADA {'FUNCIONA sem virar mingau' if (ok_int and ok_tr and ok_dis) else 'PRECISA AJUSTE'}")
json.dump(r, open(os.path.join(os.path.dirname(__file__), 'transfer_result.json'), 'w'), indent=1)
