"""
SISTEMA PLASTICO UNIFICADO — a visao do usuario: inducao + plasticidade + memoria, COLADOS por um MAESTRO APRENDIDO
e plastico (auto-modela), com auto-alteracao de pesos (fast-weights). Benchmark que EXIGE os dois mecanismos:
  final(x) = M[(x + s) % N]   onde s = shift COMPOSICIONAL (generalizavel, inducao infere de poucos exemplos)
                              e M = permutacao por-episodio (BINDING arbitrario, tem que MEMORIZAR = plasticidade).
Inducao sozinha generaliza s mas nao sabe M; plasticidade sozinha memoriza pares mas nao generaliza p/ x novo.
O MAESTRO mistura os dois e regula a plasticidade. Ablacoes: FULL vs so-inducao vs so-plastico vs sem-maestro.
Mede: o todo-colado-pelo-maestro supera as partes? A cola faz diferenca? Sem LLM. GPU. (System 1+2 juntos.)
"""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N, K, D = 16, 24, 96            # N simbolos, K demos de suporte

def episode(B):
    s = torch.randint(0, N, (B,), device=DEV)                  # shift composicional
    M = torch.argsort(torch.rand(B, N, device=DEV), 1)         # permutacao binding por-episodio
    xs = torch.randint(0, N, (B, K), device=DEV)               # suporte
    bi = torch.arange(B, device=DEV).unsqueeze(1)
    ys = M[bi, (xs + s.unsqueeze(1)) % N]                       # final = M[(x+s)%N]
    qx = torch.randint(0, N, (B,), device=DEV)                 # query (pode ser x novo)
    qy = M[torch.arange(B, device=DEV), (qx + s) % N]
    return xs, ys, qx, qy

class Unified(nn.Module):
    def __init__(s, mode='full'):
        super().__init__(); s.mode = mode
        s.ex = nn.Embedding(N, D); s.ey = nn.Embedding(N + 1, D)     # +1 = MASK
        enc = nn.TransformerEncoderLayer(D, 4, D * 2, batch_first=True)
        s.tr = nn.TransformerEncoder(enc, 3)                          # INDUCAO (in-context)
        s.ind_out = nn.Linear(D, N)
        s.key = nn.Linear(D, D); s.val = nn.Embedding(N, D)          # PLASTICIDADE (fast-weights associativo)
        s.pl_out = nn.Linear(D, N)
        s.maestro = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 2))  # MAESTRO aprendido: [gain_plast, mix]
    def forward(s, xs, ys, qx):
        B = xs.shape[0]
        # contexto do maestro = resumo do suporte
        ctx = (s.ex(xs) + s.ey(ys)).mean(1)                          # [B,D]
        m = s.maestro(ctx); gain = torch.sigmoid(m[:, 0:1]); mix = torch.sigmoid(m[:, 1:2])
        if s.mode == 'sem_maestro': gain = torch.full_like(gain, 0.5); mix = torch.full_like(mix, 0.5)
        # INDUCAO
        sup = s.ex(xs) + s.ey(ys); q = (s.ex(qx) + s.ey(torch.full((B,), N, device=DEV))).unsqueeze(1)
        h = s.tr(torch.cat([sup, q], 1)); ind_logit = s.ind_out(h[:, -1])
        # PLASTICIDADE: memoria associativa fast-weight (chave=embed x, valor=embed y), modulada pelo gain
        kx = s.key(s.ex(xs))                                          # [B,K,D]
        vy = s.val(ys)                                                # [B,K,D]
        Hfast = (gain.unsqueeze(1) * kx).transpose(1, 2) @ vy         # [B,D,D] traco Hebbiano modulado
        kq = s.key(s.ex(qx)).unsqueeze(1)                            # [B,1,D]
        read = (kq @ Hfast).squeeze(1)                               # [B,D]
        pl_logit = s.pl_out(read)
        if s.mode == 'so_inducao': return ind_logit
        if s.mode == 'so_plastico': return pl_logit
        return mix * ind_logit + (1 - mix) * pl_logit                # MAESTRO mistura (a cola)

def run(mode, iters=5000):
    net = Unified(mode).to(DEV); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-2)
    for it in range(iters):
        xs, ys, qx, qy = episode(128)
        loss = F.cross_entropy(net(xs, ys, qx), qy); opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); acc = []
    with torch.no_grad():
        for _ in range(60):
            xs, ys, qx, qy = episode(128); acc.append((net(xs, ys, qx).argmax(1) == qy).float().mean().item())
    return sum(acc) / len(acc)

if __name__ == "__main__":
    print(f"[{DEV}] SISTEMA UNIFICADO | final(x)=M[(x+s)%N] (compor s + memorizar M) | chance={1/N:.1%}")
    r = {}
    for mode in ['so_inducao', 'so_plastico', 'sem_maestro', 'full']:
        r[mode] = run(mode); print(f"  {mode:12s}: {r[mode]:.1%}", flush=True)
    print("=" * 60)
    print(f"  SO INDUCAO ............ {r['so_inducao']:.1%}   (generaliza s, mas sabe M?)")
    print(f"  SO PLASTICO ........... {r['so_plastico']:.1%}   (memoriza M, mas generaliza s?)")
    print(f"  SEM MAESTRO (mix fixo)  {r['sem_maestro']:.1%}")
    print(f"  FULL (maestro cola) ... {r['full']:.1%}   <- o todo-colado supera as partes?")
    print("=" * 60)
    best_part = max(r['so_inducao'], r['so_plastico'])
    print(f"LEITURA: full > max(partes)={best_part:.1%} e full > sem_maestro -> a COLA (maestro) faz o todo>partes.")
    print("         full ~ partes -> a cola nao agrega aqui (o transformer ja faz tudo in-context) = honesto.")
