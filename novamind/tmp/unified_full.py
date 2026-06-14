"""
SISTEMA PLASTICO UNIFICADO COMPLETO (os 7 itens de ouro, mecanismo CERTO da pesquisa):
- Benchmark JUSTO: 6 primitivas-base FIXAS (compartilhadas) sobre Z_N; tarefa = COMPOSICAO de 2 (b o a). Treino num
  subconjunto de composicoes; teste em composicoes HELD-OUT (combinacao nunca vista; cada base vista em outras).
  Query em x quaisquer (induzir a regra, nao lookup). = generalizacao SISTEMATICA (a meta do MLC, feita direito).
- IMAGINACAO = refinamento iterativo generativo (estilo TRM/generative-replay): a rede mantem um estado-hipotese e
  RE-SIMULA/refina por T passos (broadcast/re-entrada do workspace), nao 1 passada.
- MAESTRO = controlador GLOBAL WORKSPACE aprendido: a cada passo decide o GANHO da atualizacao (ignicao) e a
  TEMPERATURA (loucura). Ele mesmo e treinado (plastico).
- METAPLASTICIDADE = peso rapido Hebbiano modulado pelo maestro acumula a hipotese vencedora (auto-alteracao).
Ablacoes: BASE (1 passada, sem workspace) vs UNIFICADO (imaginacao+maestro+metaplasticidade).
Mede held-out: o sistema completo generaliza p/ composicao nova MELHOR que a passada unica? Sem LLM. GPU.
"""
import sys, os, itertools, random, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N, K, NB = 10, 20, 6                 # N estados, K demos, NB primitivas base
# 6 permutacoes base FIXAS e compartilhadas (o "alfabeto" que toda tarefa usa)
BASE = torch.stack([torch.randperm(N, device=DEV) for _ in range(NB)])     # [NB,N]
PAIRS = list(itertools.product(range(NB), repeat=2)); random.shuffle(PAIRS)
TRAIN_C, HELD_C = PAIRS[:28], PAIRS[28:]                                    # 28 vistas, 8 held-out

def comp(pair, x):                    # aplica base[a] depois base[b]
    a, b = pair; return BASE[b][BASE[a][x]]

def episode(pairs, B):
    pr = [random.choice(pairs) for _ in range(B)]
    xs = torch.randint(0, N, (B, K), device=DEV)
    ys = torch.stack([comp(pr[i], xs[i]) for i in range(B)])
    qx = torch.randint(0, N, (B,), device=DEV)
    qy = torch.stack([comp(pr[i], qx[i:i+1])[0] for i in range(B)])
    return xs, ys, qx, qy

class System(nn.Module):
    def __init__(s, unified, d=128, heads=8, layers=4, T=5):
        super().__init__(); s.unified = unified; s.T = T
        s.ex = nn.Embedding(N, d); s.ey = nn.Embedding(N + 1, d)
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True)
        s.tr = nn.TransformerEncoder(enc, layers); s.out = nn.Linear(d, N)
        s.ref = nn.GRUCell(d, d)                                  # refinador (imaginacao/re-entrada)
        s.maestro = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 2))   # [ganho, temperatura]
        s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)          # estabilidade (estabilidade-plasticidade)
    def encode(s, xs, ys, qx):
        sup = s.ex(xs) + s.ey(ys); q = (s.ex(qx) + s.ey(torch.full(qx.shape, N, device=DEV))).unsqueeze(1)
        h = s.tr(torch.cat([sup, q], 1)); return h[:, -1], h[:, :-1].mean(1)   # (query_repr, contexto)
    def forward(s, xs, ys, qx):
        qr, ctx = s.encode(xs, ys, qx)
        if not s.unified:
            return s.out(qr)                                      # BASE: 1 passada
        w = qr; B, d = w.shape; Hfast = torch.zeros(B, d, d, device=DEV)
        for t in range(s.T):                                     # IMAGINACAO: refina T passos (workspace re-entrada)
            m = s.maestro(ctx); gain = torch.sigmoid(m[:, 0:1]); temp = torch.sigmoid(m[:, 1:2])
            prop = s.ln1(s.ref(w, qr))                           # propoe hipotese (generativo) + normaliza
            prop = prop + temp * torch.randn_like(prop) * 0.3 * (1.0 - t / s.T)   # temperatura = loucura (decai)
            read = torch.tanh((prop.unsqueeze(1) @ Hfast).squeeze(1) / (d ** 0.5))  # leitura LIMITADA
            w = s.ln2(w + gain * (prop + 0.3 * read - w))        # ignicao + normaliza (estabilidade)
            Hfast = 0.8 * Hfast + 0.2 * gain.unsqueeze(2) * (prop.unsqueeze(2) * w.unsqueeze(1))  # decai + escala
        return s.out(w)

def run(unified, iters=8000):
    net = System(unified).to(DEV); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-2)
    for it in range(iters):
        xs, ys, qx, qy = episode(TRAIN_C, 96)
        loss = F.cross_entropy(net(xs, ys, qx), qy); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if it % 2000 == 0:
            tr = (net(xs, ys, qx).argmax(1) == qy).float().mean().item()
            print(f"  [{'UNIF' if unified else 'BASE'}] it{it} loss {loss.item():.3f} treino {tr:.1%}", flush=True)
    net.eval()
    def ev(pairs, n=50):
        ac = []
        with torch.no_grad():
            for _ in range(n):
                xs, ys, qx, qy = episode(pairs, 96); ac.append((net(xs, ys, qx).argmax(1) == qy).float().mean().item())
        return sum(ac) / len(ac)
    return ev(TRAIN_C), ev(HELD_C)

if __name__ == "__main__":
    print(f"[{DEV}] UNIFICADO COMPLETO | {len(TRAIN_C)} comp vistas | {len(HELD_C)} HELD-OUT | chance={1/N:.1%}")
    bt, bh = run(False); ut, uh = run(True)
    print("=" * 64)
    print(f"  BASE (1 passada)    : vistas {bt:.1%} | HELD-OUT {bh:.1%}")
    print(f"  UNIFICADO (imagin.+maestro+metaplast.): vistas {ut:.1%} | HELD-OUT {uh:.1%}")
    print("=" * 64)
    print(f"  GANHO do sistema completo no HELD-OUT: {uh-bh:+.1%}")
    print("  LEITURA: unificado >> base no held-out -> a imaginacao iterativa + workspace + metaplasticidade")
    print("           AJUDAM a generalizar composicao nova. ~igual -> a 1 passada ja basta (honesto).")
