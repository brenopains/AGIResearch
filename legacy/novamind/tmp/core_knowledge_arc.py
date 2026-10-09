"""
ATAQUE REAL AO ARC — DSL de CORE-KNOWLEDGE (Spelke: objeto/geometria/numero/topologia) + COMPOSICAO META-APRENDIDA.
A receita honesta que emergiu: cobrir o manifold CERTO (primitivas core) com o mecanismo que JA funciona (79.6%).
Primitivas same-shape parameter-free sobre grade quadrada: geometria (rot/flip/transp), translacao (shift),
fisica (gravidade), topologia (contorno/preencher), numero (ciclo de cor). Tarefa = COMPOSICAO de 2 primitivas.
Meta-treina sobre composicoes; testa em composicoes HELD-OUT (combinacao nunca vista) = generalizacao SISTEMATICA.
VERIFICA que aprende o treino ANTES de confiar no held-out. Sem LLM. GPU. Se held-out alto -> o ataque funciona.
"""
import sys, os, random, itertools, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
torch.manual_seed(0); random.seed(0); np.random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
G, C, K = 5, 4, 4                    # grade 5x5 (quadrada p/ transpose), 4 cores, 4 demos

# ---------- CORE-KNOWLEDGE primitivas (grade->grade, same-shape, parameter-free) ----------
def rot180(g): return np.rot90(g, 2)
def fliph(g): return g[:, ::-1]
def flipv(g): return g[::-1, :]
def transp(g): return g.T
def shiftU(g): return np.roll(g, -1, 0)
def shiftD(g): return np.roll(g, 1, 0)
def shiftL(g): return np.roll(g, -1, 1)
def shiftR(g): return np.roll(g, 1, 1)
def color_cycle(g): o = g.copy(); o[g != 0] = (g[g != 0] % (C - 1)) + 1; return o   # numero: cor->prox (nao-fundo)
def gravityD(g):                                                                     # fisica: nao-fundo cai
    o = np.zeros_like(g)
    for j in range(g.shape[1]):
        col = g[:, j]; nz = col[col != 0]; o[g.shape[0] - len(nz):, j] = nz
    return o
PRIMS = [rot180, fliph, flipv, transp, shiftU, shiftD, shiftL, shiftR, color_cycle, gravityD]
PAIRS = list(itertools.product(range(len(PRIMS)), repeat=2)); random.shuffle(PAIRS)
TRAIN_C, HELD_C = PAIRS[:78], PAIRS[78:]      # 78 vistas, 22 held-out

def apply_pair(pair, g):
    a, b = pair; return np.ascontiguousarray(PRIMS[b](np.ascontiguousarray(PRIMS[a](g))))

def rand_grid():                              # grade com OBJETOS (p/ gravidade/contorno fazerem sentido)
    g = np.zeros((G, G), int)
    for _ in range(random.randint(2, 4)):
        c = random.randint(1, C - 1); y, x = random.randint(0, G - 1), random.randint(0, G - 1)
        g[y, x] = c
        if random.random() < 0.5 and y + 1 < G: g[y + 1, x] = c
    return g

def make_episode(pairs, B):
    xs = np.zeros((B, K + 1, G, G), int); ys = np.zeros((B, K + 1, G, G), int)
    for i in range(B):
        pr = random.choice(pairs)
        for k in range(K + 1):
            gi = rand_grid(); xs[i, k] = gi; ys[i, k] = apply_pair(pr, gi)
    return torch.tensor(xs, device=DEV, dtype=torch.long), torch.tensor(ys, device=DEV, dtype=torch.long)

class Solver(nn.Module):
    def __init__(s, d=160, heads=8, layers=6):
        super().__init__()
        s.col = nn.Embedding(C + 1, d); s.pos = nn.Embedding(G * G, d); s.seg = nn.Embedding(3, d)
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True); s.tr = nn.TransformerEncoder(enc, layers)
        s.out = nn.Linear(d, C)
    def tok(s, g, seg):
        B = g.shape[0]; idx = torch.arange(G * G, device=DEV)
        return s.col(g.reshape(B, -1)) + s.pos(idx)[None] + s.seg(torch.full((G * G,), seg, device=DEV))[None]
    def forward(s, xs, ys):
        B = xs.shape[0]; toks = []
        for k in range(K): toks += [s.tok(xs[:, k], 0), s.tok(ys[:, k], 1)]
        toks.append(s.tok(xs[:, K], 2))
        qm = s.col(torch.full((B, G * G), C, device=DEV)) + s.pos(torch.arange(G * G, device=DEV))[None] + s.seg(torch.full((G * G,), 1, device=DEV))[None]
        toks.append(qm)
        h = s.tr(torch.cat(toks, 1)); return s.out(h[:, -G * G:]).reshape(B, G, G, C)

def run(iters=16000):
    net = Solver().to(DEV); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-2)
    for it in range(iters):
        xs, ys = make_episode(TRAIN_C, 64); logit = net(xs, ys)
        loss = F.cross_entropy(logit.reshape(-1, C), ys[:, K].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if it % 2000 == 0:
            ex = (logit.argmax(-1) == ys[:, K]).all(-1).all(-1).float().mean().item()
            print(f"  it{it} loss {loss.item():.3f} treino(grade-exata) {ex:.1%}", flush=True)
    net.eval()
    def ev(pairs, n=40):
        ac = []
        with torch.no_grad():
            for _ in range(n):
                xs, ys = make_episode(pairs, 64); ac.append((net(xs, ys).argmax(-1) == ys[:, K]).all(-1).all(-1).float().mean().item())
        return sum(ac) / len(ac)
    return ev(TRAIN_C), ev(HELD_C)

if __name__ == "__main__":
    print(f"[{DEV}] CORE-KNOWLEDGE + COMPOSICAO | {len(PRIMS)} primitivas | {len(TRAIN_C)} comp treino | {len(HELD_C)} HELD-OUT")
    tr, he = run()
    print("=" * 64)
    print(f"  composicoes de TREINO (aprendeu?) ........ {tr:.1%}")
    print(f"  composicoes HELD-OUT (generaliza?) ....... {he:.1%}   <- o ataque ao muro")
    print("=" * 64)
    print("  LEITURA: treino alto E held-out alto -> core-knowledge + composicao meta-aprendida GENERALIZA")
    print("           p/ tarefa nova = o ataque funciona (caminho viavel pro ARC, sem LLM). treino alto/held-out baixo")
    print("           -> aprendeu mas nao compoe; treino baixo -> arquitetura/treino ainda fraco (honesto).")
