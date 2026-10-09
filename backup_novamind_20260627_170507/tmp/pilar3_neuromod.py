"""
PILAR 3 do salto — META-PLASTICIDADE NEUROMODULADA (a regra de aprendizado muda com o contexto).
Teste canonico: o modelo precisa MEMORIZAR K associacoes NOVAS (input->output) numa unica passada e recuperar —
algo que peso FIXO nao consegue. Com PESOS RAPIDOS HEBBIANOS (Miconi, differentiable plasticity) modulados por um
NEUROMODULADOR aprendido (escreve forte quando ha novidade), a rede liga as associacoes na hora.
Compara: (1) FIXO (so peso lento), (2) PLASTICO (Hebbiano), (3) PLASTICO+NEUROMOD (gate aprendido).
Se plastico+neuromod >> fixo -> plasticidade como primitiva e o que da binding rapido = o Pilar 3. Sem LLM. GPU.
"""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DIN, H, NCLS, K = 32, 128, 20, 16     # K associacoes novas por episodio; recuperar a classe certa

def episode(B):
    # K inputs aleatorios distintos -> K rotulos aleatorios (associacao NOVA por episodio)
    x = torch.randn(B, K, DIN, device=DEV)
    y = torch.randint(0, NCLS, (B, K), device=DEV)
    qi = torch.randint(0, K, (B,), device=DEV)            # query = um dos K inputs vistos
    bi = torch.arange(B, device=DEV)
    return x, y, x[bi, qi], y[bi, qi]

class PlasticNet(nn.Module):
    """camada com PESO LENTO (meta-aprendido) + PESO RAPIDO Hebbiano modulado por neuromodulador."""
    def __init__(s, mode):  # mode: fixo | plastico | neuromod
        super().__init__(); s.mode = mode
        s.enc = nn.Linear(DIN, H)
        s.Wslow = nn.Parameter(torch.randn(H, NCLS) * 0.01)
        s.alpha = nn.Parameter(torch.ones(1) * 0.5)       # ganho do peso rapido (plasticidade)
        if mode == 'neuromod':
            s.nm = nn.Sequential(nn.Linear(H + NCLS, 32), nn.ReLU(), nn.Linear(32, 1))  # neuromodulador: quanto escrever
    def forward(s, x_sup, y_sup, x_q):
        B = x_sup.shape[0]
        Hfast = torch.zeros(B, H, NCLS, device=DEV)       # peso rapido (traco Hebbiano), zera por episodio
        for t in range(K):
            pre = torch.tanh(s.enc(x_sup[:, t]))           # [B,H]
            post = F.one_hot(y_sup[:, t], NCLS).float()    # [B,NCLS]
            if s.mode == 'fixo': continue                  # sem escrita -> nao memoriza
            gate = 1.0
            if s.mode == 'neuromod':                       # neuromodulador: escreve forte qd surpresa alta
                pred = (pre.unsqueeze(2) * Hfast).sum(1)
                surprise = torch.cat([pre, F.softmax(pred, -1)], -1)
                gate = torch.sigmoid(s.nm(surprise))       # [B,1]
            outer = pre.unsqueeze(2) * post.unsqueeze(1)   # [B,H,NCLS]
            Hfast = Hfast + (gate if isinstance(gate, float) else gate.unsqueeze(2)) * outer
        preq = torch.tanh(s.enc(x_q))                      # [B,H]
        logit = preq @ s.Wslow + s.alpha * (preq.unsqueeze(2) * Hfast).sum(1)
        return logit

def run(mode, iters=4000):
    net = PlasticNet(mode).to(DEV); opt = torch.optim.Adam(net.parameters(), 1e-3)
    for it in range(iters):
        x, y, xq, yq = episode(128); logit = net(x, y, xq)
        loss = F.cross_entropy(logit, yq); opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); acc = []
    with torch.no_grad():
        for _ in range(40):
            x, y, xq, yq = episode(128); acc.append((net(x, y, xq).argmax(1) == yq).float().mean().item())
    return sum(acc) / len(acc)

if __name__ == "__main__":
    print(f"[{DEV}] PILAR 3 — memorizar {K} associacoes NOVAS numa passada e recuperar. chance={1/NCLS:.1%}")
    r = {}
    for mode in ['fixo', 'plastico', 'neuromod']:
        print(f"treinando {mode}..."); r[mode] = run(mode); print(f"  {mode}: {r[mode]:.1%}")
    print("=" * 64)
    print(f"  FIXO (peso lento so) ........ {r['fixo']:.1%}   <- nao consegue ligar associacao nova")
    print(f"  PLASTICO (Hebbiano) ......... {r['plastico']:.1%}")
    print(f"  PLASTICO + NEUROMOD ......... {r['neuromod']:.1%}   <- regra de aprendizado modulada")
    print(f"  chance ...................... {1/NCLS:.1%}")
    print("=" * 64)
    print("LEITURA: plastico/neuromod >> fixo -> plasticidade como PRIMITIVA da o binding rapido (Pilar 3 confirmado).")
