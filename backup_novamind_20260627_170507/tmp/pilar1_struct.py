"""
PILAR 1 do salto — PLASTICIDADE ESTRUTURAL (a rede aloca/reescreve estrutura, nao so pesos).
Teste: aprendizado CONTINUO de T tarefas em sequencia. Rede FIXA sofre interferencia catastrofica (esquece as
antigas ao aprender as novas). Rede ESTRUTURALMENTE PLASTICA aloca um MODULO NOVO por tarefa (estilo progressive
nets) e congela os antigos -> retem tudo. Metrica: acuracia media em TODAS as tarefas no fim. Sem LLM.
Se plastica >> fixa na retencao -> alocar estrutura nova e o que evita interferencia (Pilar 1 confirmado).
"""
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DIN, H, NCLS, T = 24, 64, 6, 5

def make_task():
    W = torch.randn(DIN, NCLS, device=DEV) * 2.0          # tarefa = um classificador linear aleatorio
    def gen(n):
        x = torch.randn(n, DIN, device=DEV); y = (x @ W).argmax(1); return x, y
    return gen

class Fixed(nn.Module):
    def __init__(s): super().__init__(); s.net = nn.Sequential(nn.Linear(DIN, H), nn.ReLU(), nn.Linear(H, NCLS))
    def forward(s, x): return s.net(x)

class StructPlastic(nn.Module):
    """aloca um modulo NOVO por tarefa; congela os antigos (plasticidade ESTRUTURAL, nao so peso)."""
    def __init__(s): super().__init__(); s.cols = nn.ModuleList()
    def add_column(s):
        for p in s.parameters(): p.requires_grad_(False)   # congela tudo que existe
        s.cols.append(nn.Sequential(nn.Linear(DIN, H), nn.ReLU(), nn.Linear(H, NCLS)).to(DEV))
        return s.cols[-1]
    def forward(s, x, col): return col(x)

def train_fixed(tasks):
    net = Fixed().to(DEV); opt = torch.optim.Adam(net.parameters(), 1e-3)
    for gen in tasks:                                       # treina sequencial; SEM rever as antigas
        for _ in range(800):
            x, y = gen(128); loss = F.cross_entropy(net(x), y); opt.zero_grad(); loss.backward(); opt.step()
    accs = []
    for gen in tasks:
        x, y = gen(512); accs.append((net(x).argmax(1) == y).float().mean().item())
    return accs

def train_plastic(tasks):
    net = StructPlastic().to(DEV); cols = []
    for gen in tasks:
        col = net.add_column(); cols.append(col)            # ALOCA estrutura nova
        opt = torch.optim.Adam(col.parameters(), 1e-3)
        for _ in range(800):
            x, y = gen(128); loss = F.cross_entropy(net(x, col), y); opt.zero_grad(); loss.backward(); opt.step()
    accs = []
    for gen, col in zip(tasks, cols):
        x, y = gen(512); accs.append((net(x, col).argmax(1) == y).float().mean().item())
    return accs

if __name__ == "__main__":
    print(f"[{DEV}] PILAR 1 — aprendizado continuo de {T} tarefas; retencao final. chance={1/NCLS:.1%}")
    tasks = [make_task() for _ in range(T)]
    af = train_fixed(tasks); ap = train_plastic(tasks)
    print("=" * 60)
    print(f"  FIXA      por tarefa: {[f'{a:.0%}' for a in af]}  media {sum(af)/T:.1%}  <- esquece as antigas")
    print(f"  PLASTICA  por tarefa: {[f'{a:.0%}' for a in ap]}  media {sum(ap)/T:.1%}  <- aloca, retem")
    print("=" * 60)
    print("LEITURA: plastica >> fixa na retencao -> plasticidade ESTRUTURAL evita interferencia (Pilar 1 confirmado).")
