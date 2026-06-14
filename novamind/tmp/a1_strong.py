"""
A1-FORTE: dar PODER ao indutor. Maior (d=128, 5 camadas), mais treino (15k), MULTI-QUERY por passo (mais sinal),
weight-decay (grokking). Mesmo benchmark: induzir estrutura AFIM nova e generalizar p/ pares NUNCA vistos.
Testa se 18.8% era falta de poder (sobe) ou se inducao few-shot de estrutura e fundamentalmente dura (plato honesto).
"""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import a1_structure_inducer as A1
DEV = A1.DEV; N = A1.N; A = A1.A

class Big(nn.Module):
    def __init__(s, d=128, heads=8, layers=5):
        super().__init__()
        s.es = nn.Embedding(N, d); s.ea = nn.Embedding(A, d); s.es2 = nn.Embedding(N + 1, d)
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True, dropout=0.0)
        s.tr = nn.TransformerEncoder(enc, layers); s.out = nn.Linear(d, N)
    def forward(s, sup_s, sup_a, sup_s2, q_s, q_a):
        sup = s.es(sup_s) + s.ea(sup_a) + s.es2(sup_s2)
        q = (s.es(q_s) + s.ea(q_a) + s.es2(torch.full_like(q_s, N)))           # [B,Q,d]
        h = s.tr(torch.cat([sup, q], 1))
        return s.out(h[:, sup.shape[1]:])                                       # logits de TODAS as queries [B,Q,N]

def run():
    net = Big().to(DEV); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-2)
    s_all, a_all = A1.all_pairs()
    for it in range(15000):
        B = 128; k, b = A1.sample_domains(B); _, _, sup, qry = A1.split_support_query(B)
        S = s_all[sup]; Aa = a_all[sup]; S2 = A1.transition(k, b, S, Aa)
        QS = s_all[qry]; QA = a_all[qry]; QS2 = A1.transition(k, b, QS, QA)       # TODAS as queries (multi)
        logit = net(S, Aa, S2, QS, QA)                                           # [B,Q,N]
        loss = F.cross_entropy(logit.reshape(-1, N), QS2.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 2000 == 0:
            acc = (logit.argmax(-1) == QS2).float().mean().item()
            print(f"  it{it} loss {loss.item():.3f} acc(treino-query) {acc:.1%}", flush=True)
    # eval held-out
    net.eval(); accs = []
    with torch.no_grad():
        for _ in range(40):
            B = 64; k, b = A1.sample_domains(B); _, _, sup, qry = A1.split_support_query(B)
            S = s_all[sup]; Aa = a_all[sup]; S2 = A1.transition(k, b, S, Aa)
            QS = s_all[qry]; QA = a_all[qry]; QS2 = A1.transition(k, b, QS, QA)
            pred = net(S, Aa, S2, QS, QA).argmax(-1)
            accs.append((pred == QS2).float().mean().item())
    print("=" * 64)
    print(f"  A1-FORTE (held-out, pares nunca vistos) ... {sum(accs)/len(accs):.1%}")
    print(f"  (A1 fraco antes = 18.8% | factored = 7.9% | chance = {1/N:.1%})")
    print("  LEITURA: subiu forte -> era falta de poder (peca construivel). Plato ~20% -> inducao few-shot e dura.")

if __name__ == "__main__":
    print(f"[{DEV}] A1-FORTE: indutor turbinado induzindo estrutura afim nova")
    run()
