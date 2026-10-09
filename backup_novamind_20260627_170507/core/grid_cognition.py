"""
core/grid_cognition.py — NUCLEO DE COGNICAO DE GRADE (priors do ARC vivendo em world-model + memoria + imaginacao).

NAO e um classificador de regra. E um WORLD-MODEL GENERATIVO que:
  - infere a REGRA de uma tarefa a partir de poucos pares suporte (memoria episodica da tarefa = contexto);
  - IMAGINA a grade-resultado (decode generativo), nao apenas rotula;
  - compartilha estrutura entre as 7 familias de priors do ARC (objetidade, geometria, simetria, contagem,
    topologia/fisica, automato celular) -> o que deve transferir p/ uma familia NOVA.

7 FAMILIAS (cada uma e um "dominio"/prior; a regra varia dentro da familia -> precisa inferir do suporte):
  geom (dihedral) · translate · mirror/tiling · recolor(objetidade) · cellular(crescer) · counting(numerosidade)
  | gravity/drop (fisica-topologia) = a 7a, HELD-OUT, p/ testar transferencia a um prior NOVO.
"""
import torch, torch.nn as nn, torch.nn.functional as F

H = W = 8
C = 6           # cores (0 = fundo)
HW = H * W

# ---------- mapas de indice p/ familias espaciais (out.flat[k] = in.flat[map[k]]) ----------
def _base(): return torch.arange(HW).reshape(H, W)
def _flatmap(g): return g.reshape(-1).long()
def geom_maps():
    b = _base(); ms = [b, torch.rot90(b, 1), torch.rot90(b, 2), torch.rot90(b, 3),
                       torch.flip(b, [0]), torch.flip(b, [1]), b.t().contiguous(), torch.flip(b.t().contiguous(), [0])]
    return torch.stack([_flatmap(m) for m in ms])
def trans_maps():
    b = _base(); ms = []
    for dy in (-2, -1, 1, 2):
        for dx in (-2, -1, 1, 2): ms.append(torch.roll(b, shifts=(dy, dx), dims=(0, 1)))
    return torch.stack([_flatmap(m) for m in ms])
def mirror_maps():
    b = _base(); out = []
    g = b.clone()                       # espelho L->R
    for j in range(W): g[:, j] = b[:, j if j < W // 2 else W - 1 - j]
    out.append(g.clone())
    g = b.clone()                       # espelho T->B
    for i in range(H): g[i, :] = b[i if i < H // 2 else H - 1 - i, :]
    out.append(g.clone())
    g = b.clone()                       # quad-espelho
    for i in range(H):
        for j in range(W): g[i, j] = b[i if i < H // 2 else H - 1 - i, j if j < W // 2 else W - 1 - j]
    out.append(g.clone())
    return torch.stack([_flatmap(m) for m in out])

class GridDomains:
    """Gera tarefas das 7 familias. Cada tarefa = uma regra (variavel) + grades aleatorias de entrada."""
    def __init__(self, dev, p=0.18):
        self.dev = dev; self.p = p
        self.maps = {"geom": geom_maps().to(dev), "translate": trans_maps().to(dev), "mirror": mirror_maps().to(dev)}
    def rand_in(self, B, M, p=None):
        p = self.p if p is None else p
        g = torch.zeros(B, M, HW, dtype=torch.long, device=self.dev)
        mask = torch.rand(B, M, HW, device=self.dev) < p
        col = torch.randint(1, C, (B, M, HW), device=self.dev)
        g[mask] = col[mask]; return g
    def _spatial(self, fam, B, M):
        x = self.rand_in(B, M); maps = self.maps[fam]
        rule = torch.randint(0, maps.shape[0], (B,), device=self.dev)
        sel = maps[rule]                                          # [B,HW]
        y = torch.gather(x, 2, sel[:, None, :].expand(-1, M, -1))
        return x, y
    def _recolor(self, B, M):
        x = self.rand_in(B, M)
        lut = torch.stack([torch.randperm(C, device=self.dev) for _ in range(B)])   # [B,C] (0 fica? randperm inclui 0)
        lut[:, 0] = 0                                             # fundo continua fundo
        y = torch.gather(lut, 1, x.reshape(B, -1)).reshape(B, M, HW)
        return x, y
    def _cellular(self, B, M):
        x = self.rand_in(B, M, p=0.10).reshape(B, M, H, W)
        steps = 1 + int(torch.randint(0, 2, (1,)).item())        # 1 ou 2 passos de crescimento
        y = x.clone()
        for _ in range(steps):
            nb = torch.zeros_like(y)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = torch.maximum(nb, torch.roll(y, shifts=(dy, dx), dims=(2, 3)))
            y = torch.where(y == 0, nb, y)
        return x.reshape(B, M, HW), y.reshape(B, M, HW)
    def _counting(self, B, M):
        x = self.rand_in(B, M, p=0.12)
        cnt = (x > 0).sum(2)                                     # [B,M] qtd de celulas
        y = torch.zeros_like(x)
        ar = torch.arange(HW, device=self.dev)
        y = torch.where(ar[None, None, :] < cnt[..., None], torch.ones_like(x), torch.zeros_like(x))
        return x, y
    def _gravity(self, B, M):                                    # HELD-OUT: celulas caem p/ baixo (fisica)
        x = self.rand_in(B, M, p=0.16).reshape(B, M, H, W)
        nz = (x != 0).long()
        order = torch.argsort(nz, dim=2, stable=True)            # zeros (0) sobem, nao-zeros (1) descem
        y = torch.gather(x, 2, order)
        return x.reshape(B, M, HW), y.reshape(B, M, HW)
    def gen(self, fam, B, M):
        if fam in self.maps: return self._spatial(fam, B, M)
        return getattr(self, "_" + fam)(B, M)
    def episode(self, fam, B, K, Q):
        xs, ys = self.gen(fam, B, K); xq, yq = self.gen(fam, B, Q)
        return (xs, ys), (xq, yq)

TRAIN_FAMS = ["geom", "translate", "mirror", "recolor", "cellular", "counting"]
HELDOUT_FAM = "gravity"

def onehot(g):  # [*,HW] long -> [*,C,H,W] float
    sh = g.shape[:-1]
    return F.one_hot(g.reshape(*sh, H, W), C).permute(*range(len(sh)), len(sh) + 2, len(sh), len(sh) + 1).float()


class GridWorldModel(nn.Module):
    """World-model generativo ESPACIAL: transformer sobre as HW celulas, condicionado na REGRA inferida do
    suporte (atencao pode rotear qualquer celula -> qualquer celula => consegue rotacao/gravidade/etc. EXATAS).
    Memoria = pares suporte (episodica da tarefa); Imaginacao = gera a grade de saida celula a celula."""
    def __init__(self, d=64, heads=4, layers=3, max_hw=900, colors=11):
        super().__init__()
        self.col = nn.Embedding(colors, d)
        self.pos = nn.Embedding(max_hw, d)        # posicao em escala NORMAL (comparavel a cor) -> sinal espacial real
        self.role = nn.Embedding(3, d)        # 0=query-in, 1=suporte-in, 2=suporte-out
        self.cls = nn.Parameter(torch.randn(d) * 0.1)   # token CLS p/ resumir a regra do par
        self.ctx_tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True, dropout=0.0), 3)
        self.ctx_pool = nn.Linear(d, d)
        self.main_tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True, dropout=0.0), layers)
        self.head = nn.Linear(d, colors)
        self.d = d
        self.max_hw = max_hw
        self.colors = colors
        
    def _tok(self, grid, role):               # grid [N,hw] -> tokens [N,hw,d]
        hw = grid.shape[1]
        pos = self.pos(torch.arange(hw, device=grid.device))[None]   # [1,hw,d]
        return self.col(grid) + pos + self.role.weight[role][None, None]
        
    def context(self, sup):                   # (xs,ys): [B,K,hw] -> c:[B,d]
        xs, ys = sup; B, Kk, hw = xs.shape[0], xs.shape[1], xs.shape[2]
        ti = self._tok(xs.reshape(B * Kk, hw), 1); to = self._tok(ys.reshape(B * Kk, hw), 2)
        cls = self.cls[None, None].expand(B * Kk, 1, self.d)     # CLS resume a transformacao do par
        h = self.ctx_tf(torch.cat([cls, ti, to], 1))[:, 0]       # le o CLS (alinha in<->out por atencao)
        return self.ctx_pool(h.reshape(B, Kk, self.d).mean(1))   # media sobre os K pares -> regra [B,d]
        
    def forward(self, xq, sup):               # IN-CONTEXT: a query atende DIRETAMENTE aos pares de suporte
        xs, ys = sup; B, Kk, hw = xs.shape[0], xs.shape[1], xs.shape[2]; Q = xq.shape[1]
        sin = self._tok(xs.reshape(B * Kk, hw), 1).reshape(B, Kk * hw, self.d)   # suporte-in
        sout = self._tok(ys.reshape(B * Kk, hw), 2).reshape(B, Kk * hw, self.d)  # suporte-out
        sup_tok = torch.cat([sin, sout], 1)                      # [B, 2*K*hw, d] (bag de pares)
        sup_exp = sup_tok[:, None].expand(-1, Q, -1, -1).reshape(B * Q, Kk * hw * 2, self.d)
        qtok = self._tok(xq.reshape(B * Q, hw), 0)               # [B*Q,hw,d]
        seq = torch.cat([sup_exp, qtok], 1)                      # suporte + query numa sequencia so
        h = self.main_tf(seq)[:, -hw:, :]                        # le as posicoes da query
        logits = self.head(h).reshape(B, Q, hw, self.colors).permute(0, 1, 3, 2) # [B, Q, C, hw]
        return logits
