src = open("organismo.py", encoding="utf-8").read()
n = 0
def rep(a, b):
    global src, n
    assert src.count(a) == 1, a[:200]
    src = src.replace(a, b); n += 1
rep('''                       escopo="fixo", beta_fixo=8.0)''', '''                       escopo="fixo", beta_fixo=8.0, valor_criacao=False)''')
rep('''    # vida 2: o ALCANCE de cada estrutura aprendido x larguras fixas
    "ESCOPO": dict(escopo="aprendido"),
    "LARGURA_3": dict(beta_fixo=2.0),
    "GLOBAL": dict(beta_fixo=0.0),''', '''    # vida 2: valor futuro da criacao em todos; o ALCANCE aprendido x larguras fixas
    "LOCAL2": dict(valor_criacao=True),
    "ESCOPO": dict(escopo="aprendido", valor_criacao=True),
    "LARGURA_3": dict(beta_fixo=2.0, valor_criacao=True),
    "GLOBAL": dict(beta_fixo=0.0, valor_criacao=True),''')
rep('''REGULAR = 2          # "regularidade" = repetir em 2 problemas distintos (projeto)''', '''REGULAR = 2          # "regularidade" = repetir em 2 problemas distintos (projeto)
V0_CRIACAO = 1.0     # a priori, uma criacao rende 1 uso futuro (pseudo-contagem 1)''')
rep('''                v = th / max(c, 1e-3)
                if v > vmelhor:''', '''                v = (th + self._valor_futuro(op)) / max(c, 1e-3)
                if v > vmelhor:''')
rep('''    def _agenda_aprendida(self, ws):''', '''    def _valor_futuro(self, op):
        """Operacao que CRIA estrutura vale tambem pelos usos futuros do que cria
        (aprendido: usos bem-sucedidos posteriores dos produtos / execucoes)."""
        if not self.f["valor_criacao"] or op not in ("inventar", "derivar"):
            return 0.0
        p = self.E[f"op:{op}"].payload
        return (p.get("credito_futuro", 0.0) + V0_CRIACAO) / (p.get("execucoes", 0) + 1.0)

    def _agenda_aprendida(self, ws):''')
rep('''        oe = self.E[f"op:{op}"]
        real = ws.gasto - gasto0''', '''        oe = self.E[f"op:{op}"]
        real = ws.gasto - gasto0
        if op in ("inventar", "derivar"):
            oe.payload["execucoes"] = oe.payload.get("execucoes", 0) + 1''')
rep('''        if e.origem is not None and e.nasc < self.t:
            op = self.E.get(e.origem[0])
            if op is not None:''', '''        if e.origem is not None and e.nasc < self.t:
            op = self.E.get(e.origem[0])
            if op is not None:
                op.payload["credito_futuro"] = op.payload.get("credito_futuro", 0.0) + 1.0''')
open("organismo.py", "w", encoding="utf-8").write(src)
print("trocas", n)
