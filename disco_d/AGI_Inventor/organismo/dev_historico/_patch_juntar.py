src = open("organismo.py", encoding="utf-8").read()
n = 0
def rep(a, b):
    global src, n
    assert src.count(a) == 1, a[:200]
    src = src.replace(a, b); n += 1
rep('''                       escopo="fixo", beta_fixo=8.0, valor_criacao=False)''', '''                       escopo="fixo", beta_fixo=8.0, valor_criacao=False,
                       juntar_contrafactual=False)''')
rep('''    "LOCAL2": dict(valor_criacao=True),
    "ESCOPO": dict(escopo="aprendido", valor_criacao=True),
    "LARGURA_3": dict(beta_fixo=2.0, valor_criacao=True),
    "GLOBAL": dict(beta_fixo=0.0, valor_criacao=True),''', '''    "LOCAL2": dict(valor_criacao=True, juntar_contrafactual=True),
    "ESCOPO": dict(escopo="aprendido", valor_criacao=True, juntar_contrafactual=True),
    "LARGURA_3": dict(beta_fixo=2.0, valor_criacao=True, juntar_contrafactual=True),
    "GLOBAL": dict(beta_fixo=0.0, valor_criacao=True, juntar_contrafactual=True),''')
rep('''        juntados = [a["arg"] for a in self.acoes if a["op"] == "juntar"]
        if len(self.dados) > 1:
            realmente += juntados''', '''        if self.contrafactual:
            # so conta o que foi juntado quando ainda NAO havia resposta valida
            # e a resposta que se tornou valida usa o dado juntado
            juntados = [a["arg"] for a in self.acoes if a["op"] == "juntar"
                        and not a.get("valido_antes", False)]
        else:
            juntados = [a["arg"] for a in self.acoes if a["op"] == "juntar"]
        if len(self.dados) > 1 and (ok or not self.contrafactual):
            realmente += juntados''')
rep('''        self.leis_guardadas = []''', '''        self.leis_guardadas = []
        self.contrafactual = False''')
rep('''        elif op == "juntar":
            ws.juntados.add(e.id)''', '''        elif op == "juntar":
            acao["valido_antes"] = bool(ws.valido)
            ws.juntados.add(e.id)''')
rep('''        self._fam_atual = prob.fam      # so para o relatorio de alcance
        ws = Espaco(prob, None, orcamento)''', '''        self._fam_atual = prob.fam      # so para o relatorio de alcance
        ws = Espaco(prob, None, orcamento)
        ws.contrafactual = self.f["juntar_contrafactual"]''')
open("organismo.py", "w", encoding="utf-8").write(src)
print("trocas", n)
