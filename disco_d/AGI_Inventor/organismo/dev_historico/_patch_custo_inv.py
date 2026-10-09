src = open("organismo.py", encoding="utf-8").read()
n = 0
def rep(a, b, c=1):
    global src, n
    assert src.count(a) == c, (src.count(a), a[:200])
    src = src.replace(a, b); n += 1
rep('''                       juntar_contrafactual=False)''', '''                       juntar_contrafactual=False, custo_inv_corrigido=False)''')
rep('''dict(valor_criacao=True, juntar_contrafactual=True)''', '''dict(valor_criacao=True, juntar_contrafactual=True, custo_inv_corrigido=True)''')
rep('''dict(escopo="aprendido", valor_criacao=True, juntar_contrafactual=True)''', '''dict(escopo="aprendido", valor_criacao=True, juntar_contrafactual=True,
                   custo_inv_corrigido=True)''')
rep('''dict(beta_fixo=2.0, valor_criacao=True, juntar_contrafactual=True)''', '''dict(beta_fixo=2.0, valor_criacao=True, juntar_contrafactual=True,
                      custo_inv_corrigido=True)''')
rep('''dict(beta_fixo=0.0, valor_criacao=True, juntar_contrafactual=True)''', '''dict(beta_fixo=0.0, valor_criacao=True, juntar_contrafactual=True,
                   custo_inv_corrigido=True)''')
rep('''            cpt = self._cpt.get(op[:6])
            if cpt is not None:
                return cpt * w
            return F.custo(w * 8)''', '''            cpt = self._cpt.get(op[:6])
            if cpt is not None:
                return cpt * w
            # antes do 1o uso: o material cobra custo(subconjuntos x pontos) na
            # invencao (fator 1); o 8 era a calibracao do AJUSTE (v1 usava 8 nos dois)
            if op == "inventar" and self.f["custo_inv_corrigido"]:
                return F.custo(w)
            return F.custo(w * 8)''')
open("organismo.py", "w", encoding="utf-8").write(src)
print("trocas", n)
