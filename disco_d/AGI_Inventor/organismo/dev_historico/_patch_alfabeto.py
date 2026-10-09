src = open("organismo.py", encoding="utf-8").read()
n = 0
def rep(a, b, c=1):
    global src, n
    assert src.count(a) == c, (src.count(a), a[:200])
    src = src.replace(a, b); n += 1
rep('''juntar_contrafactual=False, custo_inv_corrigido=False)''', '''juntar_contrafactual=False, custo_inv_corrigido=False,
                       alfabeto_ctx=False)''')
rep('''custo_inv_corrigido=True)''', '''custo_inv_corrigido=True, alfabeto_ctx=True)''', 4)
rep('''        out = {}
        for e in org.E.values():
            if e.tipo == "conceito" and any(o > 0 for _t, _c, o in e.regs):
                C = self.variantes.get(e.id, e.payload["c"])
                out[C.nome] = C
                self._mapa[C.nome] = e.id''', '''        out = {}
        T = org._temperatura(self) if org.f["alfabeto_ctx"] else 0.0
        for e in org.E.values():
            if e.tipo == "conceito" and any(o > 0 for _t, _c, o in e.regs):
                # vida 2: leva para a invencao so o que a crenca acha relevante AQUI
                if org.f["alfabeto_ctx"] and org.media(e, self.x, T) <= 0.5:
                    continue
                C = self.variantes.get(e.id, e.payload["c"])
                out[C.nome] = C
                self._mapa[C.nome] = e.id''')
open("organismo.py", "w", encoding="utf-8").write(src)
print("trocas", n)
