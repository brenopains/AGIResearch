src = open("organismo.py", encoding="utf-8").read()
n = 0
def rep(a, b):
    global src, n
    assert src.count(a) == 1, a[:200]
    src = src.replace(a, b); n += 1
rep('''        usados = set(ws.melhor.get("usados", [])) if (ok and ws.melhor) else set()
        for a in ws.acoes:
            a_ok = a["id"] in anc''', '''        usados = set(ws.melhor.get("usados", [])) if (ok and ws.melhor) else set()
        if self.f["juntar_contrafactual"] and ok and ws.prob.tipo == "L":
            anc, usados = self._juntados_consistentes(ws, anc, usados)
        for a in ws.acoes:
            a_ok = a["id"] in anc''')
rep('''    def _propaga(self, sid, x):''', '''    def _juntados_consistentes(self, ws, anc, usados):
        """Contrafactual do juntar: o dado juntado so recebe credito se obedece a
        lei encontrada (mesmo mundo). Dado de outro mundo que so 'passou' porque o
        ajuste conjunto aceita a uniao das formas nao recebe."""
        lei = ws.melhor["resposta"] if ws.melhor else None
        if lei is None or not hasattr(lei, "termos") or not lei.termos:
            return anc, usados
        forma = [(v, e_, d) for v, e_, d in lei.termos]
        for a in ws.acoes:
            if a["op"] != "juntar" or a["id"] not in anc:
                continue
            fonte = self.E.get(a["arg"])
            dados = fonte.payload.get("dados", []) if fonte is not None else []
            if not dados:
                continue
            try:
                leis, _b, _c = F.ajusta_lei(dados, lei.conceitos, lei.extras, forma=forma)
                q = np.mean([F.qui2(l, S, A_) for l, (S, A_) in zip(leis, dados)])
                lim = F.limiar_qui2(sum(A_.size for _S, A_ in dados))
            except Exception:
                q, lim = np.inf, 0.0
            if not q <= lim:
                anc = anc - {a["id"]}
                usados = usados - {a["arg"]}
                self.evento("juntar_rejeitado", aberto=a["arg"], qui2=float(q) if np.isfinite(q) else None)
        return anc, usados

    def _propaga(self, sid, x):''')
open("organismo.py", "w", encoding="utf-8").write(src)
print("trocas", n)
