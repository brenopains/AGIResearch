src = open("organismo.py", encoding="utf-8").read()
n = 0
def rep(a, b):
    global src, n
    assert src.count(a) == 1, a[:200]
    src = src.replace(a, b); n += 1
rep('''        if op.startswith("ajustar") or op.startswith("prever"):
            w = self._trabalho(op, e, ws)
            cpt = self._cpt.get(op[:6])''', '''        if op.startswith("ajustar") or op.startswith("prever") or op == "inventar":
            w = self._trabalho(op, e, ws)
            cpt = self._cpt.get(op[:6])''')
rep('''    def _trabalho(self, op, e, ws):
        if op.startswith("ajustar") and op == "ajustar_com":''', '''    def _trabalho(self, op, e, ws):
        if op == "inventar":
            from math import comb
            a = 6 + len(ws.atomos_inv(self))
            m = a + a * (a + 1) // 2
            Tc = ws.trajs_atuais() if ws.trajs_atuais() is not None else (ws.trajs[-1] if ws.trajs else None)
            npts = (Tc.shape[0] * Tc.shape[1] // 2) if Tc is not None else 100
            return (m + comb(m, 2) + comb(m, 3)) * npts
        if op.startswith("ajustar") and op == "ajustar_com":''')
rep('''        if (op.startswith("ajustar") or op.startswith("prever")) and acao.get("trabalho"):''', '''        if (op.startswith("ajustar") or op.startswith("prever") or op == "inventar") and acao.get("trabalho"):''')
rep('''        elif op == "inventar":
            ws.assinaturas.add''', '''        elif op == "inventar":
            acao["trabalho"] = self._trabalho(op, e, ws)
            ws.assinaturas.add''')
open("organismo.py", "w", encoding="utf-8").write(src)
print("trocas", n)
