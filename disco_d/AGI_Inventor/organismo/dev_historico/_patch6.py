src = open("organismo.py", encoding="utf-8").read()
n = 0
def rep(a, b):
    global src, n
    assert src.count(a) == 1, a[:200]
    src = src.replace(a, b); n += 1

rep('''    def ativacao(self, e):
        tau = self.tau.get(e.tipo, TAU0)
        return float(sum(np.exp(-(self.t - u) / tau) for u in e.usos))

    def posterior(self, e, x, T):
        beta = BETA0 * (1 - T) + 1.0
        a = b = 0.0
        for (_t, c, ok) in e.regs:
            w = np.exp(beta * (hdc.sim(x, c) - 1.0))
            a += w * ok
            b += w * (1 - ok)
        esc = 1.0 / (1.0 + 3.0 * T)''', '''    def ativacao(self, e):
        if not e.usos:
            return 0.0
        tau = self.tau.get(e.tipo, TAU0)
        return float(np.exp(-(self.t - np.asarray(e.usos, float)) / tau).sum())

    def _sims(self, e, x):
        """Semelhanca do contexto x com cada registro de e (uma matmul, em cache)."""
        n = len(e.regs)
        chave = (e.id, id(x), n)
        s = self._cache.get(chave)
        if s is None:
            if getattr(e, "_C", None) is None or e._C.shape[0] != n:
                C = np.stack([c for _t, c, _o in e.regs]) if n else np.zeros((0, hdc.D))
                nr = np.linalg.norm(C, axis=1, keepdims=True)
                e._C = C / np.maximum(nr, 1e-12)
                e._ok = np.array([o for _t, _c, o in e.regs], float)
            nx = np.linalg.norm(x)
            s = e._C @ (x / max(nx, 1e-12)) if n else np.zeros(0)
            self._cache[chave] = s
        return s, e._ok

    def posterior(self, e, x, T):
        beta = BETA0 * (1 - T) + 1.0
        a = b = 0.0
        if e.regs:
            s, oks = self._sims(e, x)
            w = np.exp(beta * (s - 1.0))
            a = float(w @ oks)
            b = float(w.sum() - a)
        esc = 1.0 / (1.0 + 3.0 * T)''')
rep('''        self._cpt = {}                # custo aprendido por unidade de trabalho''', '''        self._cpt = {}                # custo aprendido por unidade de trabalho
        self._cache = {}''')
rep('''    def resolve(self, prob, orcamento):
        ws = Espaco(prob, None, orcamento)''', '''    def resolve(self, prob, orcamento):
        self._cache = {}
        ws = Espaco(prob, None, orcamento)''')
# os registros retroativos (credito atrasado) invalidam o cache dessa estrutura
rep('''                op.regs.append([e.nasc, e.ctx_nasc if e.ctx_nasc is not None else x, 1.0])''', '''                op.regs.append([e.nasc, e.ctx_nasc if e.ctx_nasc is not None else x, 1.0])
                op._C = None''')
rep('''    def _registra(self, e, x, ok):
        e.regs.append([self.t, x, float(ok)])''', '''    def _registra(self, e, x, ok):
        e.regs.append([self.t, x, float(ok)])
        e._C = None''')
open("organismo.py", "w", encoding="utf-8").write(src)
print("trocas", n)
