src = open("organismo.py", encoding="utf-8").read()
n = 0
def rep(a, b, c=1):
    global src, n
    assert src.count(a) == c, (src.count(a), a[:200])
    src = src.replace(a, b); n += 1
rep('''    def _por_largura(self, e, x, T):
        pi = self.escopo(e)
        abs_ = [self._ab(e, x, T, b * (1 - T)) for b in GRADE_BETA]
        return pi, abs_''', '''    def _por_largura(self, e, x, T):
        """As 7 larguras de uma vez (vetorizado), com memoria dentro do problema."""
        chave = ("pl", e.id, id(x), len(e.regs), round(T, 4), len(e.usos))
        r = self._cache.get(chave)
        if r is not None:
            return r
        pi = self.escopo(e)
        betas = GRADE_BETA * (1 - T)
        if e.regs:
            s, oks = self._sims(e, x)
            W = np.exp(np.outer(betas, s - 1.0))
            a = W @ oks
            b = W.sum(1) - a
        else:
            a = b = np.zeros(len(betas))
        esc = 1.0 / (1.0 + 3.0 * T)
        m = self._prior(e, T)
        act = min(self.ativacao(e), 3.0) if self.f["meso"] else 0.0
        a = a * esc + 1 + N0 * m + act
        b = b * esc + 1 + N0 * (1 - m)
        r = (pi, a, b)
        self._cache[chave] = r
        return r''')
rep('''        if self.f["escopo"] == "aprendido":
            pi, abs_ = self._por_largura(e, x, T)
            k = int(self.rng.choice(len(pi), p=pi))
            a, b = abs_[k]
            return float(self.rng.beta(a, b))''', '''        if self.f["escopo"] == "aprendido":
            pi, a, b = self._por_largura(e, x, T)
            k = int(self.rng.choice(len(pi), p=pi))
            return float(self.rng.beta(a[k], b[k]))''')
rep('''        if self.f["escopo"] == "aprendido":
            pi, abs_ = self._por_largura(e, x, T)
            return float(sum(p * a / (a + b) for p, (a, b) in zip(pi, abs_)))''', '''        if self.f["escopo"] == "aprendido":
            pi, a, b = self._por_largura(e, x, T)
            return float(pi @ (a / (a + b)))''')
# a crenca muda no credito e antes do sonho: limpar a memoria
rep('''    def _credito(self, ws, ok):
        x = ws.x''', '''    def _credito(self, ws, ok):
        self._cache = {}
        x = ws.x''')
rep('''        if self.f["sonho"] and sobra > 0:
            gasto_sonho = self._sonha(sobra)''', '''        if self.f["sonho"] and sobra > 0:
            self._cache = {}
            gasto_sonho = self._sonha(sobra)''')
open("organismo.py", "w", encoding="utf-8").write(src)
print("trocas", n)
