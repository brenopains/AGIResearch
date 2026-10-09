src = open("organismo.py", encoding="utf-8").read()
n = 0
def rep(a, b):
    global src, n
    assert src.count(a) == 1, a[:200]
    src = src.replace(a, b); n += 1

rep('''FLAGS_ORGANISMO = dict(meso=True, tetos=False, meta=True, coal=True, eco=True,
                       regime=True, proc2=True, sonho=True, rag=False,
                       placebo_meso=False, pipeline=False)''', '''FLAGS_ORGANISMO = dict(meso=True, tetos=False, meta=True, coal=True, eco=True,
                       regime=True, proc2=True, sonho=True, rag=False,
                       placebo_meso=False, pipeline=False,
                       escopo="fixo", beta_fixo=8.0)''')
rep('''    "SEM_SONHO": dict(sonho=False),''', '''    "SEM_SONHO": dict(sonho=False),
    # vida 2: o ALCANCE de cada estrutura aprendido x larguras fixas
    "ESCOPO": dict(escopo="aprendido"),
    "LARGURA_3": dict(beta_fixo=2.0),
    "GLOBAL": dict(beta_fixo=0.0),''')
rep('''BETA0 = 8.0          # largura do nucleo de contexto quando frio''', '''BETA0 = 8.0          # largura do nucleo de contexto quando frio (v1; vida 2 aprende)
GRADE_BETA = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])   # larguras possiveis''')
rep('''    origem: tuple | None = None                   # (id da operacao, indice do registro)
    ctx_nasc: np.ndarray | None = None''', '''    origem: tuple | None = None                   # (id da operacao, indice do registro)
    ctx_nasc: np.ndarray | None = None
    ll: np.ndarray | None = None                  # verossimilhanca preditiva por largura
    fams: list = field(default_factory=list)      # SO para analise: (familia, ok)''')

rep('''    def posterior(self, e, x, T):
        beta = BETA0 * (1 - T) + 1.0
        a = b = 0.0
        if e.regs:
            s, oks = self._sims(e, x)
            w = np.exp(beta * (s - 1.0))
            a = float(w @ oks)
            b = float(w.sum() - a)
        esc = 1.0 / (1.0 + 3.0 * T)
        a, b = a * esc, b * esc
        m = self._prior(e, T)
        a0, b0 = 1 + N0 * m, 1 + N0 * (1 - m)
        act = min(self.ativacao(e), 3.0) if self.f["meso"] else 0.0
        return a + a0 + act, b + b0''', '''    def _ab(self, e, x, T, beta):
        a = b = 0.0
        if e.regs:
            s, oks = self._sims(e, x)
            w = np.exp(beta * (s - 1.0))
            a = float(w @ oks)
            b = float(w.sum() - a)
        esc = 1.0 / (1.0 + 3.0 * T)
        a, b = a * esc, b * esc
        m = self._prior(e, T)
        a0, b0 = 1 + N0 * m, 1 + N0 * (1 - m)
        act = min(self.ativacao(e), 3.0) if self.f["meso"] else 0.0
        return a + a0 + act, b + b0

    def posterior(self, e, x, T):
        bf = self.f["beta_fixo"]
        return self._ab(e, x, T, bf * (1 - T) + (1.0 if bf > 0 else 0.0))

    def escopo(self, e):
        """Distribuicao posterior sobre as larguras (o ALCANCE aprendido)."""
        if e.ll is None:
            return np.full(len(GRADE_BETA), 1 / len(GRADE_BETA))
        p = np.exp(e.ll - e.ll.max())
        return p / p.sum()

    def _por_largura(self, e, x, T):
        pi = self.escopo(e)
        abs_ = [self._ab(e, x, T, b * (1 - T)) for b in GRADE_BETA]
        return pi, abs_

    def _adiciona_reg(self, e, t, ctx, ok):
        """Todo registro entra por aqui: atualiza a verossimilhanca PREDITIVA de
        cada largura (o quanto ela previu este desfecho antes de ve-lo)."""
        if self.f["escopo"] == "aprendido":
            if e.ll is None:
                e.ll = np.zeros(len(GRADE_BETA))
            if e.regs:
                C = np.stack([c for _t, c, _o in e.regs])
                C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
                s = C @ (ctx / max(np.linalg.norm(ctx), 1e-12))
                oks = np.array([o for _t, _c, o in e.regs], float)
                for k, bb in enumerate(GRADE_BETA):
                    w = np.exp(bb * (s - 1.0))
                    p = (1.0 + w @ oks) / (2.0 + w.sum())
                    e.ll[k] += ok * np.log(p) + (1 - ok) * np.log(1 - p)
            else:
                e.ll += np.log(0.5)
        e.regs.append([t, ctx, float(ok)])
        e.fams.append((getattr(self, "_fam_atual", None), float(ok)))
        e._C = None''')

rep('''    def amostra(self, e, x, T):
        a, b = self.posterior(e, x, T)
        return float(self.rng.beta(a, b))

    def media(self, e, x, T):
        a, b = self.posterior(e, x, T)
        return a / (a + b)''', '''    def amostra(self, e, x, T):
        if self.f["escopo"] == "aprendido":
            pi, abs_ = self._por_largura(e, x, T)
            k = int(self.rng.choice(len(pi), p=pi))
            a, b = abs_[k]
            return float(self.rng.beta(a, b))
        a, b = self.posterior(e, x, T)
        return float(self.rng.beta(a, b))

    def media(self, e, x, T):
        if self.f["escopo"] == "aprendido":
            pi, abs_ = self._por_largura(e, x, T)
            return float(sum(p * a / (a + b) for p, (a, b) in zip(pi, abs_)))
        a, b = self.posterior(e, x, T)
        return a / (a + b)''')

rep('''    def _registra(self, e, x, ok):
        e.regs.append([self.t, x, float(ok)])
        e._C = None''', '''    def _registra(self, e, x, ok):
        self._adiciona_reg(e, self.t, x, ok)''')
rep('''                op.regs.append([e.nasc, e.ctx_nasc if e.ctx_nasc is not None else x, 1.0])
                op._C = None''', '''                self._adiciona_reg(op, e.nasc, e.ctx_nasc if e.ctx_nasc is not None else x, 1.0)''')
rep('''    def resolve(self, prob, orcamento):
        self._cache = {}''', '''    def resolve(self, prob, orcamento):
        self._cache = {}
        self._fam_atual = prob.fam      # so para o relatorio de alcance''')
open("organismo.py", "w", encoding="utf-8").write(src)
print("trocas", n)
