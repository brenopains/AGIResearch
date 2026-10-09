src = open("organismo.py", encoding="utf-8").read()
n = 0
def rep(a, b):
    global src, n
    assert src.count(a) == 1, a[:200]
    src = src.replace(a, b); n += 1
rep('''        if ws.prob.fonte != "percebido":
            return
        for sid in ids:
            e = self.E.get(sid)
            if e is None or e.tipo != "lei" or e.fonte == "percebido":
                continue
            ab = self.fonte_cred.setdefault(e.fonte, [1.0, 1.0])
            ab[0 if ok else 1] += 1
            self.evento("fonte", fonte=e.fonte, ok=bool(ok), cred=ab[0] / sum(ab))''', '''        if ws.prob.fonte != "percebido":
            return
        for sid in ids:
            e = self.E.get(sid)
            if e is None or e.tipo != "lei" or e.fonte in ("percebido", "sonhado"):
                continue
            # so conta como teste da AFIRMACAO onde o mundo parece o mesmo em que
            # ela foi feita (peso pelo mesmo nucleo de contexto da crenca)
            w = float(np.exp(BETA0 * (hdc.sim(ws.x, e.ctx_nasc) - 1.0))) if e.ctx_nasc is not None else 0.0
            if w < 1e-3:
                continue
            ab = self.fonte_cred.setdefault(e.fonte, [1.0, 1.0])
            ab[0 if ok else 1] += w
            self.evento("fonte", fonte=e.fonte, ok=bool(ok), peso=w, cred=ab[0] / sum(ab))''')
# reativacao = algo LATENTE no inicio do problema que entra na resposta que deu certo
rep('''    def _reativacoes(self, ws):
        """Registra quando algo LATENTE e reacendido pelo contexto (o 'BUM')."""
        T = self._temperatura(ws)
        for e in self.E.values():
            if e.tipo in ("op",) or not e.regs:
                continue
            act = self.ativacao(e)
            if act < 0.05:
                p = self.media(e, ws.x_reat if e.tipo == "aberto" else ws.x, T)
                if p > 0.6:
                    gap = self.t - (max(e.usos) if e.usos else e.nasc)
                    self.evento("reativacao", id=e.id, tipo=e.tipo, ativ=act,
                                cred=p, intervalo=gap)''', '''    def _reativacoes(self, ws):
        """Guarda quem estava LATENTE no inicio do problema (ativacao ~ 0)."""
        ws.latentes = {e.id: (self.t - (max(e.usos) if e.usos else e.nasc))
                       for e in self.E.values()
                       if e.tipo != "op" and self.ativacao(e) < 0.05}

    def _registra_bum(self, ws, ok):
        """O 'BUM': algo latente voltou e entrou na resposta que deu certo."""
        if not ok or ws.melhor is None:
            return
        usados = set(ws.melhor.get("usados", []))
        usados |= {a["arg"] for a in ws.acoes if a["op"] in ("juntar", "forma", "macro")
                   and a["id"] in self._ancestrais(ws)}
        for sid in usados:
            if sid in getattr(ws, "latentes", {}) and sid in self.E:
                self.evento("reativacao", id=sid, tipo=self.E[sid].tipo,
                            intervalo=ws.latentes[sid])''')
rep('''        if prob.verificavel and ws.valido:
            self._registra(self.E["op:confiar"], ws.x, ok)''', '''        if prob.verificavel and ws.valido:
            self._registra(self.E["op:confiar"], ws.x, ok)
        self._registra_bum(ws, ok)''')
open("organismo.py", "w", encoding="utf-8").write(src)
print("trocas", n)
