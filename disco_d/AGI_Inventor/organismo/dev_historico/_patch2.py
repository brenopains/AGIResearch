src = open("organismo.py", encoding="utf-8").read()
n_ok = 0


def rep(a, b, conta=1):
    global src, n_ok
    if src.count(a) != conta:
        raise SystemExit(f"esperava {conta}, achei {src.count(a)}:\n" + a[:300])
    src = src.replace(a, b)
    n_ok += 1


# A) prior por tipo: fonte controla CRENCA (leis); potencial gerativo controla
#    IMAGINACAO (ideias); o resto comeca neutro
rep('''        cf, _n = self._cred_fonte(e.fonte) if e.tipo not in ("op",) else (0.5, 0)
        if self.f["proc2"]:
            g = e.potencial_gerativo()
            a0 = 1 + N0 * (cf * (1 - T) + g * T)
            b0 = 1 + N0 * ((1 - cf) * (1 - T) + (1 - g) * T)
        else:
            a0 = 1 + N0 * cf
            b0 = 1 + N0 * (1 - cf)
        act = min(self.ativacao(e), 3.0) if self.f["meso"] else 0.0
        return a + a0 + act, b + b0''', '''        m = self._prior(e, T)
        a0, b0 = 1 + N0 * m, 1 + N0 * (1 - m)
        act = min(self.ativacao(e), 3.0) if self.f["meso"] else 0.0
        return a + a0 + act, b + b0

    def _prior(self, e, T):
        """Dois eixos. Lei = AFIRMACAO sobre um mundo: a fonte controla a crenca
        (quente, a crenca pesa menos). Conceito/direcao = IDEIA: frio, neutra;
        quente, vale o potencial gerativo. Sem proc2: a fonte controla tudo."""
        if e.tipo == "lei":
            cf, _n = self._cred_fonte(e.fonte)
            return cf * (1 - T) + 0.5 * T if self.f["proc2"] else cf
        if e.tipo in ("conceito", "direcao"):
            if self.f["proc2"]:
                return 0.5 * (1 - T) + e.potencial_gerativo() * T
            cf, _n = self._cred_fonte(e.fonte)
            return cf
        return 0.5''')

# B) confianca calibrada por contexto: parar so quando a experiencia diz que,
#    neste tipo de situacao, resposta validada internamente costuma estar certa
rep('''        for op in ("forma", "ajustar_base", "ajustar_ctx", "ajustar_novos",
                   "ajustar_amplo", "juntar", "inventar", "reajustar", "derivar",
                   "prever_base", "prever_ctx", "prever_novos", "macro"):
            self._nova("op", {"nome": op}, "substrato", eid=f"op:{op}")''', '''        for op in ("forma", "ajustar_base", "ajustar_ctx", "ajustar_novos",
                   "ajustar_amplo", "juntar", "inventar", "reajustar", "derivar",
                   "prever_base", "prever_ctx", "prever_novos", "macro"):
            self._nova("op", {"nome": op}, "substrato", eid=f"op:{op}")
        # metacognicao: "quando eu acho que acertei, aqui, eu acerto?"
        self._nova("op", {"nome": "confiar"}, "substrato", eid="op:confiar")''')

rep('''    def _agenda_aprendida(self, ws):
        for _seg in range(SEGURANCA_ACOES):
            if ws.gasto >= ws.orcamento or ws.valido:
                break''', '''    def _confia(self, ws):
        """Parar com resposta validada? Amostra da crenca de que, NESTE contexto,
        validar internamente basta (aprendida pelo verificador)."""
        if not ws.valido:
            return False
        th = self.amostra(self.E["op:confiar"], ws.x, self._temperatura(ws))
        ws.desconfiou = ws.desconfiou or th < 0.5
        return th >= 0.5

    def _agenda_aprendida(self, ws):
        for _seg in range(SEGURANCA_ACOES):
            if ws.gasto >= ws.orcamento or self._confia(ws):
                break''')

rep('''        self._credito(ws, ok)
        self._atualiza_abertos(ws, ok)''', '''        if prob.verificavel and ws.valido:
            self._registra(self.E["op:confiar"], ws.x, ok)
        self._credito(ws, ok)
        self._atualiza_abertos(ws, ok)''')

rep('''        self.assinaturas = set()''', '''        self.assinaturas = set()
        self.desconfiou = False''')

# C) refutacao lembrada: forma ja reprovada pelo verificador NESTE contexto nao
#    conta como validada de novo (o organismo lembra o que ja deu errado)
rep('''            ok, q, c2 = F.valida_lei([(prob.S, prob.A)], conc, ext, forma, self.rng)
            ws.gasta(c + c2, False)
            acao["usa"] = usa
            acao["vocab"] = sorted(usa)''', '''            ok, q, c2 = F.valida_lei([(prob.S, prob.A)], conc, ext, forma, self.rng)
            ok = ok and not self._refutada(forma, ws)
            ws.gasta(c + c2, False)
            acao["usa"] = usa
            acao["vocab"] = sorted(usa)''')
rep('''                ok, q, c = F.valida_lei([(prob.S, prob.A)], conc, ext, forma, self.rng)
                leis, bits, c2 = F.ajusta_lei([(prob.S, prob.A)], conc, ext, forma=forma)''', '''                ok, q, c = F.valida_lei([(prob.S, prob.A)], conc, ext, forma, self.rng)
                ok = ok and not self._refutada(forma, ws)
                leis, bits, c2 = F.ajusta_lei([(prob.S, prob.A)], conc, ext, forma=forma)''')
rep('''    def _duplicata_conceito(self, sup, w, esc, pais, Tc):''', '''    def _refutada(self, forma, ws):
        if self.f["pipeline"]:
            return False
        f2 = tuple((tuple(v), tuple(e_), d) for v, e_, d in forma)
        T = self._temperatura(ws)
        for e in self._de_tipo("lei"):
            if e.payload["forma"] == f2 and e.regs:
                if self.media(e, ws.x, T) < 0.5:
                    return True
        return False

    def _duplicata_conceito(self, sup, w, esc, pais, Tc):''')
# a lei respondida e guardada mesmo quando reprovada (para lembrar a refutacao)
rep('''        if ws.melhor is not None and ws.prob.tipo == "L" and (ok or ws.valido):
            self._guarda_lei(ws, ok)''', '''        if ws.melhor is not None and ws.prob.tipo == "L" and (ok or ws.valido):
            self._guarda_lei(ws, ok)''')
rep('''        for e in self._de_tipo("lei"):
            if e.payload["forma"] == forma:
                return''', '''        for e in self._de_tipo("lei"):
            if e.payload["forma"] == forma:
                if not any(a["arg"] == e.id for a in ws.acoes):
                    self._registra(e, ws.x, ok)
                return''')

# D) juntar e barato de fazer e CARO depois: o preco inclui o encarecimento dos
#    ajustes seguintes (pontos acrescentados x custo por ponto aprendido)
rep('''        oe = self.E[f"op:{op}"]
        cm = oe.payload.get("custo_med")
        if cm is not None and op != "macro":''', '''        oe = self.E[f"op:{op}"]
        cm = oe.payload.get("custo_med")
        if op == "juntar" and e is not None:
            add = sum(len(S) for S, _A in e.payload.get("dados", []))
            ab = self.E["op:ajustar_base"].payload
            por_ponto = (ab.get("custo_med") or 10.0) / max(ab.get("n_med") or 24, 1)
            return 0.2 + add * por_ponto
        if cm is not None and op != "macro":''')

# E) nao repetir invencao/derivacao identica
rep('''    def ja_feito(self, op, org):
        if not (op.startswith("ajustar") or op.startswith("prever")):
            return False''', '''    def ja_feito(self, op, org):
        if op == "inventar":
            return ("inventar", len(self.trajs), tuple(sorted(self.atomos_inv(org)))) in self.assinaturas
        if op == "derivar":
            return ("derivar", tuple(sorted(self.dirs_em_jogo(org)))) in self.assinaturas
        if not (op.startswith("ajustar") or op.startswith("prever")):
            return False''')
rep('''        elif op == "inventar":
            Tc = ws.trajs_atuais()''', '''        elif op == "inventar":
            ws.assinaturas.add(("inventar", len(ws.trajs), tuple(sorted(ws.atomos_inv(self)))))
            Tc = ws.trajs_atuais()''')
rep('''        elif op == "derivar":
            exist = [self.E[i].payload["expr"] for i in ws.dirs_em_jogo(self)]''', '''        elif op == "derivar":
            ws.assinaturas.add(("derivar", tuple(sorted(ws.dirs_em_jogo(self)))))
            exist = [self.E[i].payload["expr"] for i in ws.dirs_em_jogo(self)]''')
open("organismo.py", "w", encoding="utf-8").write(src)
print("trocas:", n_ok)
