src = open("organismo.py", encoding="utf-8").read()
n_ok = 0


def rep(a, b, conta=1):
    global src, n_ok
    if src.count(a) != conta:
        raise SystemExit(f"esperava {conta}, achei {src.count(a)}:\n" + a[:300])
    src = src.replace(a, b)
    n_ok += 1


# 1) derivar so de direcoes que ja pagaram (mais as de base)
rep('''    def dirs_em_jogo(self, org):
        return [e.id for e in org.E.values() if e.tipo == "direcao"
                and (e.id in self.em_jogo or any(o > 0 for _t, _c, o in e.regs))]''', '''    def dirs_em_jogo(self, org):
        """Direcoes que ja pagaram em algum lugar: as unicas de onde se deriva."""
        return [e.id for e in org.E.values() if e.tipo == "direcao"
                and any(o > 0 for _t, _c, o in e.regs)]''')

# 2) operacao nova: testar UMA candidata (vocabulario de contexto + ela)
rep('''        for op in ("forma", "ajustar_base", "ajustar_ctx", "ajustar_novos",''', '''        for op in ("forma", "ajustar_base", "ajustar_ctx", "ajustar_novos", "ajustar_com",''')
rep('''            if ws.novos_atomos():
                out.append(("ajustar_novos", None))
            if T > 0.5 or not self.f["regime"]:''', '''            if ws.novos_atomos():
                out.append(("ajustar_novos", None))
            for cid in sorted(ws.em_jogo):
                ce = self.E.get(cid)
                if ce is not None and ce.tipo in ("conceito", "direcao") and cid not in ws.testados_com:
                    out.append(("ajustar_com", ce))
            if T > 0.5 or not self.f["regime"]:''')
rep('''            if ws.novos_atomos():
                out.append(("prever_novos", None))''', '''            if ws.novos_atomos():
                out.append(("prever_novos", None))
            for cid in sorted(ws.em_jogo):
                ce = self.E.get(cid)
                if ce is not None and ce.tipo == "conceito" and cid not in ws.testados_com:
                    out.append(("prever_com", ce))''')
rep('''                   "prever_base", "prever_ctx", "prever_novos", "macro"):''', '''                   "prever_base", "prever_ctx", "prever_novos", "prever_com", "macro"):''')
rep('''        self.desconfiou = False''', '''        self.desconfiou = False
        self.testados_com = set()''')
rep('''        elif op.endswith("base"):
            sel = []
        elif op.endswith("novos"):''', '''        elif op.endswith("base"):
            sel = []
        elif op.endswith("_com"):
            sel = self._vocab_ctx(org, cands, x, T)
            alvo = org.E.get(self.alvo_com)
            if alvo is not None and alvo not in sel:
                sel = sel + [alvo]
        elif op.endswith("novos"):''')
rep('''        else:   # ctx: crenca de contexto, sem numero de atomos fixo
            medias = [(org.media(e, x, T), e) for e in cands]
            medias.sort(key=lambda m: -m[0])
            sel = [e for m, e in medias if m > 0.5]
            sel += [e for e in cands if e.id in self.em_jogo and e not in sel]
            if org.f["tetos"]:
                sel = sel[:TETOS["vocab"]]''', '''        else:   # ctx: crenca de contexto, sem numero de atomos fixo
            sel = self._vocab_ctx(org, cands, x, T)''')
rep('''    def atomos_inv(self, org):''', '''    def _vocab_ctx(self, org, cands, x, T):
        """Atomos cuja crenca de pagar NESTE contexto passa de 1/2 (decisao de
        Bayes com perda simetrica), mais os ja confirmados neste problema."""
        medias = [(org.media(e, x, T), e) for e in cands if e.regs]
        medias.sort(key=lambda m: -m[0])
        sel = [e for m, e in medias if m > 0.5]
        sel += [org.E[i] for i in self.confirmados if i in org.E and org.E[i] not in sel]
        if org.f["tetos"]:
            sel = sel[:TETOS["vocab"]]
        return sel

    def atomos_inv(self, org):''')
rep('''        self.testados_com = set()''', '''        self.testados_com = set()
        self.confirmados = set()
        self.alvo_com = None''')
# execucao: ajustar_com / prever_com marcam o alvo
rep('''        elif op.startswith("ajustar"):
            conc, ext, usa = ws.vocab_para(op, self)''', '''        elif op.startswith("ajustar"):
            if op == "ajustar_com":
                ws.alvo_com = e.id
                ws.testados_com.add(e.id)
                acao["arg"] = e.id
            conc, ext, usa = ws.vocab_para(op, self)''')
rep('''        elif op.startswith("prever"):
            conc, _ext, usa = ws.vocab_para(op, self)''', '''        elif op.startswith("prever"):
            if op == "prever_com":
                ws.alvo_com = e.id
                ws.testados_com.add(e.id)
                acao["arg"] = e.id
            conc, _ext, usa = ws.vocab_para(op, self)''')
# atomo que entrou numa resposta validada fica confirmado no problema
rep('''            self.melhor = {"resposta": lei, "bits": bits, "acao": acao,
                           "usados": sorted(set(realmente))}
            self.valido = self.valido or ok''', '''            self.melhor = {"resposta": lei, "bits": bits, "acao": acao,
                           "usados": sorted(set(realmente))}
            self.valido = self.valido or ok
            if ok:
                self.confirmados |= {s for s in realmente if s.startswith(("conceito", "direcao"))}''')
# assinatura do _com inclui o alvo
rep('''    def assinatura(self, op, usa):
        tipo = "prever" if op.startswith("prever") else "ajustar"''', '''    def assinatura(self, op, usa):
        tipo = "prever" if op.startswith("prever") else "ajustar"
        if op.endswith("_com"):
            return (tipo, "com", self.alvo_com, tuple(sorted(usa)), len(self.dados))''')
rep('''    def ja_feito(self, op, org):''', '''    def ja_feito(self, op, org, e=None):
        if op.endswith("_com"):
            return e is not None and e.id in self.testados_com''')
rep('''            cands = [c for c in self._candidatas(ws) if not ws.ja_feito(c[0], self)]''', '''            cands = [c for c in self._candidatas(ws) if not ws.ja_feito(c[0], self, c[1])]''')

# 3) custo aprendido por UNIDADE DE TRABALHO (candidatos x direcoes x pontos)
rep('''        if cm is not None and op != "macro":
            if op.startswith("ajustar") or op == "derivar":
                return cm * n / max(oe.payload.get("n_med", n), 1)
            return cm
        return self._custo_formula(op, e, ws, n)''', '''        if op.startswith("ajustar") or op.startswith("prever"):
            w = self._trabalho(op, e, ws)
            cpt = self._cpt.get(op[:6])
            if cpt is not None:
                return cpt * w
            return F.custo(w * 8)
        if cm is not None and op != "macro":
            return cm
        return self._custo_formula(op, e, ws, n)

    def _trabalho(self, op, e, ws):
        if op.startswith("ajustar") and op == "ajustar_com":
            ws.alvo_com = e.id if e is not None else ws.alvo_com
        conc, ext, _u = ws.vocab_para(op, self, dry=True)
        if op.startswith("ajustar"):
            P = 3 + len(conc)
            nc = P * 8 + P * (P - 1) // 2 * 64
            n = sum(len(S) for S, _A in ws.dados)
            return nc * (3 + len(ext)) * 2 * n
        na = 7 + len(conc)
        return (na + na * (na - 1) // 2 * 24) * 32''')
rep('''        oe = self.E[f"op:{op}"]
        real = ws.gasto - gasto0''', '''        oe = self.E[f"op:{op}"]
        real = ws.gasto - gasto0
        if (op.startswith("ajustar") or op.startswith("prever")) and acao.get("trabalho"):
            k = op[:6]
            cpt = real / acao["trabalho"]
            self._cpt[k] = cpt if k not in self._cpt else 0.8 * self._cpt[k] + 0.2 * cpt''')
rep('''        elif op.startswith("ajustar"):
            if op == "ajustar_com":''', '''        elif op.startswith("ajustar"):
            acao["trabalho"] = self._trabalho(op, e, ws)
            if op == "ajustar_com":''')
rep('''        elif op.startswith("prever"):
            if op == "prever_com":''', '''        elif op.startswith("prever"):
            acao["trabalho"] = self._trabalho(op, e, ws)
            if op == "prever_com":''')
rep('''        self.admitidos = []           # pipeline: ids admitidos (booleano)''', '''        self.admitidos = []           # pipeline: ids admitidos (booleano)
        self._cpt = {}                # custo aprendido por unidade de trabalho''')
open("organismo.py", "w", encoding="utf-8").write(src)
print("trocas:", n_ok)
