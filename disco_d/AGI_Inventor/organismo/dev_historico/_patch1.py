src = open("organismo.py", encoding="utf-8").read()
n_ok = 0


def rep(a, b):
    global src, n_ok
    if a not in src:
        raise SystemExit("NAO ACHEI:\n" + a[:200])
    src = src.replace(a, b)
    n_ok += 1


# ---------- Espaco: mapas nome->id e assinaturas ----------
rep('''        self.novos, self.em_jogo = set(), set()
        self.variantes = {}
        self.juntou = []''', '''        self.novos, self.em_jogo = set(), set()
        self.variantes = {}
        self.juntou = []
        self._mapa = {}          # nome do conceito -> id da estrutura
        self._mapa_dir = {}      # id da direcao -> nome da direcao
        self.assinaturas = set()''')

rep('''        conc, ext, usa = {}, [], []
        for e in sel:
            if e.tipo == "conceito":
                C = self.variantes.get(e.id, e.payload["c"])
                conc[C.nome] = C
            else:
                ext.append(e.payload["expr"])
            usa.append(e.id)
        return conc, ext, usa''', '''        conc, ext, usa = {}, [], []
        for e in sel:
            if e.tipo == "conceito":
                C = self.variantes.get(e.id, e.payload["c"])
                conc[C.nome] = C
                self._mapa[C.nome] = e.id
            else:
                if op.startswith("prever"):
                    continue
                ext.append(e.payload["expr"])
                self._mapa_dir[e.id] = F.nome_dir(e.payload["expr"])
            usa.append(e.id)
        return conc, ext, usa''')

rep('''        out = {}
        for e in org.E.values():
            if e.tipo == "conceito" and any(o > 0 for _t, _c, o in e.regs):
                C = self.variantes.get(e.id, e.payload["c"])
                out[C.nome] = C''', '''        out = {}
        for e in org.E.values():
            if e.tipo == "conceito" and any(o > 0 for _t, _c, o in e.regs):
                C = self.variantes.get(e.id, e.payload["c"])
                out[C.nome] = C
                self._mapa[C.nome] = e.id''')

rep('''    def id_de_atomo(self, nome):
        return self._mapa.get(nome) if hasattr(self, "_mapa") else None''', '''    def id_de_atomo(self, nome):
        return self._mapa.get(nome)''')

rep('''            if a.tipo == "conceito":
                C = self.variantes.get(aid, a.payload["c"])
                conc[C.nome] = C
            else:
                ext.append(a.payload["expr"])
            usa.append(aid)''', '''            if a.tipo == "conceito":
                C = self.variantes.get(aid, a.payload["c"])
                conc[C.nome] = C
                self._mapa[C.nome] = aid
            else:
                ext.append(a.payload["expr"])
                self._mapa_dir[aid] = F.nome_dir(a.payload["expr"])
            usa.append(aid)''')

rep('''        usados = []
        nomes = {n for v, _e, _d in lei.termos for n in v}
        dn = {d for _v, _e, d in lei.termos}
        for sid in acao.get("usa", []):
            usados.append(sid)
        mapa_c = {C.nome: sid for sid, C in self._ids_conc(acao, conc)}
        realmente = [mapa_c[n] for n in nomes if n in mapa_c]
        realmente += [sid for sid in acao.get("usa", []) if sid.startswith("direcao")
                      and sid in self._mapa_dir and self._mapa_dir[sid] in dn]''', '''        nomes = {n for v, _e, _d in lei.termos for n in v}
        dn = {d for _v, _e, d in lei.termos}
        realmente = [self._mapa[n] for n in nomes if n in self._mapa]
        realmente += [sid for sid in acao.get("usa", []) if sid.startswith("direcao")
                      and self._mapa_dir.get(sid) in dn]''')

rep('''    def _ids_conc(self, acao, conc):
        out = []
        for sid in acao.get("usa", []):
            if sid.startswith("conceito"):
                for nome, C in conc.items():
                    if getattr(C, "_sid", None) == sid:
                        out.append((sid, C))
        return out

''', '')

rep('''        nomes = {desc[1]} if desc[0] == "1" else {desc[1], desc[2]}
        mapa_c = {C.nome: sid for sid, C in self._ids_conc(acao, conc)}
        usados = [mapa_c[n] for n in nomes if n in mapa_c]''', '''        nomes = {desc[1]} if desc[0] == "1" else {desc[1], desc[2]}
        usados = [self._mapa[n] for n in nomes if n in self._mapa]''')

# ---------- vocab ctx deterministico (media), para nao repetir acao ----------
rep('''            if dry:
                medias = [(org.media(e, x, T), e) for e in cands]
            else:
                medias = [(org.amostra(e, x, T), e) for e in cands]''', '''            medias = [(org.media(e, x, T), e) for e in cands]''')

# ---------- custo aprendido pelo uso + assinatura de acao ----------
rep('''    def _custo_previsto(self, op, e, ws):
        n = sum(len(S) for S, _A in ws.dados) if ws.dados else 16''', '''    def _custo_previsto(self, op, e, ws):
        """Custo APRENDIDO pelo uso (media movel do custo real da operacao,
        escalada pelo tamanho do dado); a formula so vale antes do primeiro uso."""
        n = sum(len(S) for S, _A in ws.dados) if ws.dados else 16
        oe = self.E[f"op:{op}"]
        cm = oe.payload.get("custo_med")
        if cm is not None and op != "macro":
            if op.startswith("ajustar") or op == "derivar":
                return cm * n / max(oe.payload.get("n_med", n), 1)
            return cm
        return self._custo_formula(op, e, ws, n)

    def _custo_formula(self, op, e, ws, n):''')

rep('''        acao["progresso"] = ws.chave_melhor() != antes''', '''        oe = self.E[f"op:{op}"]
        real = ws.gasto - gasto0
        nd = sum(len(S) for S, _A in ws.dados) if ws.dados else 16
        if oe.payload.get("custo_med") is None:
            oe.payload["custo_med"], oe.payload["n_med"] = real, nd
        else:
            oe.payload["custo_med"] = 0.8 * oe.payload["custo_med"] + 0.2 * real
            oe.payload["n_med"] = 0.8 * oe.payload["n_med"] + 0.2 * nd
        acao["custo"] = real
        acao["progresso"] = ws.chave_melhor() != antes''')

rep('''        prob = ws.prob
        antes = ws.chave_melhor()
        if op == "forma":''', '''        prob = ws.prob
        antes = ws.chave_melhor()
        gasto0 = ws.gasto
        if op == "forma":''')

rep('''        elif op.startswith("ajustar"):
            conc, ext, usa = ws.vocab_para(op, self)''', '''        elif op.startswith("ajustar"):
            conc, ext, usa = ws.vocab_para(op, self)
            ws.assinaturas.add(ws.assinatura(op, usa))''')
rep('''        elif op.startswith("prever"):
            conc, _ext, usa = ws.vocab_para(op, self)''', '''        elif op.startswith("prever"):
            conc, _ext, usa = ws.vocab_para(op, self)
            ws.assinaturas.add(ws.assinatura(op, usa))''')

rep('''    def gasta(self, c, progresso):''', '''    def assinatura(self, op, usa):
        tipo = "prever" if op.startswith("prever") else "ajustar"
        return (tipo, tuple(sorted(usa)), len(self.dados),
                tuple(sorted(self.variantes)))

    def ja_feito(self, op, org):
        if not (op.startswith("ajustar") or op.startswith("prever")):
            return False
        _c, _e, usa = self.vocab_para(op, org, dry=True)
        return self.assinatura(op, usa) in self.assinaturas

    def gasta(self, c, progresso):''')

rep('''    def _agenda_aprendida(self, ws):
        while ws.gasto < ws.orcamento and not ws.valido:
            T = self._temperatura(ws)
            cands = self._candidatas(ws)''', '''    def _agenda_aprendida(self, ws):
        for _seg in range(SEGURANCA_ACOES):
            if ws.gasto >= ws.orcamento or ws.valido:
                break
            T = self._temperatura(ws)
            cands = [c for c in self._candidatas(ws) if not ws.ja_feito(c[0], self)]''')
rep('''        for op, e in ordem:
            if ws.valido or ws.gasto >= ws.orcamento:
                break
            if e is not None and e.id not in self.E:
                continue''', '''        for op, e in ordem:
            if ws.valido or ws.gasto >= ws.orcamento:
                break
            if e is not None and e.id not in self.E:
                continue
            if ws.ja_feito(op, self):
                continue''')
rep('''MEM_PONTOS = 40000   # memoria externa (pontos de dado guardados), como o orcamento''', '''MEM_PONTOS = 40000   # memoria externa (pontos de dado guardados), como o orcamento
SEGURANCA_ACOES = 80 # protecao de laco por problema (nunca deve limitar)''')

# ---------- a percepcao reaproveita o ajuste de base como 1a acao ----------
rep('''    def resolve(self, prob, orcamento):
        x, feats, c_perc = self.percebe(prob)
        ws = Espaco(prob, x, orcamento)
        ws.gasta(c_perc, progresso=False)''', '''    def resolve(self, prob, orcamento):
        ws = Espaco(prob, None, orcamento)
        if prob.tipo == "L":
            ws.x = hdc.codifica({"tipo": "L", "fonte": prob.fonte})
            self._executa(ws, "ajustar_base", None)
            x, feats = self.percebe(prob, ws.melhor["resposta"])
        else:
            x, feats = self.percebe(prob, None)
        ws.x = x''')
rep('''    def percebe(self, prob):
        feats = {"tipo": prob.tipo, "fonte": prob.fonte}
        custo = 0.0
        if prob.tipo == "L":
            leis, _b, c = F.ajusta_lei([(prob.S, prob.A)], {}, [])
            custo += c
            ts = leis[0].termos''', '''    def percebe(self, prob, lei_base):
        feats = {"tipo": prob.tipo, "fonte": prob.fonte}
        if prob.tipo == "L":
            ts = lei_base.termos''')
rep('''                lam = max(F._lam(q), 1e-6)
                feats[nome] = int(np.round(np.log10(lam) * 2))
        return hdc.codifica(feats), feats, custo''', '''                lam = max(F._lam(q), 1e-6)
                feats[nome] = int(np.round(np.log10(lam) * 2))
        return hdc.codifica(feats), feats''')

# sonho: todos os conceitos que ja pagaram + novos; todo o dado do aberto
rep('''            novos = [e for e in self.E.values() if e.tipo in ("conceito", "direcao")
                     and e.nasc > ult]
            if not novos:
                continue''', '''            novos = [e for e in self.E.values() if e.tipo in ("conceito", "direcao")
                     and e.nasc > ult]
            if not novos:
                continue
            novos += [e for e in self.E.values() if e.tipo in ("conceito", "direcao")
                      and e.nasc <= ult and any(o > 0 for _t, _c, o in e.regs)]''')
rep('''            dados = ab.payload["dados"][-6:]''', '''            dados = ab.payload["dados"]''')
open("organismo.py", "w", encoding="utf-8").write(src)
print("trocas:", n_ok)
