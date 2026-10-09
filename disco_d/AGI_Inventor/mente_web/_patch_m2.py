s = open("mente.py", encoding="utf-8").read()
n = 0
def rep(a, b, c=1):
    global s, n
    assert s.count(a) == c, (s.count(a), a[:120]); s = s.replace(a, b); n += 1
# 2) estrutura: volta em >= 2 DOMINIOS independentes e tem >= 3 bytes
rep('''        for (a, b) in segs:
            p = bloco[a:b]
            if len(p) < 2:
                continue
            depois = float(sur[b:b + 8].mean()) if b < len(sur) else float(sur[-1])
            e = self.lex.get(p)
            if e is None:
                self.lex[p] = [1, {pid}, self.t]
            else:
                e[0] += 1
                e[2] = self.t
                if len(e[1]) < 50:
                    e[1].add(pid)
                if p not in self.E and len(e[1]) >= 2:
                    self._nasce(p, pg)''', '''        for (a, b) in segs:
            p = bloco[a:b]
            if len(p) < 3:
                continue
            # quanto custa ANTECIPAR este pedaco (surpresa media dos proprios bytes)
            depois = float(sur[a:b].mean())
            e = self.lex.get(p)
            if e is None:
                self.lex[p] = [1, {pg.dominio}, self.t]
            else:
                e[0] += 1
                e[2] = self.t
                if len(e[1]) < 12:
                    e[1].add(pg.dominio)
                # regularidade = volta em >= 2 dominios INDEPENDENTES
                if p not in self.E and len(e[1]) >= 2:
                    self._nasce(p, pg)''')
# 3) pergunta: pedaco de >= 4 bytes que eu nao consigo antecipar
rep('''        cands = [(p, s) for p, s in self.E.items() if s["incompr"] is not None and len(p) >= 3]''', '''        cands = [(p, s) for p, s in self.E.items() if s["incompr"] is not None and len(p) >= 4
                 and len(s["doms"]) >= 2]''')
# 1) acao que falha custa e ensina
rep('''        dt = time.time() - t0
        self.vida_s += dt
        self.custo[estr] = 0.8 * self.custo[estr] + 0.2 * dt
        hist = self.hist_prog[modal]
        ok = prog is not None and len(hist) > 5 and prog > np.median(hist[-200:])
        if prog is not None or nova is None:
            self.regs[estr].append([x, float(ok)])''', '''        dt = time.time() - t0
        self.vida_s += dt
        falhou = (nova is not None and nova.tipo == "erro") or \
            (prog is None and estr not in ("voltar",) and nova is None and alvo is None)
        # uma tentativa nunca custa zero: abrir a mao e olhar ja gasta vida
        self.custo[estr] = 0.8 * self.custo[estr] + 0.2 * max(dt, 1.0)
        hist = self.hist_prog[modal]
        ok = prog is not None and len(hist) > 5 and prog > np.median(hist[-200:])
        if prog is not None or falhou or nova is None:
            self.regs[estr].append([x, float(ok and not falhou)])''')
open("mente.py", "w", encoding="utf-8").write(s)
print("trocas", n)
