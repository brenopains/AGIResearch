src = open("organismo.py", encoding="utf-8").read()
n_ok = 0


def rep(a, b, conta=1):
    global src, n_ok
    if src.count(a) != conta:
        raise SystemExit(f"esperava {conta}, achei {src.count(a)}:\n" + a[:300])
    src = src.replace(a, b)
    n_ok += 1


# 1) macro guarda o alvo das operacoes "_com"
rep('''                if a["op"].startswith("ajustar") or a["op"].startswith("prever"):
                    arg = None''', '''                if (a["op"].startswith("ajustar") or a["op"].startswith("prever")) \\
                        and not a["op"].endswith("_com"):
                    arg = None''')

# 2) orcamento: toda acao precisa caber no que sobra (juntar inclusive, pelo
#    custo a jusante); so 'forma' e 'reajustar' (centesimos) ficam isentas
rep('''                c = self._custo_previsto(op, e, ws)
                if c > ws.orcamento - ws.gasto + 1e-9 and op not in ("forma", "juntar", "reajustar"):
                    continue
                v = th / max(c, 1e-3)''', '''                c = self._custo_previsto(op, e, ws)
                if c > ws.orcamento - ws.gasto + 1e-9 and op not in ("forma", "reajustar"):
                    continue
                v = th / max(c, 1e-3)''')
rep('''            c = self._custo_previsto(op, e, ws)
            if c > ws.orcamento - ws.gasto + 1e-9 and op not in ("forma", "juntar", "reajustar"):
                continue
            self._executa(ws, op, e)''', '''            c = self._custo_previsto(op, e, ws)
            if c > ws.orcamento - ws.gasto + 1e-9 and op not in ("forma", "reajustar"):
                continue
            self._executa(ws, op, e)''')

# 3) memoria por MUNDO: o problema resolvido por uma lei lembrada engorda o
#    aberto de onde a lei veio, em vez de abrir outro
rep('''            if juntou:
                ab = self.E[juntou[0]]''', '''            if not juntou and ws.melhor is not None and ws.melhor["acao"]["op"] == "forma":
                lid = ws.melhor["acao"]["arg"]
                for e in self._de_tipo("aberto"):
                    if lid in e.payload.get("leis", []):
                        juntou = [e.id]
                        break
            if juntou:
                ab = self.E[juntou[0]]''')
rep('''                if ws.melhor is not None and prob.tipo == "L":
                    ab.payload["melhor"] = repr(ws.melhor["resposta"])''', '''                if ws.melhor is not None and prob.tipo == "L":
                    ab.payload["melhor"] = repr(ws.melhor["resposta"])
                ab.payload.setdefault("leis", [])
                ab.payload["leis"] += [i for i in ws.leis_guardadas if i not in ab.payload["leis"]]''')
rep('''                ab = self._nova("aberto", {"dados": dados, "trajs": trajs,
                                           "candidatos": cands, "encontros": 1,
                                           "resolvido": ok,''', '''                ab = self._nova("aberto", {"dados": dados, "trajs": trajs,
                                           "candidatos": cands, "encontros": 1,
                                           "resolvido": ok, "leis": list(ws.leis_guardadas),''')
rep('''        nova = self._nova("lei", {"forma": forma, "repr": repr(lei), "atomos": usados},
                          ws.prob.fonte, pais=usados, ctx=ws.x)
        nova.origem = (f"op:{m['acao']['op']}", None)''', '''        nova = self._nova("lei", {"forma": forma, "repr": repr(lei), "atomos": usados},
                          ws.prob.fonte, pais=usados, ctx=ws.x)
        nova.origem = (f"op:{m['acao']['op']}", None)
        ws.leis_guardadas.append(nova.id)''')
rep('''        for e in self._de_tipo("lei"):
            if e.payload["forma"] == forma:
                if not any(a["arg"] == e.id for a in ws.acoes):
                    self._registra(e, ws.x, ok)
                return''', '''        for e in self._de_tipo("lei"):
            if e.payload["forma"] == forma:
                if not any(a["arg"] == e.id for a in ws.acoes):
                    self._registra(e, ws.x, ok)
                ws.leis_guardadas.append(e.id)
                return''')
rep('''        self.confirmados = set()
        self.alvo_com = None''', '''        self.confirmados = set()
        self.alvo_com = None
        self.leis_guardadas = []''')
# _credito chama _guarda_lei ANTES de _atualiza_abertos: ordem ja certa.
open("organismo.py", "w", encoding="utf-8").write(src)
print("trocas:", n_ok)
