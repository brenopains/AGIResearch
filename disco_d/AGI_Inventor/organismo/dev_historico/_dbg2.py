import time, collections, numpy as np
import organismo as O, vida
orig = O.Organismo._executa
estouros = collections.Counter(); pior = []
def cron(self, ws, op, e):
    resto = ws.orcamento - ws.gasto
    prev = self._custo_previsto(op, e, ws) if op != "ajustar_base" or ws.acoes else 0
    a = orig(self, ws, op, e)
    real = a.get("custo", 0)
    if real > resto + 1e-6:
        estouros[op] += 1
        pior.append((round(real - resto, 1), op, round(prev, 1), round(real, 1)))
    return a
O.Organismo._executa = cron
import shutil; shutil.rmtree("resultados/dbg", ignore_errors=True)
t0 = time.time()
r = vida.vive("ORGANISMO", 3, (), "resultados/dbg", n_max=80, ablacao=False)
print("tempo", round(time.time() - t0, 1), "s;  acertos", sum(l["ok"] for l in r["linhas"]), "/", len(r["linhas"]))
print("estouros por op:", dict(estouros))
for p in sorted(pior, reverse=True)[:12]: print("  excesso", p[0], p[1], "previsto", p[2], "real", p[3])
