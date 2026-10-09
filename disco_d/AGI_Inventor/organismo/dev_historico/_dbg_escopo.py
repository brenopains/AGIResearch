import sys, json, numpy as np
sys.path.insert(0, ".")
import organismo as O, vida, mundo_fisico as M
orig = O.Organismo._agenda_aprendida
vistos = [0]
def espia(self, ws):
    if ws.prob.tipo == "L" and ws.prob.T is not None and vistos[0] < 4 and ws.prob.idx >= 30:
        vistos[0] += 1
        T = self._temperatura(ws)
        linhas = []
        for op, e in self._candidatas(ws):
            if ws.ja_feito(op, self, e):
                continue
            th = self.media(self.E[f"op:{op}"], ws.x, T)
            if e is not None:
                th *= self.media(e, ws.x_reat if e.tipo == "aberto" else ws.x, T)
            c = self._custo_previsto(op, e, ws)
            v = (th + self._valor_futuro(op)) / max(c, 1e-3)
            linhas.append((v, op, e.id if e else "", round(th, 3), round(c, 2), round(self._valor_futuro(op), 2)))
        linhas.sort(reverse=True)
        print(f"problema {ws.prob.idx} {ws.prob.fam} orcamento restante {ws.orcamento-ws.gasto:.0f}, {len(linhas)} candidatas; cpt={ {k: round(v*1e4,3) for k, v in self._cpt.items()} } trab_inv={self._trabalho("inventar", None, ws)} exec_inv={self.E["op:inventar"].payload}")
        for l in linhas[:6]: print("   ", l)
        inv = [l for l in linhas if l[1] == "inventar"]
        print("    inventar:", inv[:1], "posicao", [l[1] for l in linhas].index("inventar") if inv else None)
    return orig(self, ws)
O.Organismo._agenda_aprendida = espia
import shutil; shutil.rmtree("resultados/_dbg", ignore_errors=True)
vida.vive("ESCOPO", 1, ("periodo_longo",), "resultados/_dbg", n_max=70, ablacao=False)
