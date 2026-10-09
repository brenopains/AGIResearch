"""novaphys.planeja2 — a CONDICAO DE HABILITACAO e ela mesma um objetivo.

O planeja.py tinha as transicoes (minerio -> placa) e nao as condicoes que as
habilitam (esta transicao so ocorre DENTRO de um forno). Medido: 0/6, e o
`begin_crafting` recusou tudo porque fundicao nao se faz na mao.

Isso e geral: reacao precisa de catalisador, regra de inferencia precisa da
forma da premissa, transicao precisa de maquina. Transicao sem condicao de
habilitacao e modelo INCOMPLETO em qualquer dominio.

Aqui a condicao vira SUBOBJETIVO: para fundir preciso de forno; para ter forno
preciso fabricar um; para fabricar preciso de pedra; pedra eu extraio.
"""
from __future__ import annotations


class Plano2:
    def __init__(self, grafo, categorias, maquinas, primitivos, na_mao=("crafting",)):
        self.g = grafo
        self.cat = categorias
        self.maq = maquinas              # {categoria: [maquinas que fazem]}
        self.prim = set(primitivos)
        self.na_mao = set(na_mao)
        self.por_saida = {}
        for nome, r in grafo.items():
            for p in r["pro"]:
                self.por_saida.setdefault(p, []).append(nome)

    def _melhor_maquina(self, cat):
        opc = self.maq.get(cat, [])
        return sorted(opc, key=len)[0] if opc else None

    def decompoe(self, alvo, qtd=1, prof=0, visto=None, teto=14):
        visto = visto or set()
        if alvo in self.prim:
            return [("extrair", alvo, qtd, prof, None)], prof, []
        if prof >= teto or alvo in visto:
            return [], prof, [alvo]
        rs = self.por_saida.get(alvo)
        if not rs:
            return [], prof, [alvo]
        melhor, md, mf = None, 10**9, None
        for nome in rs:
            r = self.g[nome]
            cat = self.cat.get(nome, "crafting")
            sq = r["pro"].get(alvo, 1)
            n = max(1, int(-(-qtd // sq)))
            passos, dmax, falta = [], prof, []
            # CONDICAO DE HABILITACAO como SUBOBJETIVO
            onde = None
            if cat not in self.na_mao:
                onde = self._melhor_maquina(cat)
                if onde is None:
                    falta.append(f"maquina:{cat}")
                elif onde not in visto:
                    p3, d3, f3 = self.decompoe(onde, 1, prof + 1, visto | {alvo}, teto)
                    passos += p3; dmax = max(dmax, d3); falta += f3
            for ing, q in r["ing"].items():
                p2, d2, f2 = self.decompoe(ing, q * n, prof + 1, visto | {alvo}, teto)
                passos += p2; dmax = max(dmax, d2); falta += f2
            passos.append(("fazer", nome, n, prof, onde))
            if not falta and dmax < md:
                melhor, md, mf = passos, dmax, falta
            elif melhor is None:
                melhor, md, mf = passos, dmax, falta
        return melhor or [], md, mf or []

    def plano(self, alvo, qtd=1):
        passos, prof, falta = self.decompoe(alvo, qtd)
        vistos, fora = set(), []
        for p in passos:
            ch = (p[0], p[1])
            if ch not in vistos:
                vistos.add(ch); fora.append(p)
        return {"alvo": alvo, "qtd": qtd, "passos": fora, "profundidade": prof,
                "faltou": sorted(set(falta)), "n_passos": len(fora),
                "precisa_maquina": sorted({p[4] for p in fora if p[4]})}
