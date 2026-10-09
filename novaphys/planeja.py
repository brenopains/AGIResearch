"""novaphys.planeja — encadeamento para tras sobre um grafo de transicoes.

GERAL POR CONSTRUCAO: recebe {saida: {entradas}} e um objetivo, devolve um plano.
Nao ha nada de Factorio aqui. O mesmo algoritmo serve para receita, para sintese
quimica, para prova, para qualquer cadeia de producao.

E o regime que a gente MEDIU funcionar: com modelo de mundo DADO, planejar vale
62x; com modelo APRENDIDO PARCIAL, 1,0x. Aqui o modelo e dado (as receitas sao
lidas do jogo), que e a condicao do usuario: "ja com conhecimento do que e o que".
"""
from __future__ import annotations


class Plano:
    def __init__(self, grafo, primitivos):
        """grafo: {saida: {"ing": {item: qtd}, "pro": {item: qtd}}}
        primitivos: itens que se obtem direto do mundo (minerio, pedra...)"""
        self.g = grafo
        self.prim = set(primitivos)
        self.por_saida = {}
        for nome, r in grafo.items():
            for p in r["pro"]:
                self.por_saida.setdefault(p, []).append(nome)

    def decompoe(self, alvo, qtd=1, prof=0, visto=None, teto=12):
        """Volta a arvore de subobjetivos. Devolve (passos, profundidade, faltou)."""
        visto = visto or set()
        if alvo in self.prim:
            return [("extrair", alvo, qtd, prof)], prof, []
        if prof >= teto or alvo in visto:
            return [], prof, [alvo]
        rs = self.por_saida.get(alvo)
        if not rs:
            return [], prof, [alvo]
        # escolhe a receita de MENOR profundidade total — criterio de descricao curta
        melhor, md, mf = None, 10**9, None
        for nome in rs:
            r = self.g[nome]
            saida_qtd = r["pro"].get(alvo, 1)
            n = max(1, int(-(-qtd // saida_qtd)))
            passos, dmax, falta = [], prof, []
            for ing, q in r["ing"].items():
                p2, d2, f2 = self.decompoe(ing, q * n, prof + 1, visto | {alvo}, teto)
                passos += p2; dmax = max(dmax, d2); falta += f2
            passos.append(("fazer", nome, n, prof))
            if not falta and dmax < md:
                melhor, md, mf = passos, dmax, falta
            elif melhor is None:
                melhor, md, mf = passos, dmax, falta
        return melhor or [], md, mf or []

    def plano(self, alvo, qtd=1):
        passos, prof, falta = self.decompoe(alvo, qtd)
        # junta passos iguais preservando a ordem
        vistos, fora = set(), []
        for p in passos:
            ch = (p[0], p[1])
            if ch not in vistos:
                vistos.add(ch); fora.append(p)
        return {"alvo": alvo, "qtd": qtd, "passos": fora,
                "profundidade": prof, "faltou": falta,
                "n_passos": len(fora)}
