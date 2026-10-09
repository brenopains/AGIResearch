"""novaphys.alcance_total — o 0x e COBERTURA ou EPISTEMOLOGIA?

DE ONDE VEM
-------------
`alcance.py` mostrou que a vantagem 30-54x do discriminador desaba quando ha
uma porta, e que com modelo APRENDIDO o planejador vira identico a novidade
(1,0x). A causa apontada foi ALCANCABILIDADE. Mas o `Modelo` de la tem uma
propriedade que nunca foi variada:

    "Transicoes deduzidas. Nao inventa: estado nao visto e lacuna."

Ele se RECUSA a chutar. Busca em largura nao atravessa lacuna, entao o
planejador morre na fronteira da experiencia. Isso confunde duas explicacoes:

  COBERTURA      o modelo nao esta DEFINIDO fora do visto -> busca nao anda
  EPISTEMOLOGIA  o modelo esta ERRADO de forma categorica -> planejar amplifica

O TESTE
---------
`ModeloTotal`: mesma aprendizagem, mas generaliza o deslocamento TIPICO de cada
acao (aprendido dos proprios dados) para estado nunca visto. Fica TOTAL — e
fica ERRADO na porta, porque preve que da para atravessar.

O mecanismo que isso liberta, se a leitura estiver certa: o planejador traca
rota ate a celula decisiva ATRAVESSANDO a porta; executar a rota e caminhar ate
a porta; a chave esta no meio do caminho; ele pega a chave POR ACIDENTE, sem
nunca ter tido motivo informacional para quere-la. Otimismo sob ignorancia
vira motor de exploracao.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  T0  PRE-CONDICAO. Sem porta, total e parcial tem que empatar (o modelo ja
      cobre quase tudo). Se total ganhar muito SEM porta, o chute esta dando
      informacao que nao deveria e o teste nao vale.
  T1  COM porta, total recupera >= metade da vantagem que a parcial perdeu
      -> COBERTURA e a variavel; chutar e barato e nos dois erramos.
  T2  COM porta, total nao recupera nada (ou piora)
      -> EPISTEMOLOGIA; modelo total e errado e pior que modelo com buraco.
  T3  total piora TAMBEM sem porta -> o chute e ruim demais, teste inconclusivo.

MINHA PREVISAO, escrita antes de rodar: T1, com 60% de confianca, e espero
recuperacao PARCIAL e nao total — o planejador vai bater na porta fechada
repetidas vezes e desperdicar acoes, mas vai pegar a chave no caminho.
Calibragem: das minhas ultimas previsoes nesta sessao, a de poder do duto de
farmaco saiu certa e a de "o limiar zero basta" saiu errada.
"""
from __future__ import annotations
import argparse, random, statistics
from collections import deque
from typing import Dict, Optional

from novaphys.alcance import (LADO, Mundo, hipoteses, Modelo, esc_novidade,
                              esc_planejada, esc_acaso)


class ModeloTotal(Modelo):
    """Total: generaliza o deslocamento tipico da acao para o nao visto."""

    def __init__(self) -> None:
        super().__init__()
        self.desl: Dict[int, Dict[tuple, int]] = {a: {} for a in range(4)}

    def aprender(self, e0, a, e1) -> None:
        super().aprender(e0, a, e1)
        d = (e1[0][0] - e0[0][0], e1[0][1] - e0[0][1])
        self.desl[a][d] = self.desl[a].get(d, 0) + 1

    def prever(self, e0, a):
        p = self.tab.get((e0, a))
        if p is not None:
            return p
        if not self.desl[a]:
            return None
        d = max(self.desl[a].items(), key=lambda kv: kv[1])[0]
        i = min(LADO - 1, max(0, e0[0][0] + d[0]))
        j = min(LADO - 1, max(0, e0[0][1] + d[1]))
        return ((i, j), e0[1])                 # chuta: bandeira nao muda


class ModeloTotalErrado(ModeloTotal):
    """TOTAL mas sistematicamente ERRADO fora do visto: ao chutar para a acao a,
    usa o deslocamento tipico de OUTRA acao (permutacao fixa sem ponto fixo).
    Exato no visto (consulta a tabela antes). Isola COBERTURA de CORRECAO:
    se este ainda esmagar o parcial, a variavel e estar DEFINIDO, nao estar certo.
    """
    TROCA = {0: 1, 1: 0, 2: 3, 3: 2}

    def prever(self, e0, a):
        p = self.tab.get((e0, a))
        if p is not None:
            return p
        b = self.TROCA[a]
        if not self.desl[b]:
            return None
        d = max(self.desl[b].items(), key=lambda kv: kv[1])[0]
        i = min(LADO - 1, max(0, e0[0][0] + d[0]))
        j = min(LADO - 1, max(0, e0[0][1] + d[1]))
        return ((i, j), e0[1])


def rodar2(d, porta, explorador, max_acoes, semente, klass=Modelo):
    rng = random.Random(semente)
    m = Mundo(d, porta)
    hip = hipoteses(d)
    est = [((i, j), t) for i in range(LADO) for j in range(LADO) for t in (0, 1)]
    assin = [tuple(f(e) for e in est) for _n, f in hip]
    vivos = list(range(len(hip)))
    cont: Dict[tuple, int] = {}
    mod = klass()
    esc = {"acaso": esc_acaso, "novidade": esc_novidade,
           "planejada": esc_planejada}[explorador]
    for t in range(1, max_acoes + 1):
        e0 = m.estado()
        a = esc(e0, cont, vivos, hip, rng, mod)
        e1 = m.passo(a)
        mod.aprender(e0, a, e1)
        cont[e1] = cont.get(e1, 0) + 1
        v = m.venceu(e1)
        vivos = [k for k in vivos if hip[k][1](e1) == v]
        if vivos and len({assin[k] for k in vivos}) == 1:
            return t
        if v:
            m.reiniciar()
    return None


def med(vals, teto):
    vv = [v if v is not None else teto for v in vals]
    return statistics.median(vv), sum(1 for v in vals if v is None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=15)
    ap.add_argument("--acoes", type=int, default=20000)
    ap.add_argument("--semente", type=int, default=20260905)
    a = ap.parse_args()
    print("=" * 74)
    print("ALCANCE TOTAL — chutar na lacuna recupera o planejamento?")
    print("=" * 74)
    for porta in (False, True):
        print(f"\n{'COM PORTA (alcance cresce)' if porta else 'SEM PORTA (alcance fixo)'}")
        print(f"  {'d':>3} {'NOVIDADE':>10} {'PLA.parcial':>12} {'TOT.correto':>11}"
              f" {'TOT.ERRADO':>12} {'nov/parc':>9} {'nov/corr':>10} {'nov/ERR':>10}")
        print("  " + "-" * 88)
        for d in (9, 27):
            r = {}
            for nome, kl, ex in (("nov", Modelo, "novidade"),
                                 ("parc", Modelo, "planejada"),
                                 ("tot", ModeloTotal, "planejada"),
                                 ("err", ModeloTotalErrado, "planejada")):
                vals = [rodar2(d, porta, ex, a.acoes, a.semente + s, kl)
                        for s in range(a.sementes)]
                r[nome] = med(vals, a.acoes)
            n, p, t, e = (r["nov"][0], r["parc"][0], r["tot"][0], r["err"][0])
            print(f"  {d:3d} {n:10.0f} {p:12.0f} {t:11.0f} {e:12.0f}"
                  f" {n/max(p,1):9.2f} {n/max(t,1):10.2f} {n/max(e,1):10.2f}"
                  f"   (falhas {r['nov'][1]}/{r['parc'][1]}/{r['tot'][1]}/{r['err'][1]})")


if __name__ == "__main__" and not set(["--limiar","--separa"]) & set(__import__("sys").argv):
    main()


class ModeloTotalP(ModeloTotal):
    """Total, com probabilidade P de o chute estar CERTO (senao usa o
    deslocamento de outra acao). P=1 -> correto; P=0 -> errado.
    Acha o LIMIAR: quao certo o modelo aprendido precisa estar na fronteira
    para planejar pagar mais que nao planejar."""
    P = 1.0
    TROCA = {0: 1, 1: 0, 2: 3, 3: 2}

    def __init__(self):
        super().__init__()
        self._r = random.Random(12345)

    def prever(self, e0, a):
        p = self.tab.get((e0, a))
        if p is not None:
            return p
        b = a if self._r.random() < self.P else self.TROCA[a]
        if not self.desl[b]:
            return None
        d = max(self.desl[b].items(), key=lambda kv: kv[1])[0]
        i = min(LADO - 1, max(0, e0[0][0] + d[0]))
        j = min(LADO - 1, max(0, e0[0][1] + d[1]))
        return ((i, j), e0[1])


def varredura_p(sementes=6, acoes=6000, semente=20260905, d=9):
    print("=" * 66)
    print("LIMIAR DE CORRECAO — quanto o chute precisa acertar para pagar?")
    print("=" * 66)
    for porta in (False, True):
        print(f"\n{'COM PORTA' if porta else 'SEM PORTA'}  (d={d})")
        base = med([rodar2(d, porta, "novidade", acoes, semente + s, Modelo)
                    for s in range(sementes)], acoes)[0]
        print(f"  novidade (referencia): {base:.0f} acoes")
        print(f"  {'P(chute certo)':>16}{'acoes':>9}{'nov/pla':>10}{'falhas':>8}   veredito")
        for P in (globals().get("PS") or (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)):
            kl = type("M", (ModeloTotalP,), {"P": P})
            v = [rodar2(d, porta, "planejada", acoes, semente + s, kl)
                 for s in range(sementes)]
            m, f = med(v, acoes)
            r = base / max(m, 1)
            ver = "PAGA" if r > 1.15 else ("empata" if r > 0.85 else "PIOR que nao planejar")
            print(f"  {P:16.2f}{m:9.0f}{r:10.2f}{f:8d}   {ver}")


if __name__ == "__main__" and "--limiar" in __import__("sys").argv:
    varredura_p()


class ModeloErradoVariado(ModeloTotal):
    """0% correto, DETERMINISTICO, mas o erro varia por estado (hash), em vez de
    um vies global unico. Separa 'taxa de acerto' de 'consistencia do erro'."""
    def prever(self, e0, a):
        p = self.tab.get((e0, a))
        if p is not None:
            return p
        h = (hash((e0[0][0], e0[0][1], a)) % 3) + 1          # nunca 0 -> nunca certo
        b = (a + h) % 4
        if not self.desl[b]:
            return None
        d = max(self.desl[b].items(), key=lambda kv: kv[1])[0]
        i = min(LADO - 1, max(0, e0[0][0] + d[0]))
        j = min(LADO - 1, max(0, e0[0][1] + d[1]))
        return ((i, j), e0[1])


class ModeloErradoSorteado(ModeloTotal):
    """0% correto, ESTOCASTICO: sorteia entre as 3 acoes erradas a cada chamada."""
    def __init__(self):
        super().__init__(); self._r = random.Random(999)
    def prever(self, e0, a):
        p = self.tab.get((e0, a))
        if p is not None:
            return p
        b = (a + self._r.randint(1, 3)) % 4
        if not self.desl[b]:
            return None
        d = max(self.desl[b].items(), key=lambda kv: kv[1])[0]
        i = min(LADO - 1, max(0, e0[0][0] + d[0]))
        j = min(LADO - 1, max(0, e0[0][1] + d[1]))
        return ((i, j), e0[1])


def separa(sementes=4, acoes=6000, semente=20260905, d=9):
    print("=" * 74)
    print("TAXA DE ACERTO ou CONSISTENCIA DO ERRO? (todos 0% corretos, exceto base)")
    print("=" * 74)
    base = med([rodar2(d, False, "novidade", acoes, semente + s, Modelo)
                for s in range(sementes)], acoes)[0]
    print(f"  novidade (referencia): {base:.0f} acoes\n")
    print(f"  {'modelo':<34}{'acerto':>8}{'acoes':>9}{'nov/pla':>10}{'falhas':>8}")
    alvos = [("parcial (recusa chutar)", Modelo, "-"),
             ("errado SISTEMATICO global", ModeloTotalErrado, "0%"),
             ("errado DETERMINISTICO variado", ModeloErradoVariado, "0%"),
             ("errado ESTOCASTICO", ModeloErradoSorteado, "0%"),
             ("correto", ModeloTotal, "100%")]
    for nome, kl, ac in alvos:
        v = [rodar2(d, False, "planejada", acoes, semente + s, kl) for s in range(sementes)]
        m, f = med(v, acoes)
        print(f"  {nome:<34}{ac:>8}{m:9.0f}{base/max(m,1):10.2f}{f:8d}")


if __name__ == "__main__" and "--separa" in __import__("sys").argv:
    separa()
