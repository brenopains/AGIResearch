"""
novaphys.alvo_proprio — o sistema escolhe o que atacar em seguida
==================================================================

DE ONDE VEM
-------------
Do usuario: "pense sobre como a mente humana define objetivos por si so... como
pensa pra definir o que atacar em que momento".

E da minha propria lista do que falta: "o sistema nao quer nada, nao escolhe o
que atacar". Este modulo ataca esse item.

O QUE A LITERATURA DIZ, E O QUE ESTE PROJETO JA TINHA
-------------------------------------------------------
O campo se chama aprendizagem aberta com motivacao intrinseca; agentes que
escolhem os proprios objetivos sao AUTOTELICOS. O sinal que a literatura usa e
PROGRESSO DE APRENDIZAGEM: escolhe-se o objetivo em que se preve MELHORAR mais
— nem o ja dominado, nem o impossivel (MAGELLAN, arXiv 2502.07709).

Este projeto ja tinha uma versao medida disso sem eu conectar:
[[acerto-na-fronteira]] mediu que planejar paga a partir de ~10% de acerto onde
nunca se esteve, e a 0% nunca paga (6 falhas em 6). Mesma forma: existe uma
zona de dificuldade produtiva, com limiar medido.

O DESENHO
-----------
Ha 12 objetivos candidatos (material x transformacao). Alguns sao quase
dominados, outros impossiveis para o encoder (canal dominante quadratico), e
outros estao no meio. O sistema tem ORCAMENTO de rodadas; a cada rodada escolhe
UM objetivo e gasta episodios nele, o que melhora a regra daquele objetivo.

  competencia(objetivo) = acuracia BALANCEADA do preditor atual
  progresso(objetivo)   = competencia agora - competencia ha 1 visita
  escolha por PA        = maior progresso recente, com bonus de exploracao
                          para quem nunca foi visitado

Metrica final: COBERTURA — soma das competencias em TODOS os objetivos ao fim
do orcamento. Cobertura mede o que se ganhou no repertorio, nao num alvo so.

PRE-REGISTRO
--------------
  W0  PRE-CONDICAO. Os candidatos tem de cobrir faixa de dificuldade: pelo
      menos 2 objetivos acima de 0,75 e 2 abaixo de 0,60 na competencia
      inicial. Sem faixa, nao ha o que escolher e a comparacao e vazia.
  W1  PA > sorteio em cobertura final, por >= 2 desvios do sorteio
      -> o sistema escolhe melhor que o acaso o que atacar
  W2  PA ~ sorteio -> o sinal de progresso nao informa neste substrato
  W3  PA < sorteio -> o sinal e anti-informativo

  Rivais, todos com o MESMO orcamento:
    sorteio        escolhe ao acaso              (o placebo)
    facil-primeiro sempre a maior competencia    (exploracao nula)
    dificil-1o     sempre a menor competencia    (a armadilha do impossivel)
    PA             maior progresso recente       (o que se testa)

  10 sementes, media e desvio. Nada de n=1.

PREVISAO: W1, 0,6. Mecanismo: `dificil-primeiro` deve ser o pior porque gasta
todo o orcamento nos objetivos de canal quadratico, onde o encoder nao expressa
a regra e nenhum episodio ajuda — a armadilha do impossivel e real e medida.
`facil-primeiro` deve empatar com sorteio porque satura rapido. O PA deve
ganhar por evitar os dois extremos. RISCO NOMEADO: com orcamento pequeno o
progresso e ruidoso demais para estimar, e o PA vira sorteio caro.
"""

from __future__ import annotations

import random
import statistics
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from novaphys.hdc_indice import Espaco, de_fluxo, feixe, sim
from novaphys.nuvem import codifica, preve
from novaphys.nuvem_matriz import (TRANSFS, _canal, _latente, episodio,
                                   sorteia_materiais)
from novaphys.segunda_ordem import _ac

N_MAT = 3
T_EP = 150
ORCAMENTO = 24          # rodadas de escolha
EP_POR_RODADA = 3       # episodios gastos por rodada escolhida
N_TESTE = 6
N_SEM = 10
SEMENTE = 20260918


class Objetivo:
    """Um alvo candidato: prever a transformacao `tn` no material `mi`."""

    def __init__(self, e: Espaco, rng, mats, mi: int, tn: str):
        self.e, self.rng, self.mats, self.mi, self.tn = e, rng, mats, mi, tn
        self.nome = f"m{mi}|{tn}"
        self.vetores: List[np.ndarray] = []
        self.visitas = 0
        self.hist: List[float] = []
        self.lim = rng.uniform(-0.3, 0.3)     # FIXO por objetivo
        self.testes = [self._episodio() for _ in range(N_TESTE)]
        self.gasta(1)                       # semente minima, igual para todos

    def _episodio(self):
        """
        Episodio com o limiar FIXO do objetivo.

        CONSERTO DE GERADOR, e ele explica todos os fracassos anteriores deste
        modulo. `nuvem_matriz._alvo` sorteia `lim ~ U(-0,3;0,3)` a CADA
        episodio — entao "alvo +1 quando s > lim" e uma REGRA DIFERENTE em cada
        um, e nao ha o que acumular entre episodios do mesmo objetivo. Medido:
        1 episodio da o mesmo que 40, e feixar muitos PIORA (m1|extremo vai de
        0,788 com 3 para 0,520 com 20).

        Um objetivo so e um objetivo se a regra dele for a MESMA toda vez.
        Aqui o limiar e sorteado UMA vez, na criacao, e vale para todos os
        episodios daquele objetivo — inclusive os de teste.
        """
        s = _latente(self.rng, T_EP)
        A = np.stack([_canal(c, s, self.rng) for c in self.mats[self.mi]],
                     axis=1)
        d = np.diff(s, prepend=s[0])
        dd = np.diff(d, prepend=d[0])
        L = self.lim
        if self.tn == "regime":
            y = np.where(s > L, 1, -1)
        elif self.tn == "deriva":
            y = np.where(d > 0, 1, -1)
        elif self.tn == "extremo":
            y = np.where(np.abs(s) > abs(L) + 0.5, 1, -1)
        else:
            y = np.where(dd > 0, 1, -1)
        return A, y

    def gasta(self, n: int) -> None:
        for _ in range(n):
            A, y = self._episodio()
            # DOIS vetores por episodio: um de BUSCA (so o fluxo) e um de
            # CONTEUDO (fluxo + alvo). A versao anterior buscava com o fluxo
            # codificado com alvo TODO +1 contra vetores que tinham o alvo
            # verdadeiro dentro — a similaridade era dominada pelo descasamento
            # de alvo e nao pela semelhanca de fluxo. E o padrao do `Acervo`,
            # que ja mediu que separar busca de conteudo funciona.
            self.vetores.append((de_fluxo(self.e, A), codifica(self.e, A, y)))
        self.visitas += 1

    def competencia(self, k_viz: int = 3) -> float:
        """
        Competencia por RECUPERACAO, nao por feixe de tudo.

        CONSERTO, e ele vem de medida. A versao anterior feixava TODOS os
        episodios numa regra so, e a curva de aprendizado media assim NAO SOBE:
        ```
        m1|regime  1ep 0,501  3ep 0,512  8ep 0,883  20ep 0,754  40ep 0,796
        m1|extremo 1ep 0,623  3ep 0,788  8ep 0,739  20ep 0,520  40ep 0,578
        ```
        Tres dos quatro objetivos PIORAM com mais episodios. O `feixe` e soma
        com sinal e cada episodio tem limiar proprio sorteado — empilhar muitos
        nao acumula evidencia, ela se CANCELA.

        E a quinta vez que o mesmo padrao aparece no projeto: FUNDIR destroi,
        GUARDAR SEPARADO E RECUPERAR funciona ([[sob-demanda-funciona]]). Aqui
        se recuperam os `k_viz` episodios guardados mais parecidos com o de
        teste e so eles entram no feixe.
        """
        if not self.vetores:
            return 0.5

        def preditor(A: np.ndarray) -> np.ndarray:
            sonda = de_fluxo(self.e, A)
            viz = sorted(self.vetores, key=lambda vc: -sim(sonda, vc[0]))
            regras = [c for _b, c in viz[:k_viz]]
            return preve(self.e, feixe(regras, self.e.dim), A, 1.0)

        return _ac(preditor, self.testes)

    def progresso(self) -> float:
        """Quanto melhorou desde a visita anterior. E o sinal da literatura."""
        if len(self.hist) < 2:
            return 1.0                      # nunca visitado: bonus de exploracao
        return self.hist[-1] - self.hist[-2]


def roda(estrategia: str, e: Espaco, semente: int) -> float:
    rng = random.Random(semente)
    mats = [m for m in sorteia_materiais(rng, 18)
            if m[0]["tipo"] in ("linear", "saturado", "quadratico")][:N_MAT]
    objs = [Objetivo(e, rng, mats, mi, tn)
            for mi in range(len(mats)) for tn in TRANSFS]
    for o in objs:
        o.hist.append(o.competencia())

    for _ in range(ORCAMENTO):
        if estrategia == "sorteio":
            alvo = rng.choice(objs)
        elif estrategia == "facil":
            alvo = max(objs, key=lambda o: o.hist[-1])
        elif estrategia == "dificil":
            alvo = min(objs, key=lambda o: o.hist[-1])
        else:                                # progresso de aprendizagem
            alvo = max(objs, key=lambda o: o.progresso())
        alvo.gasta(EP_POR_RODADA)
        alvo.hist.append(alvo.competencia())
    return sum(o.competencia() for o in objs)


def experimento() -> None:
    e = Espaco()
    print("=" * 78)
    print("ALVO PROPRIO — o sistema escolhe o que atacar em seguida")
    print("=" * 78)
    print(f"  {N_MAT} materiais x {len(TRANSFS)} transformacoes = "
          f"{N_MAT*len(TRANSFS)} objetivos candidatos")
    print(f"  orcamento {ORCAMENTO} rodadas, {EP_POR_RODADA} episodios cada")
    print(f"  metrica: COBERTURA = soma das competencias em TODOS os "
          f"objetivos\n")

    # W0: a faixa de dificuldade existe?
    rng0 = random.Random(SEMENTE)
    mats0 = [m for m in sorteia_materiais(rng0, 18)
             if m[0]["tipo"] in ("linear", "saturado", "quadratico")][:N_MAT]
    ini = []
    for mi in range(len(mats0)):
        for tn in TRANSFS:
            o = Objetivo(e, rng0, mats0, mi, tn)
            ini.append((o.nome, mats0[mi][0]["tipo"][:4], o.competencia()))
    ini.sort(key=lambda x: -x[2])
    print("  competencia inicial dos candidatos:")
    for nome, tipo, c in ini:
        print(f"    {nome:14s} ({tipo})  {c:.3f}")
    altos = sum(1 for _n, _t, c in ini if c > 0.75)
    baixos = sum(1 for _n, _t, c in ini if c < 0.60)
    print(f"\n  {altos} acima de 0,75 e {baixos} abaixo de 0,60")
    if altos < 2 or baixos < 2:
        print("W0 FALHOU: sem faixa de dificuldade, a escolha nao tem o que "
              "decidir.")
        return
    print("W0 . ha faixa de dificuldade.\n")

    res: Dict[str, List[float]] = {}
    for est in ("sorteio", "facil", "dificil", "progresso"):
        res[est] = [roda(est, e, SEMENTE + s) for s in range(N_SEM)]
        m, sd = statistics.mean(res[est]), statistics.pstdev(res[est])
        print(f"  {est:12s} cobertura {m:6.3f} +- {sd:.3f}", flush=True)

    print("\n" + "=" * 78)
    ms = {k: statistics.mean(v) for k, v in res.items()}
    sd_s = statistics.pstdev(res["sorteio"])
    ganho = ms["progresso"] - ms["sorteio"]
    z = ganho / max(sd_s, 1e-9)
    print(f"  progresso - sorteio = {ganho:+.3f}  ({z:+.1f} desvios do sorteio)")
    print(f"  facil - sorteio     = {ms['facil']-ms['sorteio']:+.3f}")
    print(f"  dificil - sorteio   = {ms['dificil']-ms['sorteio']:+.3f}")
    print("")
    if z >= 2.0:
        print("W1 . O SISTEMA ESCOLHE MELHOR QUE O ACASO o que atacar.")
        print("     O sinal de progresso de aprendizagem informa neste")
        print("     substrato, e a escolha e do sistema, nao minha.")
    elif z > -2.0:
        print(f"W2 . o sinal nao informa ({z:+.1f} desvios). Escolher por")
        print("     progresso empata com sortear.")
    else:
        print(f"W3 . o sinal e ANTI-informativo ({z:+.1f} desvios).")


if __name__ == "__main__":
    experimento()
