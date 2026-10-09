"""
novaphys.lente_ativa_rep — a replica, e o controle que faltou
===============================================================

DE ONDE VEM
-------------
`lente_ativa_limpa` (semente 20260910, 10 episodios por braco) deu A1 pela regra
pre-registrada:

    lente    passivo  descoberta 3/10
    lente    ativo    descoberta 5/10
    sorteado          0/10 nos dois bracos

A diferenca de 2 bateu EXATAMENTE no limiar, com n = 10. Um teste de sinal sobre
isso nao passaria de p ~ 0,3. E o resultado contraria um prior medido do projeto
("desenho de experimento so paga com modelo de mundo DADO"). Prior medido nao se
derruba com n = 10.

E faltou um controle. O braco ativo escolhe a cena de maior DESACORDO do comite.
Mas cenas de maior desacordo tendem a ser cenas mais CHEIAS (mais particulas, mais
massas), e cena cheia carrega mais informacao sem que hipotese nenhuma entre em
jogo. Se for isso, o ganho e de uma heuristica de riqueza, nao de agencia guiada
por modelo.

O DESENHO
-----------
Tres bracos, orcamento casado de 5 observacoes, 30 episodios, dois mundos:

    PASSIVO   5 cenas sorteadas
    ATIVO     3 sorteadas + 2 escolhidas por desacordo do comite (como antes)
    RICO      3 sorteadas + 2 escolhidas entre as MESMAS 10 candidatas do ativo,
              pela que tem mais celulas ocupadas — sem hipotese nenhuma

ATIVO e RICO veem as mesmas 3 cenas iniciais E os mesmos conjuntos de candidatas
(o comite usa um gerador separado). A unica diferenca entre eles e o criterio de
escolha. Pareamento por episodio: teste de sinal exato nos pares discordantes.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  A0  PRE-CONDICAO: passivo no mundo lente entre 15% e 60%. Fora disso a
      comparacao nao tem folga.
  N   NULO: descoberta no mundo sorteado <= 1/30 em TODOS os bracos. Senao, algum
      criterio de escolha manufatura lei.

  R1  REPLICA: ativo - passivo >= 4/30 E teste de sinal unilateral p <= 0,05
      nos pares discordantes -> o efeito do n = 10 era real
  R1x se falhar: o A1 anterior era ruido de amostra pequena; o prior se mantem

  R2  E O COMITE: ativo - rico >= 3/30 -> o que paga e escolher pelo desacordo
      das proprias hipoteses, nao por riqueza da cena
  R2x se R1 passa e R2 falha: o ganho e de CURIOSIDADE POR RIQUEZA, uma
      heuristica sem modelo. Ainda e escolher o experimento, mas nao e o
      mecanismo que o prior do projeto negava.

SEMENTE PRE-REGISTRADA: 20260911 (nova; a 20260910 ja foi gasta)

MINHA CALIBRAGEM: sessenta e sete previsoes, VINTE E CINCO certas. A ultima
errada foi PESSIMISTA (previ empate, veio A1) — a primeira em muito tempo nessa
direcao.

PREVISAO: R1 passa por pouco ou falha por pouco — regressao a media e o esperado
depois de um resultado no limiar, entao aposto em ganho menor (+2 a +4 de 30) e
R1x. E R2x: o comite de 3 bootstraps discorda mais onde ha mais coisa na cena,
entao espero ativo ~ rico.
"""

from __future__ import annotations

import argparse
import random
import time
from math import comb

from novaphys import descricao as D
from novaphys.lente_ativa_limpa import (cena, _comite, _desacordo, _le_massa,
                                        N_INI, N_ESCOLHE, N_CAND, N_TESTE)

SEMENTE = 20260911
BRACOS = ("passivo", "ativo", "rico")


def _ocupadas(g):
    return sum(1 for lin in g for x in lin if x)


def _episodio(args):
    modo, braco, k, semente = args
    rng = random.Random(semente)
    rc = random.Random(semente + 1)
    rt = random.Random(semente + 2)
    pares = [cena(rng, modo) for _ in range(N_INI)]
    for _ in range(N_ESCOLHE):
        if braco == "passivo":
            pares.append(cena(rng, modo))
            continue
        cands = [cena(rng, modo) for _ in range(N_CAND)]
        if braco == "ativo":
            hips = _comite(pares, rc)
            pares.append(max(cands, key=lambda ab: _desacordo(hips, ab[0])))
        else:
            pares.append(max(cands, key=lambda ab: _ocupadas(ab[0])))
    te = [cena(rt, modo) for _ in range(N_TESTE)]
    try:
        fm, p, e = D.montar(pares, refoco=True, feixe=4)
        if not p or e != 0:
            return modo, braco, k, 0, 0, 0
        teste = all(D.aplicar((fm, p), x) == y for x, y in te)
        return modo, braco, k, 1, int(teste), int(teste and _le_massa(p))
    except Exception:
        return modo, braco, k, 0, 0, 0


def _sinal(x, y):
    """Teste de sinal exato unilateral: x ganha de y nos pares discordantes."""
    b = sum(1 for u, v in zip(x, y) if u > v)
    c = sum(1 for u, v in zip(x, y) if u < v)
    n = b + c
    if n == 0:
        return b, c, 1.0
    return b, c, sum(comb(n, i) for i in range(b, n + 1)) / 2 ** n


def _ini():
    D.usar_biblioteca(None)


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=3)
    ap.add_argument("--episodios", type=int, default=30)
    a = ap.parse_args()
    n = a.episodios

    print("=" * 76)
    print("LENTE ATIVA — REPLICA com controle de riqueza")
    print("=" * 76)
    trab = [(m, b, k, SEMENTE + 101 * k + (0 if m == "lente" else 50000))
            for m in ("lente", "sorteado") for b in BRACOS for k in range(n)]
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_episodio, trab, chunksize=1)
    print(f"\n  ({time.time()-t0:.0f}s)\n")

    desc = {}
    print("  mundo      braco     fecha   acerta teste   DESCOBERTA")
    print("  --------   -------   -----   ------------   ----------")
    for m in ("lente", "sorteado"):
        for b in BRACOS:
            rs = sorted((r for r in res if r[0] == m and r[1] == b),
                        key=lambda r: r[2])
            desc[(m, b)] = [r[5] for r in rs]
            f, t, d = (sum(r[i] for r in rs) for i in (3, 4, 5))
            print(f"  {m:9s}  {b:8s}  {f:2d}/{n:<3d}  {t:5d}/{n:<6d}   {d:4d}/{n}")

    L = {b: desc[("lente", b)] for b in BRACOS}
    print("\n  pares discordantes no mundo lente (x ganha, y ganha, p unilateral):")
    for x, y in (("ativo", "passivo"), ("ativo", "rico"), ("rico", "passivo")):
        bx, cy, p = _sinal(L[x], L[y])
        print(f"    {x:7s} vs {y:7s}   {bx:2d}  {cy:2d}   p = {p:.3f}")

    print("")
    print("=" * 76)
    sp, sa, sr = (sum(L[b]) for b in BRACOS)
    if not (0.15 * n <= sp <= 0.60 * n):
        print(f"A0 . passivo deu {sp}/{n}, fora de 15-60%: comparacao sem folga.")
    nulo = max(sum(desc[("sorteado", b)]) for b in BRACOS)
    if nulo > 1:
        print(f"N  FALHOU . {nulo}/{n} 'descobertas' no mundo sem lei.")
    _, _, p_ap = _sinal(L["ativo"], L["passivo"])
    r1 = sa - sp >= 4 and p_ap <= 0.05
    if r1:
        print("R1 . replica: escolher o experimento paga com modelo APRENDIDO.")
    else:
        print("R1x . nao replicou: o A1 com n = 10 era ruido; o prior se mantem.")
    # conserto (2026-09-10, depois da rodada): a primeira versao testava R2 ANTES
    # de checar R1 e imprimiu "e o COMITE" sem R1 — contra o pre-registro, que diz
    # que sem R1 a comparacao e so descritiva. Os numeros nao mudam; so a frase.
    if not r1:
        print(f"R2 . sem R1, comite x riqueza e so descritivo (ativo-rico = {sa-sr}).")
    elif sa - sr >= 3:
        print("R2 . e o COMITE: o desacordo das hipoteses escolhe melhor que a riqueza.")
    else:
        print("R2x . o ganho e de curiosidade por RIQUEZA, sem modelo.")


if __name__ == "__main__":
    main()
