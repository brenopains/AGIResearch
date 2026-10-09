"""
novaphys.lente_ativa — um mundo que PERMITE a descoberta: o agente escolhe o experimento
==========================================================================================

A PECA QUE NINGUEM TOCOU
--------------------------
Onze agentes com vieses diferentes produziram ~35 propostas para este projeto.
NENHUMA tocou agencia. Todas sao sobre induzir melhor uma regra a partir de pares
DADOS. E agencia e uma das cinco ausencias categoricas do sistema (junto com
objetivo, planejamento, linguagem e quantidade continua).

A frase do usuario foi "um mundo que PERMITA a descoberta". E o paper
"SciLaws-Parallel" (arXiv 2609.01552) tem isso como nucleo do desenho: o agente
comeca com ZERO observacoes e escolhe as proprias consultas ao simulador.

O EXPERIMENTO
---------------
O mesmo mundo do `lente.py`: particulas caem reto (Newton), exceto ao lado de uma
celula de MASSA, onde desviam em direcao a ela (a lente — a previsao mais famosa da
relatividade geral).

Orcamento fixo de observacoes. Dois bracos, orcamento CASADO:

    PASSIVO   as cenas sao sorteadas — o que o `lente.py` ja faz
    ATIVO     depois de 3 cenas sorteadas, cada nova cena e ESCOLHIDA entre 12
              candidatas: a que mais divide um COMITE de hipoteses

O comite e por bootstrap: `montar` roda em 3 reamostragens dos pares ja vistos, e
cada uma vira uma hipotese. A cena escolhida e a de maior DESACORDO entre elas —
consulta por comite (Seung, Opper & Sompolinsky 1992), a forma classica e mais
simples de desenho de experimento.

E ISTO NAO CONTRABANDEIA CONHECIMENTO
---------------------------------------
O braco ativo nao sabe que a massa importa. Ele so sabe onde as proprias hipoteses
DISCORDAM. Se a massa for a causa da anomalia, as hipoteses vao discordar perto
dela, e as consultas vao se concentrar la sem ninguem ter dito. Se nao for, nao.

PRIOR MEDIDO NESTE PROJETO, E ELE E DURO
------------------------------------------
A memoria do projeto registra: "desenho de experimento da 30-54x SO com modelo de
mundo DADO — com modelo APRENDIDO, vantagem zero." Aqui o modelo e aprendido. Esse
prior diz que o braco ativo deve empatar.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  A0  PRE-CONDICAO. O braco passivo tem que reproduzir a taxa do `lente.py` na mesma
      semente e orcamento. Se nao, a comparacao nao vale.

  A1  ATIVO > PASSIVO em descoberta (fecha treino + acerta teste + le a massa), no
      mundo com lei -> escolher o experimento acelera descoberta com modelo
      APRENDIDO, contradizendo o prior do projeto
  A2  empate -> o prior se mantem: desenho de experimento so paga com modelo dado
  A3  ativo "descobre" no mundo SORTEADO -> consultas concentradas manufaturam lei
      falsa; e o pior desfecho e o que o nulo existe para pegar

SEMENTE PRE-REGISTRADA: 20260910

MINHA CALIBRAGEM: sessenta e duas previsoes, VINTE E QUATRO certas, e o prior do
projeto sobre este tema exato e negativo.

PREVISAO: A2, empate. Apostar contra um prior medido do proprio projeto exigiria
motivo, e eu nao tenho: o comite por bootstrap de poucas amostras tende a discordar
em ruido, nao em estrutura. Se der A1, e a primeira evidencia de que agencia paga
aqui — e o prior "vantagem zero com modelo aprendido" precisa ser refinado.
"""

from __future__ import annotations

import argparse
import random
import time

from novaphys.arc import dims
from novaphys import descricao as D
from novaphys import lente as LE

SEMENTE = 20260910
N_INI, N_ATIVO, N_CAND, N_COMITE = 3, 4, 12, 3
TAR = None


def _desacordo(hips, a, rng):
    """Quantas celulas as hipoteses do comite preveem diferente para a cena `a`."""
    preds = []
    for fm, pilha in hips:
        try:
            p = D.aplicar((fm, pilha), a)
        except Exception:
            p = None
        preds.append(p)
    preds = [p for p in preds if p is not None]
    if len(preds) < 2:
        return 0
    h, w = dims(a)
    return sum(1 for i in range(h) for j in range(w)
               if len({p[i][j] for p in preds}) > 1)


def _comite(pares, rng):
    hips = []
    for _ in range(N_COMITE):
        amostra = [rng.choice(pares) for _ in range(len(pares))]
        try:
            fm, pilha, _e = D.montar(amostra, refoco=True, feixe=2)
            if pilha:
                hips.append((fm, pilha))
        except Exception:
            pass
    return hips


def _episodio(args):
    modo, braco, semente = args
    rng = random.Random(semente)
    pares = []
    while len(pares) < N_INI:
        a = LE._cena(rng)
        b = LE._passo(a, modo, rng)
        if a != b:
            pares.append((a, b))
    for _ in range(N_ATIVO):
        if braco == "passivo":
            while True:
                a = LE._cena(rng)
                b = LE._passo(a, modo, rng)
                if a != b:
                    break
        else:
            hips = _comite(pares, rng)
            cands = [LE._cena(rng) for _ in range(N_CAND)]
            a = max(cands, key=lambda c: _desacordo(hips, c, rng))
            b = LE._passo(a, modo, rng)
        pares.append((a, b))
    te = []
    while len(te) < 2:
        a = LE._cena(rng)
        b = LE._passo(a, modo, rng)
        if a != b:
            te.append((a, b))
    try:
        fm, pilha, err = D.montar(pares, refoco=True, feixe=4)
        if not pilha or err != 0:
            return modo, braco, 0, 0, 0
        teste = all(D.aplicar((fm, pilha), x) == y for x, y in te)
        le = LE._le_massa(pilha)
        return modo, braco, 1, 1 if teste else 0, 1 if (teste and le) else 0
    except Exception:
        return modo, braco, 0, 0, 0


def _ini():
    D.usar_biblioteca(None)


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=3)
    ap.add_argument("--episodios", type=int, default=8)
    a = ap.parse_args()

    print("=" * 76)
    print("LENTE ATIVA — o agente escolhe o experimento")
    print("=" * 76)
    print(f"  orcamento {N_INI + N_ATIVO} observacoes por episodio; comite de "
          f"{N_COMITE}; {N_CAND} cenas candidatas por consulta")
    trab = [(m, b, SEMENTE + 1000 * k + i)
            for i, m in enumerate(("lente", "sorteado"))
            for b in ("passivo", "ativo")
            for k in range(a.episodios)]
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_episodio, trab, chunksize=1)

    print(f"\n  ({time.time()-t0:.0f}s)\n")
    print("  mundo      braco     fecha   acerta teste   DESCOBERTA")
    print("  --------   -------   -----   ------------   ----------")
    tab = {}
    for m in ("lente", "sorteado"):
        for b in ("passivo", "ativo"):
            rs = [r for r in res if r[0] == m and r[1] == b]
            n = len(rs)
            f, t, d = (sum(r[k] for r in rs) for k in (2, 3, 4))
            tab[(m, b)] = d
            print(f"  {m:9s}  {b:8s}  {f:2d}/{n:<3d}  {t:5d}/{n:<6d}   {d:4d}/{n}")

    print("")
    print("=" * 76)
    la, lp = tab[("lente", "ativo")], tab[("lente", "passivo")]
    sa = tab[("sorteado", "ativo")]
    if sa > tab[("sorteado", "passivo")]:
        print("A3 . consultas concentradas MANUFATURAM lei no mundo sem lei.")
    if la > lp:
        print("A1 . escolher o experimento acelera descoberta com modelo")
        print("     APRENDIDO — contradiz o prior do projeto.")
    elif la == lp:
        print("A2 . empate: desenho de experimento so paga com modelo dado.")
    else:
        print("ativo PIOR que passivo: o comite escolhe cenas enganosas.")


if __name__ == "__main__":
    main()
