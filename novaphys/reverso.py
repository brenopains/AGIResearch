"""
novaphys.reverso — e se a seta estiver apontando para o lado difícil?
=====================================================================

O INSIGHT, E POR QUE NINGUEM VIU
----------------------------------
Todo o projeto e unidirecional: descreve `entrada -> saida`. Nove agentes
discutiram o que dizer, como buscar, como pontuar. Nenhum questionou a DIRECAO.

Mas o censo diz que 199 tarefas nao resolvidas exigem FORMAS NOVAS na saida.
Para essas, o mapa direto e um-para-muitos — dada a entrada, "desenhe algo que
nao estava la" e mal-posto por construcao, e nenhum pintor conserta isso.

O mapa INVERSO e destrutivo: dada a saida, apagar o que foi desenhado e
recuperar a entrada e uma funcao. E funcao destrutiva e DEDUZIVEL — e o mesmo
motivo pelo qual `eter_inferencia.py` teve 100% de precisao deduzindo da
diferenca em vez de sortear.

"Nada no projeto desenha" foi tratado como falta de pintor. Pode ser escolha de
direcao que torna desenhar indeduzivel.

O EXPERIMENTO, DESCRITIVO PRIMEIRO
------------------------------------
Rodar os MESMOS codecs sobre `(saida, entrada)` e comparar contra `(entrada,
saida)`. Sem mudar arquitetura, sem escrever pintor novo.

    razao direta   bits(a->b) / literal(b)
    razao reversa  bits(b->a) / literal(a)

E o controle pareado esta embutido no proprio dado, o que e raro:

    tarefas com FORMAS NOVAS   -> assimetria esperada, reverso mais barato
    tarefas com FORMAS IGUAIS  -> deve ser SIMETRICO

Se o reverso vencer nas DUAS, e vies generico ou bug, nao insight.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  V0  PRE-CONDICAO. Nas tarefas de dims iguais e formas iguais, a diferenca
      media |razao_direta - razao_reversa| tem que ser pequena (< 0,10). Se nao
      for, ha assimetria generica e o resto da medida nao significa nada.

  V1  ASSIMETRIA. Nas tarefas com formas novas, razao_reversa < razao_direta
      com margem > 0,10 -> a direcao dificil e a que escolhemos, e ha um
      atalho: aprender o inverso e usa-lo para ordenar o feixe direto.
  V2  empate -> a direcao nao importa; desenhar e dificil nos dois sentidos.
  V3  reverso PIOR -> o insight esta invertido.

MINHA CALIBRAGEM: cinquenta e oito previsoes, VINTE E DUAS certas, e hoje uma
afirmacao central minha foi retirada por tautologia e a regua de bits estava
torta em 64% dos pares.

PREVISAO: V1 fraco. Aposto que a assimetria EXISTE mas e menor que 0,10 na
media, porque muitas "formas novas" nao sao desenho puro — sao copia de algo
que ja esta na entrada noutro lugar, e nesse caso as duas direcoes sao
igualmente deduziveis. Se V1 der forte, e a melhor noticia do dia.
"""

from __future__ import annotations

import argparse
import time

from novaphys.arc import carregar, dims
from novaphys.compositor import fundo
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D

TAR = None


def _ini():
    global TAR
    TAR = carregar(PASTA_AVAL)
    D.usar_biblioteca(None)


def _formas_novas(tr):
    """A saida contem forma que nao esta na entrada?"""
    for a, b in tr:
        ha, wa = dims(a)
        hb, wb = dims(b)
        ca = {a[i][j] for i in range(ha) for j in range(wa)}
        cb = {b[i][j] for i in range(hb) for j in range(wb)}
        if cb - ca:
            return True
        if (ha, wa) == (hb, wb):
            novas = sum(1 for i in range(ha) for j in range(wa)
                        if a[i][j] == fundo(a) and b[i][j] != fundo(a))
            if novas > 0.05 * ha * wa:
                return True
    return False


def _uma(ti):
    n, tr, te = TAR[ti]
    try:
        dir_ = [(a, b) for a, b in tr]
        rev = [(b, a) for a, b in tr]
        rd = rr = 1.0
        fm, p, _e = D.montar(dir_, refoco=True, feixe=2, criterio="compressao")
        if p:
            rd = D.razao(p, [(D.forma.base(fm, a), b) for a, b in dir_])
        fm2, p2, _e2 = D.montar(rev, refoco=True, feixe=2,
                                criterio="compressao")
        if p2:
            rr = D.razao(p2, [(D.forma.base(fm2, a), b) for a, b in rev])
        return n, _formas_novas(tr), rd, rr
    except Exception:
        return n, False, 1.0, 1.0


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=4)
    ap.add_argument("--amostra", type=int, default=200)
    a = ap.parse_args()
    _ini()

    print("=" * 76)
    print("REVERSO — a seta esta apontando para o lado dificil?")
    print("=" * 76)
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_uma, range(min(a.amostra, len(TAR))), chunksize=2)

    nov = [(n, d, r) for n, f, d, r in res if f]
    igu = [(n, d, r) for n, f, d, r in res if not f]
    print(f"\n  ({time.time()-t0:.0f}s)   amostra {len(res)}\n")
    print("  grupo          n     razao direta   razao reversa   assimetria")
    print("  -----------   ---    ------------   -------------   ----------")
    for nome, gr in (("formas novas", nov), ("formas iguais", igu)):
        if not gr:
            continue
        md = sum(d for _n, d, _r in gr) / len(gr)
        mr = sum(r for _n, _d, r in gr) / len(gr)
        print(f"  {nome:13s} {len(gr):3d}    {md:12.3f}   {mr:13.3f}   "
              f"{md-mr:+10.3f}")

    an = (sum(d for _n, d, _r in nov) / max(len(nov), 1)
          - sum(r for _n, _d, r in nov) / max(len(nov), 1))
    ai = (sum(d for _n, d, _r in igu) / max(len(igu), 1)
          - sum(r for _n, _d, r in igu) / max(len(igu), 1))
    print(f"\n  V0 . assimetria nas formas IGUAIS: {ai:+.3f} "
          f"(tem que ser pequena)")
    print(f"  V1 . assimetria nas formas NOVAS : {an:+.3f}")

    print("")
    print("=" * 76)
    if abs(ai) > 0.10:
        print("V0 FALHOU: ha assimetria generica; a medida nao isola direcao.")
    elif an > 0.10:
        print("V1 . A DIRECAO DIFICIL E A QUE ESCOLHEMOS. Aprender o inverso e")
        print("     usa-lo para ordenar o feixe direto e um atalho real.")
    elif an < -0.10:
        print("V3 . o insight esta invertido.")
    else:
        print("V2 . a direcao nao importa; desenhar e dificil nos dois lados.")


if __name__ == "__main__":
    main()
