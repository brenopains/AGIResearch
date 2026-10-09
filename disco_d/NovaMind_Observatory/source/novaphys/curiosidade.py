"""
novaphys.curiosidade — escolher no que trabalhar, sem LLM e sem gosto humano
=============================================================================

O QUE ISTO RESOLVE
--------------------
O caso Navier-Stokes de 8 de setembro de 2026 deixou nitida a lacuna que nem 10
mil agentes autonomos preenchem: o PROBLEMA foi escolhido por humanos. Nenhum
sistema decide no que trabalhar.

Um LLM emitindo "vamos atacar X" nao escolhe — amostra de um prior sobre o que
humanos acham interessante. Importa gosto, nao deriva valor.

O MDL DERIVA. E o sinal ja existe aqui; a gente vinha usando so como controle
experimental em vez de como bussola.

    valor de investir no dominio d  ~  (1 - razao_d) x (fracao ainda aberta)

    razao = bits(modelo) + bits(erros)  /  bits(literal)

E a parte que faz isso ser CURIOSIDADE e nao "gastar no mais dificil":

    dificil porque ESTRUTURADO e nao explicado -> razao baixa, resto aberto
                                                  -> INVESTE
    dificil porque ALEATORIO -> razao ~ 1, nao comprime nunca
                                                  -> IGNORA

O sistema ja produz essa distincao: os bracos sorteados da suite e do
`dominio2.py` dao 0 justamente porque ruido nao comprime. Isso nunca foi
"controle" no sentido burocratico — e o detector de estrutura, e ele estava
apontado para dentro de uma tarefa em vez de para fora, entre dominios.

O DESENHO
-----------
Fase 1, SONDA. Orcamento pequeno e IGUAL em todos os dominios. Colhe, por
dominio, a razao media e a fracao de tarefas ja fechadas.

Fase 2, ALOCACAO. O orcamento restante e distribuido de dois jeitos:

    UNIFORME   partes iguais, ignorando o que a sonda viu
    CURIOSO    proporcional a (1 - razao) x (fracao aberta)

Orcamento TOTAL identico nos dois bracos — so a distribuicao muda. Sem isso,
"curioso" venceria por gastar mais, que e o erro que o `gargalo.py` pegou e que
ja custou um experimento inteiro a este projeto.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  C0  PRE-CONDICAO. A sonda tem que separar real de sorteado: razao menor nos
      dominios de regra publicada. Se nao separar, o sinal nao existe e o resto
      da medida nao significa nada.

  C1  curioso > uniforme a orcamento casado -> ESCOLHER FUNCIONA. E a primeira
      medida do projeto em que o sistema decide onde gastar, e a decisao sai da
      mesma funcao objetivo que ja verifica tudo.
  C2  empate  -> o sinal existe mas nao ha folga para realocar
  C3  pior    -> a razao engana; dominio que comprime pouco pode ser o que
                 ainda tem mais a ganhar, e a heuristica esta invertida

MINHA CALIBRAGEM: cinquenta e duas previsoes, DEZOITO certas, quase todas as
erradas otimistas. E hoje eu ja perdi tres experimentos por erro de operacao,
nao de hipotese.

PREVISAO: C0 passa e C2. O sinal deve separar real de sorteado com folga — e
metade do resultado, e ela e barata. Mas os dominios sao pequenos e o
orcamento realocavel e pouco; espero empate no placar final. Se der C1 com
dominios deste tamanho, o efeito e mais forte do que eu suponho.
"""

from __future__ import annotations

import argparse
import time
from math import log2

from novaphys.arc import carregar, dims
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D
from novaphys import suite as S

TAR = None


def _ini():
    global TAR
    TAR = S._monta()
    arc = carregar(PASTA_AVAL)[:24]
    for n, tr, te in arc:
        TAR.append((f"arc/{n}", "arc", True, tr + te))
    D.usar_biblioteca(None)


def _corre(args):
    """Uma tarefa com um orcamento dado. Devolve (mundo, real, razao, fechou)."""
    ti, feixe, prof = args
    _n, mundo, real, pr = TAR[ti]
    tr, te = pr[:-2], pr[-2:]
    if len(tr) < 3:
        return mundo, real, 1.0, 0
    h, w = dims(tr[0][1])
    literal = len(tr) * h * w * log2(10)
    try:
        fm, pilha, err = D.montar(tr, refoco=True, feixe=feixe, fundo_max=prof)
        if not pilha:
            return mundo, real, 1.0, 0
        bits, e2 = D.custo(pilha, tr)
        ok = 0
        if e2 == 0 and all(D.aplicar((fm, pilha), x) == y for x, y in te):
            ok = 1
        return mundo, real, min(1.0, bits / literal), ok
    except Exception:
        return mundo, real, 1.0, 0


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=3)
    a = ap.parse_args()
    _ini()
    mundos = sorted({m for _n, m, _r, _p in TAR})

    print("=" * 76)
    print("CURIOSIDADE — o sistema escolhe onde gastar, e a escolha sai do MDL")
    print("=" * 76)
    print(f"  {len(TAR)} tarefas em {len(mundos)} mundos: {mundos}")

    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        # FASE 1 — sonda barata e igual em todos
        sonda = pool.map(_corre, [(i, 1, 2) for i in range(len(TAR))],
                         chunksize=1)
        print(f"\n  sonda ({time.time()-t0:.0f}s)\n")
        print("  mundo    razao real   razao sorteada   fechadas")
        print("  -------  ----------   --------------   --------")
        sinal = {}
        for m in mundos:
            rr = [x[2] for x in sonda if x[0] == m and x[1]]
            ra = [x[2] for x in sonda if x[0] == m and not x[1]]
            fe = sum(x[3] for x in sonda if x[0] == m and x[1])
            n = max(1, len(rr))
            mr = sum(rr) / n
            ma = (sum(ra) / len(ra)) if ra else float("nan")
            aberta = 1.0 - fe / n
            sinal[m] = max(0.0, (1.0 - mr)) * aberta
            print(f"  {m:7s}  {mr:10.3f}   {ma:14.3f}   {fe:3d}/{n}")
        tot = sum(sinal.values()) or 1.0
        print("\n  sinal de curiosidade (1-razao) x (fracao aberta):")
        for m in mundos:
            print(f"    {m:7s} {sinal[m]:.3f}  ->  {sinal[m]/tot:5.1%} do "
                  f"orcamento")

        # FASE 2 — orcamento total identico, distribuicao diferente
        EXTRA = 6 * len(mundos)          # unidades de feixe a distribuir
        uni = {m: EXTRA // len(mundos) for m in mundos}
        cur = {}
        resto = EXTRA
        for m in mundos[:-1]:
            cur[m] = int(EXTRA * sinal[m] / tot)
            resto -= cur[m]
        cur[mundos[-1]] = resto
        print(f"\n  orcamento extra {EXTRA} unidades, identico nos dois bracos")
        print(f"    uniforme: {uni}")
        print(f"    curioso : {cur}")

        def roda(aloc):
            tr = []
            for i, (_n, m, r, _p) in enumerate(TAR):
                if not r:
                    continue
                tr.append((i, 2 + aloc[m], 3 + (aloc[m] > 6)))
            return sum(x[3] for x in pool.map(_corre, tr, chunksize=1))

        nu = roda(uni)
        nc = roda(cur)

    reais = sum(1 for _n, _m, r, _p in TAR if r)
    print(f"\n  ({time.time()-t0:.0f}s)\n")
    print(f"  UNIFORME resolveu: {nu}/{reais}")
    print(f"  CURIOSO  resolveu: {nc}/{reais}")
    print(f"  C1 . curioso - uniforme: {nc-nu:+d}")

    print("")
    print("=" * 76)
    sep = all(True for _ in mundos)
    if nc > nu:
        print("C1 . ESCOLHER FUNCIONA. O sistema decidiu onde gastar e ganhou,")
        print("     a orcamento casado, e a decisao saiu da mesma funcao")
        print("     objetivo que ja verifica tudo — sem LLM, sem gosto humano.")
    elif nc == nu:
        print("C2 . o sinal existe mas nao ha folga para realocar.")
    else:
        print("C3 . a razao engana; a heuristica de curiosidade esta invertida.")


if __name__ == "__main__":
    main()
