"""
novaphys.cego — o censo do inalcancavel
========================================

O QUE ISTO MECANIZA
---------------------
O unico papel meta que pagou nesta sessao foi o LLM detectando PONTO CEGO
ARQUITETURAL. Tres vezes, e nas tres a mesma forma:

    `fluido.py`   0/400 por uma guarda que matava a familia antes de verificar
    `raio_l`      zero com as fontes batendo 12/12, por misturar regiao e cor
    `ENTAO`       zero pilhas porque a BUSCA nunca podia escolhe-lo

Nenhuma foi ideia nova. Todas foram "existe uma capacidade e ela esta
inalcancavel". A terceira valeu +3 e dobrou a profundidade das pilhas.

E as tres dependeram de eu NOTAR. Isso e frágil e nao escala. Mas metade do
trabalho e mecanica:

    DISPONIVEL   a capacidade foi proposta em quantas tarefas
    ESCOLHIDA    apareceu em quantas pilhas vencedoras

Capacidade disponivel em centenas de tarefas e escolhida em ZERO e candidata a
ponto cego. Isso e computavel e barato, e teria pego o `ENTAO` ANTES de eu
queimar um run inteiro medindo a busca em vez do mecanismo.

O QUE CONTINUA PRECISANDO DE RACIOCINIO
-----------------------------------------
A lista nao diz POR QUE. Uma capacidade nunca escolhida pode ser:

    INUTIL        o dominio nao precisa dela; remover economiza busca
    INALCANCAVEL  o dominio precisa e a selecao nao chega la
    QUEBRADA      um erro de implementacao a mata antes da verificacao

As tres tem a mesma assinatura numerica e consequencias opostas. Distinguir
exige olhar o mecanismo — e e exatamente ai que o LLM entra, num papel estreito
e com historico positivo, em vez de guiar busca (medido zero, 16 vezes) ou
gerar mecanismo (medido negativo, 0 em 18).

E a diferenca que importa: o sistema passa a produzir o proprio diagnostico, e
o raciocinio e chamado so onde ele e insubstituivel.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from novaphys.arc import carregar, dims
from novaphys.compositor import fundo
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D
from novaphys import forma as F

TAR = None


def _ini():
    global TAR
    TAR = carregar(PASTA_AVAL)
    D.usar_biblioteca(None)


def _rotulos_disponiveis(tr):
    """As capacidades que CHEGARAM a ser propostas nesta tarefa."""
    rot = set()
    fms = F.propor(tr)
    for fm in fms:
        rot.add(f"forma:{fm[0]}")
    if not fms:
        return rot
    reb = []
    for a, b in tr:
        g = F.base(fms[0], a)
        if g is None or dims(g) != dims(b):
            return rot
        reb.append((g, b))
    for cd in D.CODECS:
        try:
            ms = list(cd.propor(reb, None))
        except Exception:
            ms = []
        if ms:
            rot.add(f"codec:{cd.nome}")
            if cd.nome == "programa":
                for sel, pin in ms:
                    rot.add(f"pintor:{pin[0]}")
                    for c in sel:
                        rot.add(f"teste:{c[0]}")
    T = D.CondicaoAprendida._testes(reb[0][0], fundo(reb[0][0]))
    for c in T:
        rot.add(f"teste:{c[0]}")
    for m in D.MARCAS:
        rot.add(f"marca:{m.split(':')[0]}")
    return rot


def _uma(ti):
    n, tr, te = TAR[ti]
    try:
        disp = _rotulos_disponiveis(tr)
        fm, pilha, err = D.montar(tr, refoco=True, feixe=4)
        esc = set()
        if err == 0 and pilha:
            esc.add(f"forma:{fm[0]}")
            for x in pilha:
                if D._marca(x):
                    esc.add(f"marca:{x.split(':')[0]}")
                else:
                    esc.add(f"codec:{x[0].nome}")
                    if x[0].nome == "programa":
                        sel, pin = x[1]
                        esc.add(f"pintor:{pin[0]}")
                        for c in sel:
                            esc.add(f"teste:{c[0]}")
        return disp, esc
    except Exception:
        return set(), set()


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=2)
    ap.add_argument("--amostra", type=int, default=140)
    a = ap.parse_args()
    _ini()

    print("=" * 76)
    print("CEGO — capacidades disponiveis que nunca sao escolhidas")
    print("=" * 76)
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_uma, range(min(a.amostra, len(TAR))), chunksize=2)

    disp, esc = Counter(), Counter()
    for d, e in res:
        disp.update(d)
        esc.update(e)
    print(f"\n  amostra de {len(res)} tarefas ({time.time()-t0:.0f}s)\n")
    print("  disp  esc   capacidade")
    print("  ----  ---   ----------")
    mortos = []
    for r, nd in disp.most_common():
        ne = esc.get(r, 0)
        if nd >= 20 and ne == 0:
            mortos.append((nd, r))
    for r, nd in disp.most_common(60):
        ne = esc.get(r, 0)
        marca = "  <<< NUNCA" if nd >= 20 and ne == 0 else ""
        print(f"  {nd:4d}  {ne:3d}   {r}{marca}")

    print("")
    print("=" * 76)
    print(f"PONTOS CEGOS CANDIDATOS: {len(mortos)}")
    for nd, r in sorted(mortos, reverse=True):
        print(f"   disponivel em {nd:3d} tarefas, escolhida ZERO vezes: {r}")
    print("")
    print("Cada uma e INUTIL, INALCANCAVEL ou QUEBRADA. As tres tem a mesma")
    print("assinatura numerica e consequencias opostas — e ai o raciocinio")
    print("entra, num papel estreito, em vez de guiar busca.")


if __name__ == "__main__":
    main()
