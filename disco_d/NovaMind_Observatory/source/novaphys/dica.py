"""
novaphys.dica — quantos BITS de dica convertem falha em acerto?
================================================================

A PERGUNTA QUE DECIDE A LINHA DO PROFESSOR
--------------------------------------------
O usuario argumenta que cognicao e andaimada socialmente: professor da dica
quando o aluno trava, e isso cria caminho. Nosso sistema nao tem professor,
nem dica, nem feedback alem de certo/errado.

O argumento e bom e tem duas leituras que fazem previsoes OPOSTAS:

    (a) a resposta esta no nosso espaco e o sistema nao consegue ESCOLHER
        -> dica converte falha em acerto; professor e curriculo mordem
    (b) a resposta NAO esta no nosso espaco
        -> nenhuma dica ajuda; dizer "use ladrilho" mil vezes nao torna a
           regra exprimivel

Este arquivo separa as duas com um numero.

O ORACULO, E O QUE ELE NAO E
------------------------------
Para cada tarefa, testam-se TODAS as 8 lentes x 10 propriedades = 80
combinacoes, e pergunta-se se ALGUMA produz uma regra que reproduz o treino E
acerta o teste.

Isso e o teto que um professor perfeito compraria: ele conhece a resposta e
aponta a combinacao certa.

    ISTO USA O TESTE PARA MEDIR. NAO E UM PLACAR.

O numero do oraculo nunca entra na uniao e nunca e reportado como resultado do
sistema. E diagnostico — mede a DISTANCIA entre o que o sistema acha e o que
ele poderia achar com orientacao perfeita. Confundir os dois seria o vazamento
que `propositor.py` cometeu e que custou uma conclusao inteira.

A CURVA DE BITS
-----------------
    0 bits    o que o sistema resolve sozinho, com portao imunologico
    3 bits    dica de LENTE  (log2(8) = 3) — "olhe por componente conexa",
              "olhe por coluna", "olhe por painel". O sistema ainda escolhe a
              propriedade sozinho.
    6,3 bits  dica de LENTE + PROPRIEDADE (log2(80)) — a forma da regra
              inteira e apontada; so os valores do mapa sao deduzidos.

A diferenca entre 0 e 6,3 bits e exatamente o que um professor pode comprar. O
que sobra ate 400 e o que professor nenhum compra.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  D0  PRE-CONDICAO. O oraculo tem que ser >= 4, senao ele nao reproduz nem o
      que o sistema ja faz sozinho e a medida esta quebrada.

  D1  oraculo >= 20
      -> as respostas ESTAO no espaco e o gargalo e SELECAO. Professor e
         curriculo tem onde morder, e a prioridade passa a ser escolher melhor.
  D2  oraculo 8 a 19
      -> parte esta no espaco; ganho moderado com orientacao
  D3  oraculo <= 7
      -> as respostas NAO estao no espaco. Nenhuma dica ajuda, e a linha do
         professor fecha: o problema e EXPRESSIVIDADE, nao direcao.

  D4  QUANTAS combinacoes funcionam por tarefa. Se for tipicamente 1, a dica e
      essencial e a selecao e agulha no palheiro. Se forem varias, o sistema
      erra a escolha tendo varias saidas certas — e ai o problema e o
      CRITERIO de escolha, nao a busca.

MINHA CALIBRAGEM: dezoito previsoes na sessao, SEIS certas, as doze erradas
todas otimistas. Aplico a correcao ao meu palpite.

PREVISAO: D2, na faixa baixa, 10 a 15. Raciocinio: 127 tarefas ajustam o treino
com ALGUMA combinacao, e o motor pega a primeira. Se ~10% delas tiverem uma
combinacao correta entre as 80, sao ~13. Espero tambem D4 apontando
tipicamente 1 combinacao por tarefa — o que significaria que o portao
imunologico nao esta rejeitando a certa, e sim que a certa quase nunca existe.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL
from novaphys.segmenta import LENTES, PROPS, aplicar, fundo_de
from novaphys.segmenta import deduzir as deduzir_livre


def deduzir_em(treino, lente, prop: Optional[str]):
    """
    Deduz restrito a uma propriedade (ou livre, se `prop` for None).
    Reimplementa o laco final de `segmenta.deduzir` com a propriedade fixada.
    """
    from novaphys.segmenta import deltas, props
    from novaphys.arc import dims
    obs: Dict = {}
    telas: Set[str] = set()
    alvo = prop
    for a, b in treino:
        bg = fundo_de(a)
        ci, co = lente(a, bg), lente(b, bg)
        if not ci.objs:
            return None
        ha, wa = dims(a)
        hb, wb = dims(b)
        telas.add("igual" if (hb, wb) == (ha, wa) else "corte")
        d = deltas(ci, co)
        if d is None:
            return None
        pi = props(ci)
        for oid, lst in d.items():
            obs.setdefault(pi[oid][alvo], set()).add(lst)
    if len(telas) != 1 or not obs:
        return None
    if any(len(v) > 1 for v in obs.values()):
        return None
    return (alvo, {k: next(iter(v)) for k, v in obs.items()},
            next(iter(telas)))


def _uma(args):
    nome, tr, te = args
    certas: List[Tuple[str, str]] = []
    lentes_ok: Set[str] = set()
    for ln, lente in LENTES:
        for p in PROPS:
            try:
                r = deduzir_em(tr, lente, p)
                if r is None:
                    continue
                if any(aplicar(r, a, lente) != b for a, b in tr):
                    continue
                if all(aplicar(r, x, lente) == y for x, y in te):
                    certas.append((ln, p))
                    lentes_ok.add(ln)
            except Exception:
                continue
    return nome, certas, sorted(lentes_ok)


def _tres_bits(args):
    """
    Dica de LENTE apenas: a lente certa e apontada, e o sistema escolhe a
    propriedade sozinho, do jeito que ja escolhe (primeira consistente).
    """
    nome, tr, te, lentes_ok = args
    for ln, lente in LENTES:
        if ln not in lentes_ok:
            continue
        try:
            r = deduzir_livre(tr, lente)
            if r is None:
                continue
            if any(aplicar(r, a, lente) != b for a, b in tr):
                continue
            if all(aplicar(r, x, lente) == y for x, y in te):
                return nome, True
        except Exception:
            continue
    return nome, False


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=6)
    a = ap.parse_args()

    print("=" * 80)
    print("DICA — quantos bits de orientacao convertem falha em acerto?")
    print("=" * 80)
    print("\nATENCAO: o oraculo USA O TESTE para medir. NAO e placar, e")
    print("diagnostico. Nunca entra na uniao.\n")
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=4)
    oraculo = {n: c for n, c, _l in res if c}
    lentes = {n: l for n, c, l in res if c}
    print(f"  ORACULO (6,3 bits: lente + propriedade): {len(oraculo):3d}/400 "
          f"({time.time()-t0:.0f}s)")

    with Pool(a.nucleos) as pool:
        r3 = pool.map(_tres_bits, [(n, tr, te, lentes.get(n, []))
                                   for n, tr, te in tarefas], chunksize=4)
    tres = {n for n, ok in r3 if ok}
    print(f"  DICA de 3 bits (so a lente)            : {len(tres):3d}/400")
    print(f"  ZERO bits (portao imunologico, `ockham`):   4/400")

    cont = Counter(len(c) for c in oraculo.values())
    print(f"\nD4 · combinacoes CERTAS por tarefa (das 80):")
    for k in sorted(cont):
        print(f"     {k:2d} combinacao(oes): {cont[k]:3d} tarefas")
    uma_so = cont.get(1, 0)

    lc = Counter(l for c in oraculo.values() for l, _p in c)
    pc = Counter(p for c in oraculo.values() for _l, p in c)
    print(f"\n  lentes que aparecem: {dict(lc.most_common())}")
    print(f"  propriedades       : {dict(pc.most_common())}")
    print(f"\n  ganho de 0 -> 3 bits  : {len(tres) - 4:+d}")
    print(f"  ganho de 3 -> 6,3 bits: {len(oraculo) - len(tres):+d}")
    print(f"  inalcancavel mesmo com dica: {400 - len(oraculo)}")

    print("\n" + "=" * 80)
    if len(oraculo) < 4:
        print("D0 FALHOU: o oraculo nao reproduz nem o que o sistema faz.")
        return
    if len(oraculo) >= 20:
        print("D1 · as respostas ESTAO no espaco e o gargalo e SELECAO.")
        print("     Professor e curriculo tem onde morder.")
    elif len(oraculo) >= 8:
        print("D2 · parte esta no espaco. Orientacao compra ganho moderado.")
    else:
        print("D3 · as respostas NAO estao no espaco. Nenhuma dica ajuda —")
        print("     o problema e EXPRESSIVIDADE, nao direcao, e a linha do")
        print("     professor fecha.")
    if uma_so > len(oraculo) * 0.6:
        print(f"\nD4 · {uma_so} das {len(oraculo)} tem UMA UNICA combinacao")
        print("     certa entre 80. A selecao e agulha no palheiro, e o")
        print("     portao imunologico nao esta rejeitando a certa — ela")
        print("     quase nunca existe.")
    else:
        print(f"\nD4 · a maioria tem VARIAS combinacoes certas. O sistema erra")
        print("     a escolha tendo saidas corretas disponiveis: o problema e")
        print("     o CRITERIO de escolha, nao a busca.")


if __name__ == "__main__":
    main()
