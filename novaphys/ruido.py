"""
novaphys.ruido — o criterio de sucesso para dado que nao e de quebra-cabeca
============================================================================

O PROBLEMA QUE ISTO RESOLVE, E QUE EU QUASE NAO VI
----------------------------------------------------
A arquitetura inteira gateia em `residuo == 0`: a pilha so conta se reproduz o
treino EXATAMENTE. Isso e certo para o ARC, onde a saida e construida por um
humano e nao tem ruido.

Dado real nunca fecha em zero. Uma concha de molusco tem ruido de deposicao,
bordas de crescimento imprecisas, dano fisico. Um tecido tem celula morta. Uma
medida tem erro. Com o criterio atual, TODO dominio real da zero — e a
conclusao seria "a arquitetura nao casa com o mundo", quando o que nao casa e
o CRITERIO.

E o conserto nao exige mecanismo novo, porque o sistema JA calcula a quantidade
certa:

    literal      bits para codificar a saida celula a celula
    modelo       bits do modelo + bits das celulas que ele erra
    RAZAO        modelo / literal

Razao < 1 significa que a regra COMPRIME os dados. Isso e o criterio do MDL de
verdade — o que o `custo` ja devolve — e o `residuo == 0` e so o caso especial
em que a compressao e perfeita.

E crucialmente: uma regra que explica 95% de uma concha e paga poucos bits VENCE
a codificacao literal, mesmo sem nunca fechar em zero.

O QUE ESTE EXPERIMENTO MEDE
-----------------------------
Pega as regras PUBLICADAS de Wolfram (30, 90, 110, 184) sobre uma grade de uma
linha — o mundo `fita` da suite — e injeta ruido de p% nas saidas. Depois
pergunta:

    a que nivel de ruido a regra verdadeira deixa de comprimir?

Isso e pre-condicao para qualquer plano com dado real. Se a arquitetura perder
o sinal a 1% de ruido, nao ha concha, nao ha tecido, nao ha medida — e a
conversa acaba barata, que era o combinado.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  N0  PRE-CONDICAO. Com ruido ZERO, a razao tem que ser < 1 e o residuo 0. Se
      nao, o mundo `fita` esta fora de alcance e a medida nao existe.

  N1  a razao continua < 1 ate 10% de ruido -> a arquitetura suporta dado real,
      e o gargalo do mundo real nao e o ruido
  N2  degrada entre 1% e 5%  -> suporta dado limpo de laboratorio, nao campo
  N3  quebra em 1%           -> `residuo == 0` nao e um detalhe do criterio, e
                                uma dependencia estrutural, e todo dominio real
                                esta fora ate isso mudar

SEMENTE PRE-REGISTRADA: 20260909

MINHA CALIBRAGEM: cinquenta e duas previsoes, DEZOITO certas.

PREVISAO: N2. A razao deve degradar rapido porque o `propor` dos codecs exige
SOLIDEZ — um programa e rejeitado se pinta UMA celula errada. Com 1% de ruido,
quase todo programa bom passa a errar em algum lugar e some da geracao. Ou seja,
espero que o problema NAO esteja no criterio de sucesso (que este experimento
conserta) e sim na GERACAO, que continua binaria por dentro. Se for isso, o
conserto e o mesmo padrao de sempre: trocar exigencia de perfeicao por bits,
uma camada mais fundo.
"""

from __future__ import annotations

import argparse
import random
import time
from math import log2

from novaphys.arc import dims
from novaphys import descricao as D

SEMENTE = 20260909
LARG = 24
REGRAS = (30, 90, 110, 184)
RUIDOS = (0.0, 0.01, 0.03, 0.05, 0.10, 0.20)
TAR = None


def _passo1d(lin, regra, cor):
    w = len(lin)
    fora = []
    for j in range(w):
        e = 1 if (j > 0 and lin[j - 1]) else 0
        c = 1 if lin[j] else 0
        d = 1 if (j + 1 < w and lin[j + 1]) else 0
        fora.append(cor if (regra >> ((e << 2) | (c << 1) | d)) & 1 else 0)
    return tuple(fora)


def _monta():
    rng = random.Random(SEMENTE)
    T = []
    for r in REGRAS:
        for p in RUIDOS:
            pr = []
            for _ in range(8):
                a = (tuple(5 if rng.random() < 0.4 else 0
                           for _ in range(LARG)),)
                b = _passo1d(a[0], r, 5)
                if p:
                    b = tuple(0 if (rng.random() < p and c) else
                              (5 if (rng.random() < p and not c) else c)
                              for c in b)
                if a[0] != b:
                    pr.append((a, (b,)))
            if len(pr) >= 5:
                T.append((r, p, pr))
    return T


def _ini():
    global TAR
    TAR = _monta()
    D.usar_biblioteca(None)


def _uma(ti):
    r, p, pr = TAR[ti]
    tr = pr[:-2]
    h, w = dims(tr[0][0])
    literal = len(tr) * h * w * log2(10)
    try:
        _fm, pilha, err = D.montar(tr, refoco=True, feixe=4)
        if not pilha:
            return r, p, 1.0, err, 0
        # `custo` ja devolve bits do modelo + bits das celulas erradas
        bits, e2 = D.custo(pilha, tr)
        return r, p, bits / literal, e2, len(pilha)
    except Exception:
        return r, p, 1.0, -1, 0


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=2)
    a = ap.parse_args()
    _ini()

    print("=" * 76)
    print("RUIDO — a regra ainda COMPRIME quando o dado nao e limpo?")
    print("=" * 76)
    print(f"  regras publicadas de Wolfram {REGRAS}, grade 1x{LARG}")
    print(f"  criterio: bits(modelo) + bits(erros)  contra  bits(literal)")
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_uma, range(len(TAR)), chunksize=1)

    print(f"\n  ({time.time()-t0:.0f}s)\n")
    print("  ruido    razao media   comprime   residuo 0")
    print("  ------   -----------   --------   ---------")
    for p in RUIDOS:
        rs = [x for x in res if abs(x[1] - p) < 1e-9]
        if not rs:
            continue
        med = sum(x[2] for x in rs) / len(rs)
        comp = sum(1 for x in rs if x[2] < 1.0)
        zero = sum(1 for x in rs if x[3] == 0)
        print(f"  {p:5.0%}    {med:11.3f}   {comp:3d}/{len(rs):<4d}   "
              f"{zero:3d}/{len(rs)}")

    lim = None
    for p in RUIDOS:
        rs = [x for x in res if abs(x[1] - p) < 1e-9]
        if rs and sum(1 for x in rs if x[2] < 1.0) * 2 < len(rs):
            lim = p
            break
    print("")
    print("=" * 76)
    z0 = [x for x in res if x[1] == 0.0]
    if not z0 or sum(1 for x in z0 if x[3] == 0) == 0:
        print("N0 FALHOU: sem ruido a `fita` nao fecha; a medida nao existe.")
    elif lim is None or lim > 0.10:
        print("N1 . a arquitetura suporta dado real; ruido nao e o gargalo.")
    elif lim > 0.01:
        print(f"N2 . degrada em {lim:.0%}: dado limpo de laboratorio serve,")
        print("     dado de campo nao.")
    else:
        print("N3 . quebra em 1%. `residuo == 0` nao e detalhe de criterio, e")
        print("     dependencia estrutural — todo dominio real esta fora ate")
        print("     a GERACAO deixar de exigir perfeicao.")


if __name__ == "__main__":
    main()
