"""
novaphys.lente — descobrir o termo novo quando a teoria velha quase funciona
=============================================================================

A INTUICAO DO USUARIO
-----------------------
"Imagine dar a ela toda a fisica e informacao antes da relatividade geral e um
mundo que permita a possibilidade da descoberta."

E a forma certa de testar AGI e nao so ARC: nao induzir uma regra do zero, mas
REVISAR uma teoria que ja funciona quase sempre, a partir da anomalia que ela
nao consegue absorver. O periélio de Mercurio era um resíduo de 43 segundos de
arco por seculo numa teoria que acertava todo o resto.

O QUE A LITERATURA JA MEDIU
-----------------------------
"Can LLMs Discover Scientific Laws in Real and Parallel Worlds?" (arXiv
2609.01552, set/2026) constroi exatamente isso: lei publicada MAIS um termo novo
adicionado, com ruido calibrado. Tres achados, e os tres batem com medidas nossas:

  1. ajuste diverge de validade cientifica — 54,9% de concordancia
     (e o nosso "compressividade nao implica transferencia")
  2. recuperacao da lei COMPLETA fica em ~8% mesmo em modelos de fronteira:
     acham a forma publicada e falham no termo novo
  3. gargalo de SELECAO: pools maiores contem leis melhores e a auto-selecao
     captura so uma fracao do ganho (e a nossa cegueira do ganho imediato)

O MUNDO
---------
A previsao mais famosa da relatividade geral e a LUZ SE CURVAR PERTO DE MASSA.
Traduzido para o paradigma discreto:

    teoria velha   toda particula cai uma linha, reto       (Newton)
    lei verdadeira cai reto, EXCETO se ha uma celula de MASSA ao lado —
                   ai desvia na diagonal EM DIRECAO a ela   (lente)

A teoria velha o sistema JA TEM: foi o que resolveu o mundo `queda`, 6/6 em K=1
com `desl(N, ocupada)` e `vizinho`. Entao a pergunta nao e "induz queda", e
"quando queda quase funciona, o sistema acha o termo que falta?"

O CONTROLE PAREADO, QUE E O QUE DA SENTIDO A TUDO
---------------------------------------------------
Mundo gemeo com a MESMA frequencia de desvios diagonais, mas SORTEADOS — sem
relacao nenhuma com a massa. O residuo da teoria velha tem o mesmo tamanho nos
dois mundos. So muda se ele tem ESTRUTURA.

    lente     o residuo vive inteiro ao lado de massa    -> ha lei a descobrir
    sorteado  o residuo e espalhado sem causa             -> nao ha lei

Um sistema que "descobre uma lei" no mundo sorteado nao descobriu nada: esta
decorando ruido. E exatamente a divergencia ajuste-validade do paper, e o nulo e
o que a detecta.

A MEDIDA E ESTRUTURAL, NAO DE AJUSTE
--------------------------------------
Seguindo o paper: nao basta fechar o treino. Conta como descoberta so se
  (a) fecha o treino E acerta pares de TESTE com condicoes iniciais novas, e
  (b) a pilha vencedora REFERENCIA A MASSA — algum teste ou pintor que le a cor
      da massa. Uma pilha que acerta sem olhar para a massa achou atalho, nao lei.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  L0  PRE-CONDICAO. A teoria velha sozinha (pilha que so faz queda reta) tem que
      deixar residuo > 0 nos DOIS mundos. Senao nao ha anomalia e o teste e vazio.

  L1  DESCOBERTA. lente > sorteado em episodios com (a) e (b) satisfeitos.
          lente >= 50% e sorteado ~0  -> o sistema REVISA teoria a partir de
                                         anomalia estruturada
  L2  lente ~ sorteado -> o sistema ajusta qualquer residuo; e decoracao
  L3  lente ~ 0 -> o termo novo e inexprimivel no vocabulario atual — e ai a
                   pergunta vira QUAL primitivo falta, que e informacao

SEMENTE PRE-REGISTRADA: 20260910

MINHA CALIBRAGEM: cinquenta e oito previsoes, VINTE E DUAS certas.

PREVISAO: L1 parcial. `viz_cor` conta vizinhos de cor especifica e `desl` le a
celula ao lado — em principio "ha massa a esquerda" e dizivel. Mas o desvio e um
MOVIMENTO diagonal condicionado, e os nossos pintores de movimento (`prop`) sao
incondicionais. Aposto que o sistema fecha o treino na maioria dos episodios de
lente, com o criterio (b) — referenciar a massa — satisfeito em menos da metade.
E aposto que o sorteado fica perto de zero, porque o nulo tem dado zero em tudo
desde que o critério de compressao foi corrigido.
"""

from __future__ import annotations

import argparse
import random
import time

from novaphys.arc import dims
from novaphys import descricao as D

SEMENTE = 20260910
LADO = 10
PART, MASSA = 5, 8


def _passo(g, modo, rng):
    """Um passo do mundo. `modo` = 'lente' (desvio causado pela massa) ou
    'sorteado' (desvio com a mesma frequencia, sem causa)."""
    h, w = dims(g)
    m = [[c if c == MASSA else 0 for c in lin] for lin in g]
    parts = [(i, j) for i in range(h) for j in range(w) if g[i][j] == PART]
    # quantos desviariam no mundo lente — o sorteado usa a mesma quantidade
    for i, j in sorted(parts, key=lambda p: -p[0]):
        dx = 0
        if modo == "lente":
            if j > 0 and g[i][j - 1] == MASSA:
                dx = -1
            elif j < w - 1 and g[i][j + 1] == MASSA:
                dx = 1
        else:
            if rng.random() < 0.18:
                dx = rng.choice((-1, 1))
        y, x = i + 1, j + dx
        if 0 <= y < h and 0 <= x < w and m[y][x] == 0:
            m[y][x] = PART
        else:
            m[i][j] = PART
    return tuple(tuple(l) for l in m)


def _cena(rng):
    g = [[0] * LADO for _ in range(LADO)]
    for _ in range(rng.randint(2, 4)):
        g[rng.randint(3, LADO - 2)][rng.randint(1, LADO - 2)] = MASSA
    for _ in range(rng.randint(5, 9)):
        i, j = rng.randint(0, LADO - 4), rng.randint(0, LADO - 1)
        if g[i][j] == 0:
            g[i][j] = PART
    return tuple(tuple(l) for l in g)


def _episodios(modo, n, rng):
    fora = []
    for _ in range(n):
        pr = []
        while len(pr) < 7:
            a = _cena(rng)
            b = _passo(a, modo, rng)
            if a != b:
                pr.append((a, b))
        fora.append(pr)
    return fora


def _le_massa(pilha) -> bool:
    """A pilha vencedora referencia a massa em algum teste ou pintor?"""
    for x in pilha:
        if D._marca(x):
            continue
        r = repr(x[1])
        if str(MASSA) in r:
            return True
    return False


TAR = None


def _ini():
    global TAR
    rng = random.Random(SEMENTE)
    TAR = ([("lente", p) for p in _episodios("lente", 12, rng)]
           + [("sorteado", p) for p in _episodios("sorteado", 12, rng)])
    D.usar_biblioteca(None)


def _uma(ti):
    modo, pr = TAR[ti]
    tr, te = pr[:5], pr[5:]
    try:
        fm, pilha, err = D.montar(tr, refoco=True, feixe=4)
        if not pilha:
            return modo, 0, 0, 0, 0
        fecha = 1 if err == 0 else 0
        teste = 1 if (fecha and all(D.aplicar((fm, pilha), x) == y
                                    for x, y in te)) else 0
        le = 1 if _le_massa(pilha) else 0
        return modo, fecha, teste, le, 1 if (teste and le) else 0
    except Exception:
        return modo, 0, 0, 0, 0


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=3)
    a = ap.parse_args()
    _ini()

    print("=" * 76)
    print("LENTE — achar o termo novo quando a teoria velha quase funciona")
    print("=" * 76)
    # L0: a teoria velha deixa residuo nos dois mundos?
    rng = random.Random(SEMENTE)
    for modo in ("lente", "sorteado"):
        desv = tot = 0
        for _ in range(40):
            a = _cena(rng)
            b = _passo(a, modo, rng)
            for i in range(LADO):
                for j in range(LADO):
                    if b[i][j] == PART and (i == 0 or a[i - 1][j] != PART):
                        tot += 1
                        if not (i > 0 and a[i - 1][j] == PART):
                            desv += 1
        print(f"  {modo:9s}: {desv}/{tot} particulas fora da queda reta")

    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_uma, range(len(TAR)), chunksize=1)

    print(f"\n  ({time.time()-t0:.0f}s)\n")
    print("  mundo      fecha treino   acerta teste   le a massa   DESCOBERTA")
    print("  --------   ------------   ------------   ----------   ----------")
    tot = {}
    for modo in ("lente", "sorteado"):
        rs = [r for r in res if r[0] == modo]
        f, t, l, d = (sum(r[k] for r in rs) for k in (1, 2, 3, 4))
        n = len(rs)
        tot[modo] = d
        print(f"  {modo:9s}  {f:5d}/{n:<6d}   {t:5d}/{n:<6d}   {l:4d}/{n:<5d}"
              f"   {d:4d}/{n}")

    print("")
    print("=" * 76)
    dl, ds = tot["lente"], tot["sorteado"]
    if dl > ds and dl >= 6:
        print("L1 . DESCOBERTA. O sistema revisa a teoria velha a partir da")
        print("     anomalia ESTRUTURADA, e nao a partir de ruido.")
    elif dl > ds:
        print("L1 parcial . descobre mais no mundo com lei, mas pouco.")
    elif dl == 0 and ds == 0:
        print("L3 . o termo novo e inexprimivel no vocabulario atual.")
    else:
        print("L2 . descobre igual nos dois: e decoracao, nao descoberta.")


if __name__ == "__main__":
    main()
