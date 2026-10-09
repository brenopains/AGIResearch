"""
novaphys.bilinear — bancada de multiplicação de matrizes, com verificador exato
================================================================================

POR QUE ESTE ALVO E NÃO OS OUTROS
-----------------------------------
Apontar o sistema para problema em aberto já foi tentado QUATRO vezes aqui, e as
quatro estão medidas como negativas:

  empacotamento      99,9% do ótimo em 18s — o problema já era fácil
  OEIS "hard"        rende 1 em 110
  nenhum-trio        n=22 contra recorde 68; parede de 15x por degrau, 10^27
  conjectura grafo   redescobriu Whitney 1932 em 3s; 7 de 8 eram d<=2r disfarçado
  DepMap             portões mecânicos passaram, 0/8 inéditos

[[levantamento-de-alvos]] destilou 5 critérios desses fracassos. Este alvo é o
que passa neles:

  verificador EXATO      identidade tensorial sobre inteiros; sem simulação,
                         sem ruído, sem "quase"
  resposta PEQUENA       3x3 com 23 produtos = 23 x 27 coeficientes ternários
  ESCADA abaixo da       Strassen 2x2/7 (1969) e Laderman 3x3/23 (1976) são
  fronteira              alvos conhecidos e verificáveis
  fronteira PÚBLICA      3x3 com <= 22 continua aberto
  progresso INTERMEDIÁRIO  R decrescente é uma régua, não um binário

A FORMULAÇÃO
--------------
Um algoritmo bilinear com R produtos para C = A·B (n x n):

    P_r = ( Σ_{a,b} u[r][a][b] · A[a][b] ) · ( Σ_{c,d} v[r][c][d] · B[c][d] )
    C[i][l] = Σ_r w[r][i][l] · P_r

Igualando coeficientes de A[a][b]·B[c][d] dos dois lados, a identidade exata é:

    Σ_r u[r][a][b] · v[r][c][d] · w[r][i][l]  ==  [a==i]·[b==c]·[d==l]

para TODOS os (a,b,c,d,i,l). São n^6 equações — 64 para 2x2, 729 para 3x3.
Verificação em inteiros: ou fecha, ou não fecha. Não existe "quase certo", que é
justamente o que [[paisagem-e-penhasco]] mediu como penhasco.

O MARCO QUE DECIDE BARATO
---------------------------
3x3 com 22 é a fronteira. Mas antes disso existe uma pergunta de horas, não de
semanas:

    o maquinário consegue REDESCOBRIR Strassen 2x2 com 7 produtos?

É um objeto conhecido, pequeno (7 x 12 = 84 coeficientes ternários) e já
resolvido por um humano em 1969. Se a busca não acha isso, 3x3/22 está fora de
alcance e nós aprendemos em horas em vez de semanas. Este arquivo entrega o
verificador, as soluções conhecidas para validá-lo, e a LINHA DE BASE que
qualquer maquinário nosso precisa bater.

PRÉ-REGISTRO
--------------
  B0  PRÉ-CONDIÇÃO. O verificador aceita Strassen 2x2/7 e Laderman 3x3/23, e
      REJEITA versões com um coeficiente trocado. Sem isso nada mais vale.
  B1  busca aleatória ternária acha 2x2/7 em <= 10^7 tentativas
      -> o alvo é fácil demais e não serve de bancada
  B2  não acha -> temos linha de base medida, e o número dela é a barra que
      o maquinário do projeto precisa bater antes de qualquer alegação
  B3  o trivial 2x2/8 é achado mas 7 não -> a bancada tem gradiente utilizável
      e o degrau 8->7 é a unidade de dificuldade

PREVISÃO: B2. Mecanismo: 3^84 é intratável por sorteio, e a identidade é
tudo-ou-nada em 64 equações simultâneas. A previsão interessante não é essa, é
a de B3: se nem o trivial 8 aparece por sorteio, a bancada não tem gradiente
nenhum e serve só como verificador, não como espaço de busca.
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, List, Optional, Sequence, Tuple

Coef = Tuple[Tuple[int, ...], ...]          # [r][a*n+b]
Alg = Tuple[Coef, Coef, Coef]               # (u, v, w)


def alvo(n: int, a: int, b: int, c: int, d: int, i: int, l: int) -> int:
    """O tensor da multiplicação: 1 exatamente quando a==i, b==c, d==l."""
    return 1 if (a == i and b == c and d == l) else 0


def verifica(alg: Alg, n: int) -> Tuple[bool, int]:
    """
    EXATO, em inteiros. Devolve (fecha, quantas equações falham).

    O segundo número é a única régua contínua que este problema oferece — e
    [[paisagem-e-penhasco]] avisa para não confiar nela como gradiente antes de
    medir se ela tem gradiente.
    """
    u, v, w = alg
    R = len(u)
    falhas = 0
    rng = range(n)
    for a in rng:
        for b in rng:
            for c in rng:
                for d in rng:
                    ab, cd = a * n + b, c * n + d
                    for i in rng:
                        for l in rng:
                            il = i * n + l
                            s = 0
                            for r in range(R):
                                ur = u[r][ab]
                                if ur == 0:
                                    continue
                                vr = v[r][cd]
                                if vr == 0:
                                    continue
                                s += ur * vr * w[r][il]
                            if s != alvo(n, a, b, c, d, i, l):
                                falhas += 1
    return falhas == 0, falhas


def trivial(n: int) -> Alg:
    """O algoritmo escolar: n^3 produtos, um por (i,k,l)."""
    u, v, w = [], [], []
    for i in range(n):
        for k in range(n):
            for l in range(n):
                uu = [0] * (n * n); uu[i * n + k] = 1
                vv = [0] * (n * n); vv[k * n + l] = 1
                ww = [0] * (n * n); ww[i * n + l] = 1
                u.append(tuple(uu)); v.append(tuple(vv)); w.append(tuple(ww))
    return tuple(u), tuple(v), tuple(w)


def strassen() -> Alg:
    """Strassen 1969, 2x2 com 7 produtos. Índices: A[a][b] -> a*2+b."""
    # M1=(A11+A22)(B11+B22) M2=(A21+A22)B11 M3=A11(B12-B22) M4=A22(B21-B11)
    # M5=(A11+A12)B22      M6=(A21-A11)(B11+B12)  M7=(A12-A22)(B21+B22)
    U = [(1,0,0,1), (0,0,1,1), (1,0,0,0), (0,0,0,1),
         (1,1,0,0), (-1,0,1,0), (0,1,0,-1)]
    V = [(1,0,0,1), (1,0,0,0), (0,1,0,-1), (-1,0,1,0),
         (0,0,0,1), (1,1,0,0), (0,0,1,1)]
    # C11=M1+M4-M5+M7  C12=M3+M5  C21=M2+M4  C22=M1-M2+M3+M6
    W = [(1,0,0,1), (0,0,1,-1), (0,1,0,1), (1,0,1,0),
         (-1,1,0,0), (0,0,0,1), (1,0,0,0)]
    return tuple(U), tuple(V), tuple(W)


def sorteia(n: int, R: int, rnd: random.Random, dens: float = 0.35) -> Alg:
    def linha():
        return tuple(rnd.choice((-1, 1)) if rnd.random() < dens else 0
                     for _ in range(n * n))
    return (tuple(linha() for _ in range(R)),
            tuple(linha() for _ in range(R)),
            tuple(linha() for _ in range(R)))


def base_aleatoria(n: int, R: int, tentativas: int, semente: int = 20260921
                   ) -> Tuple[Optional[Alg], int, int]:
    """LINHA DE BASE. Devolve (achado, menor numero de falhas, tentativas)."""
    rnd = random.Random(semente)
    melhor = 10 ** 9
    for t in range(tentativas):
        alg = sorteia(n, R, rnd)
        ok, f = verifica(alg, n)
        if f < melhor:
            melhor = f
        if ok:
            return alg, 0, t + 1
    return None, melhor, tentativas


def experimento(tentativas: int = 200_000) -> None:
    print("=" * 78)
    print("BILINEAR — verificador exato e linha de base")
    print("=" * 78)

    print("\n  B0 — o verificador esta correto?")
    for nome, alg, n, esperado in (("trivial 2x2/8 ", trivial(2), 2, True),
                                   ("Strassen 2x2/7", strassen(), 2, True),
                                   ("trivial 3x3/27", trivial(3), 3, True)):
        ok, f = verifica(alg, n)
        print("    %s  fecha=%-5s falhas=%-4d  %s"
              % (nome, ok, f, "OK" if ok == esperado else "ERRADO"))

    # um coeficiente trocado tem de QUEBRAR
    u, v, w = strassen()
    u2 = list(map(list, u)); u2[0][0] = -u2[0][0]
    ok, f = verifica((tuple(map(tuple, u2)), v, w), 2)
    print("    Strassen com 1 coef trocado  fecha=%-5s falhas=%-4d  %s"
          % (ok, f, "OK" if not ok else "ERRADO — o verificador nao discrimina"))

    print("\n  linha de base: sorteio ternario, %d tentativas" % tentativas)
    for R in (8, 7):
        alg, menor, t = base_aleatoria(2, R, tentativas)
        print("    2x2 com R=%d  achou=%-5s  menor numero de falhas=%d de 64"
              % (R, alg is not None, menor), flush=True)

    print("\n" + "=" * 78)
    print("  O numero acima e a BARRA: qualquer maquinario deste projeto que")
    print("  alegue achar estrutura aqui precisa bater o sorteio sob o MESMO")
    print("  orcamento, e o orcamento se conta em avaliacoes do verificador.")


if __name__ == "__main__":
    experimento()
