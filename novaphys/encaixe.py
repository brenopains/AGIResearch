"""
novaphys.encaixe — a peca solta volta para o buraco que tem a forma dela
=========================================================================

DE ONDE VEM
-------------
NAO veio de palpite meu. Veio de `fluido.py` FALHAR.

O motor de fluido deu F0: zero tarefas ajustando o treino. Ao rastrear a
`184a9768` — que o censo tinha marcado como `novas_enchem`, "o conjunto pintado
e exatamente a regiao de fundo cercada" — o diff mostrou outra coisa:

    par 0: mudou=96  de_fundo=45  de_ocupada=51
    ocupada->: TODAS para 0

As 51 celulas ocupadas que mudam viram FUNDO. Nao e liquido enchendo copo. E
uma peca que SOME de um lugar e APARECE em outro. O censo tinha casado pelo
motivo errado: buraco de quebra-cabeca tambem e regiao cercada.

O CENSO QUE ISSO ABRIU
------------------------
Assinatura: o multiconjunto de FORMAS removidas == o de formas pintadas.

    ENCAIXE EXATO ................... 9 tarefas
    + tolerancia a peca sem destino .. 14 tarefas
    ja resolvidas por outro motor .... 1  (8ee62060)

    transformacoes: id 195, r90 9, r180 3
    so translacao: 11 das 14

E a maior familia com MECANISMO EXATO que apareceu na sessao inteira. As outras
vieram de perfil frouxo (novas_uma_cor 56, preserva 77) que nunca virou regra.
Esta veio com o mecanismo junto: congruencia de forma sob D4.

E e a familia que o usuario descreveu falando de relatos psicodelicos, muito
antes de qualquer censo: "as transformacoes atingem uma simetria quase perfeita,
formas que se completam e SE ENCAIXAM".

AS VARIANTES
--------------
    transf     so translacao, ou o grupo D4 inteiro
    desempate  exigir peca congruente UNICA, ou pegar a mais PERTO
    sobra      peca sem destino: mantem, ou apaga (ruido)
    cor        o buraco recebe a cor da peca, ou uma constante lida do diff

Dezesseis combinacoes. Nada e sorteado; a escolha e por reproducao exata do
treino, e depois passa pelo imunologico (leave-one-out).

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  E0  PRE-CONDICAO. Alguma das 13 tarefas do censo tem que ajustar o treino.

      E0 JA DISPAROU UMA VEZ, e estava certo. A versao 1 casava UM buraco com
      UMA peca e deu 0/400. O traco na `184a9768` mostrou por que:

          buracos = [1, 4, 6, 8, 262]   pecas = [1x12, 4x3, 6, 89]

      o buraco de 8 e fechado por DUAS pecas. Buraco nao e peca virada do
      avesso; buraco e AREA A COBRIR. A versao 2 ladrilha (cobertura exata).

      Trocando a pre-condicao, e registrando a troca: a v1 nomeava a
      `184a9768` porque era a que eu tinha lido a mao — mas ela e o membro
      MAIS DIFICIL, com 11 celulas soltas de ruido e uma peca 1x1 de cor 4
      indistinguivel das 1x1 de cor 5. Ela nao esta nas 14 do censo. Cobrar a
      pre-condicao no membro mais dificil da familia foi erro meu de desenho.

  E1  >= 7 inedidas -> encaixe era o degrau; conservacao de forma e o primeiro
                       mecanismo desta escala que o projeto expressa
  E2  3 a 6         -> ajuda, no patamar das mudancas de representacao
  E3  <= 2          -> nem tendo o mecanismo exato e o censo na mao; a cauda
                       longa vale ate para familia com assinatura fechada

MINHA CALIBRAGEM: trinta e uma previsoes, DEZ certas, as vinte e uma erradas
todas otimistas. E o refino mais recente e mais duro: o que veio de MEDIDA
entregou (+2, +3, +3); o que veio de palpite meu deu zero, dezoito vezes.
Esta veio de medida — mas a medida diz que a ASSINATURA existe em 13 tarefas
nao resolvidas, e assinatura nunca foi solucao.

PREVISAO: E2, 3 a 6.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.compositor import fundo
from novaphys.eter import PASTA_AVAL

ORTO = ((-1, 0), (1, 0), (0, -1), (0, 1))
Cel = Tuple[int, int]


def componentes(cels: Set[Cel]) -> List[FrozenSet[Cel]]:
    s = set(cels)
    fora = []
    while s:
        p = s.pop()
        g = [p]
        f = deque([p])
        while f:
            y, x = f.popleft()
            for dy, dx in ORTO:
                q = (y + dy, x + dx)
                if q in s:
                    s.discard(q)
                    g.append(q)
                    f.append(q)
        fora.append(frozenset(g))
    return fora


def normal(g) -> FrozenSet[Cel]:
    my = min(y for y, _x in g)
    mx = min(x for _y, x in g)
    return frozenset((y - my, x - mx) for y, x in g)


def _r90(g) -> FrozenSet[Cel]:
    return normal(frozenset((x, -y) for y, x in g))


def orbita(g, grupo: str) -> Dict[FrozenSet[Cel], str]:
    """Formas equivalentes da peca. Chave = forma, valor = nome da transf."""
    c = normal(g)
    if grupo == "id":
        return {c: "id"}
    fora: Dict[FrozenSet[Cel], str] = {}
    for nome in ("id", "r90", "r180", "r270"):
        fora.setdefault(c, nome)
        c = _r90(c)
    c = normal(frozenset((y, -x) for y, x in g))
    for nome in ("e", "er90", "er180", "er270"):
        fora.setdefault(c, nome)
        c = _r90(c)
    return fora


def pecas_e_buracos(g: Grid):
    """Objetos monocromaticos e regioes de fundo — as duas metades do mundo."""
    bg = fundo(g)
    h, w = dims(g)
    por_cor: Dict[int, Set[Cel]] = {}
    vazio: Set[Cel] = set()
    for i in range(h):
        for j in range(w):
            if g[i][j] == bg:
                vazio.add((i, j))
            else:
                por_cor.setdefault(g[i][j], set()).add((i, j))
    pecas = []
    for k, s in por_cor.items():
        for c in componentes(s):
            pecas.append((c, k))
    return bg, pecas, componentes(vazio)


def centro(g) -> Tuple[float, float]:
    return (sum(y for y, _x in g) / len(g), sum(x for _y, x in g) / len(g))


def posicoes(forma, alvo: Set[Cel], cel: Cel):
    """Todos os jeitos de por `forma` dentro de `alvo` cobrindo `cel`."""
    fora = []
    for s in forma:
        for o in s:
            p = frozenset((cel[0] + y - o[0], cel[1] + x - o[1]) for y, x in s)
            if cel in p and p <= alvo:
                fora.append(p)
    return set(fora)


def cobrir(alvo: Set[Cel], disp, grupo: str, limite: int = 2, teto: int = 40000):
    """
    LADRILHA o buraco com as pecas disponiveis, cada peca usada no maximo uma
    vez. Devolve ate `limite` coberturas distintas, cada uma como
    (pintura {cel: cor}, indices das pecas usadas).

    E aqui que o modelo anterior errava: ele casava UM buraco com UMA peca. Na
    `184a9768` um buraco de 8 celulas e fechado por DUAS pecas — a barra de 4
    e a coluna de 4. Buraco nao e peca virada do avesso; buraco e area a cobrir.
    """
    orbs = [(i, orbita(c, grupo), k, len(c)) for i, (c, k) in disp]
    achadas = []
    nos = [0]

    def rec(falta: Set[Cel], usadas, pint):
        if len(achadas) >= limite or nos[0] > teto:
            return
        nos[0] += 1
        if not falta:
            achadas.append((dict(pint), frozenset(usadas)))
            return
        cel = min(falta)
        for i, orb, k, n in orbs:
            if i in usadas or n > len(falta):
                continue
            for p in posicoes(orb, falta, cel):
                for q in p:
                    pint[q] = k
                rec(falta - p, usadas | {i}, pint)
                for q in p:
                    del pint[q]
                if len(achadas) >= limite or nos[0] > teto:
                    return

    rec(set(alvo), frozenset(), {})
    return achadas


def encaixar(g: Grid, grupo: str, desempate: str):
    """
    Devolve (pintura, sobra). None quando alguma cobertura e ambigua sob o
    criterio `unico` — duas ladrilhagens distintas do mesmo buraco.
    """
    _bg, pecas, buracos = pecas_e_buracos(g)
    area = sum(len(c) for c, _k in pecas)
    disp = list(enumerate(pecas))
    usadas: Set[int] = set()
    pintura: Dict[Cel, int] = {}
    achou = False
    for bur in sorted(buracos, key=lambda b: (-len(b), sorted(b)[0])):
        if len(bur) > area:
            continue                      # o fundo de fora, nao um buraco
        livres = [(i, p) for i, p in disp if i not in usadas]
        sols = cobrir(set(bur), livres, grupo)
        if not sols:
            continue
        if len(sols) > 1:
            if desempate == "unico":
                return None
            if {tuple(sorted(s[0].items())) for s in sols}.__len__() > 1:
                sols.sort(key=lambda s: sorted(s[0].items()))
        pint, us = sols[0]
        pintura.update(pint)
        usadas |= set(us)
        achou = True
    if not achou:
        return None
    sobra = [(i, c, k) for i, (c, k) in disp if i not in usadas]
    movidas = [(i, c, k) for i, (c, k) in disp if i in usadas]
    return pintura, movidas, sobra


def aplicar(regra, g: Grid) -> Optional[Grid]:
    try:
        grupo, desempate, sobra, cor = regra
        r = encaixar(g, grupo, desempate)
        if r is None:
            return None
        pintura, movidas, livres = r
        bg = fundo(g)
        fora = [list(l) for l in g]
        for _i, pc, _k in movidas:
            for y, x in pc:
                fora[y][x] = bg
        for (y, x), k in pintura.items():
            fora[y][x] = k if cor is None else cor
        if sobra == "apaga":
            maior = max((len(c) for c, _k in pecas_e_buracos(g)[1]), default=0)
            for _i, pc, _k in livres:
                if len(pc) < maior:
                    for y, x in pc:
                        fora[y][x] = bg
        return tuple(tuple(l) for l in fora)
    except Exception:
        return None


def deduzir(treino):
    if any(dims(a) != dims(b) for a, b in treino):
        return None
    cores = set()
    for a, b in treino:
        bg = fundo(a)
        h, w = dims(a)
        rem = any(a[i][j] != bg and b[i][j] == bg
                  for i in range(h) for j in range(w))
        pin = [b[i][j] for i in range(h) for j in range(w)
               if a[i][j] == bg and b[i][j] != bg]
        if not rem or not pin:
            return None
        cores |= set(pin)
    fixa = [next(iter(cores))] if len(cores) == 1 else []
    for grupo in ("id", "d4"):
        for desempate in ("unico", "perto"):
            for sobra in ("mantem", "apaga"):
                for cor in [None] + fixa:
                    r = (grupo, desempate, sobra, cor)
                    if all(aplicar(r, a) == b for a, b in treino):
                        return r
    return None


def imune(treino) -> bool:
    if len(treino) < 3:
        return False
    for i in range(len(treino)):
        resto = treino[:i] + treino[i + 1:]
        r = deduzir(resto)
        if r is None or aplicar(r, treino[i][0]) != treino[i][1]:
            return False
    return True


def _uma(args):
    nome, tr, te = args
    try:
        r = deduzir(tr)
        if r is None:
            return nome, False, False, False, None
        ok = all(aplicar(r, x) == y for x, y in te)
        return nome, True, imune(tr), ok, f"{r[0]}/{r[1]}/{r[2]}"
    except Exception:
        return nome, False, False, False, None


# as 14 da assinatura, menos a `8ee62060` que outro motor ja resolve
CENSO = {"1c56ad9f", "42a15761", "64a7c07e", "67c52801", "85b81ff1",
         "90347967", "917bccba", "9b4c17c4", "dd2401ed", "df8cc377",
         "e1d2900e", "ecaa0ec1", "f3e62deb"}

JA = {"195ba7dc", "5d2a5c43", "d282b262", "37d3e8b2", "66e6c45b", "8ee62060",
      "903d1b4a", "981571dc", "9ddd00f0", "ae58858e", "af22c60d", "c7d4e6ad",
      "ca8f78db", "fc754716", "32e9702f", "332efdb3", "47996f11", "00dbd492",
      "72a961c9", "b1fc8b8e", "f45f5ca7", "358ba94e", "5ffb2104", "e0fb7511",
      "00576224", "59341089", "bc4146bd", "070dd51e", "aa18de87", "3194b014",
      "ccd554ac", "cd3c21df", "55059096"}


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()

    print("=" * 76)
    print("ENCAIXE — a peca solta volta para o buraco que tem a forma dela")
    print("=" * 76)
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=4)
    ajust = [n for n, aj, _i, _o, _f in res if aj]
    imunes = [n for n, _aj, i, _o, _f in res if i]
    sem = {n: f for n, _aj, _i, o, f in res if o}
    com = {n: f for n, _aj, i, o, f in res if i and o}

    print("")
    print(f"  ajustou o TREINO       : {len(ajust):3d}/400  "
          f"({time.time()-t0:.0f}s)")
    print(f"  sobreviveu ao IMUNE    : {len(imunes):3d}/400")
    print(f"  SEM portao passou teste: {len(sem):3d}/400   "
          f"precisao {len(sem)/max(len(ajust),1):.1%}")
    print(f"  COM portao passou teste: {len(com):3d}/400")
    print("")
    bate = sorted(CENSO & set(ajust))
    print(f"  E0 . do censo ajustaram o treino: {len(bate)}/13  {bate}")
    print(f"  variantes: {dict(Counter(sem.values()).most_common())}")
    print(f"  tarefas: {sorted(sem)}")
    novas = sorted(set(sem) - JA)
    print(f"  INEDITAS: {len(novas)}  {novas}")

    print("")
    print("=" * 76)
    if not bate:
        print("E0 FALHOU: nenhuma tarefa do censo ajustou. Erro de")
        print("     implementacao, nao resultado.")
    if len(novas) >= 7:
        print("E1 . encaixe era o degrau.")
    elif len(novas) >= 3:
        print("E2 . ajuda, no patamar das mudancas de representacao.")
    else:
        print("E3 . a cauda longa vale ate para familia com assinatura fechada.")


if __name__ == "__main__":
    main()
