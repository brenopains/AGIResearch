"""
novaphys.inventario — o que as 393 inalcancaveis EXIGEM?
==========================================================

POR QUE MEDIR EM VEZ DE LISTAR
--------------------------------
`dica.py` mediu que um professor perfeito, apontando a forma exata da regra,
compra 3 tarefas: de 4 para 7. As outras 393 sao inalcancaveis por orientacao
— falta REPRESENTACAO, nao direcao.

A pergunta seguinte e "quais representacoes faltam". E a tentacao e listar as
plausiveis. Mas nesta sessao eu fiz 19 previsoes sobre mecanismos e acertei 7,
com as 12 erradas TODAS otimistas. Listar do que eu acho que falta e
exatamente o procedimento que errou doze vezes.

Entao mede-se. Para cada tarefa, computam-se assinaturas estruturais baratas da
relacao entrada->saida, exigindo que valham em TODOS os pares de treino. A
distribuicao delas e o mapa.

AS ASSINATURAS, e cada uma corresponde a uma familia de representacao
----------------------------------------------------------------------
    dim_igual        entrada e saida do mesmo tamanho
    dim_subgrade     a saida e uma SUBGRADE exata da entrada  -> familia de
                     SELECAO (recortar a parte certa)
    dim_ladrilho     saida = entrada repetida k x m           -> LADRILHO
    dim_escala       saida = cada celula virou bloco k x k    -> ESCALA
    dim_outra        muda de tamanho de outro jeito

    so_recolore      saida = entrada com um mapa de cor por celula -> mapa de
                     cor puro, sem geometria
    formas_iguais    o multiconjunto de formas normalizadas dos objetos e o
                     mesmo -> objetos preservados, so movem/pintam
    formas_novas     aparecem formas que nao existiam -> CRIACAO de estrutura
    cores_novas      aparecem cores que nao estavam na entrada
    menos_objetos    a saida tem menos objetos -> selecao/apagamento
    mais_objetos     a saida tem mais -> copia/desenho
    saida_constante  a mesma saida para todas as entradas -> a resposta nao
                     depende da entrada
    simetrico        a saida e simetrica (H, V ou ambas) e a entrada nao ->
                     SIMETRIZACAO

Cada tarefa pode cair em varias. O que interessa nao e classificar
perfeitamente — e saber quais familias concentram as 393.

O QUE ISTO NAO E
------------------
Nao usa o teste. As assinaturas sao computadas so nos pares de TREINO, que sao
a especificacao. E nao e um solver: e um censo.

PRE-REGISTRO
--------------
  I0  PRE-CONDICAO. As assinaturas tem que cobrir a maioria: se mais de 40%
      das tarefas nao caem em nenhuma, o vocabulario de assinaturas e cego e o
      censo nao informa.

  I1  uma familia concentra >= 60 tarefas nao resolvidas
      -> ha um alvo grande e nomeado
  I2  a maior fica entre 25 e 59
  I3  nenhuma passa de 25
      -> o ARC e uma cauda longa de familias pequenas, e nao existe UMA
         representacao que compre muito; a estrategia de "achar a proxima
         representacao" tem retorno decrescente por construcao

MINHA PREVISAO: I2. Espero `dim_subgrade` (selecao) e `formas_novas` (criacao)
entre as maiores, e espero que `dim_igual` + `formas_iguais` — que e a familia
que nosso motor JA cobre — seja pequena, porque se fosse grande nos
estariamos resolvendo mais.

E registro o que me faria mudar de estrategia: se der I3, parar de procurar a
proxima representacao e passar a atacar o que e comum a TODAS elas.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from typing import Dict, List, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.eter import PASTA_AVAL

RESOLVIDAS = {
    "195ba7dc", "5d2a5c43", "d282b262", "37d3e8b2", "66e6c45b", "8ee62060",
    "903d1b4a", "981571dc", "9ddd00f0", "ae58858e", "af22c60d", "c7d4e6ad",
    "ca8f78db", "fc754716", "32e9702f", "332efdb3", "47996f11", "00dbd492",
    "72a961c9", "b1fc8b8e", "f45f5ca7", "358ba94e", "5ffb2104", "e0fb7511",
    "00576224", "59341089", "bc4146bd",
}


def _fundo(g: Grid) -> int:
    return Counter(v for l in g for v in l).most_common(1)[0][0]


def _objs(g: Grid, bg: int) -> List[frozenset]:
    h, w = dims(g)
    vis = [[False] * w for _ in range(h)]
    fora = []
    for i in range(h):
        for j in range(w):
            if vis[i][j] or g[i][j] == bg:
                continue
            cor = g[i][j]
            cel, fila = set(), deque([(i, j)])
            vis[i][j] = True
            while fila:
                y, x = fila.popleft()
                cel.add((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w and not vis[ny][nx]
                            and g[ny][nx] == cor):
                        vis[ny][nx] = True
                        fila.append((ny, nx))
            fora.append(frozenset(cel))
    return fora


def _forma(cel: frozenset) -> frozenset:
    y0 = min(y for y, _ in cel)
    x0 = min(x for _, x in cel)
    return frozenset((y - y0, x - x0) for y, x in cel)


def _subgrade(a: Grid, b: Grid) -> bool:
    ha, wa = dims(a)
    hb, wb = dims(b)
    if hb > ha or wb > wa:
        return False
    for i in range(ha - hb + 1):
        for j in range(wa - wb + 1):
            if all(a[i + y][j + x] == b[y][x]
                   for y in range(hb) for x in range(wb)):
                return True
    return False


def _ladrilho(a: Grid, b: Grid) -> bool:
    ha, wa = dims(a)
    hb, wb = dims(b)
    if hb % ha or wb % wa or (hb // ha) * (wb // wa) < 2:
        return False
    return True


def _escala(a: Grid, b: Grid) -> bool:
    ha, wa = dims(a)
    hb, wb = dims(b)
    if hb % ha or wb % wa or hb // ha != wb // wa or hb // ha < 2:
        return False
    k = hb // ha
    return all(b[i * k + u][j * k + v] == a[i][j]
               for i in range(ha) for j in range(wa)
               for u in range(k) for v in range(k))


def _so_recolore(a: Grid, b: Grid) -> bool:
    if dims(a) != dims(b):
        return False
    m: Dict[int, int] = {}
    for la, lb in zip(a, b):
        for x, y in zip(la, lb):
            if m.setdefault(x, y) != y:
                return False
    return True


def _simetrico(g: Grid) -> bool:
    return (g == tuple(tuple(l[::-1]) for l in g)) or (g == g[::-1])


def assinaturas(tr) -> Set[str]:
    """So vale a assinatura que se sustenta em TODOS os pares de treino."""
    cand = {
        "dim_igual": lambda a, b: dims(a) == dims(b),
        "dim_subgrade": lambda a, b: dims(a) != dims(b) and _subgrade(a, b),
        "dim_ladrilho": lambda a, b: _ladrilho(a, b),
        "dim_escala": lambda a, b: _escala(a, b),
        "so_recolore": _so_recolore,
        "cores_novas": lambda a, b: bool(
            {c for l in b for c in l} - {c for l in a for c in l}),
        "saida_simetrica": lambda a, b: _simetrico(b) and not _simetrico(a),
    }
    fora = {k for k, f in cand.items() if all(f(a, b) for a, b in tr)}

    fi = fn = mo = me = 0
    for a, b in tr:
        bg = _fundo(a)
        oa, ob = _objs(a, bg), _objs(b, bg)
        sa = Counter(_forma(c) for c in oa)
        sb = Counter(_forma(c) for c in ob)
        fi += sa == sb
        fn += bool(set(sb) - set(sa))
        mo += len(ob) > len(oa)
        me += len(ob) < len(oa)
    n = len(tr)
    if fi == n:
        fora.add("formas_iguais")
    if fn == n:
        fora.add("formas_novas")
    if mo == n:
        fora.add("mais_objetos")
    if me == n:
        fora.add("menos_objetos")
    if len({b for _a, b in tr}) == 1 and n > 1:
        fora.add("saida_constante")
    return fora


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    print("=" * 78)
    print("INVENTARIO — o que as inalcancaveis exigem?")
    print("=" * 78)
    tarefas = carregar(PASTA_AVAL)
    todas: Dict[str, Set[str]] = {}
    for n, tr, _te in tarefas:
        try:
            todas[n] = assinaturas(tr)
        except Exception:
            todas[n] = set()

    nao = {n: s for n, s in todas.items() if n not in RESOLVIDAS}
    sem = [n for n, s in nao.items() if not s]
    print(f"\n{len(tarefas)} tarefas, {len(RESOLVIDAS)} resolvidas, "
          f"{len(nao)} nao resolvidas")
    print(f"I0 · sem nenhuma assinatura: {len(sem)} "
          f"({len(sem)/max(len(nao),1):.0%} das nao resolvidas)")

    c = Counter(a for s in nao.values() for a in s)
    cr = Counter(a for n, s in todas.items() if n in RESOLVIDAS for a in s)
    print(f"\n{'assinatura':18s} {'nao resolvidas':>15s} {'resolvidas':>11s}")
    print("-" * 47)
    for k, v in c.most_common():
        print(f"{k:18s} {v:>15d} {cr.get(k,0):>11d}")
    print("-" * 47)

    maior = c.most_common(1)[0] if c else ("", 0)
    print("\n" + "=" * 78)
    if len(sem) > len(nao) * 0.40:
        print("I0 FALHOU: as assinaturas sao cegas para a maioria. O censo")
        print("     nao informa.")
        return
    if maior[1] >= 60:
        print(f"I1 · `{maior[0]}` concentra {maior[1]} nao resolvidas. Ha um")
        print("     alvo grande e nomeado.")
    elif maior[1] >= 25:
        print(f"I2 · a maior familia e `{maior[0]}` com {maior[1]}.")
    else:
        print("I3 · cauda longa: nenhuma familia passa de 25. Nao existe UMA")
        print("     representacao que compre muito, e procurar a proxima tem")
        print("     retorno decrescente por construcao.")


if __name__ == "__main__":
    main()
