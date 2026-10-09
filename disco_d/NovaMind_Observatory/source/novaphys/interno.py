"""
novaphys.interno — a CENA vira vocabulario, em vez de eu escrever vocabulario
==============================================================================

A IDEIA DO USUARIO, E A FORMULACAO MECANICA DELA
--------------------------------------------------
"A mente nao pega o problema ou a cena literal, ela traz pra dentro e trabalha
internamente. Se o que falta e composicao e nao vocabulario, talvez isso
resolva."

Ha uma razao mecanica para isso morder exatamente onde dói, e ela vem de um
resultado medido: composicao falha quando o resultado INTERMEDIARIO nao tem
nome. `relaxado.py` mostrou que exigir cobertura 100% do operador da ZERO
composicoes de dois passos (mesmo com 2,4x o orcamento) e que aceitar 85% da
CINCO. Estrutura parcial funciona porque da a busca algo para PASSAR ADIANTE.

O gargalo nao e ter operadores. E ter INTERMEDIARIOS que valham a pena.

E e disso que trazer a cena para dentro trata: a tarefa fornece os
intermediarios que a composicao precisa. "Faca virar aquele motivo que aparece
no canto" e inexpressivel no meu vocabulario e trivial no vocabulario da
propria tarefa.

O QUE JA FOI TESTADO E POR QUE NAO CONTA
------------------------------------------
`eter.py` aterra operadores por tarefa: deduz fatores de escala das dimensoes
e instancia 48 a 137 operadores. Rendeu +1 tarefa. Mas aquilo PARAMETRIZA
operadores meus — nao transforma CONTEUDO da cena em vocabulario. A forma de
um objeto que aparece na grade nunca vira uma peca manipulavel.

O QUE ENTRA AQUI
------------------
Chaves de mapeamento que sao a propria cena:

    forma        o conjunto de celulas do objeto, normalizado para a origem
    forma_d4     a mesma forma canonizada pela orbita do grupo diedral D4 —
                 mesma forma girada ou espelhada cai na MESMA chave
    caixa        (altura, largura) da caixa envolvente
    forma_cor    (forma, cor) juntas

O mapa chave -> cor continua sendo DEDUZIDO dos pares de treino, resolvendo a
restricao. Nenhuma amostragem. E a canonizacao por D4 e a resposta operacional
a intuicao de simetria: em vez de enumerar as oito rotacoes como programas
diferentes, colapsa-se a orbita numa chave so.

LIMITE DECLARADO, e ele e severo e estrutural
-----------------------------------------------
Um mapa chaveado por forma so responde sobre formas que APARECERAM no treino.
Se a grade de teste tem uma forma inedita, o mapa nao tem entrada e a leitura
devolve -1, que nunca casa. Entao isto NAO generaliza sobre formas novas —
generaliza sobre a REGRA que liga forma conhecida a resposta.

Isso e exatamente certo para a familia "estas formas viram estas cores" e
exatamente errado para "descubra o padrao e extrapole". E preciso dizer qual
das duas se esta medindo, senao o numero mente.

A protecao contra decoreba continua sendo o verificador: igualdade EXATA nas
grades de TESTE, nunca vistas no ajuste do mapa.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  I0  PRE-CONDICAO. O braco sem chaves de forma tem que dar 6/400,
      reproduzindo `perobj.py`. Se nao reproduzir, divergiu.

  I1  +5 ou mais    a cena como vocabulario e o mecanismo; a intuicao de
                    "trazer para dentro" tem lastro
  I2  +2 a +4       ajuda, na ordem dos outros mecanismos
  I3  +0 ou +1      conteudo da cena como chave nao e o que falta

E o gate que importa mais que o placar:

  I4  a canonizacao D4 resolve tarefas que a forma crua NAO resolve
      -> a orbita do grupo carrega informacao, e elegancia e eficiencia
         tambem aqui, nao so na busca

MINHA PREVISAO: I2, entre +2 e +4, e I4 positivo com 1 a 2 tarefas. Modesto de
proposito: hoje eu ja errei uma aposta alta neste mesmo terreno (previ 12-20
para quantificador e deu 4), e o limite declarado acima e severo — metade das
tarefas de "forma vira cor" do ARC usa formas de teste que nao aparecem no
treino, e essas ficam fora por construcao.

O numero que vai importar de verdade e este vocabulario COM composicao de dois
passos, nao num passo. Um passo e teto duro: a linguagem por celula da 3 num
passo e 8 em dois.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims, fundo
from novaphys.eter import PASTA_AVAL
from novaphys.eter_inferencia import todas_leituras, todos_testes
from novaphys.ordem1 import (aplicar, inferir, leituras_quantificadas,
                             objetos, testes_quantificados)
from novaphys.perobj import PROPS, _props, leituras_por_objeto


# ══════════════════════════════════════════════ a cena como chave
def _norm(cel: Set[Tuple[int, int]]) -> frozenset:
    y0 = min(y for y, _ in cel)
    x0 = min(x for _, x in cel)
    return frozenset((y - y0, x - x0) for y, x in cel)


def _d4(cel: frozenset) -> frozenset:
    """
    Chave canonica sob a orbita do grupo diedral D4: as oito simetrias do
    quadrado. Mesma forma girada ou espelhada cai na mesma chave.

    E a resposta operacional a intuicao de simetria — em vez de enumerar as
    oito rotacoes como programas diferentes (que e o que a busca crua faz, e
    que o quociente observacional mediu como 2,06x de desperdicio), colapsa-se
    a orbita inteira numa chave so.
    """
    h = max(y for y, _ in cel) + 1
    w = max(x for _, x in cel) + 1
    fora = []
    for k in range(4):
        if k == 0:
            v = {(y, x) for y, x in cel}
            hh, ww = h, w
        elif k == 1:
            v = {(x, hh - 1 - y) for y, x in v}
            hh, ww = ww, hh
        elif k == 2:
            v = {(x, hh - 1 - y) for y, x in v}
            hh, ww = ww, hh
        else:
            v = {(x, hh - 1 - y) for y, x in v}
            hh, ww = ww, hh
        fora.append(frozenset(v))
        fora.append(frozenset((y, ww - 1 - x) for y, x in v))
    return min(fora, key=lambda s: sorted(s))


CHAVES = ("forma", "forma_d4", "caixa", "forma_cor")


def _chaves(objs) -> List[Dict[str, Any]]:
    fora = []
    for cel, cor in objs:
        n = _norm(cel)
        ys = [y for y, _ in cel]
        xs = [x for _, x in cel]
        fora.append({
            "forma": n,
            "forma_d4": _d4(n),
            "caixa": (max(ys) - min(ys) + 1, max(xs) - min(xs) + 1),
            "forma_cor": (n, cor),
        })
    return fora


class LeituraCena:
    """Valor da celula = mapa[chave da FORMA do objeto a que ela pertence]."""

    __slots__ = ("chave", "diag", "mapa", "const")

    def __init__(self, chave: str, diag: bool, mapa: Dict[Any, int]):
        self.chave, self.diag, self.mapa = chave, diag, mapa
        self.const = False

    def valores(self, g: Grid, bg: int) -> List[List[int]]:
        h, w = dims(g)
        fora = [[bg] * w for _ in range(h)]
        objs = objetos(g, bg, self.diag)
        if len(objs) > 60:
            return [[-1] * w for _ in range(h)]
        for (cel, _cor), k in zip(objs, _chaves(objs)):
            v = self.mapa.get(k[self.chave], -1)
            for (i, j) in cel:
                fora[i][j] = v
        return fora

    def __str__(self) -> str:
        return f"porForma({self.chave}){'8' if self.diag else '4'}"


def ajustar_cena(treino, chave: str, diag: bool) -> Optional[LeituraCena]:
    votos: Dict[Any, Counter] = {}
    for a, b in treino:
        if dims(a) != dims(b):
            return None
        bg = fundo(a)
        objs = objetos(a, bg, diag)
        if not objs or len(objs) > 60:
            continue
        for (cel, _cor), k in zip(objs, _chaves(objs)):
            c = Counter(b[i][j] for (i, j) in cel)
            votos.setdefault(k[chave], Counter()).update(c)
    if not votos:
        return None
    return LeituraCena(chave, diag,
                       {k: v.most_common(1)[0][0] for k, v in votos.items()})


def leituras_cena(treino, so_d4: Optional[bool] = None) -> List[LeituraCena]:
    fora = []
    for chave in CHAVES:
        if so_d4 is True and chave != "forma_d4":
            continue
        if so_d4 is False and chave == "forma_d4":
            continue
        for diag in (False, True):
            L = ajustar_cena(treino, chave, diag)
            if L is not None:
                fora.append(L)
    return fora


# ══════════════════════════════════════════════ avaliacao
def vocabulario(treino, cena: str):
    cores = sorted({c for a, b in treino for l in (a + b) for c in l})
    L = list(todas_leituras(cores))
    T = list(todos_testes(cores)) + testes_quantificados()
    L += leituras_quantificadas()
    try:
        L += leituras_por_objeto(treino)
    except Exception:
        pass
    if cena == "cheio":
        L += leituras_cena(treino)
    elif cena == "sem_d4":
        L += leituras_cena(treino, so_d4=False)
    return L, T


def uma(args):
    nome, tr, te, cena = args
    try:
        L, T = vocabulario(tr, cena)
        op, _g = inferir(tr, L, T)
        if op is None:
            return nome, False, None
        ok = all(aplicar(op, x) == y for x, y in te)
    except Exception:
        return nome, False, None
    return nome, ok, f"[{op[0]} ? {op[1]} : {op[2]}]"


def avaliar(tarefas, cena, rotulo, nucleos):
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(nucleos) as pool:
        res = pool.map(uma, [(n, tr, te, cena) for n, tr, te in tarefas],
                       chunksize=2)
    ok = {n for n, o, _t in res if o}
    ex = {n: t for n, o, t in res if o}
    n = len(tarefas)
    print(f"  {rotulo:28s} {len(ok):3d}/{n}  ({len(ok)/n:5.2%})  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return ok, ex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--nucleos", type=int, default=6)
    a = ap.parse_args()

    print("=" * 80)
    print("INTERNO — a CENA vira vocabulario")
    print("=" * 80)
    tarefas = carregar(a.pasta)
    print("\nmapa chave->cor DEDUZIDO; verificador exato no teste\n")

    base, _ = avaliar(tarefas, "nenhum", "sem chave de forma (perobj)",
                      a.nucleos)
    semd4, _ = avaliar(tarefas, "sem_d4", "+ forma crua, caixa, forma+cor",
                       a.nucleos)
    cheio, ex = avaliar(tarefas, "cheio", "+ canonizacao D4", a.nucleos)

    print(f"\ncena - base   = {len(cheio) - len(base):+d}")
    print(f"D4  - forma crua = {len(cheio) - len(semd4):+d}   <- gate I4")
    print(f"\nso a cena resolve : {sorted(cheio - base)}")
    print(f"so o D4 resolve   : {sorted(cheio - semd4)}")
    print("\nleis novas:")
    for n in sorted(cheio - base)[:16]:
        print(f"  {n}  {ex[n]}")

    d = len(cheio) - len(base)
    print("\n" + "=" * 80)
    if len(base) != 6:
        print(f"I0 · base deu {len(base)}/400 e `perobj` deu 6. Divergiu.")
    if d >= 5:
        print("I1 · a CENA como vocabulario e o mecanismo. Trazer o problema")
        print("     para dentro e transforma-lo em peca manipulavel tem lastro.")
    elif d >= 2:
        print("I2 · ajuda, na ordem dos outros mecanismos da sessao.")
    else:
        print("I3 · conteudo da cena como chave nao e o que falta.")
    if len(cheio) > len(semd4):
        print(f"I4 · a canonizacao D4 resolve {len(cheio)-len(semd4)} tarefa(s)")
        print("     que a forma crua nao resolve. A orbita do grupo carrega")
        print("     informacao: elegancia e eficiencia tambem no vocabulario.")
    else:
        print("I4 · a canonizacao D4 nao acrescenta sobre a forma crua.")


if __name__ == "__main__":
    main()
