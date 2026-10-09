"""
novaphys.propriedade_inventada — o que faltava na grade era a GRANDEZA, não o codec
=====================================================================================

O DIAGNOSTICO, MEDIDO
-----------------------
Em 60 tarefas: 12 resolvidas, 3 decoraram, 5 sem pilha, e **40 nao conseguiram
nem DESCREVER o treino**. Nao e traducao ruim, e vocabulario ausente.

E este projeto ja sabe o que NAO resolve isso:

    compor primitivas existentes numa biblioteca    +0 com placebo pareado
    reordenar/repriorizar as que existem            40/400 nos 4 bracos
    LLM local escrevendo codecs novos               0 adotados em 18 tarefas

E sabe o que resolve, duas vezes, as duas na fisica: INVENTAR UM TERMO NOVO POR
CONSTRUCAO A PARTIR DOS DADOS, julgado pela utilidade rio abaixo. O agente
escreveu `h = x*vy - y*vx` (10/10) e derivou `n^ = rot(v^)` (0% -> 95%). Nenhuma
das duas veio de compor primitivas nem de alguem propondo de fora.

O PONTO DE ALAVANCA NA GRADE
------------------------------
No nucleo:

    class CorPorPropriedade:  PROPS = (tam, forma, buracos, altura, largura, posto)
    class ApagaPorPropriedade: PROPS = (tam, cor, forma)

Seis propriedades, escritas por mim. E varios codecs recebem propriedade como
PARAMETRO. Entao o alfabeto de propriedades e um MULTIPLICADOR, nao um somatorio:
crescer essa lista fortalece `cor_por_prop`, `apaga_prop`, `condicao` e `relacao`
de uma vez. Na fisica nao faltava a FORMA DA LEI, faltava a GRANDEZA para por
dentro. Na grade nao falta CODEC, falta a PROPRIEDADE para por dentro dele. E o
mesmo gap, no mesmo lugar da estrutura.

A REGRESSAO NAO ACABA, E ISSO E HONESTO
-----------------------------------------
Na fisica o agente inventou `h`, mas as operacoes de derivacao (produto, rotacao,
autodecomposicao) fui eu que dei. Aqui e igual: eu dou os OPERADORES DE
CONSTRUCAO e ele gera as propriedades. Voce nunca para de fornecer algo — o que
se faz e subir de nivel ate o que se fornece ser pequeno e independente de
dominio, enquanto a parte grande e especifica passa a ser gerada.

    11 observaveis base  x  7 operadores  =  ~80 propriedades derivadas

Os 7 operadores (posto, unicidade, extremalidade, paridade, contagem, distancia
ao maximo, razao) nao tem nada de grade — valem em qualquer dominio com objetos.

O CONFUNDIDOR QUE ESTE PROJETO JA CONHECE
-------------------------------------------
Mais propriedades = mais candidatos = mais AVALIACOES gastas. Com teto em
avaliacoes, o braco rico gasta o orcamento mais rapido. Entao os dois bracos
correm com o MESMO teto, e existe um PLACEBO com o mesmo NUMERO de propriedades,
so que arbitrarias. Sem isso a medida nao vale nada.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  Q1  NAO QUEBRA: o codec estendido nao perde nenhuma tarefa que o original
      resolvia (e superconjunto do vocabulario, entao perda = confundidor de
      orcamento, e precisa aparecer)
  Q2  GANHA: >= +3 tarefas sobre a base, teto casado
  Q3  PLACEBO: o mesmo numero de propriedades ARBITRARIAS ganha <= 0
  Q4  QUEM PAGOU: pelo menos 1 propriedade INVENTADA aparece nas pilhas
      vencedoras — se todas forem as minhas seis antigas, a invencao nao
      contribuiu e o ganho veio de outro lugar
PREVISAO: Q1 e o risco real e aposto que FALHA por 1 ou 2 tarefas, porque 80
propriedades queimam o teto. Q2 marginal, +1 a +4. Q3 passa. Q4 passa com 1 a 3
propriedades. Se Q2 passar e Q4 nomear uma propriedade inventada, o gap de
alfabeto abriu na grade pelo mesmo mecanismo que abriu na fisica.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D
from novaphys.descricao import (CorPorPropriedade, ApagaPorPropriedade,
                                objetos, fundo, dims)

# ─── os observaveis base ─────────────────────────────────────────────────────

BASE = ("tam", "altura", "largura", "buracos", "cor", "perimetro",
        "densidade", "sim_h", "sim_v", "toca_borda", "n_cores")

# ─── os 7 operadores de construcao, que sao o que EU forneco ─────────────────

OPS = ("posto", "unico", "maximo", "minimo", "paridade", "conta", "difmax")


def _base(prop, c, a, bg):
    ys = [y for y, _x in c]
    xs = [x for _y, x in c]
    h0, w0 = max(ys) - min(ys) + 1, max(xs) - min(xs) + 1
    if prop == "tam":
        return len(c)
    if prop == "altura":
        return h0
    if prop == "largura":
        return w0
    if prop == "cor":
        return min(a[y][x] for y, x in c)
    if prop == "n_cores":
        return len({a[y][x] for y, x in c})
    if prop == "densidade":
        return int(round(10 * len(c) / max(h0 * w0, 1)))
    if prop == "toca_borda":
        H, W = dims(a)
        return int(min(ys) == 0 or min(xs) == 0 or max(ys) == H - 1
                   or max(xs) == W - 1)
    if prop == "perimetro":
        n = 0
        for y, x in c:
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (y + dy, x + dx) not in c:
                    n += 1
        return n
    if prop == "sim_h":
        my, mx = min(ys), min(xs)
        cx = {(y - my, x - mx) for y, x in c}
        return int(all((i, w0 - 1 - j) in cx for i, j in cx))
    if prop == "sim_v":
        my, mx = min(ys), min(xs)
        cx = {(y - my, x - mx) for y, x in c}
        return int(all((h0 - 1 - i, j) in cx for i, j in cx))
    if prop == "buracos":
        return CorPorPropriedade._valor("buracos", c, a, bg)
    return 0


def _derivada(nome, c, a, bg):
    """`op_base`: aplica um operador de construcao a um observavel base.
    Operadores relacionais precisam de TODOS os objetos da grade — e e nisso que
    eles carregam informacao que nenhum observavel isolado tem."""
    op, prop = nome.split("_", 1)
    v = _base(prop, c, a, bg)
    if op == "paridade":
        return v % 2 if isinstance(v, int) else 0
    todos = [_base(prop, cc, a, bg) for cc, _k in objetos(a, bg)]
    if not todos:
        return v
    if op == "posto":
        return sorted(set(todos), reverse=True).index(v)
    if op == "unico":
        return int(todos.count(v) == 1)
    if op == "maximo":
        return int(v == max(todos))
    if op == "minimo":
        return int(v == min(todos))
    if op == "conta":
        return todos.count(v)
    if op == "difmax":
        return max(todos) - v
    return v


def vocabulario():
    """As ~80 propriedades: as base, mais cada operador aplicado a cada base."""
    fora = list(BASE)
    for op in OPS:
        for prop in BASE:
            if op == "paridade" and prop in ("sim_h", "sim_v", "toca_borda"):
                continue           # paridade de um booleano e ele mesmo
            fora.append(f"{op}_{prop}")
    return tuple(fora)


VOCAB = vocabulario()
INVENTADAS = set(VOCAB) - {"tam", "altura", "largura", "buracos", "cor", "posto"}


# ─── os codecs estendidos: entram como EXTRA, o nucleo nao muda ──────────────

def _valor_estendido(prop, c, a, bg):
    if "_" in prop and prop.split("_", 1)[0] in OPS:
        return _derivada(prop, c, a, bg)
    if prop in BASE:
        return _base(prop, c, a, bg)
    return CorPorPropriedade._valor(prop, c, a, bg)


class CorPorPropMais(CorPorPropriedade):
    nome = "cor_por_prop_mais"
    PROPS = VOCAB

    @staticmethod
    def _valor(prop, c, a, bg):
        return _valor_estendido(prop, c, a, bg)


class ApagaPorPropMais(ApagaPorPropriedade):
    nome = "apaga_prop_mais"
    PROPS = VOCAB

    @staticmethod
    def _valor(prop, c, a, bg):
        return _valor_estendido(prop, c, a, bg)


# ─── placebo pareado: mesmo NUMERO de propriedades, arbitrarias ──────────────

def _arbitraria(nome, c, a, bg):
    """Funcao do objeto sem sentido nenhum, mas deterministica e com a mesma
    cardinalidade tipica. Controla 'ganhou porque tem mais bilhete'."""
    s = hash((nome, len(c), min(c), max(c))) % 9973
    return s % 12


class CorPorPropPlacebo(CorPorPropriedade):
    nome = "cor_por_prop_placebo"
    PROPS = tuple(f"fake{k}" for k in range(len(VOCAB)))

    @staticmethod
    def _valor(prop, c, a, bg):
        return _arbitraria(prop, c, a, bg)


class ApagaPorPropPlacebo(ApagaPorPropriedade):
    nome = "apaga_prop_placebo"
    PROPS = tuple(f"fake{k}" for k in range(len(VOCAB)))

    @staticmethod
    def _valor(prop, c, a, bg):
        return _arbitraria(prop, c, a, bg)


# ─── a corrida ───────────────────────────────────────────────────────────────

def roda(tarefas, codecs, teto, etiqueta, usadas=None):
    ok, t0 = set(), time.time()
    for k, (tarefa, tr, te) in enumerate(tarefas):
        try:
            fm, pilha, err = D.montar(tr, codecs=codecs, teto=teto)
            if pilha and err == 0 and all(D.aplicar((fm, pilha), x) == y
                                          for x, y in te):
                ok.add(tarefa)
                if usadas is not None:
                    for it in pilha:
                        if (not D._marca(it) and it[0].nome.endswith("_mais")
                                and isinstance(it[1], tuple) and it[1]):
                            usadas.append((tarefa, it[0].nome, it[1][0]))
        except Exception:
            pass
        if (k + 1) % 25 == 0:
            print(f"    [{etiqueta}] {k+1} tarefas, {len(ok)} resolvidas, "
                  f"{time.time()-t0:.0f}s", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefas", type=int, default=120)
    ap.add_argument("--teto", type=int, default=400)
    a = ap.parse_args()
    tarefas = carregar(PASTA_AVAL)[: a.tarefas]
    print("=" * 78)
    print(f"PROPRIEDADE INVENTADA — {len(tarefas)} tarefas, teto {a.teto} "
          f"(CASADO nos tres bracos)")
    print(f"  {len(BASE)} observaveis base x {len(OPS)} operadores = "
          f"{len(VOCAB)} propriedades ({len(INVENTADAS)} novas)")
    print("=" * 78)

    base = roda(tarefas, D.CODECS, a.teto, "base")
    usadas = []
    ric = roda(tarefas, D.CODECS + [CorPorPropMais(), ApagaPorPropMais()],
               a.teto, "inventadas", usadas)
    pla = roda(tarefas, D.CODECS + [CorPorPropPlacebo(), ApagaPorPropPlacebo()],
               a.teto, "placebo")

    nov = [(t, c, p) for t, c, p in usadas if p in INVENTADAS]
    print("")
    print("=" * 78)
    print(f"  base         {len(base):3d}/{len(tarefas)}")
    print(f"  inventadas   {len(ric):3d}/{len(tarefas)}   "
          f"ganhou {len(ric-base)}, perdeu {len(base-ric)}")
    print(f"  placebo      {len(pla):3d}/{len(tarefas)}   "
          f"ganhou {len(pla-base)}, perdeu {len(base-pla)}")
    print(f"\n  propriedades que apareceram em pilha vencedora: "
          f"{Counter(p for _t, _c, p in usadas).most_common(10)}")
    print(f"  dessas, INVENTADAS: {sorted({p for _t, _c, p in nov})}")
    print("")
    print(f"Q1 {'.' if not (base - ric) else 'FALHOU .'} nao quebrou: "
          f"{len(base - ric)} perdas (0)")
    print(f"Q2 {'.' if len(ric)-len(base) >= 3 else 'FALHOU .'} ganha "
          f"{len(ric)-len(base):+d} (>= +3)")
    print(f"Q3 {'.' if len(pla)-len(base) <= 0 else 'FALHOU .'} placebo "
          f"{len(pla)-len(base):+d} (<= 0)")
    print(f"Q4 {'.' if nov else 'FALHOU .'} propriedade inventada pagou em "
          f"{len({t for t, _c, _p in nov})} tarefa(s) (>= 1)")


if __name__ == "__main__":
    main()
