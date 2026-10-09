"""
novaphys.eter — operadores parametrizados aterrados sob demanda
================================================================

O DIAGNOSTICO QUE MOTIVA ESTE ARQUIVO
--------------------------------------
As 24 primitivas de `arc.py` sao TODAS `Grid -> Grid` sem argumento. Nenhuma
aceita "qual cor", "quanto deslocar", "qual fator de escala", "qual objeto".
A DSL nao consegue nem enunciar "pinte o objeto azul de vermelho" — nao e que
a busca nao acha, e que o programa NAO EXISTE na gramatica.

Isso e o teto, e nao adianta aumentar profundidade: profundidade 4 sobre um
vocabulario que nao expressa a operacao continua nao expressando.

O ETER
-------
O espaco de operadores aqui e INFINITO e nao ha lista fixa. Cada familia e um
molde com buracos:

    recolore(de, para)      desloca(dy, dx)      escala(fh, fw)
    mantem(cor)             remove(cor)          recortaCor(cor)
    obj(k)                  preenche(cor)        reduz(fh, fw)

Uma familia so vira operador CONCRETO quando a tarefa diz quais argumentos
fazem sentido. Isso e o "buscar sob demanda dentro do eter":

    aterrar(tarefa) -> lista de operadores concretos para ESTA tarefa

E o dominio nao e chutado, e DEDUZIDO dos pares de treino:

    escala(fh,fw)   nao e enumerado. As dimensoes de saida dividem as de
                    entrada e o fator sai unico, ou a familia nem e criada.
    recolore(a,b)   `a` percorre so as cores que EXISTEM nas entradas, `b` so
                    as que existem nas saidas. Cor ausente nao gera candidato.

Por isso o espaco e infinito mas o aterramento e pequeno: a tarefa poda o
eter antes da busca comecar. O tamanho do vocabulario passa a ser uma FUNCAO
DA TAREFA, nao uma constante minha.

CONTABILIDADE DE ORCAMENTO — a peca que ja me enganou antes
-------------------------------------------------------------
`gargalo.py` estabeleceu que ampliar vocabulario vence por construcao se o
orcamento nao for casado. Aqui e pior que o normal: o eter aterra um
vocabulario MAIOR, entao a profundidade 2 dele ja avalia mais candidatos que
a profundidade 2 da base.

Regra desta medicao: os dois bracos avaliam EXATAMENTE o mesmo numero de
programas completos. O orcamento e contado em candidatos verificados contra
os pares de treino, nao em profundidade. Se o eter vencer, venceu escolhendo
melhor os mesmos K candidatos — nao olhando mais.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  E0  PRE-CONDICAO. Com as familias parametrizadas desligadas, o eter tem que
      reproduzir o numero da base. Se nao reproduzir, ha bug no arcabouco e o
      resto nao vale nada. Se E0 falhar, nao interpretar o resto.

  E1  eter >= base + 4 tarefas com orcamento casado
      -> a parametrizacao era a peca que faltava; o teto era expressividade

  E2  eter ~ base (diferenca < 4)
      -> parametros nao sao o gargalo; o teto esta em outro lugar

  E3  eter < base
      -> a explosao de argumentos dilui a busca; com orcamento fixo, aterrar
         mais operadores EMPURRA para fora candidatos que resolviam

MINHA PREVISAO, escrita antes de rodar: espero E1, ganho entre 4 e 15
tarefas. A razao concreta e `escala`/`reduz`: as dimensoes de saida deduzem o
fator sem custo de busca, e ha uma fatia de ARC que e so escalar com fator
irregular que a base nao alcanca.

Sobre `recolore` estou cetico: `ajustar_mapa` ja resolve recoloracao global
no fim de cada candidato. Se o ganho vier todo de recolore, e sinal de que
estou medindo redundancia e nao expressividade nova.

Risco que assumo: E3 e desfecho real e nao seria surpresa. Aterrar ~60
operadores em vez de 24 corta a profundidade alcancavel de 3 para 2 com o
mesmo orcamento, e se as solucoes da base moravam na profundidade 3, o eter
PERDE tarefas que a base ganhava. Esse e exatamente o risco que o `gargalo`
mediu em fisica, agora em grade.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from novaphys.arc import (Grid, PRIMITIVAS, ajustar_mapa, aplicar_mapa,
                          carregar, dims, fundo, valida)

OpConcreto = Tuple[str, Callable[[Grid], Optional[Grid]]]

PASTA_AVAL = (('C:/Users/usuario/Documents/AGI 2.0/backup_novamind_20260627_170507/data/ARC/data/evaluation').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))


# ══════════════════════════════════════════════ contexto deduzido da tarefa
class Contexto:
    """O que a tarefa revela sobre quais argumentos fazem sentido."""

    def __init__(self, treino: Sequence[Tuple[Grid, Grid]]):
        self.treino = list(treino)
        ent: set = set()
        sai: set = set()
        for a, b in treino:
            for l in a:
                ent.update(l)
            for l in b:
                sai.update(l)
        self.cores_ent = sorted(ent)
        self.cores_sai = sorted(sai)
        self.fundo = fundo(treino[0][0])

        self.escalas = self._razao(lambda ha, wa, hb, wb:
                                   (hb // ha, wb // wa)
                                   if ha and wa and hb % ha == 0
                                   and wb % wa == 0 else None)
        self.reducoes = self._razao(lambda ha, wa, hb, wb:
                                    (ha // hb, wa // wb)
                                    if hb and wb and ha % hb == 0
                                    and wa % wb == 0 else None)

    def _razao(self, f) -> List[Tuple[int, int]]:
        """Fator deduzido dos pares. So instancia se for o MESMO em todos."""
        vistos = set()
        for a, b in self.treino:
            ha, wa = dims(a)
            hb, wb = dims(b)
            vistos.add(f(ha, wa, hb, wb))
        if len(vistos) == 1:
            r = vistos.pop()
            if r is not None and r != (1, 1):
                return [r]
        return []


# ══════════════════════════════════════════════ familias do eter
def _objetos(g: Grid, cor_fundo: int) -> List[List[Tuple[int, int]]]:
    """Componentes conexos 4-vizinhos que nao sao fundo."""
    h, w = dims(g)
    visto = [[False] * w for _ in range(h)]
    fora: List[List[Tuple[int, int]]] = []
    for i in range(h):
        for j in range(w):
            if visto[i][j] or g[i][j] == cor_fundo:
                continue
            pilha = [(i, j)]
            visto[i][j] = True
            comp = []
            while pilha:
                y, x = pilha.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w and not visto[ny][nx]
                            and g[ny][nx] != cor_fundo):
                        visto[ny][nx] = True
                        pilha.append((ny, nx))
            fora.append(comp)
    return fora


def _caixa(g: Grid, celulas: Sequence[Tuple[int, int]]) -> Optional[Grid]:
    if not celulas:
        return None
    ys = [y for y, _ in celulas]
    xs = [x for _, x in celulas]
    return tuple(tuple(g[y][x] for x in range(min(xs), max(xs) + 1))
                 for y in range(min(ys), max(ys) + 1))


def f_recolore(de: int, para: int):
    def fn(g: Grid) -> Optional[Grid]:
        return tuple(tuple(para if c == de else c for c in l) for l in g)
    return fn


def f_mantem(cor: int, bg: int):
    def fn(g: Grid) -> Optional[Grid]:
        return tuple(tuple(c if c == cor else bg for c in l) for l in g)
    return fn


def f_remove(cor: int, bg: int):
    def fn(g: Grid) -> Optional[Grid]:
        return tuple(tuple(bg if c == cor else c for c in l) for l in g)
    return fn


def f_recorta_cor(cor: int):
    def fn(g: Grid) -> Optional[Grid]:
        cel = [(y, x) for y, l in enumerate(g) for x, c in enumerate(l)
               if c == cor]
        return _caixa(g, cel)
    return fn


def f_escala(fh: int, fw: int):
    def fn(g: Grid) -> Optional[Grid]:
        fora = []
        for l in g:
            linha = tuple(c for c in l for _ in range(fw))
            for _ in range(fh):
                fora.append(linha)
        return tuple(fora)
    return fn


def f_reduz(fh: int, fw: int):
    """Reduz por blocos, exigindo que cada bloco seja de cor unica."""
    def fn(g: Grid) -> Optional[Grid]:
        h, w = dims(g)
        if h % fh or w % fw:
            return None
        fora = []
        for i in range(0, h, fh):
            linha = []
            for j in range(0, w, fw):
                bloco = {g[i + a][j + b] for a in range(fh) for b in range(fw)}
                if len(bloco) != 1:
                    return None
                linha.append(bloco.pop())
            fora.append(tuple(linha))
        return tuple(fora)
    return fn


def f_obj(k: int, bg: int):
    """k-esimo maior objeto, recortado na caixa."""
    def fn(g: Grid) -> Optional[Grid]:
        objs = _objetos(g, bg)
        if len(objs) <= k:
            return None
        objs.sort(key=len, reverse=True)
        return _caixa(g, objs[k])
    return fn


def f_obj_raro(bg: int):
    """O objeto cuja COR e a mais rara — 'ache o diferente', comum no ARC."""
    def fn(g: Grid) -> Optional[Grid]:
        objs = _objetos(g, bg)
        if not objs:
            return None
        cont: Dict[int, int] = {}
        for o in objs:
            c = g[o[0][0]][o[0][1]]
            cont[c] = cont.get(c, 0) + 1
        alvo = min(cont, key=lambda c: cont[c])
        for o in objs:
            if g[o[0][0]][o[0][1]] == alvo:
                return _caixa(g, o)
        return None
    return fn


def f_preenche(cor: int, bg: int):
    """Pinta de `cor` as regioes de fundo que NAO tocam a borda."""
    def fn(g: Grid) -> Optional[Grid]:
        h, w = dims(g)
        livre = [[g[i][j] == bg for j in range(w)] for i in range(h)]
        pilha = [(i, j) for i in range(h) for j in range(w)
                 if livre[i][j] and (i in (0, h - 1) or j in (0, w - 1))]
        for i, j in pilha:
            livre[i][j] = False
        while pilha:
            y, x = pilha.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and livre[ny][nx]:
                    livre[ny][nx] = False
                    pilha.append((ny, nx))
        return tuple(tuple(cor if (g[i][j] == bg and livre[i][j])
                           else g[i][j] for j in range(w)) for i in range(h))
    return fn


def f_desloca(dy: int, dx: int, bg: int):
    def fn(g: Grid) -> Optional[Grid]:
        h, w = dims(g)
        fora = [[bg] * w for _ in range(h)]
        for i in range(h):
            for j in range(w):
                ni, nj = i + dy, j + dx
                if 0 <= ni < h and 0 <= nj < w:
                    fora[ni][nj] = g[i][j]
        return tuple(tuple(l) for l in fora)
    return fn


# ══════════════════════════════════════════════ aterramento
def aterrar(ctx: Contexto, parametrizadas: bool = True) -> List[OpConcreto]:
    """
    Instancia o eter para ESTA tarefa. E aqui que o espaco infinito vira uma
    lista finita — e a tarefa que decide o tamanho dela, nao eu.
    """
    ops: List[OpConcreto] = list(PRIMITIVAS)
    if not parametrizadas:
        return ops                                  # braco BASE / gate E0

    bg = ctx.fundo

    for fh, fw in ctx.escalas:
        ops.append((f"escala{fh}x{fw}", f_escala(fh, fw)))
    for fh, fw in ctx.reducoes:
        ops.append((f"reduz{fh}x{fw}", f_reduz(fh, fw)))

    for c in ctx.cores_ent:
        if c != bg:
            ops.append((f"mantem{c}", f_mantem(c, bg)))
            ops.append((f"remove{c}", f_remove(c, bg)))
            ops.append((f"recorta{c}", f_recorta_cor(c)))

    for de in ctx.cores_ent:
        for para in ctx.cores_sai:
            if de != para:
                ops.append((f"rec{de}p{para}", f_recolore(de, para)))

    for c in ctx.cores_sai:
        ops.append((f"preenche{c}", f_preenche(c, bg)))

    for k in range(3):
        ops.append((f"obj{k}", f_obj(k, bg)))
    ops.append(("objRaro", f_obj_raro(bg)))

    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        ops.append((f"desl{dy}_{dx}", f_desloca(dy, dx, bg)))

    return ops


# ══════════════════════════════════════════════ busca com orcamento casado
def resolver(treino: Sequence[Tuple[Grid, Grid]], orcamento: int,
             parametrizadas: bool = True,
             prof_max: int = 3) -> Tuple[Optional[List[str]],
                                         Optional[Dict[int, int]], int, int]:
    """
    Enumera por profundidade crescente ate gastar `orcamento` candidatos.
    Devolve (operacoes, mapa de cor final ou None, candidatos gastos,
    tamanho do vocabulario aterrado).

    O contador incrementa por PROGRAMA COMPLETO verificado contra os pares —
    mesma unidade nos dois bracos, senao a comparacao nao vale nada.
    """
    ctx = Contexto(treino)
    ops = aterrar(ctx, parametrizadas)
    nomes = [n for n, _ in ops]
    d = dict(ops)
    gasto = 0

    for prof in range(0, prof_max + 1):
        for seq in itertools.product(nomes, repeat=prof):
            if gasto >= orcamento:
                return None, None, gasto, len(nomes)
            gasto += 1
            saidas = []
            ok = True
            for a, b in treino:
                r: Optional[Grid] = a
                for nome in seq:
                    r = d[nome](r)                  # type: ignore[arg-type]
                    if not valida(r):
                        ok = False
                        break
                if not ok:
                    break
                saidas.append(r)
            if not ok:
                continue
            if all(r == b for r, (_, b) in zip(saidas, treino)):
                return list(seq), None, gasto, len(nomes)
            mapa = ajustar_mapa(list(zip(saidas, [b for _, b in treino])))
            if mapa is not None:
                if all(aplicar_mapa(r, mapa) == b       # type: ignore[arg-type]
                       for r, (_, b) in zip(saidas, treino)):
                    return list(seq), mapa, gasto, len(nomes)
    return None, None, gasto, len(nomes)


def aplicar_seq(seq: Sequence[str], mapa: Optional[Dict[int, int]], g: Grid,
                d: Dict[str, Callable]) -> Optional[Grid]:
    r: Optional[Grid] = g
    for nome in seq:
        if nome not in d:
            return None
        r = d[nome](r)
        if not valida(r):
            return None
    if mapa is not None:
        return aplicar_mapa(r, mapa)                 # type: ignore[arg-type]
    return r


# ══════════════════════════════════════════════ avaliacao
def avaliar(pasta: str, orcamento: int, parametrizadas: bool,
            rotulo: str) -> Dict[str, object]:
    tarefas = carregar(pasta)
    t0 = time.time()
    acertos: List[str] = []
    gasto_total = 0
    vocab_total = 0
    for nome, tr, te in tarefas:
        seq, mapa, gasto, nv = resolver(tr, orcamento, parametrizadas)
        gasto_total += gasto
        vocab_total += nv
        if seq is None:
            continue
        d = dict(aterrar(Contexto(tr), parametrizadas))
        if all(aplicar_seq(seq, mapa, a, d) == b for a, b in te):
            txt = " . ".join(seq) or "id"
            if mapa:
                txt += f" +recolore{mapa}"
            acertos.append(f"{nome}|{txt}")
    dt = time.time() - t0
    n = len(tarefas)
    print(f"  {rotulo:6s} {len(acertos):3d}/{n}  ({len(acertos)/n:6.2%})   "
          f"vocab medio {vocab_total/n:5.1f}   "
          f"{gasto_total/n:8.0f} cand/tarefa   {dt:5.0f}s")
    return {"rotulo": rotulo, "acertos": acertos, "n": n,
            "gasto": gasto_total, "vocab": vocab_total / n, "segundos": dt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--orcamento", type=int, default=14424)
    a = ap.parse_args()

    print("=" * 78)
    print("ETER — operadores parametrizados aterrados sob demanda")
    print("=" * 78)
    print(f"orcamento casado: {a.orcamento} candidatos/tarefa nos DOIS bracos")
    print("(14424 = 24 + 24^2 + 24^3 = profundidade 3 completa da base)\n")

    base = avaliar(a.pasta, a.orcamento, False, "BASE")
    eter = avaliar(a.pasta, a.orcamento, True, "ETER")

    sb = {x.split("|")[0] for x in base["acertos"]}    # type: ignore[union-attr]
    se = {x.split("|")[0] for x in eter["acertos"]}    # type: ignore[union-attr]
    d = len(se) - len(sb)

    print("\n" + "=" * 78)
    print(f"so a BASE : {len(sb - se):2d}  {sorted(sb - se)}")
    print(f"so o ETER : {len(se - sb):2d}  {sorted(se - sb)}")
    print(f"ambos     : {len(sb & se):2d}")
    print(f"\ndelta = {d:+d} tarefas")
    if d >= 4:
        print("E1 · a parametrizacao era a peca que faltava")
    elif d <= -1:
        print("E3 · dilui — aterrar mais operadores empurrou solucoes para fora")
    else:
        print("E2 · parametros nao sao o gargalo; o teto esta em outro lugar")

    print("\nprogramas novos que o ETER achou:")
    for x in sorted(eter["acertos"]):                  # type: ignore[union-attr]
        if x.split("|")[0] in se - sb:
            print("   " + x.replace("|", "  ->  "))

    with open("eter_resultado.json", "w") as f:
        json.dump({"base": base, "eter": eter}, f, indent=1)


if __name__ == "__main__":
    main()
