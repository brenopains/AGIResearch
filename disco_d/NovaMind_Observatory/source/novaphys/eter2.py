"""
novaphys.eter2 — o gerador gera os PROPRIOS operadores
========================================================

A CRITICA QUE ORIGINA ESTE ARQUIVO, E ELA ESTA CERTA
------------------------------------------------------
"As primitivas estao especializadas na dimensao do ARC somente. Uma AGI nunca
vai nascer sendo moldada a mao."

Procede, e o `eter.py` de hoje nao escapa disso. Ele aterra sob demanda, mas
as FAMILIAS (recolore, escala, preenche, desloca...) fui eu que escrevi. O
espaco e infinito em argumentos e finito em ideias — e as ideias sao minhas.
O mesmo vale para os 14 atomos do `multiverso.py`.

O QUE MUDA AQUI
-----------------
Nada de lista de familias. Um operador e um PROGRAMA sorteado de uma
meta-gramatica que descreve como uma celula pode depender da vizinhanca e da
posicao. Da meta-gramatica saem, sem eu nomear nenhuma:

    reflexoes, rotacoes, translacoes         (leitura reposicionada)
    automatos celulares                       (leitura da vizinhanca)
    mascaras e recoloracoes condicionais      (teste sobre a cor)
    listras, xadrezes, gradientes, diagonais  (teste sobre a posicao)
    e muita coisa sem nome, que e o ponto

    celula(i,j) = ESCOLHA( teste, ramo_verdadeiro, ramo_falso )
    teste       = comparacao entre duas LEITURAS, ou paridade de posicao
    leitura     = grade[ POS ] | constante
    POS         = (i,j) transformado por uma matriz inteira pequena + deslocamento

A matriz inteira 2x2 com entradas em {-1,0,1} ja contem identidade, as quatro
rotacoes, as reflexoes, transposicao, projecoes e cisalhamentos — sem que eu
escreva "rot90" em lugar nenhum. Elas EMERGEM do sorteio.

A PARTE PSICODELICA, NO SENTIDO LITERAL E NAO DECORATIVO
----------------------------------------------------------
O que a psicodelia faz, mecanicamente, e afrouxar o filtro que descarta
interpretacoes improvaveis. Aqui isso tem forma exata: a meta-gramatica NAO
tem prior de plausibilidade. Ela sorteia matrizes degeneradas, testes que
nunca sao verdadeiros, composicoes que colapsam tudo numa cor. A maioria e
lixo — e o lixo tem que passar, porque um filtro de "razoavel" seria o meu
gosto entrando pela porta dos fundos, que e exatamente o que se quer evitar.

O verificador continua incorruptivel. Gerar sem filtro nao afrouxa a
aceitacao: afrouxa so o que chega a ser considerado.

O QUE ISTO PODE E O QUE NAO PODE PROVAR
-----------------------------------------
NAO prova que o sistema "cria dominios novos". Prova ou refuta uma coisa
menor e verificavel: operadores que eu nao escolhi resolvem tarefas que os
que eu escolhi nao resolvem?

Se sim, a minha escolha manual era um teto, e da para subir sem me consultar.
Se nao, o problema nao esta em quem escolhe as primitivas — e a honestidade
manda dizer isso e parar de culpar o desenho manual.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  G0  PRE-CONDICAO. A meta-gramatica tem que REDESCOBRIR primitivas
      conhecidas. Se entre milhares de sorteios nao aparecer nenhuma que se
      comporte como rot90 / espelho / identidade nas grades de teste, o
      gerador nao alcanca nem o que ja temos, e o resto nao vale ler.

  G1  operadores sorteados resolvem >= 3 tarefas do ARC que a base de 24
      primitivas NAO resolve, com orcamento casado
      -> a escolha manual era teto, e da para subir sem mim

  G2  resolvem as mesmas, ou menos
      -> o gargalo nao e quem escolhe as primitivas

  G3  o sorteio nem reproduz a base (G0 falha)
      -> a meta-gramatica esta mal desenhada; consertar antes de concluir

MINHA PREVISAO, escrita antes de rodar: espero G2, e por um motivo concreto
que os numeros de hoje deram. 33 das 51 solucoes do ARC tem UM passo. O que
falta nao e vocabulario — ja mostrei hoje que ampliar de 24 para 80
operadores rendeu +1 tarefa liquida. Um vocabulario sorteado e maior ainda, e
com orcamento fixo isso dilui mais.

Mas ha uma razao real para rodar mesmo assim: um operador sorteado pode
resolver em UM passo o que a base so faria em quatro, e e exatamente esse
caso que os numeros de hoje nao conseguem excluir. E o unico jeito de saber
qual das duas forcas ganha e medindo.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from novaphys.arc import (Grid, PRIMITIVAS, ajustar_mapa, aplicar_mapa,
                          carregar, dims, fundo, valida)
from novaphys.eter import PASTA_AVAL

MATS = [(a, b, c, d)
        for a in (-1, 0, 1) for b in (-1, 0, 1)
        for c in (-1, 0, 1) for d in (-1, 0, 1)]


# ══════════════════════════════════════════════ meta-gramatica
class Leitura:
    """De onde vem um valor: da grade numa posicao transformada, ou constante."""

    def __init__(self, rng: random.Random, cores: Sequence[int]):
        self.tipo = rng.choice(["grade", "grade", "grade", "const"])
        if self.tipo == "const":
            self.k = rng.choice(list(cores))
        else:
            self.m = rng.choice(MATS)
            self.dy = rng.randint(-1, 1)
            self.dx = rng.randint(-1, 1)
            self.borda = rng.choice(["fundo", "toro", "recorta"])

    def valor(self, g: Grid, i: int, j: int, h: int, w: int,
              bg: int) -> Optional[int]:
        if self.tipo == "const":
            return self.k
        a, b, c, d = self.m
        y = a * i + b * j + self.dy
        x = c * i + d * j + self.dx
        if self.borda == "toro":
            return g[y % h][x % w]
        if 0 <= y < h and 0 <= x < w:
            return g[y][x]
        return bg if self.borda == "fundo" else None

    def __str__(self) -> str:
        if self.tipo == "const":
            return f"={self.k}"
        a, b, c, d = self.m
        return (f"g[{a}i{b:+d}j{self.dy:+d}, {c}i{d:+d}j{self.dx:+d}]"
                f"·{self.borda[:3]}")


class Teste:
    """Uma condicao booleana sobre a celula: compara leituras, ou olha a posicao."""

    def __init__(self, rng: random.Random, cores: Sequence[int]):
        self.tipo = rng.choice(["igual", "igual", "difere", "paridade",
                                "borda", "sempre"])
        if self.tipo in ("igual", "difere"):
            self.a = Leitura(rng, cores)
            self.b = Leitura(rng, cores)
        elif self.tipo == "paridade":
            self.p = rng.choice(["i", "j", "i+j", "i-j"])
            self.mod = rng.choice([2, 2, 3])
            self.r = rng.randrange(self.mod)

    def vale(self, g: Grid, i: int, j: int, h: int, w: int, bg: int) -> bool:
        if self.tipo == "sempre":
            return True
        if self.tipo == "borda":
            return i in (0, h - 1) or j in (0, w - 1)
        if self.tipo == "paridade":
            v = {"i": i, "j": j, "i+j": i + j, "i-j": i - j}[self.p]
            return v % self.mod == self.r
        va = self.a.valor(g, i, j, h, w, bg)
        vb = self.b.valor(g, i, j, h, w, bg)
        if va is None or vb is None:
            return False
        return (va == vb) if self.tipo == "igual" else (va != vb)

    def __str__(self) -> str:
        if self.tipo == "sempre":
            return "T"
        if self.tipo == "borda":
            return "borda"
        if self.tipo == "paridade":
            return f"({self.p})%{self.mod}=={self.r}"
        s = "==" if self.tipo == "igual" else "!="
        return f"{self.a}{s}{self.b}"


class Operador:
    """celula(i,j) = teste ? ramo_v : ramo_f. Nenhum nome foi dado por mim."""

    def __init__(self, rng: random.Random, cores: Sequence[int]):
        self.t = Teste(rng, cores)
        self.v = Leitura(rng, cores)
        self.f = Leitura(rng, cores)

    def aplicar(self, g: Grid, bg: int) -> Optional[Grid]:
        h, w = dims(g)
        fora = []
        for i in range(h):
            linha = []
            for j in range(w):
                l = self.v if self.t.vale(g, i, j, h, w, bg) else self.f
                x = l.valor(g, i, j, h, w, bg)
                if x is None:
                    return None
                linha.append(x)
            fora.append(tuple(linha))
        return tuple(fora)

    def __str__(self) -> str:
        return f"[{self.t} ? {self.v} : {self.f}]"


def sortear(n: int, cores: Sequence[int], semente: int) -> List[Operador]:
    rng = random.Random(semente)
    return [Operador(rng, cores) for _ in range(n)]


# ══════════════════════════════════════════════ G0: redescobre o conhecido?
def assinatura(fn: Callable[[Grid], Optional[Grid]],
               provas: Sequence[Grid]) -> Optional[Tuple]:
    """Comportamento observavel numa bateria fixa de grades."""
    fora = []
    for g in provas:
        r = fn(g)
        if r is None or not valida(r):
            return None
        fora.append(r)
    return tuple(fora)


def gate_g0(n_sorteios: int, semente: int) -> Dict[str, int]:
    rng = random.Random(semente + 7)
    provas = []
    for _ in range(6):
        h, w = rng.randint(3, 5), rng.randint(3, 5)
        provas.append(tuple(tuple(rng.randrange(5) for _ in range(w))
                            for _ in range(h)))

    alvo: Dict[Tuple, str] = {}
    for nome, f in PRIMITIVAS:
        a = assinatura(f, provas)
        if a is not None:
            alvo.setdefault(a, nome)

    achados: Dict[str, int] = {}
    ops = sortear(n_sorteios, range(10), semente)
    for o in ops:
        a = assinatura(lambda g, o=o: o.aplicar(g, fundo(g)), provas)
        if a is not None and a in alvo:
            achados[alvo[a]] = achados.get(alvo[a], 0) + 1
    return achados


# ══════════════════════════════════════════════ peneira pela TAREFA
def peneirar(cand: List["Operador"], treino, quantos: int
             ) -> Tuple[List["Operador"], int]:
    """
    G0 mediu a densidade de operador util no sorteio cego: ~1 em 3000. Com
    orcamento para ~120 operadores no vocabulario, o numero esperado de uteis
    entre eles e 0,04. Sortear e buscar direto e aritmeticamente condenado.

    A saida NAO e reintroduzir o meu gosto ("parece uma primitiva decente").
    E filtrar pelo que a TAREFA responde. Um operador sobrevive a peneira se,
    nas grades de treino desta tarefa, ele:

        produz grade valida
        nao devolve a entrada intacta      (nao e identidade disfarcada)
        nao colapsa tudo numa cor          (nao destroi informacao)
        preserva as dimensoes, se a tarefa preserva

    Nada disso e opiniao minha sobre o que e um bom operador: sao propriedades
    medidas contra os dados da tarefa. E o mesmo movimento do `eter.py` —
    aterrar sob demanda — agora sobre um espaco que eu nao escrevi.

    CUSTO: cada teste aplica o operador uma vez, o mesmo trabalho de avaliar
    um candidato. Por isso a peneira e COBRADA do orcamento e devolvida no
    contador. Peneirar de graca seria a fraude que o `gargalo.py` pega.
    """
    a0, b0 = treino[0]
    bg = fundo(a0)
    mesma_forma = all(dims(x) == dims(y) for x, y in treino)
    vivos, gasto = [], 0
    for o in cand:
        gasto += 1
        r = o.aplicar(a0, bg)
        if r is None or not valida(r):
            continue
        if r == a0:
            continue
        if len({c for l in r for c in l}) < 2:
            continue
        if mesma_forma and dims(r) != dims(a0):
            continue
        vivos.append(o)
        if len(vivos) >= quantos:
            break
    return vivos, gasto


# ══════════════════════════════════════════════ busca
def resolver(treino: Sequence[Tuple[Grid, Grid]], ops: List[Operador],
             orcamento: int, prof_max: int = 2
             ) -> Tuple[Optional[List[int]], int]:
    """
    Cadeias de operadores SORTEADOS. Profundidade 2 porque o vocabulario e
    grande; o orcamento e o mesmo dos outros bracos e e ele que manda.
    """
    bg = fundo(treino[0][0])
    gasto = 0

    # profundidade 1
    for k, o in enumerate(ops):
        if gasto >= orcamento:
            return None, gasto
        gasto += 1
        saidas = []
        ok = True
        for a, b in treino:
            r = o.aplicar(a, bg)
            if r is None or not valida(r):
                ok = False
                break
            saidas.append(r)
        if not ok:
            continue
        if all(r == b for r, (_, b) in zip(saidas, treino)):
            return [k], gasto
        mapa = ajustar_mapa(list(zip(saidas, [b for _, b in treino])))
        if mapa is not None and all(
                aplicar_mapa(r, mapa) == b
                for r, (_, b) in zip(saidas, treino)):
            return [k, -1], gasto

    # profundidade 2
    for k1, o1 in enumerate(ops):
        pre = []
        bom = True
        for a, b in treino:
            r = o1.aplicar(a, bg)
            if r is None or not valida(r):
                bom = False
                break
            pre.append(r)
        if not bom:
            continue
        for k2, o2 in enumerate(ops):
            if gasto >= orcamento:
                return None, gasto
            gasto += 1
            saidas = []
            ok = True
            for r0 in pre:
                r = o2.aplicar(r0, bg)
                if r is None or not valida(r):
                    ok = False
                    break
                saidas.append(r)
            if not ok:
                continue
            if all(r == b for r, (_, b) in zip(saidas, treino)):
                return [k1, k2], gasto
    return None, gasto


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--ops", type=int, default=85,
                    help="operadores que sobrevivem a peneira")
    ap.add_argument("--sorteados", type=int, default=7000,
                    help="sorteados antes da peneira; custa do orcamento")
    ap.add_argument("--orcamento", type=int, default=14424)
    ap.add_argument("--semente", type=int, default=20260903)
    ap.add_argument("--sementes", type=int, default=3)
    a = ap.parse_args()

    print("=" * 78)
    print("ETER 2 — o gerador gera os proprios operadores")
    print("=" * 78)

    print("\nG0 · PRE-CONDICAO — o sorteio redescobre primitivas conhecidas?")
    ach = gate_g0(4000, a.semente)
    if not ach:
        print("     G3 · o sorteio nao reproduz NENHUMA primitiva conhecida.")
        print("     A meta-gramatica esta mal desenhada. Nao ler o resto.")
        return
    print(f"     em 4000 sorteios, reencontrou {len(ach)} primitivas por "
          f"COMPORTAMENTO,")
    print("     sem que eu nomeasse nenhuma:")
    for n, c in sorted(ach.items(), key=lambda x: -x[1]):
        print(f"       {n:12s} {c:4d}x")
    print("     G0 · passa")

    tarefas = carregar(a.pasta)
    base_ids = set()
    try:
        ev = json.load(open("eter_resultado.json"))
        base_ids = {x.split("|")[0] for x in ev["base"]["acertos"]}
        print(f"\n(base de 24 primitivas resolve {len(base_ids)} destas tarefas)")
    except Exception:
        print("\n(sem eter_resultado.json — sem comparacao com a base)")

    print(f"\nG1 · {a.sementes} sorteios independentes de {a.ops} operadores, "
          f"orcamento {a.orcamento}")
    print(f"{'semente':>9} {'resolve':>9} {'novas vs base':>15}   tarefas novas")
    todas_novas = set()
    for s in range(a.sementes):
        cand = sortear(a.sorteados, range(10), a.semente + s * 101)
        t0 = time.time()
        ok = set()
        n_vivos = []
        for nome, tr, te in tarefas:
            ops, g_pen = peneirar(cand, tr, a.ops)
            n_vivos.append(len(ops))
            if not ops:
                continue
            seq, _g = resolver(tr, ops, a.orcamento - g_pen)
            if seq is None:
                continue
            bg = fundo(tr[0][0])

            def apl(g, seq=seq, ops=ops, bg=bg, tr=tr):
                r = g
                for k in seq:
                    if k == -1:
                        saidas = [ops[seq[0]].aplicar(x, bg) for x, _ in tr]
                        m = ajustar_mapa(list(zip(saidas,
                                                  [y for _, y in tr])))
                        return aplicar_mapa(r, m) if m else None
                    r = ops[k].aplicar(r, bg)
                    if r is None or not valida(r):
                        return None
                return r
            if all(apl(x) == y for x, y in te):
                ok.add(nome)
        novas = ok - base_ids
        todas_novas |= novas
        med = sum(n_vivos) / len(n_vivos) if n_vivos else 0
        print(f"{a.semente + s * 101:>9} {med:>9.1f} {len(ok):>9} "
              f"{len(novas):>15}   {sorted(novas)[:5]}  "
              f"({time.time()-t0:.0f}s)")

    print(f"\nuniao das novas em {a.sementes} sorteios: {len(todas_novas)}  "
          f"{sorted(todas_novas)}")
    if len(todas_novas) >= 3:
        print("G1 · operadores que eu NAO escolhi resolvem o que os meus nao")
        print("     resolvem. A escolha manual era teto.")
    else:
        print("G2 · o gargalo nao e quem escolhe as primitivas. Sortear o")
        print("     vocabulario nao passa do que a escolha manual alcanca.")


if __name__ == "__main__":
    main()
