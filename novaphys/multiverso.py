"""
novaphys.multiverso — a que ponto o mundo precisa ter estrutura para que
abstrair valha a pena?
========================================================================

CORRECAO DE UM ERRO MEU, ANTES DE QUALQUER NUMERO
---------------------------------------------------
A primeira versao deste arquivo gerava leis UNIFORMEMENTE ao acaso sobre 14
atomos, e o cabecalho afirmava que aquilo era "o cenario mais favoravel
possivel para biblioteca". Rodei, deu diluicao, e a afirmacao estava errada
pelo motivo oposto: leis uniformes sao o cenario MENOS favoravel que existe.

A conta que eu devia ter feito antes: um fragmento de 2 atomos cobre 1/196 do
espaco de pares. Oito fragmentos cobrem 4%. A chance de uma lei de
profundidade 4 se decompor em dois fragmentos DESSES oito e ~0,16%. A
biblioteca nao tinha o que descobrir porque eu nao coloquei nada la para
descobrir. Medir diluicao naquele desenho nao dizia nada sobre biblioteca —
dizia so que ruido nao comprime.

O EXPERIMENTO CERTO, QUE SO APARECEU POR CAUSA DO ERRO
--------------------------------------------------------
Se leis uniformes matam a biblioteca e leis totalmente estruturadas
obviamente a favorecem, entao existe um LIMIAR no meio. E a posicao desse
limiar e a pergunta interessante, porque ela explica todos os nulos que
tomamos nesta sessao:

    quanta estrutura recorrente um dominio precisa ter para que
    "abstrair do que ja resolveu" pague o proprio custo?

Isso e medivel. O gerador ganha um parametro `c` (concentracao):

    c = 0.0   cada posicao da lei e um atomo sorteado uniformemente
              — sem motivos, nada a abstrair
    c = 1.0   a lei e uma concatenacao de MOTIVOS de um repertorio pequeno
              — estrutura maxima, a biblioteca deveria varrer
    0 < c < 1 mistura: com prob. `c` a proxima peca vem do repertorio de
              motivos, senao e um atomo solto

Varrendo `c` acha-se o ponto de virada. E dai sai uma leitura que vale para
o ARC e para a fisica: se o limiar for alto, biblioteca so funciona em
dominios muito repetitivos, e os nulos de `arc_biblioteca` e `gargalo` deixam
de ser misteriosos e viram consequencia esperada.

O REPERTORIO DE MOTIVOS E ESCONDIDO DO SOLUCIONADOR
-----------------------------------------------------
O solucionador nunca ve a lista de motivos. Ele so ve pares (entrada, saida)
e tem que redescobrir os motivos a partir das solucoes que ele mesmo achou.
Se ele reconstruir o repertorio, isso aparece na taxa; e da para conferir
diretamente quantos motivos verdadeiros ele recuperou, o que e uma checagem
que o ARC nao permite porque la nao existe gabarito de motivo.

TRES BRACOS, ORCAMENTO CASADO
-------------------------------
    B0  vocabulario base, 14 atomos
    B1  base + moldes extraidos das solucoes de TREINO
    BP  base + moldes ALEATORIOS, pareados em quantidade e comprimento

Sem BP nao da para separar "abstrair funciona" de "ter mais simbolos
funciona" — `placebo.py` e `gargalo.py` ja mostraram que a segunda coisa
acontece sozinha. Os tres avaliam o MESMO numero de candidatos.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  M0  PRE-CONDICAO. Em prof 1 a taxa tem que ser ~100% e em c=0 os tres
      bracos tem que empatar dentro do IC. Se B1 ganhar em c=0, ha vazamento
      (motivo entrando por outro caminho) e nada abaixo pode ser lido.

  M1  existe c* onde B1 passa B0 em >= 10pp E passa BP em >= 10pp
      -> abstrair funciona ACIMA de um limiar de estrutura, e o limiar tem
         um valor que da para reportar

  M2  B1 nunca passa BP em nenhum c
      -> abstrair por frequencia nao funciona nem com estrutura garantida;
         o mecanismo esta errado, nao o dominio

  M3  B1 passa BP em todo c, inclusive c=0
      -> vazamento. Ver M0.

  M4  recuperacao de motivos alta mas taxa nao sobe
      -> o sistema ACHA a estrutura certa e mesmo assim nao lucra; o custo de
         carregar o molde na busca supera o ganho de profundidade

MINHA PREVISAO, escrita antes de rodar: espero M1 com c* alto, entre 0,6 e
0,8. Ou seja: biblioteca funciona, mas so em dominios bem mais repetitivos
que o ARC. Se c* sair baixo (< 0,3) eu estava errado sobre a causa dos nulos
anteriores e o problema era a minha extracao, nao o dominio.

Aposto que o mecanismo funciona e que o ARC simplesmente nao e repetitivo o
bastante. Essa e uma previsao que pode dar errado dos dois lados.
"""

from __future__ import annotations

import argparse
import itertools
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Grid = Tuple[Tuple[int, ...], ...]

LADO = 8
N_CORES = 6
PARES_TREINO = 4
PARES_TESTE = 2


# ══════════════════════════════════════════════ atomos
def a_rot90(g: Grid) -> Grid:
    return tuple(zip(*g[::-1]))                       # type: ignore[return-value]


def a_espH(g: Grid) -> Grid:
    return tuple(tuple(l[::-1]) for l in g)


def a_espV(g: Grid) -> Grid:
    return tuple(g[::-1])


def a_transp(g: Grid) -> Grid:
    return tuple(zip(*g))                             # type: ignore[return-value]


def _ciclo(k: int) -> Callable[[Grid], Grid]:
    def fn(g: Grid) -> Grid:
        return tuple(tuple((c + k) % N_CORES for c in l) for l in g)
    return fn


def _troca(a: int, b: int) -> Callable[[Grid], Grid]:
    def fn(g: Grid) -> Grid:
        return tuple(tuple(b if c == a else (a if c == b else c) for c in l)
                     for l in g)
    return fn


def _desl(dy: int, dx: int) -> Callable[[Grid], Grid]:
    def fn(g: Grid) -> Grid:
        h, w = len(g), len(g[0])
        return tuple(tuple(g[(i - dy) % h][(j - dx) % w] for j in range(w))
                     for i in range(h))
    return fn


def _mascara(cor: int) -> Callable[[Grid], Grid]:
    def fn(g: Grid) -> Grid:
        return tuple(tuple(0 if c == cor else c for c in l) for l in g)
    return fn


def _borda(cor: int) -> Callable[[Grid], Grid]:
    def fn(g: Grid) -> Grid:
        h, w = len(g), len(g[0])
        return tuple(tuple(cor if (i in (0, h - 1) or j in (0, w - 1))
                           else g[i][j] for j in range(w)) for i in range(h))
    return fn


def _diagonal(cor: int) -> Callable[[Grid], Grid]:
    def fn(g: Grid) -> Grid:
        return tuple(tuple(cor if i == j else g[i][j]
                           for j in range(len(g[0]))) for i in range(len(g)))
    return fn


ATOMOS: List[Tuple[str, Callable[[Grid], Grid]]] = [
    ("rot90", a_rot90), ("espH", a_espH), ("espV", a_espV),
    ("transp", a_transp),
    ("cic1", _ciclo(1)), ("cic2", _ciclo(2)),
    ("tr12", _troca(1, 2)), ("tr34", _troca(3, 4)),
    ("dl01", _desl(0, 1)), ("dl10", _desl(1, 0)),
    ("msk1", _mascara(1)), ("msk3", _mascara(3)),
    ("brd5", _borda(5)), ("dia2", _diagonal(2)),
]
DIC = dict(ATOMOS)
NOMES = [n for n, _ in ATOMOS]


# ══════════════════════════════════════════════ o mundo tem motivos
def sortear_motivos(n: int, comp: int, rng: random.Random) -> List[List[str]]:
    """
    O repertorio ESCONDIDO. O solucionador nunca ve isto — tem que
    redescobrir a partir das proprias solucoes.
    """
    vistos = set()
    fora = []
    while len(fora) < n:
        m = tuple(rng.choice(NOMES) for _ in range(comp))
        if m not in vistos:
            vistos.add(m)
            fora.append(list(m))
    return fora


def sortear_lei(prof: int, motivos: List[List[str]], c: float,
                rng: random.Random) -> List[str]:
    """
    Com probabilidade `c` a proxima peca da lei e um motivo inteiro; senao e
    um atomo solto. `c` e a concentracao de estrutura do mundo.
    """
    lei: List[str] = []
    while len(lei) < prof:
        if motivos and rng.random() < c and len(lei) + len(motivos[0]) <= prof:
            lei.extend(rng.choice(motivos))
        else:
            lei.append(rng.choice(NOMES))
    return lei[:prof]


def grade_aleatoria(rng: random.Random) -> Grid:
    return tuple(tuple(rng.randrange(N_CORES) for _ in range(LADO))
                 for _ in range(LADO))


def aplicar(lei: Sequence[str], g: Grid) -> Grid:
    for n in lei:
        g = DIC[n](g)
    return g


def universo(prof: int, motivos: List[List[str]], c: float,
             rng: random.Random
             ) -> Tuple[List[str], List[Tuple[Grid, Grid]],
                        List[Tuple[Grid, Grid]]]:
    """Lei escondida + exemplos. Leis degeneradas (saida==entrada) sao
    rejeitadas, senao o universo seria trivial e inflaria a taxa."""
    for _ in range(300):
        lei = sortear_lei(prof, motivos, c, rng)
        tr, te = [], []
        deg = False
        for i in range(PARES_TREINO + PARES_TESTE):
            a = grade_aleatoria(rng)
            b = aplicar(lei, a)
            if a == b:
                deg = True
                break
            (tr if i < PARES_TREINO else te).append((a, b))
        if not deg:
            return lei, tr, te
    raise RuntimeError("nao consegui gerar universo nao-degenerado")


# ══════════════════════════════════════════════ solucionador
def resolver(treino: Sequence[Tuple[Grid, Grid]], vocab: List[str],
             moldes: Dict[str, List[str]], orcamento: int,
             prof_max: int = 6) -> Tuple[Optional[List[str]], int]:
    """
    Busca por profundidade crescente. Um molde expande em varios atomos: e
    por isso que ele pode achar em profundidade 2 uma lei que a base so
    alcanca em profundidade 4. O contador conta PROGRAMAS COMPLETOS
    verificados — mesma unidade nos tres bracos.
    """
    gasto = 0
    for prof in range(1, prof_max + 1):
        for seq in itertools.product(vocab, repeat=prof):
            if gasto >= orcamento:
                return None, gasto
            gasto += 1
            plano: List[str] = []
            for s in seq:
                plano.extend(moldes.get(s, [s]))
            if all(aplicar(plano, a) == b for a, b in treino):
                return plano, gasto
    return None, gasto


# ══════════════════════════════════════════════ extracao
def extrair_moldes(solucoes: List[List[str]], n: int,
                   comp: int) -> Dict[str, List[str]]:
    cont: Dict[Tuple[str, ...], int] = {}
    for s in solucoes:
        for i in range(len(s) - comp + 1):
            f = tuple(s[i:i + comp])
            cont[f] = cont.get(f, 0) + 1
    top = sorted(cont, key=lambda f: -cont[f])[:n]
    return {f"M{i}": list(f) for i, f in enumerate(top)}


def moldes_placebo(n: int, comp: int,
                   rng: random.Random) -> Dict[str, List[str]]:
    """Pareado em quantidade e comprimento. So o CONTEUDO e aleatorio."""
    return {f"P{i}": [rng.choice(NOMES) for _ in range(comp)]
            for i in range(n)}


# ══════════════════════════════════════════════ medicao
def taxa(prof: int, n: int, motivos: List[List[str]], c: float,
         vocab: List[str], moldes: Dict[str, List[str]],
         orcamento: int, semente: int) -> float:
    rng = random.Random(semente)
    ok = 0
    for _ in range(n):
        _lei, tr, te = universo(prof, motivos, c, rng)
        p, _g = resolver(tr, vocab, moldes, orcamento)
        if p is not None and all(aplicar(p, a) == b for a, b in te):
            ok += 1
    return ok / n


def erro(p: float, n: int) -> float:
    return 1.96 * (p * (1 - p) / n) ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--treino", type=int, default=250)
    ap.add_argument("--orcamento", type=int, default=25000)
    ap.add_argument("--prof", type=int, default=5, help="prof. de TESTE")
    ap.add_argument("--prof-treino", type=int, default=3,
                    help="prof. de TREINO — tem que ser resolvivel pela BASE, "
                         "senao a fase de abstracao nao extrai nada")
    ap.add_argument("--motivos", type=int, default=6)
    ap.add_argument("--moldes", type=int, default=6)
    ap.add_argument("--semente", type=int, default=20260903)
    a = ap.parse_args()

    print("=" * 78)
    print("MULTIVERSO — quanta estrutura o mundo precisa ter para que")
    print("             abstrair do proprio trabalho valha a pena?")
    print("=" * 78)
    print(f"{len(NOMES)} atomos · TREINO prof {a.prof_treino} -> "
          f"TESTE prof {a.prof} · {a.motivos} motivos escondidos de 2 atomos")
    print(f"orcamento casado {a.orcamento} cand/universo · "
          f"{a.n} universos por ponto\n")

    motivos = sortear_motivos(a.motivos, 2, random.Random(a.semente + 11))
    print("repertorio ESCONDIDO (o solucionador nunca ve isto):")
    print("   " + " | ".join(" . ".join(m) for m in motivos) + "\n")

    print(f"{'c':>5} {'B0 base':>9} {'B1 moldes':>11} {'BP placebo':>11} "
          f"{'B1-B0':>8} {'B1-BP':>8} {'motivos':>9}  veredito")
    print("-" * 78)

    linhas = []
    for c in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = a.semente + int(c * 1000)

        # fase de abstracao: resolve universos de TREINO deste mundo
        # TREINO em leis curtas que a BASE resolve, do MESMO mundo (mesmos
        # motivos, mesmo c). Treinar em prof 5 nao funciona: a base nao
        # resolve, nao ha solucao de onde extrair, e a fase fica vazia.
        # Aprender no facil e transferir para o dificil e o desenho certo.
        rng = random.Random(s + 999)
        sol: List[List[str]] = []
        for _ in range(a.treino):
            _lei, tr, _te = universo(a.prof_treino, motivos, c, rng)
            p, _g = resolver(tr, NOMES, {}, a.orcamento)
            if p is not None:
                sol.append(p)
        taxa_tr = len(sol) / a.treino
        m1 = extrair_moldes(sol, a.moldes, 2)
        mp = moldes_placebo(a.moldes, 2, random.Random(s + 555))

        verdade = {tuple(m) for m in motivos}
        rec = len({tuple(v) for v in m1.values()} & verdade)

        t0 = taxa(a.prof, a.n, motivos, c, NOMES, {}, a.orcamento, s)
        t1 = taxa(a.prof, a.n, motivos, c, NOMES + list(m1), m1,
                  a.orcamento, s)
        tp = taxa(a.prof, a.n, motivos, c, NOMES + list(mp), mp,
                  a.orcamento, s)
        d10, d1p = (t1 - t0) * 100, (t1 - tp) * 100

        if d10 >= 10 and d1p >= 10:
            v = "abstrair PAGA"
        elif t1 < t0 - 5:
            v = "dilui"
        else:
            v = "nulo"
        linhas.append((c, t0, t1, tp, d10, d1p, rec, v))
        print(f"{c:>5.2f} {t0:>9.1%} {t1:>11.1%} {tp:>11.1%} "
              f"{d10:>+7.1f}p {d1p:>+7.1f}p {rec:>4d}/{a.motivos}  {v}"
              f"   (treino {taxa_tr:.0%} resolvido)")

    print("-" * 78)
    c0 = linhas[0]
    print(f"\nM0 · em c=0 (sem motivos) B1-BP = {c0[5]:+.1f}pp  "
          f"{'ok, sem vazamento' if abs(c0[5]) < 10 else 'VAZAMENTO — abortar leitura'}")

    paga = [l for l in linhas if l[7] == "abstrair PAGA"]
    if paga:
        print(f"M1 · limiar c* = {paga[0][0]:.2f}  "
              f"(abaixo disso abstrair nao paga)")
        print(f"     motivos verdadeiros recuperados ali: "
              f"{paga[0][6]}/{a.motivos}")
    else:
        altos = [l for l in linhas if l[6] >= a.motivos * 0.5]
        if altos:
            print("M4 · o sistema RECUPERA os motivos verdadeiros "
                  f"({altos[-1][6]}/{a.motivos} em c={altos[-1][0]:.2f}) "
                  "e mesmo assim")
            print("     a taxa nao sobe: carregar o molde na busca custa mais")
            print("     do que o ganho de profundidade")
        else:
            print("M2 · abstrair por frequencia nao paga em nenhum nivel de")
            print("     estrutura testado — o mecanismo esta errado, nao o mundo")


if __name__ == "__main__":
    main()
