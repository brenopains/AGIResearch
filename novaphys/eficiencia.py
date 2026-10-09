"""
novaphys.eficiencia — abstracao ou parametros: quem generaliza com menos dado?
==============================================================================

A AFIRMACAO SOB TESTE
-----------------------
"Obter mais capacidade por unidade de dado gracas a abstracao e equivalencia,
em vez de comprimir tudo em parametros."

Isso e falsificavel e as duas pecas estao na maquina. Cabeca a cabeca, MESMO
corpus, MESMA tarefa, testados nos MESMOS itens nunca vistos:

    ABSTRACAO    enumera leis candidatas e elimina por equivalencia
                 observacional; sobra a classe compativel com tudo que viu
    PARAMETROS   uma rede pequena treinada a prever a saida a partir da
                 entrada, por gradiente

A TAREFA
----------
Uma lei escondida leva grade em grade. Os dois bracos recebem N pares
(entrada, saida) da MESMA lei, e sao avaliados em 40 grades novas: acertou a
saida inteira, celula por celula?

Varre-se N de 2 a 200. O que interessa nao e quem vence num ponto — e a
FORMA das duas curvas e onde elas se cruzam, se cruzarem.

POR QUE NADA PRE-TREINADO ENTRA AQUI
--------------------------------------
`sentence_transformers` esta instalado e seria facil usar embeddings prontos.
Seria importar o prior de um LLM pela porta dos fundos: a rede chegaria
sabendo o que a abstracao teria que descobrir sozinha, e a comparacao
perderia o sentido. A rede aqui comeca de pesos aleatorios.

O QUE CADA DESFECHO SIGNIFICA
-------------------------------
  E1  abstracao vence com folga em dado pequeno e a rede so alcanca com
      muito mais dado
      -> a tese vale, e a posicao do cruzamento e o numero que a mede

  E2  a rede vence mesmo com dado minimo
      -> gradiente e mais eficiente em dado do que estrutura, mesmo aqui

  E3  as duas empatam em toda a curva
      -> nao ha vantagem de representacao; o limite e a tarefa

  E4  nenhuma das duas generaliza
      -> a tarefa esta mal posta e nada abaixo pode ser lido

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  F0  PRE-CONDICAO. Com N grande (200 pares), pelo menos UMA das duas tem que
      passar de 50% de acerto. Se nenhuma passar, e E4 e o resto nao vale.

  F1  a abstracao chega a >= 80% de acerto com N <= 5 pares
      -> eliminacao por equivalencia e brutalmente eficiente em dado, que e
         o que a tese preve

MINHA PREVISAO, escrita antes de rodar: E1, e por um motivo que quase torna o
teste injusto — a abstracao aqui SABE a gramatica das leis, entao ela procura
no espaco certo. A rede tem que descobrir a estrutura e os pesos ao mesmo
tempo, do zero.

Por isso este teste NAO responde "estrutura vence gradiente em geral". Ele
responde algo mais estreito e ainda assim util: "quando a familia de
hipoteses e conhecida, quanto dado a eliminacao economiza?". Se a resposta
for 2 exemplos contra 200, isso quantifica o valor de ter a familia certa —
e o custo de nao ter e exatamente o que separa nosso sistema de um LLM.

A comparacao honesta viria depois: dar a rede a mesma vantagem, ou tirar da
abstracao a gramatica. Isso fica declarado como limite, nao escondido.
"""

from __future__ import annotations

import argparse
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LADO = 6
N_CORES = 5
Grid = Tuple[Tuple[int, ...], ...]


# ══════════════════════════════════════════════ atomos da lei escondida
def a_rot90(g: Grid) -> Grid:
    return tuple(zip(*g[::-1]))                    # type: ignore[return-value]


def a_espH(g: Grid) -> Grid:
    return tuple(tuple(l[::-1]) for l in g)


def a_espV(g: Grid) -> Grid:
    return tuple(g[::-1])


def a_transp(g: Grid) -> Grid:
    return tuple(zip(*g))                          # type: ignore[return-value]


def _ciclo(k: int):
    return lambda g: tuple(tuple((c + k) % N_CORES for c in l) for l in g)


def _troca(a: int, b: int):
    return lambda g: tuple(tuple(b if c == a else (a if c == b else c)
                                 for c in l) for l in g)


def _desl(dy: int, dx: int):
    def fn(g: Grid) -> Grid:
        h, w = len(g), len(g[0])
        return tuple(tuple(g[(i - dy) % h][(j - dx) % w] for j in range(w))
                     for i in range(h))
    return fn


ATOMOS: List[Tuple[str, Callable]] = [
    ("rot90", a_rot90), ("espH", a_espH), ("espV", a_espV),
    ("transp", a_transp),
    ("cic1", _ciclo(1)), ("cic2", _ciclo(2)),
    ("tr12", _troca(1, 2)), ("tr34", _troca(3, 4)),
    ("dl01", _desl(0, 1)), ("dl10", _desl(1, 0)),
]
DIC = dict(ATOMOS)
NOMES = [n for n, _ in ATOMOS]


def aplicar(lei: Sequence[str], g: Grid) -> Grid:
    for n in lei:
        g = DIC[n](g)
    return g


def grade(rng: random.Random) -> Grid:
    return tuple(tuple(rng.randrange(N_CORES) for _ in range(LADO))
                 for _ in range(LADO))


def leis(prof: int) -> List[Tuple[str, ...]]:
    """Todas as leis ate profundidade `prof`. E o espaco de hipoteses."""
    import itertools
    fora: List[Tuple[str, ...]] = []
    for d in range(1, prof + 1):
        fora.extend(itertools.product(NOMES, repeat=d))
    return fora


# ══════════════════════════════════════════════ braco ABSTRACAO
def abstracao(treino: List[Tuple[Grid, Grid]], espaco: List[Tuple[str, ...]],
              teste: List[Tuple[Grid, Grid]]) -> float:
    """
    Elimina toda lei incompativel com algum par de treino. Sobra uma CLASSE
    de equivalencia; prediz com qualquer membro dela — todos concordam no que
    foi observado, e a pergunta e se concordam tambem no que nao foi.
    """
    vivos = [L for L in espaco
             if all(aplicar(L, a) == b for a, b in treino)]
    if not vivos:
        return 0.0
    L = vivos[0]
    return sum(1 for a, b in teste if aplicar(L, a) == b) / len(teste)


# ══════════════════════════════════════════════ braco PARAMETROS
def cod(g: Grid) -> np.ndarray:
    t = np.zeros((N_CORES, LADO, LADO), dtype=np.float32)
    for i in range(LADO):
        for j in range(LADO):
            t[g[i][j], i, j] = 1.0
    return t


def parametros(treino: List[Tuple[Grid, Grid]],
               teste: List[Tuple[Grid, Grid]], epocas: int,
               semente: int, arq: str = "densa") -> float:
    """
    Rede pequena, pesos aleatorios, treinada a prever a cor de cada celula de
    saida. Nada pre-treinado entra aqui.

    ARQUITETURA, e por que ela mudou. A primeira versao usava convolucao 3x3.
    Ela saturava em EXATAMENTE 33,3% de N=35 a N=200 — quatro pontos iguais ao
    decimo, que e assinatura de fronteira de expressividade e nao de curva de
    aprendizado. A auditoria confirmou: acerto 100% nas 4 leis que sao funcao
    POR CELULA (cic1, tr12...) e 0% nas 8 ESPACIAIS (transp, rot90, dl10).
    Convolucao e equivariante a translacao por construcao e nao pode
    expressar uma rotacao. Eu tinha posto a rede numa tarefa matematicamente
    impossivel para ela e ia reportar isso como eficiencia de dado.

    A densa sobre a grade achatada pode expressar qualquer permutacao de
    celulas, entao a comparacao passa a ser sobre DADO e nao sobre
    arquitetura. A convolucional fica disponivel como `arq="conv"` para
    documentar a diferenca.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(semente)
    n = LADO * LADO
    if arq == "conv":
        rede = nn.Sequential(
            nn.Conv2d(N_CORES, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, N_CORES, 1),
        )
    else:
        rede = nn.Sequential(
            nn.Flatten(),
            nn.Linear(N_CORES * n, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, N_CORES * n),
            nn.Unflatten(1, (N_CORES, LADO, LADO)),
        )
    X = torch.from_numpy(np.stack([cod(a) for a, _ in treino]))
    Y = torch.from_numpy(np.stack([np.array(b, dtype=np.int64)
                                   for _, b in treino]))
    opt = torch.optim.Adam(rede.parameters(), lr=3e-3)
    perda = nn.CrossEntropyLoss()
    for _ in range(epocas):
        opt.zero_grad()
        perda(rede(X), Y).backward()
        opt.step()

    rede.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(np.stack([cod(a) for a, _ in teste]))
        p = rede(Xt).argmax(1).numpy()
    ok = 0
    for k, (_a, b) in enumerate(teste):
        if (p[k] == np.array(b)).all():
            ok += 1
    return ok / len(teste)


# ══════════════════════════════════════════════ medicao
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prof", type=int, default=2)
    ap.add_argument("--leis", type=int, default=12, help="leis sorteadas")
    ap.add_argument("--teste", type=int, default=40)
    ap.add_argument("--epocas", type=int, default=400)
    ap.add_argument("--semente", type=int, default=20260905)
    ap.add_argument("--arq", default="densa", choices=["densa", "conv"])
    a = ap.parse_args()

    import torch
    torch.set_num_threads(10)

    espaco = leis(a.prof)
    print("=" * 70)
    print("EFICIENCIA — abstracao ou parametros, com quanto dado?")
    print("=" * 70)
    print(f"grade {LADO}x{LADO}, {N_CORES} cores, {len(NOMES)} atomos")
    print(f"espaco de hipoteses da abstracao: {len(espaco)} leis ate "
          f"profundidade {a.prof}")
    print(f"rede: {a.arq}, pesos aleatorios, {a.epocas} epocas")
    print(f"{a.leis} leis sorteadas, teste em {a.teste} grades novas\n")

    tamanhos = [2, 3, 5, 8, 12, 20, 35, 60, 100, 200]
    print(f"{'N pares':>8} {'ABSTRACAO':>11} {'PARAMETROS':>12}")
    print("-" * 34)
    res: Dict[int, Tuple[float, float]] = {}
    for n in tamanhos:
        ab, pa = [], []
        for k in range(a.leis):
            rng = random.Random(a.semente + k)
            prof = rng.randint(1, a.prof)
            lei = tuple(rng.choice(NOMES) for _ in range(prof))
            tr = []
            while len(tr) < n:
                g = grade(rng)
                s = aplicar(lei, g)
                if s != g:
                    tr.append((g, s))
            te = []
            rng2 = random.Random(a.semente + 9000 + k)
            while len(te) < a.teste:
                g = grade(rng2)
                s = aplicar(lei, g)
                if s != g:
                    te.append((g, s))
            ab.append(abstracao(tr, espaco, te))
            pa.append(parametros(tr, te, a.epocas, a.semente + k, a.arq))
        ma, mp = sum(ab) / len(ab), sum(pa) / len(pa)
        res[n] = (ma, mp)
        print(f"{n:>8} {ma:>10.1%} {mp:>11.1%}", flush=True)
    print("-" * 34)

    grande = res[tamanhos[-1]]
    print()
    if max(grande) < 0.50:
        print("F0 · FALHA — nenhuma passa de 50% com dado grande. E4, tarefa")
        print("     mal posta, nada acima pode ser lido.")
        return
    print(f"F0 · com {tamanhos[-1]} pares, melhor braco em {max(grande):.0%} · passa")

    pequenos = [n for n in tamanhos if res[n][0] >= 0.80]
    if pequenos and pequenos[0] <= 5:
        print(f"F1 · abstracao chega a 80% com {pequenos[0]} pares.")
    elif pequenos:
        print(f"F1 · abstracao so chega a 80% com {pequenos[0]} pares "
              f"(previa <= 5).")
    else:
        print("F1 · abstracao nunca chega a 80%.")

    cruz = [n for n in tamanhos if res[n][1] > res[n][0]]
    print()
    if not cruz:
        print("E1 · a rede NUNCA alcanca a abstracao na faixa testada.")
        print(f"     razao em {tamanhos[0]} pares: "
              f"{res[tamanhos[0]][0]:.0%} contra {res[tamanhos[0]][1]:.0%}")
    elif cruz[0] <= tamanhos[1]:
        print("E2 · a rede vence quase desde o inicio.")
    else:
        print(f"E1 · cruzamento em N = {cruz[0]} pares. Abaixo disso a")
        print("     abstracao vence; acima, o gradiente alcanca.")
    print()
    print("LIMITE DECLARADO: a abstracao conhece a gramatica das leis e")
    print("procura no espaco certo; a rede descobre estrutura e pesos do")
    print("zero. Isto mede quanto dado a eliminacao economiza QUANDO a")
    print("familia e conhecida — nao 'estrutura vence gradiente em geral'.")


if __name__ == "__main__":
    main()
