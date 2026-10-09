"""
novaphys.arc_guia — a busca guiada, com o verificador mantendo a honestidade
=============================================================================

O GARGALO, MEDIDO
-------------------
    profundidade 3     14.425 programas    586s    3,5%
    profundidade 4    ~346.000 programas   ~4h     inviavel
    profundidade 5      ~8,3M programas    dias    —

Forca bruta morre na profundidade 4. E `arc_biblioteca.py` mostrou que a
biblioteca nao resolve isso: as macros COMPRIMIRAM profundidade (1,71 -> 1,57)
mas nao estenderam alcance (14/400 nos tres bracos). Comprimir nao basta
quando o problema e o tamanho do espaco.

A PECA QUE FALTA, E ONDE UMA REDE PERTENCE
--------------------------------------------
Uma distribuicao de proposta aprendida: uma rede pequena olha o par
(entrada, saida) e ordena QUAIS primitivas tentar primeiro. Converte
enumeracao exaustiva em busca guiada.

    a rede PROPOE     ordena o espaco de busca
    o verificador DISPOE   igualdade exata de grade, nao ha nota parcial

A rede nunca tem a palavra final. Se ela ordenar mal, o pior caso e a ordem
arbitraria de antes — e isso aparece no numero, porque o orcamento e casado.
E a diferenca entre usar rede como raciocinador (nao auditavel) e como guia
(auditavel pelo verificador).

O DATASET SE ROTULA SOZINHO
-----------------------------
Nao ha dataset humano aqui, e nao precisa haver:

    sorteia programa da DSL  ->  aplica a grades reais de TREINO
    ->  par (entrada, saida) COM o programa que o gerou

A DSL e geradora e rotuladora ao mesmo tempo. Rotulo perfeito, custo zero,
quantidade ilimitada. Cabe em 4 GB porque a rede e minuscula.

SEPARACAO
----------
Os grids de origem vem SO do conjunto de TREINO do ARC. O conjunto de
AVALIACAO nao entra em nenhuma etapa de treino — nem como grade, nem como
tarefa. E onde tudo e medido.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  N0  PRE-CONDICAO. A rede tem que prever primitivas acima do acaso num
      holdout sintetico. Se nao passar disso, o guia nao tem o que oferecer
      e nao se roda o resto. (E o mesmo portao de `sanidade_do_solucionador`:
      um resultado negativo por defeito do instrumento e pior que nenhum.)

  N1  guiado > base em >= 1pp com ORCAMENTO CASADO -> o guia funciona
  N2  guiado ~ base -> a rede nao ordena melhor que a ordem arbitraria
  N3  guiado < base -> o guia atrapalha; a ordem aprendida e pior que
      nenhuma ordem, e isso e informativo sobre o sinal disponivel no par

  N4  o teste que importa de verdade: com o guia, a profundidade 4 fica
      alcancavel dentro de um orcamento que a forca bruta nao alcanca?
      Medido como tarefas resolvidas com programa de profundidade >= 4.

MINHA PREVISAO, escrita antes de rodar: espero N1 fraco na profundidade 3 —
com 24 primitivas e orcamento de 14 mil, a busca exaustiva ja cobre quase
tudo, entao ordenar melhor rende pouco. O ganho real, se houver, deve
aparecer em N4, onde a forca bruta simplesmente nao chega. Se der N3 eu vou
olhar a pre-condicao antes de concluir qualquer coisa sobre o metodo.
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .arc import (Grid, PRIMITIVAS, DICIONARIO, Programa, ajustar_mapa,
                  aplicar_mapa, carregar, dims, valida)

MAX_LADO = 30
N_CORES = 11
PAD = 10
TREINO = (('C:/Users/usuario/Documents/AGI 2.0/backup_novamind_20260627_170507/data/ARC/data/training').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
AVALIACAO = (('C:/Users/usuario/Documents/AGI 2.0/backup_novamind_20260627_170507/data/ARC/data/evaluation').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
NOMES = [n for n, _ in PRIMITIVAS]
SEMENTE = 20260813


# ══════════════════════════════════════════════ codificacao
def codificar(a: Grid, b: Grid) -> np.ndarray:
    """Duas grades -> 22 canais de 30x30 (11 cores x 2 grades), com pad."""
    t = np.zeros((2 * N_CORES, MAX_LADO, MAX_LADO), dtype=np.float32)
    for k, g in enumerate((a, b)):
        h, w = dims(g)
        t[k * N_CORES + PAD] = 1.0                      # tudo pad por padrao
        for i in range(min(h, MAX_LADO)):
            for j in range(min(w, MAX_LADO)):
                c = g[i][j]
                if 0 <= c < N_CORES:
                    t[k * N_CORES + PAD, i, j] = 0.0
                    t[k * N_CORES + c, i, j] = 1.0
    return t


class Guia(nn.Module):
    """Pequena de proposito: roda em 4 GB e treina em minutos."""

    def __init__(self, n_saida: int):
        super().__init__()
        self.corpo = nn.Sequential(
            nn.Conv2d(2 * N_CORES, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.cabeca = nn.Linear(64, n_saida)

    def forward(self, x):
        return self.cabeca(self.corpo(x).flatten(1))


# ══════════════════════════════════════════════ dado que se rotula sozinho
def gerar(rng: random.Random, grades: Sequence[Grid], n: int,
          prof_max: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sorteia programa, aplica a uma grade REAL de treino, e o rotulo e o
    conjunto de primitivas usadas. Nenhuma anotacao humana.
    """
    X, Y = [], []
    tentativas = 0
    while len(X) < n and tentativas < n * 40:
        tentativas += 1
        d = rng.randint(1, prof_max)
        ops = [rng.choice(NOMES) for _ in range(d)]
        a = rng.choice(grades)
        b: Optional[Grid] = a
        for o in ops:
            b = DICIONARIO[o](b)                # type: ignore[arg-type]
            if not valida(b):
                b = None
                break
        if b is None or b == a:
            continue
        y = np.zeros(len(NOMES), dtype=np.float32)
        for o in set(ops):
            y[NOMES.index(o)] = 1.0
        X.append(codificar(a, b))
        Y.append(y)
    return np.stack(X), np.stack(Y)


def treinar(modelo, X, Y, device, epocas: int = 6, lote: int = 128):
    opt = torch.optim.Adam(modelo.parameters(), lr=2e-3)
    perda = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    n = len(Xt)
    for ep in range(epocas):
        modelo.train()
        idx = torch.randperm(n)
        soma = 0.0
        for i in range(0, n, lote):
            b = idx[i:i + lote]
            xb, yb = Xt[b].to(device), Yt[b].to(device)
            opt.zero_grad()
            l = perda(modelo(xb), yb)
            l.backward()
            opt.step()
            soma += l.item() * len(b)
        print(f"    epoca {ep+1}  perda {soma/n:.4f}", flush=True)


def avaliar_guia(modelo, X, Y, device) -> Tuple[float, float]:
    """Precisao no topo-1 e no topo-3, contra o acaso."""
    modelo.eval()
    with torch.no_grad():
        p = torch.sigmoid(modelo(torch.from_numpy(X).to(device))).cpu().numpy()
    t1 = t3 = 0
    for pi, yi in zip(p, Y):
        ordem = np.argsort(-pi)
        verd = set(np.where(yi > 0)[0])
        if ordem[0] in verd:
            t1 += 1
        if verd & set(ordem[:3].tolist()):
            t3 += 1
    return t1 / len(Y), t3 / len(Y)


# ══════════════════════════════════════════════ busca, guiada ou nao
def buscar(treino: Sequence[Tuple[Grid, Grid]], ordem: Sequence[str],
           orcamento: int, prof: int) -> Optional[Tuple[Programa, int]]:
    """
    Enumera por profundidade crescente, e DENTRO de cada profundidade na
    ordem dada. Guiar = mudar a ordem, nada mais. O verificador e o mesmo.
    """
    n = 0
    for d in range(0, prof + 1):
        for ops in itertools.product(ordem, repeat=d):
            n += 1
            if n > orcamento:
                return None
            saidas, ok = [], True
            for a, b in treino:
                r: Optional[Grid] = a
                for o in ops:
                    r = DICIONARIO[o](r)        # type: ignore[arg-type]
                    if not valida(r):
                        ok = False
                        break
                if not ok:
                    break
                saidas.append(r)
            if not ok:
                continue
            if all(r == b for r, (_, b) in zip(saidas, treino)):
                return Programa(ops), d
            mapa = ajustar_mapa(list(zip(saidas, [b for _, b in treino])))
            if mapa is not None and all(aplicar_mapa(r, mapa) == b
                                        for r, (_, b) in zip(saidas, treino)):
                return Programa(ops, mapa), d
    return None


def ordem_guiada(modelo, treino, device) -> List[str]:
    """Media da predicao sobre os pares da tarefa -> ordem das primitivas."""
    X = np.stack([codificar(a, b) for a, b in treino])
    modelo.eval()
    with torch.no_grad():
        p = torch.sigmoid(modelo(torch.from_numpy(X).to(device))).cpu().numpy()
    m = p.mean(0)
    return [NOMES[i] for i in np.argsort(-m)]


def relatorio(prof: int = 3, n_amostras: int = 40000,
              orcamento: Optional[int] = None) -> None:
    rng = random.Random(SEMENTE)
    torch.manual_seed(SEMENTE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if orcamento is None:
        orcamento = sum(len(NOMES) ** d for d in range(prof + 1))
    espaco = sum(len(NOMES) ** d for d in range(prof + 1))

    print("=" * 78)
    print("ARC — busca guiada: a rede propoe, o verificador dispoe")
    print("=" * 78)
    print("PRE-REGISTRO:")
    print("  N0  pre-condicao: a rede preve acima do acaso, senao nao roda")
    print("  N1  guiado > base em >=1pp, orcamento casado -> o guia funciona")
    print("  N2  guiado ~ base -> nao ordena melhor que ordem arbitraria")
    print("  N3  guiado < base -> o guia atrapalha")
    print("  N4  o guia alcanca profundidade que a forca bruta nao alcanca")
    print()
    print(f"device {device} · orcamento casado {orcamento} · profundidade {prof}")
    print(f"espaco total nessa profundidade: {espaco} "
          f"({orcamento/espaco:.0%} visivel)")
    if orcamento >= espaco:
        print("  AVISO: orcamento >= espaco, a busca e EXAUSTIVA e a ordem")
        print("  nao pode mudar o resultado. Este ajuste nao mede nada.")
    print("grades de origem: SO do conjunto de treino. Avaliacao nao entra.")

    tr = carregar(TREINO)
    grades = [a for _n, pares, _t in tr for a, _b in pares]
    print(f"\ngrades reais de treino disponiveis: {len(grades)}")

    print(f"\nFASE 1 · gerar {n_amostras} pares auto-rotulados")
    t0 = time.time()
    X, Y = gerar(rng, grades, n_amostras, prof_max=prof)
    corte = int(0.9 * len(X))
    Xtr, Ytr, Xho, Yho = X[:corte], Y[:corte], X[corte:], Y[corte:]
    print(f"  gerados {len(X)} · treino {len(Xtr)} · holdout {len(Xho)}"
          f" · {time.time()-t0:.0f}s")

    print("\nFASE 2 · treinar o guia")
    modelo = Guia(len(NOMES)).to(device)
    npar = sum(p.numel() for p in modelo.parameters())
    print(f"  parametros: {npar:,}")
    treinar(modelo, Xtr, Ytr, device)

    t1, t3 = avaliar_guia(modelo, Xho, Yho, device)
    acaso1 = 1.0 / len(NOMES)
    acaso3 = 3.0 / len(NOMES)
    print(f"\nN0 · holdout sintetico — topo-1 {t1:.1%} (acaso {acaso1:.1%})"
          f" · topo-3 {t3:.1%} (acaso {acaso3:.1%})")
    if t1 < 2 * acaso1:
        print("     PRE-CONDICAO REPROVADA. A rede nao preve acima do acaso;")
        print("     qualquer resultado negativo abaixo seria defeito do")
        print("     instrumento, nao do metodo. Nao rodo o resto.")
        return
    print("     aprovada — a rede tem sinal para oferecer")

    print("\nFASE 3 · AVALIACAO (400 tarefas nunca vistas), orcamento casado")
    av = carregar(AVALIACAO)
    res: Dict[str, Dict[str, object]] = {}
    for nome in ('base', 'guiado'):
        t0 = time.time()
        ok, profs = 0, []
        for _n, trp, te in av:
            ordem = NOMES if nome == 'base' else ordem_guiada(modelo, trp, device)
            r = buscar(trp, ordem, orcamento, prof)
            if r is None:
                continue
            p, d = r
            if all(p.aplicar(a) == b for a, b in te):
                ok += 1
                profs.append(d)
        res[nome] = {'ok': ok, 'profs': profs, 's': round(time.time() - t0)}
        pm = (sum(profs) / len(profs)) if profs else 0.0
        print(f"  {nome:<8} {ok:>3}/{len(av)} = {ok/len(av):>5.1%}"
              f"   prof media {pm:.2f}   {res[nome]['s']}s")

    b, g = res['base']['ok'], res['guiado']['ok']
    d = (g - b) / len(av)
    print("\n" + "=" * 78)
    if d >= 0.01:
        print(f"N1 · O GUIA FUNCIONA. {d:+.1%} com o mesmo orcamento —")
        print("     mesma gramatica, mesmo verificador, so a ORDEM mudou.")
    elif abs(d) < 0.01:
        print(f"N2 · EMPATE ({d:+.1%}). A ordem aprendida nao supera a")
        print("     arbitraria neste orcamento. Com 14 mil programas a busca")
        print("     exaustiva ja cobre quase tudo — ordenar rende pouco.")
    else:
        print(f"N3 · O GUIA ATRAPALHA ({d:+.1%}). A ordem aprendida e pior que")
        print("     nenhuma ordem. Verificar a pre-condicao antes de concluir.")
    print()
    print("N4 · e a pergunta que decide se isto escala: rodar de novo com")
    print("     --prof 4, onde a forca bruta nao chega. Este arquivo mede o")
    print("     piso; o teto e la.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prof", type=int, default=3)
    ap.add_argument("--amostras", type=int, default=40000)
    ap.add_argument("--orcamento", type=int, default=None)
    a = ap.parse_args()
    relatorio(prof=a.prof, n_amostras=a.amostras, orcamento=a.orcamento)
