"""
novaphys.propositor — trocar enumeracao por uma distribuicao aprendida
=======================================================================

O QUE OS LLM FAZEM QUE NOS NAO FAZIAMOS
-----------------------------------------
Nosso sistema propoe candidatos por enumeracao UNIFORME. A chance de acertar um
programa de profundidade d e 1/24^d. Por isso paramos em 3: o orcamento de
14.424 e exatamente 24 + 24² + 24³.

Um LLM nao enumera. Ele amostra de uma distribuicao APRENDIDA sobre programas
plausiveis. A chance dele nao depende de 24^d — depende da qualidade da
distribuicao.

    enumeracao uniforme      custo para alcancar profundidade d  =  24^d
    propositor + feixe       custo para alcancar profundidade d  =  d · W · 24

Exponencial vira linear. Profundidade 8 com feixe 50 custa ~9.600 avaliacoes;
por enumeracao custaria 110 bilhoes.

E ISSO NAO CONTRADIZ OS TRES NULOS DE GUIA DESTA SESSAO
---------------------------------------------------------
`arc_guia.py`, `prion_hdc.py` e `guia.py` mediram GUIAS: coisas que reordenam
uma enumeracao. Reordenar um espaco exponencial nao adianta — o item certo
continua fora do alcance. Um PROPOSITOR e outra coisa: ele gera os candidatos
em vez de ordenar os que ja existiriam. Eu tratei as duas como a mesma coisa
o dia inteiro, e nao sao.

DE ONDE VEM O TREINO, SEM LLM E SEM ROTULO HUMANO
---------------------------------------------------
Do proprio sistema. Amostra-se um programa ao acaso, aplica-se numa grade REAL
de entrada do ARC, e guarda-se (entrada, saida, programa). Milhoes de exemplos
supervisionados de graca. E o mesmo truque do LLM — prior massivo — fabricado
em casa.

A grade de entrada e real de proposito: a distribuicao de GRADES passa a bater
com a do ARC. O que continua sem bater e a distribuicao de PROGRAMAS, porque
tarefas do ARC foram desenhadas por humanos e nao sorteadas do nosso conjunto
de primitivas. Esse descasamento e o modo de falha esperado deste arquivo, e
esta declarado antes de rodar, nao depois.

TRES BRACOS, ORCAMENTO IDENTICO
---------------------------------
    ENUM      enumeracao por largura, o que ja temos
    FEIXE0    busca em feixe guiada por uma rede de PESOS ALEATORIOS
    FEIXE     busca em feixe guiada pela rede TREINADA

`FEIXE0` e obrigatorio. Sem ele nao da para separar "aprender ajuda" de "busca
em feixe ajuda". A rede aleatoria da a mesma estrutura de busca, o mesmo
alcance de profundidade e o mesmo orcamento — so nao sabe nada. Se os dois
empatarem, o ganho e da busca em feixe e o aprendizado nao entrou.

A moeda e chamada ao verificador (um programa completo conferido contra todos
os pares de treino), identica nos tres.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  N0  PRE-CONDICAO. A rede tem que prever a proxima primitiva acima do acaso
      (1/24 = 4,2%) em dados sinteticos SEPARADOS do treino. Se nao aprender
      nem a propria distribuicao que a gerou, nao ha propositor e o resto nao
      deve ser lido.

  N1  FEIXE resolve >= 5 tarefas a mais que ENUM com o mesmo orcamento,
      E supera FEIXE0
      -> a distribuicao aprendida transfere; o expoente foi atacado

  N2  FEIXE ~ FEIXE0, ambos > ENUM
      -> o ganho e do feixe (alcance de profundidade), nao do aprendizado.
         Ainda seria util, mas por outro motivo que o alegado

  N3  FEIXE ~ ENUM
      -> a distribuicao de programas sorteados nao transfere para tarefas
         desenhadas por humanos. O descasamento previsto se confirmou

MINHA PREVISAO, escrita antes de rodar: espero N2. A rede deve aprender bem a
prever programas sorteados (N0 passa com folga), e o feixe deve dar algum
ganho por alcancar profundidade 8 onde a enumeracao para na 3 — mas a
distribuicao aprendida sera a de programas ALEATORIOS, e tarefas do ARC nao
sao programas aleatorios.

Se N1 aparecer, e o primeiro ganho de mecanismo real da sessao e eu vou
querer ver FEIXE0 bem atras antes de acreditar.

O QUE ISTO NAO E
------------------
Nao e "resolver o ARC". E testar uma afirmacao especifica: distribuicao
aprendida sobre programas bate enumeracao uniforme com orcamento igual. A
afirmacao vale para qualquer dominio com verificador, nao so para grades.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from novaphys.arc import (Grid, DICIONARIO, PRIMITIVAS, ajustar_mapa,
                          aplicar_mapa, carregar, dims, valida)
from novaphys.eter import PASTA_AVAL

PASTA_TREINO = (('C:/Users/usuario/Documents/AGI 2.0/backup_novamind_20260627_170507/data/ARC/data/training').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))

NOMES = [n for n, _ in PRIMITIVAS]
NOP = len(NOMES)
LADO = 30
K = 14424


# ══════════════════════════════════════════════ codificacao das grades
def cod(g: Grid) -> np.ndarray:
    """
    Grade -> tensor 11 x 30 x 30. Dez canais de cor mais um de mascara, para
    que a rede saiba onde a grade acaba (as grades variam de 1x1 a 30x30 e o
    preenchimento nao pode ser confundido com a cor 0, que e preta e real).
    """
    t = np.zeros((11, LADO, LADO), dtype=np.float32)
    h, w = dims(g)
    h, w = min(h, LADO), min(w, LADO)
    for i in range(h):
        for j in range(w):
            t[g[i][j], i, j] = 1.0
    t[10, :h, :w] = 1.0
    return t


def par(estado: Grid, alvo: Grid) -> np.ndarray:
    """22 canais: o estado atual e o alvo, empilhados."""
    return np.concatenate([cod(estado), cod(alvo)], axis=0)


# ══════════════════════════════════════════════ dados fabricados
def _roda(ops: Sequence[str], g: Grid) -> Optional[Grid]:
    r: Optional[Grid] = g
    for n in ops:
        r = DICIONARIO[n](r)
        if not valida(r):
            return None
    return r


def fabricar(grades: Sequence[Grid], n: int, prof_max: int,
             semente: int) -> List[Tuple[Grid, Grid, int]]:
    """
    Amostra um programa, aplica numa grade REAL do ARC, e gera um exemplo por
    PASSO: (estado atual, alvo final) -> proxima primitiva.

    Programas degenerados (saida == entrada) sao descartados: eles ensinariam
    a rede a prever operacoes que nao fazem nada, que e o oposto do util.

    MEMORIA. Guarda as GRADES, nao os tensores. A primeira versao materializava
    160.000 x 22 x 30 x 30 em float32 = 11,8 GiB e estourou. A codificacao
    passou a ser por lote, na hora do uso: 256 exemplos ocupam 19 MiB.
    """
    rng = random.Random(semente)
    amostras: List[Tuple[Grid, Grid, int]] = []
    tent = 0
    while len(amostras) < n and tent < n * 40:
        tent += 1
        g0 = rng.choice(grades)
        prof = rng.randint(1, prof_max)
        ops = [rng.choice(NOMES) for _ in range(prof)]
        alvo = _roda(ops, g0)
        if alvo is None or alvo == g0:
            continue
        est = g0
        bom = True
        passos = []
        for o in ops:
            passos.append((est, alvo, NOMES.index(o)))
            est = DICIONARIO[o](est)
            if not valida(est):
                bom = False
                break
        if not bom:
            continue
        for e, al, y in passos:
            amostras.append((e, al, y))
            if len(amostras) >= n:
                break
    return amostras


def lote(amostras: Sequence[Tuple[Grid, Grid, int]],
         idx: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Codifica so o lote pedido. E aqui que os 11,8 GiB viram 19 MiB."""
    X = np.stack([par(amostras[i][0], amostras[i][1]) for i in idx])
    Y = np.array([amostras[i][2] for i in idx], dtype=np.int64)
    return X, Y


# ══════════════════════════════════════════════ rede
def construir(semente: int):
    import torch
    import torch.nn as nn
    torch.manual_seed(semente)
    return nn.Sequential(
        nn.Conv2d(22, 48, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, NOP),
    )


def treinar(modelo, amostras, epocas: int, tam: int = 256):
    import torch
    import torch.nn as nn
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    perda = nn.CrossEntropyLoss()
    n = len(amostras)
    for ep in range(epocas):
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, tam):
            idx = perm[i:i + tam]
            X, Y = lote(amostras, idx)
            opt.zero_grad()
            l = perda(modelo(torch.from_numpy(X)), torch.from_numpy(Y))
            l.backward()
            opt.step()
            tot += float(l) * len(idx)
        print(f"     epoca {ep + 1}/{epocas}  perda {tot / n:.4f}", flush=True)
    return modelo


def acerto(modelo, amostras, tam: int = 512) -> float:
    import torch
    modelo.eval()
    ok = tot = 0
    with torch.no_grad():
        for i in range(0, len(amostras), tam):
            idx = list(range(i, min(i + tam, len(amostras))))
            X, Y = lote(amostras, idx)
            p = modelo(torch.from_numpy(X)).argmax(1).numpy()
            ok += int((p == Y).sum())
            tot += len(idx)
    modelo.train()
    return ok / max(tot, 1)


# ══════════════════════════════════════════════ busca
def _aceita(saidas, treino) -> Optional[Dict[int, int]]:
    if all(r == b for r, (_, b) in zip(saidas, treino)):
        return {}
    m = ajustar_mapa(list(zip(saidas, [b for _, b in treino])))
    if m is not None and all(aplicar_mapa(r, m) == b
                             for r, (_, b) in zip(saidas, treino)):
        return m
    return None


def por_enumeracao(treino, orc: int, prof_max: int = 3):
    import itertools
    gasto = 0
    for prof in range(0, prof_max + 1):
        for seq in itertools.product(NOMES, repeat=prof):
            if gasto >= orc:
                return None, None, gasto
            gasto += 1
            saidas, ok = [], True
            for a, _b in treino:
                r = _roda(seq, a)
                if r is None:
                    ok = False
                    break
                saidas.append(r)
            if not ok:
                continue
            m = _aceita(saidas, treino)
            if m is not None:
                return list(seq), m, gasto
    return None, None, gasto


def por_feixe(treino, modelo, orc: int, largura: int, prof_max: int):
    """
    Feixe guiado pela rede. Cada filho gerado custa UMA chamada ao verificador
    e e cobrado. O custo cresce como prof · largura · 24, linear na
    profundidade — e essa e a diferenca inteira em relacao a enumeracao.
    """
    import torch
    alvo = treino[0][1]
    gasto = 0
    feixe: List[Tuple[List[str], List[Grid]]] = [([], [a for a, _ in treino])]

    saidas0 = [a for a, _ in treino]
    gasto += 1
    m = _aceita(saidas0, treino)
    if m is not None:
        return [], m, gasto

    for _prof in range(prof_max):
        if not feixe or gasto >= orc:
            break
        lotes = np.stack([par(s[1][0], alvo) for s in feixe])
        with torch.no_grad():
            logit = modelo(torch.from_numpy(lotes)).numpy()
        ordem = np.argsort(-logit, axis=1)

        cand: List[Tuple[float, List[str], List[Grid]]] = []
        for bi, (ops, saidas) in enumerate(feixe):
            for rank in range(NOP):
                if gasto >= orc:
                    break
                o = NOMES[ordem[bi][rank]]
                gasto += 1
                novas, ok = [], True
                for r in saidas:
                    r2 = DICIONARIO[o](r)
                    if not valida(r2):
                        ok = False
                        break
                    novas.append(r2)
                if not ok:
                    continue
                mm = _aceita(novas, treino)
                if mm is not None:
                    return ops + [o], mm, gasto
                cand.append((float(logit[bi][ordem[bi][rank]]),
                             ops + [o], novas))
        if not cand:
            break
        cand.sort(key=lambda c: -c[0])
        feixe = [(c[1], c[2]) for c in cand[:largura]]
    return None, None, gasto


def confere(ops, mapa, teste) -> bool:
    for a, b in teste:
        r = _roda(ops, a)
        if r is None:
            return False
        if mapa:
            r = aplicar_mapa(r, mapa)
        if r != b:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aval", default=PASTA_AVAL)
    ap.add_argument("--amostras", type=int, default=160000)
    ap.add_argument("--epocas", type=int, default=4)
    ap.add_argument("--prof-treino", type=int, default=6)
    ap.add_argument("--largura", type=int, default=48)
    ap.add_argument("--prof-feixe", type=int, default=10)
    ap.add_argument("--orcamento", type=int, default=K)
    ap.add_argument("--semente", type=int, default=20260903)
    a = ap.parse_args()

    import torch
    torch.set_num_threads(10)

    print("=" * 76)
    print("PROPOSITOR — distribuicao aprendida contra enumeracao uniforme")
    print("=" * 76)
    print(f"orcamento IDENTICO nos 3 bracos: {a.orcamento} verificacoes")
    print(f"feixe: largura {a.largura}, profundidade ate {a.prof_feixe}")
    print(f"  custo do feixe = {a.prof_feixe} x {a.largura} x {NOP} = "
          f"{a.prof_feixe * a.largura * NOP:,} (cabe no orcamento)")
    print(f"  enumeracao ate prof 3 = {NOP + NOP**2 + NOP**3:,}")
    print(f"  enumeracao ate prof {a.prof_feixe} seria "
          f"{NOP ** a.prof_feixe:,.0f}\n")

    tr_tarefas = carregar(PASTA_TREINO)
    grades = [g for _n, tr, _te in tr_tarefas for g, _ in tr]
    print(f"{len(grades)} grades reais do ARC (treino) como distribuicao "
          f"de entrada")

    print(f"\nfabricando {a.amostras:,} exemplos "
          f"(programas sorteados ate prof {a.prof_treino})...")
    t0 = time.time()
    tre = fabricar(grades, a.amostras, a.prof_treino, a.semente)
    val = fabricar(grades, 12000, a.prof_treino, a.semente + 777)
    print(f"  {len(tre):,} treino, {len(val):,} validacao  ({time.time()-t0:.0f}s)")

    modelo = construir(a.semente)
    modelo0 = construir(a.semente + 1)          # pesos aleatorios, controle
    print(f"\nN0 · acerto antes de treinar: {acerto(modelo, val):.1%} "
          f"(acaso = {1/NOP:.1%})")
    print("treinando...")
    t0 = time.time()
    treinar(modelo, tre, a.epocas)
    ac = acerto(modelo, val)
    print(f"\nN0 · acerto em validacao SEPARADA: {ac:.1%}  "
          f"(acaso {1/NOP:.1%}, {ac*NOP:.1f}x)  ({time.time()-t0:.0f}s)")
    if ac < 2.0 / NOP:
        print("     N0 · FALHA — a rede nem aprende a distribuicao que a")
        print("     gerou. Sem propositor. Nao ler o resto.")
        return
    print("     N0 · passa")

    modelo.eval()
    modelo0.eval()

    tarefas = carregar(a.aval)
    res: Dict[str, Set[str]] = {"ENUM": set(), "FEIXE0": set(), "FEIXE": set()}
    prof_achada: Dict[str, List[int]] = {k: [] for k in res}

    print(f"\nrodando {len(tarefas)} tarefas de avaliacao nos 3 bracos...")
    t0 = time.time()
    for nome, tr, te in tarefas:
        for br in res:
            if br == "ENUM":
                ops, m, _g = por_enumeracao(tr, a.orcamento)
            else:
                mdl = modelo if br == "FEIXE" else modelo0
                ops, m, _g = por_feixe(tr, mdl, a.orcamento, a.largura,
                                       a.prof_feixe)
            if ops is not None and confere(ops, m, te):
                res[br].add(nome)
                prof_achada[br].append(len(ops))
    print(f"({time.time()-t0:.0f}s)\n")

    print(f"{'braco':10s} {'resolve':>8} {'%':>8}   profundidades achadas")
    print("-" * 60)
    for br in ("ENUM", "FEIXE0", "FEIXE"):
        p = sorted(prof_achada[br])
        print(f"{br:10s} {len(res[br]):>8} {len(res[br])/len(tarefas):>8.2%}   "
              f"{p}")
    print("-" * 60)

    E, F0, F = len(res["ENUM"]), len(res["FEIXE0"]), len(res["FEIXE"])
    print(f"\nFEIXE - ENUM   = {F - E:+d}")
    print(f"FEIXE - FEIXE0 = {F - F0:+d}")
    print(f"\nso FEIXE resolve : {sorted(res['FEIXE'] - res['ENUM'])}")
    print(f"so ENUM resolve  : {sorted(res['ENUM'] - res['FEIXE'])}")
    uniao = res["ENUM"] | res["FEIXE"]
    print(f"\nuniao ENUM+FEIXE : {len(uniao)}  ({len(uniao)/len(tarefas):.2%})"
          f"   [gasta 2x o orcamento, nao comparavel]")

    print("\n" + "=" * 76)
    if F - E >= 5 and F > F0:
        print("N1 · a distribuicao aprendida transfere. Conferir FEIXE0.")
    elif abs(F - F0) < 3 and F > E:
        print("N2 · o ganho e do feixe (alcance), nao do aprendizado.")
    else:
        print("N3 · programas sorteados nao transferem para tarefas humanas.")
        print("     O descasamento de distribuicao previsto se confirmou.")


if __name__ == "__main__":
    main()
