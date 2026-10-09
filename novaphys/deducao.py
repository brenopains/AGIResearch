"""
novaphys.deducao — DEDUZIR a lei do mundo, em vez de filtrar uma lista minha
=============================================================================

A TROCA QUE ESTA SENDO FEITA
------------------------------
Os dois unicos achados desta sessao que deram ordem de grandeza foram trocas de
METODO DE ACESSO, nao de conteudo:

    sortear -> DEDUZIR          no mesmo espaco de 178M operadores, amostrar
                                acha 1 util em 3.000; deduzir a partir da
                                diferenca entrada/saida acha o exato, 100% de
                                precisao  (`eter_inferencia.py`)
    exigir tudo -> ACEITAR PARCIAL   3 tarefas e ZERO composicoes viram 8
                                tarefas e 5 composicoes, com 41% do orcamento
                                (`relaxado.py`)

E ha um lugar onde essa troca ainda NAO foi feita: a aquisicao de prior. Em
`fisica.py` e no experimento da dobra, eu escrevo a lista de leis candidatas e
o mundo apenas FILTRA qual delas e consistente. Isso e sortear. O mundo nunca
me ensina uma lei que eu nao tenha escrito antes — por construcao.

Aqui o mundo e um SIMULADOR DE PARTICULAS. Eu escrevo apenas regras LOCAIS:
uma particula cai, colide, empilha, e particulas coesas se movem juntas. O que
se observa e a grade antes e a grade depois de K passos de simulacao. Esse
mapa grade->grade eu NAO escrevi e nao saberia escrever: e consequencia
emergente de muitas interacoes locais.

O agente entao DEDUZ o operador que explica os pares, na meta-gramatica

    celula(i,j) = TESTE ? LEITURA_v : LEITURA_f

resolvendo restricoes, nunca sorteando. O operador deduzido pode nao ser
nenhum dos que eu escreveria — e essa e a coisa toda.

O CONTROLE
------------
Mesmo simulador, mesmas particulas, mesma quantidade de tinta e mesmo numero de
passos, mas com as regras locais EMBARALHADAS: cada particula recebe um
deslocamento fixo sorteado em vez de cair. O mundo continua deterministico,
consistente e aprendivel — so nao tem fisica. Se a deducao extrair a mesma
quantidade de estrutura util dos dois, o ganho e da DEDUCAO e nao do MUNDO, e
o resultado e nulo.

Alem disso, dois bracos de disciplina:

    EXATO    so aceita operador que explica 100% dos pares
    PARCIAL  aceita a partir de 85% de cobertura — o limiar que venceu em
             `relaxado.py`, com a curva em U ja medida (0,50 e pior que 0,85)

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  D0  PRE-CONDICAO. O simulador tem que produzir pares nao-triviais (antes
      diferente de depois) e a deducao tem que devolver ALGUM operador no
      braco de fisica. Se nao devolver nada, nao ha o que medir.

  D1  FISICA - RUIDO >= +3 no ARC, orcamento casado
      -> deduzir do mundo entrega prior que filtrar do mundo nao entrega. A
         troca de metodo de acesso funciona tambem na aquisicao.

  D2  FISICA - RUIDO = +1 ou +2
      -> funciona, no mesmo patamar de escrever a lista a mao (a dobra deu
         +3 com 14 leis escritas). Deduzir nao piora e nao multiplica.

  D3  FISICA - RUIDO <= 0
      -> a deducao so recupera o que eu ja teria escrito, ou menos. Nulo
         limpo, e fecha a linha de aquisicao de prior por mundo.

MINHA PREVISAO, escrita antes de rodar: D2, e sem entusiasmo. A meta-gramatica
`TESTE ? LEITURA : LEITURA` e por-celula com leitura de vizinhanca limitada, e
K passos de queda com empilhamento e um efeito de ALCANCE LONGO — uma particula
desce ate encontrar apoio, o que depende de tudo que esta abaixo dela. Aposto
que a deducao exata falha quase sempre e que a parcial captura versoes de um
passo so (a particula desce UMA casa), que e justamente o `w_cai_baixo` que eu
ja tinha escrito a mao.

Se eu estiver certo, o achado nao e o numero: e a razao. Seria a medida de que
a meta-gramatica tem alcance curto demais para capturar consequencia emergente,
e o gargalo da aquisicao de prior passa a ter endereco — nao e o mundo, nao e o
metodo de acesso, e o ALCANCE da linguagem que descreve a lei.
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from novaphys.arc import Grid, carregar, dims, fundo, valida
from novaphys.eter import PASTA_AVAL
from novaphys.eter_inferencia import aplicar, inferir
from novaphys.fisica import (PRIMITIVAS, ajustar_mapa, aplicar_mapa)
from novaphys.relaxado import cobertura_bits
from novaphys.feixe import parciais

LADO = 12
N_CORES = 10


# ══════════════════════════════════════════════ o simulador de particulas
class Simulador:
    """
    Regras LOCAIS apenas. O mapa grade->grade depois de K passos e emergente:
    eu nao o escrevi e nao saberia escrever em forma fechada.

    `fisico=False` mantem tudo — mesmas particulas, mesma tinta, mesmo numero
    de passos, mesmo determinismo — e troca a queda por um deslocamento fixo
    sorteado por cor. Mundo consistente e aprendivel, sem fisica.
    """

    def __init__(self, rng: random.Random, fisico: bool):
        self.fisico = fisico
        self.g = [[0] * LADO for _ in range(LADO)]
        self.desloc = {c: (rng.randrange(-1, 2), rng.randrange(-1, 2))
                       for c in range(1, N_CORES)}
        for _ in range(rng.randint(8, 16)):
            y, x = rng.randrange(LADO // 2), rng.randrange(LADO)
            self.g[y][x] = rng.randint(1, N_CORES - 1)

    def _livre(self, y: int, x: int) -> bool:
        return 0 <= y < LADO and 0 <= x < LADO and self.g[y][x] == 0

    def passo(self) -> None:
        if self.fisico:
            # queda com empilhamento e escorregamento lateral: cada particula
            # desce se puder, senao tenta a diagonal. E so isso — a forma da
            # pilha e emergente.
            for y in range(LADO - 2, -1, -1):
                for x in range(LADO):
                    c = self.g[y][x]
                    if not c:
                        continue
                    if self._livre(y + 1, x):
                        self.g[y][x], self.g[y + 1][x] = 0, c
                    elif self._livre(y + 1, x - 1):
                        self.g[y][x], self.g[y + 1][x - 1] = 0, c
                    elif self._livre(y + 1, x + 1):
                        self.g[y][x], self.g[y + 1][x + 1] = 0, c
        else:
            novo = [[0] * LADO for _ in range(LADO)]
            for y in range(LADO):
                for x in range(LADO):
                    c = self.g[y][x]
                    if not c:
                        continue
                    dy, dx = self.desloc[c]
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < LADO and 0 <= nx < LADO and not novo[ny][nx]:
                        novo[ny][nx] = c
                    else:
                        novo[y][x] = c
            self.g = novo

    def grade(self) -> Grid:
        return tuple(tuple(l) for l in self.g)


def pares(fisico: bool, n_cenas: int, passos: int, semente: int
          ) -> List[List[Tuple[Grid, Grid]]]:
    """
    Cada episodio e um fenomeno estavel observado n vezes: mesmo numero de
    passos, cenas diferentes. O agente nunca recebe o NOME da lei nem a lista
    de leis possiveis — so os pares.
    """
    rng = random.Random(semente)
    eps: List[List[Tuple[Grid, Grid]]] = []
    for k in range(1, passos + 1):
        obs: List[Tuple[Grid, Grid]] = []
        for _ in range(n_cenas):
            s = Simulador(rng, fisico)
            a = s.grade()
            for _t in range(k):
                s.passo()
            b = s.grade()
            if a != b:
                obs.append((a, b))
        if len(obs) >= 3:
            eps.append(obs)
    return eps


# ══════════════════════════════════════════════ deducao (nao amostragem)
def deduzir(eps, limiar: float, teto: int, gasto_max: int
            ) -> Tuple[List[Tuple[str, Callable]], int, int]:
    """
    Para cada episodio, RESOLVE para o operador em vez de sortear. Primeiro
    exato; se falhar, parcial no limiar de `relaxado.py`.
    """
    fora: List[Tuple[str, Callable]] = []
    gasto, exatos = 0, 0
    for i, obs in enumerate(eps):
        op, g = inferir(obs)
        gasto += g
        achou = []
        if op is not None:
            achou = [op]
            exatos += 1
        else:
            cand, g2 = parciais(obs, limiar, teto, gasto_max, True, 0.9)
            gasto += g2
            achou = cand[:teto]
        for j, p in enumerate(achou):
            def como_grid(g_: Grid, p=p) -> Optional[Grid]:
                try:
                    return aplicar(p, g_)
                except Exception:
                    return None
            fora.append((f"d{i}_{j}", como_grid))
    return fora, gasto, exatos


# ══════════════════════════════════════════════ busca no ARC
def resolver(treino, extras, orcamento: int, prof: int):
    d = dict(PRIMITIVAS)
    d.update(dict(extras))
    nomes = list(d)
    gasto = 0
    for p in range(0, prof + 1):
        for seq in itertools.product(nomes, repeat=p):
            if gasto >= orcamento:
                return None, None
            gasto += 1
            saidas, ok = [], True
            for a, _b in treino:
                r: Optional[Grid] = a
                for n in seq:
                    r = d[n](r)
                    if not valida(r):
                        ok = False
                        break
                if not ok:
                    break
                saidas.append(r)
            if not ok:
                continue
            if all(r == b for r, (_, b) in zip(saidas, treino)):
                return list(seq), None
            m = ajustar_mapa(list(zip(saidas, [b for _, b in treino])))
            if m is not None and all(aplicar_mapa(r, m) == b
                                     for r, (_, b) in zip(saidas, treino)):
                return list(seq), m
    return None, None


def confere(seq, mapa, extras, teste) -> bool:
    d = dict(PRIMITIVAS)
    d.update(dict(extras))
    for a, b in teste:
        r: Optional[Grid] = a
        for n in seq:
            r = d[n](r)
            if not valida(r):
                return False
        if mapa:
            r = aplicar_mapa(r, mapa)
        if r != b:
            return False
    return True


def avaliar(tarefas, extras, orcamento, prof, rotulo) -> Set[str]:
    ok: Set[str] = set()
    t0 = time.time()
    for nome, tr, te in tarefas:
        seq, mapa = resolver(tr, extras, orcamento, prof)
        if seq is not None and confere(seq, mapa, extras, te):
            ok.add(nome)
    print(f"  {rotulo:10s} {len(ok):3d}/{len(tarefas)}  "
          f"({len(ok)/len(tarefas):5.2%})  {len(extras):2d} deduzidas  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--cenas", type=int, default=6)
    ap.add_argument("--passos", type=int, default=8)
    ap.add_argument("--limiar", type=float, default=0.85)
    ap.add_argument("--teto", type=int, default=3)
    ap.add_argument("--orcamento", type=int, default=14424)
    ap.add_argument("--prof", type=int, default=2)
    ap.add_argument("--semente", type=int, default=20260907)
    a = ap.parse_args()

    print("=" * 74)
    print("DEDUCAO — deduzir a lei do mundo, em vez de filtrar uma lista minha")
    print("=" * 74)

    ef = pares(True, a.cenas, a.passos, a.semente)
    er = pares(False, a.cenas, a.passos, a.semente + 1)
    print(f"\nD0 · episodios com pares nao-triviais: "
          f"fisica {len(ef)}, ruido {len(er)}")
    if not ef:
        print("     D0 FALHOU: o simulador de fisica nao produziu pares.")
        return

    df, gf, xf = deduzir(ef, a.limiar, a.teto, 200000)
    dr, gr, xr = deduzir(er, a.limiar, a.teto, 200000)
    print(f"     FISICA {len(df):2d} operadores deduzidos "
          f"({xf} exatos, resto parcial)  {gf} candidatos gastos")
    print(f"     RUIDO  {len(dr):2d} operadores deduzidos "
          f"({xr} exatos, resto parcial)  {gr} candidatos gastos")
    if not dr:
        print("\n     D0 · o braco RUIDO nao deduziu nada. Sem controle a")
        print("     comparacao vira 'com operadores contra sem'. Abortando.")
        return
    k = min(len(df), len(dr))
    df, dr = df[:k], dr[:k]
    print(f"     casados em {k} operadores cada\n")

    tarefas = carregar(a.pasta)
    print(f"orcamento casado {a.orcamento}, profundidade {a.prof}\n")
    base = avaliar(tarefas, [], a.orcamento, a.prof, "BASE")
    rui = avaliar(tarefas, dr, a.orcamento, a.prof, "RUIDO")
    fis = avaliar(tarefas, df, a.orcamento, a.prof, "FISICA")

    d = len(fis) - len(rui)
    print(f"\nFISICA - BASE  = {len(fis) - len(base):+d}")
    print(f"FISICA - RUIDO = {d:+d}   <- o que decide")
    print(f"\nso a FISICA resolve : {sorted(fis - base - rui)}")
    print(f"so o RUIDO resolve  : {sorted(rui - base - fis)}")

    print("\n" + "=" * 74)
    if d >= 3:
        print("D1 · DEDUZIR do mundo entrega prior que FILTRAR do mundo nao")
        print("     entrega. A troca de metodo de acesso funciona tambem na")
        print("     aquisicao, e nao so na busca.")
    elif d >= 1:
        print("D2 · funciona, no mesmo patamar de escrever a lista a mao")
        print("     (a dobra deu +3 com 14 leis escritas). Deduzir nao piora")
        print("     e nao multiplica.")
    else:
        print("D3 · a deducao so recupera o que eu ja teria escrito, ou menos.")
        print("     Se os exatos forem poucos, o endereco do gargalo e o")
        print("     ALCANCE da meta-gramatica: `TESTE ? LEITURA : LEITURA` e")
        print("     por-celula e consequencia emergente tem alcance longo.")


if __name__ == "__main__":
    main()
