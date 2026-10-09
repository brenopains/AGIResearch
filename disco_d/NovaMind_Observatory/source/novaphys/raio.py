"""
novaphys.raio — o codec que escreve no VAZIO, e nao sobre objeto
==================================================================

COMO SE CHEGOU AQUI (duas correcoes de rota no mesmo dia)
----------------------------------------------------------
1. Eu afirmei por ANALOGIA com a fisica que faltava PROPRIEDADE no vocabulario.
   Construi 85 (80 novas, por 7 operadores relacionais). Medido em 80 tarefas:
   base 12/80, com as 85 -> 9/80, corte por MDL -> 12/80, corte ao acaso ->
   12/80, e **nenhuma propriedade apareceu em pilha vencedora**. Diagnostico
   errado.
2. Ai sim medi POR QUE as tarefas falham, em vez de supor:

       D. erra no FUNDO (o que devia aparecer do nada)   26   <- maior grupo
       C. dimensao muda e a pilha erra o conteudo        16
       E. erra DENTRO de objeto                          14
       A. nao montou pilha nenhuma                        8
       residuo no FUNDO: 34/64 das falhas (53%)
       pilha mais comum na falha: `copia` SOZINHA, 11 vezes

   `copia` sozinha = "a saida repete a entrada", acerta o fundo todo, e nao tem
   vocabulario para dizer O QUE APARECE. Todo o meu alfabeto le objeto e escreve
   SOBRE objeto; falta a classe que le objeto e escreve no VAZIO.
3. E medi O QUE aparece, nas 31 tarefas nao resolvidas com aparicao:

       raio        12/31 (39%)   5 delas com assinatura UNICA  <- este modulo
       copia_desl  10/31 (32%)   5 unicas
       molde       10/31 (32%)   2 unicas
       halo         7/31 (23%)
       preenche     5/31 (16%)   (o codec `preenche` JA existe: falha por outro
                                  motivo, provavelmente a regra da cor)
       sem assinatura conhecida  5/31

POR QUE NAO E O `cruz` QUE JA EXISTE
--------------------------------------
`LinhaObjeto` (nome "cruz") e o caso DEGENERADO de raio:

    pinta a linha e a coluna INTEIRAS          -> nunca para em obstaculo
    so para objetos de len(c) <= 4             -> exclui raio de objeto grande
    sempre com a cor do proprio objeto          -> nao aprende outra cor
    sem diagonal, sem direcao unica             -> 3 marcadores apenas

O que a medida pede e exatamente o que falta nele: o raio que PARA ao bater, em
direcao unica ou diagonal, com cor possivelmente diferente da fonte.

A LICAO DO ORCAMENTO, APLICADA NO DESENHO
-------------------------------------------
Hoje ficou medido que ampliar vocabulario com teto fixo e ESTRITAMENTE
prejudicial — o placebo com 85 propriedades arbitrarias perdeu quase o mesmo que
as 85 boas, entao o dano e do TAMANHO e nao do conteudo. Logo `propor` aqui nao
enumera o espaco de marcadores: ele testa cada combinacao contra o treino e
devolve SO as consistentes, deduzindo a cor como o `mapa_cor` faz. Em tarefa que
nao e de raio, devolve lista vazia e custa zero avaliacao.

PRE-REGISTRO, ANTES DE RODAR — 80 tarefas, teto 400
-----------------------------------------------------
  R1  NAO INFLA: mediana de marcadores propostos por tarefa <= 2, e em pelo
      menos 60% das tarefas propoe ZERO (a maioria nao e de raio)
  R2  GANHA: >= +2 tarefas sobre a base, teto casado
  R3  NAO QUEBRA: 0 perdas
  R4  PLACEBO: `raio` com as direcoes EMBARALHADAS (mesma cardinalidade de
      marcadores, geometria destruida) ganha 0
PREVISAO: R1 passa. R2 aposto +2 a +4 — das 12 com assinatura de raio, 5 sao
assinatura unica e essas sao as que realmente dependem so disto; as outras 7
co-ocorrem com molde/copia_desl e provavelmente precisam das outras classes
tambem. R3 passa. R4 passa. Se R2 falhar com R1 passando, o problema nao e
orcamento nem expressividade: e a REGRA do raio (fonte, parada ou cor) estar
mal capturada, e o proximo passo e olhar as 5 unicas uma por uma.

RESULTADO (16/09/2026, 80 tarefas, teto 400):
  base 12/80 | com raio 11/80 (ganhou 0, perdeu 1) | placebo 10/80 (0, perdeu 2)
  raio entrou em pilha vencedora em 1 tarefa (17b80ad2); GANHOU zero.
  R1 FALHOU (zero em 55%, precisava 60%; MAXIMO 166 marcadores numa tarefa)
  R2 FALHOU (-1)  R3 FALHOU (1 perda)  R4 . (placebo ganhou 0)

DIAGNOSTICO: 12 tarefas tinham assinatura de raio e o codec entrou na pilha em 1.
Nao e que ele e caro e perde na avaliacao — a descricao dele NAO COMPRIME melhor
que o que ja existe, mesmo onde a assinatura estava. R1 tambem falhou na cauda
(166 marcadores numa tarefa), entao ha inflacao de orcamento junto.

O PADRAO, com 8 pontos (6 nulos + 2 sucessos): abducao funciona quando o termo e
CONSTRUIDO do dado compondo operacoes dadas (h = x*vy - y*vx 10/10; n^ = rot(v^)
0%->95%), e falha quando e PROPOSTO DE FORA — por mim (propriedades, esta classe
raio), pela LLM (0/30) ou por enumeracao (biblioteca +0, reordenacao 40/400).
Nenhuma das tres tentativas de 16/09 construiu o termo OLHANDO O RESIDUO.
"""

from __future__ import annotations

import argparse
import random
import time

from novaphys.arc import carregar, dims
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D
from novaphys.descricao import objetos, fundo, BITS_COR

SEMENTE = 20261024

DIRS = {
    "4":     ((1, 0), (-1, 0), (0, 1), (0, -1)),
    "8":     ((1, 0), (-1, 0), (0, 1), (0, -1),
              (1, 1), (1, -1), (-1, 1), (-1, -1)),
    "diag":  ((1, 1), (1, -1), (-1, 1), (-1, -1)),
    "baixo": ((1, 0),),
    "cima":  ((-1, 0),),
    "dir":   ((0, 1),),
    "esq":   ((0, -1),),
}


class Raio:
    """De cada celula de objeto sai um feixe, que pode PARAR ao bater.

    marcador = (dirs, para, so_pixel, cor)
        dirs       chave de DIRS
        para       True = o feixe morre na primeira celula nao-fundo
        so_pixel   True = so objetos de uma celula emitem
        cor        None = usa a cor da fonte; int = cor fixa aprendida
    """

    nome = "raio"

    def __init__(self, embaralha=False, semente=SEMENTE):
        self.embaralha = embaralha
        if embaralha:
            # PLACEBO PAREADO: mesma cardinalidade de marcadores, geometria
            # destruida — as direcoes deixam de ser vizinhancas coerentes.
            r = random.Random(semente)
            self.dirs = {}
            for k, v in DIRS.items():
                todas = list(DIRS["8"])
                r.shuffle(todas)
                self.dirs[k] = tuple(todas[: len(v)])
        else:
            self.dirs = DIRS

    def _feixe(self, a, dirs, para, fonte):
        """As celulas de fundo que o feixe atinge, com a cor da fonte.

        DEFEITO CORRIGIDO, achado olhando `05a7bcf2` celula por celula: o raio
        cobria 211 de 211 celulas que apareceram — a geometria estava certa — e
        escrevia 352 EXTRAS. A causa: a linha horizontal de 8s daquela tarefa e
        um objeto, entao ela TAMBEM emitia raios para baixo e preenchia a metade
        inferior inteira. A tarefa exige distinguir QUEM EMITE de QUEM BARRA, e
        o meu discriminador era `so_pixel`, que nao tem nada a ver com isso.
        Agora `fonte` diz qual cor emite (None = todas, "pixel" = so objetos de
        uma celula), e a barreira e qualquer celula nao-fundo quando `para`."""
        bg = fundo(a)
        h, w = dims(a)
        fora = {}
        for c, k in objetos(a, bg):
            if fonte == "pixel":
                if len(c) != 1:
                    continue
            elif fonte is not None and k != fonte:
                continue
            for y, x in c:
                for dy, dx in self.dirs[dirs]:
                    i, j = y + dy, x + dx
                    while 0 <= i < h and 0 <= j < w:
                        if a[i][j] != bg:
                            if para:
                                break
                            i, j = i + dy, j + dx
                            continue
                        if (i, j) not in fora:
                            fora[(i, j)] = k
                        i, j = i + dy, j + dx
        return fora

    def propor(self, treino, alvo=None):
        """So devolve marcador CONSISTENTE com o treino. Em tarefa que nao e de
        raio, devolve [] e nao custa avaliacao nenhuma."""
        cores = set()
        for a, _b in treino:
            bg = fundo(a)
            for _c, k in objetos(a, bg):
                cores.add(k)
        fontes = [None, "pixel"] + sorted(cores)
        fora = []
        for dirs in self.dirs:
            for para in (True, False):
                for so_pixel in fontes:
                    cor_fixa, ok, mudou = None, True, 0
                    for kk, (a, b) in enumerate(treino):
                        if dims(a) != dims(b):
                            ok = False
                            break
                        bg = fundo(a)
                        al = None if alvo is None else alvo[kk]
                        f = self._feixe(a, dirs, para, so_pixel)
                        if not f:
                            continue
                        for p, k in f.items():
                            if al is not None and p not in al:
                                continue
                            d = b[p[0]][p[1]]
                            # DEFEITO CORRIGIDO: eu contava como "consistente"
                            # a celula de fundo que CONTINUOU fundo. Assim
                            # qualquer direcao passava (28 marcadores numa
                            # tarefa, com cor fixa 0 = o proprio fundo), e como
                            # quem escreve primeiro ganha, esses marcadores
                            # degenerados iam BLOQUEAR outros codecs escrevendo
                            # o que ja estava la. O raio tem de explicar celula
                            # que MUDOU.
                            if d == bg:
                                continue
                            mudou += 1
                            if d == k:
                                continue
                            if cor_fixa is None:
                                cor_fixa = d
                            elif cor_fixa != d:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok and mudou and cor_fixa != fundo(treino[0][0]):
                        fora.append((dirs, para, so_pixel, cor_fixa))
        return fora

    def bits(self, m):
        # 3 bits para a direcao, 1 para parada, 1 para so_pixel, cor se houver
        return 5.0 + (BITS_COR if m[3] is not None else 0.0)

    def prever(self, m, a):
        dirs, para, so_pixel, cor = m
        f = self._feixe(a, dirs, para, so_pixel)
        if cor is None:
            return f
        return {p: cor for p in f}


# ─── o portao ────────────────────────────────────────────────────────────────

def roda(tarefas, codecs, teto, etiqueta, usou=None):
    ok, t0 = set(), time.time()
    for k, (tarefa, tr, te) in enumerate(tarefas):
        try:
            fm, pilha, err = D.montar(tr, codecs=codecs, teto=teto)
            if pilha and err == 0 and all(D.aplicar((fm, pilha), x) == y
                                          for x, y in te):
                ok.add(tarefa)
                if usou is not None and any(not D._marca(it) and it[0].nome == "raio"
                                            for it in pilha):
                    usou.append(tarefa)
        except Exception:
            pass
        if (k + 1) % 25 == 0:
            print(f"    [{etiqueta}] {k+1} tarefas, {len(ok)} resolvidas, "
                  f"{time.time()-t0:.0f}s", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefas", type=int, default=80)
    ap.add_argument("--teto", type=int, default=400)
    a = ap.parse_args()
    tarefas = carregar(PASTA_AVAL)[: a.tarefas]
    print("=" * 78)
    print(f"RAIO — {len(tarefas)} tarefas, teto {a.teto} (casado nos tres bracos)")
    print("=" * 78)

    # R1: quantos marcadores ele propoe, por tarefa
    r = Raio()
    quantos = []
    for _t, tr, _te in tarefas:
        try:
            quantos.append(len(r.propor(tr, None)))
        except Exception:
            quantos.append(0)
    quantos.sort()
    med = quantos[len(quantos) // 2]
    zeros = sum(1 for q in quantos if q == 0)
    print(f"  marcadores por tarefa: mediana {med}, maximo {max(quantos)}, "
          f"zero em {zeros}/{len(tarefas)}")

    base = roda(tarefas, D.CODECS, a.teto, "base")
    usou = []
    com = roda(tarefas, D.CODECS + [Raio()], a.teto, "com raio", usou)
    plac = roda(tarefas, D.CODECS + [Raio(embaralha=True)], a.teto, "placebo")

    print("")
    print("=" * 78)
    print(f"  base      {len(base):3d}/{len(tarefas)}")
    print(f"  com raio  {len(com):3d}/{len(tarefas)}   ganhou "
          f"{len(com-base)}, perdeu {len(base-com)}")
    print(f"  placebo   {len(plac):3d}/{len(tarefas)}   ganhou "
          f"{len(plac-base)}, perdeu {len(base-plac)}")
    print(f"\n  raio entrou na pilha em: {sorted(usou)}")
    print(f"  ganhou: {sorted(com - base)}")
    print("")
    print(f"R1 {'.' if med <= 2 and zeros >= 0.6*len(tarefas) else 'FALHOU .'} "
          f"nao infla: mediana {med} (<= 2), zero em "
          f"{100*zeros/len(tarefas):.0f}% (>= 60%)")
    print(f"R2 {'.' if len(com)-len(base) >= 2 else 'FALHOU .'} ganha "
          f"{len(com)-len(base):+d} (>= +2)")
    print(f"R3 {'.' if not (base-com) else 'FALHOU .'} nao quebra: "
          f"{len(base-com)} perdas (0)")
    print(f"R4 {'.' if len(plac-base) == 0 else 'FALHOU .'} placebo ganhou "
          f"{len(plac-base)} (== 0)")


if __name__ == "__main__":
    main()
