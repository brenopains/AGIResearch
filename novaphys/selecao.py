"""
novaphys.selecao — o gargalo nao era vocabulario, era escolher dentro dele
===========================================================================

O QUE DUAS MEDIDAS DE HOJE DERAM, JUNTAS
------------------------------------------
  GERAR funciona   `operador_inventado.py`: modelo local de 1,8 GB produziu 9
                   operadores relacionais bem definidos, 5 deles particoes
                   genuinamente novas, numa CATEGORIA que eu nao tinha (os meus
                   6 sao ordinais; ele acrescentou metricos — distancia a
                   distribuicao).
  SELECIONAR nao   `propriedade_inventada.py`, 80 tarefas, teto casado:

                       base          12/80
                       inventadas     9/80   ganhou 0, perdeu 3
                       placebo       11/80   ganhou 1, perdeu 2

O placebo com 85 propriedades ARBITRARIAS perdeu quase o mesmo que as 85 boas.
Logo o conteudo do vocabulario nao importou: **o TAMANHO importou**. Cada
propriedade consistente vira candidato, e cada candidato custa uma avaliacao do
teto. Ampliar vocabulario com orcamento fixo e estritamente prejudicial, mesmo
quando o vocabulario e bom.

O QUE ESTE MODULO MEDE
------------------------
Tres formas de escolher POUCAS propriedades entre as 85, antes de gastar
avaliacao. Todas com o MESMO teto, e o mesmo vocabulario de 85.

  tudo        oferece as 85 (o braco que ja perdeu 3 — linha de base do problema)
  barato      ordena por MDL DO MAPA e corta no top-K. O preco em bits do mapa
              propriedade->cor e calculavel SEM avaliar: uma propriedade que
              manda 10 objetos em 3 cores comprime; uma que manda 10 em 10 nao
              explica nada e custa a mesma avaliacao. Nao precisa de LLM.
  sorteado    corta no top-K ao ACASO (placebo pareado da selecao: controla
              "melhorou so porque sao menos candidatos")

A pergunta e limpa: a melhora vem de ESCOLHER BEM, ou so de escolher MENOS?
Se `sorteado` empatar com `barato`, o ganho e do corte e nao do criterio — e
entao o prior da LLM tambem nao vai servir, porque ele e um criterio.

PRE-REGISTRO, ANTES DE RODAR — 80 tarefas, teto 400, K = 8
------------------------------------------------------------
  S0  SANIDADE: `tudo` reproduz a perda ja medida (entre 8 e 10 de 80)
  S1  O CORTE PAGA: `barato` >= base (12/80), isto e, recupera as 3 perdidas
  S2  O CRITERIO PAGA: `barato` - `sorteado` >= +2  (o teste que separa escolher
      bem de escolher menos)
  S3  E ABRE: `barato` > base, ou seja >= 13/80 — o vocabulario ampliado passa a
      GANHAR quando ha selecao
PREVISAO: S0 passa. S1 passa (o corte devolve o orcamento). S2 e o que importa e
eu aposto que PASSA por pouco, +2 a +4, porque MDL do mapa e um sinal forte e
quase gratuito. S3 FALHA — aposto empate com a base, porque as 5 propriedades
metricas novas provavelmente nao sao o que falta nestas 80 tarefas. Se S2 passar
e S3 falhar, a conclusao pre-registrada e: a maquina de duas pecas (gerar +
selecionar) esta completa e correta, e o que falta e vocabulario CERTO, nao mais
vocabulario — o que muda o alvo do proximo experimento de "gerar mais" para
"gerar direcionado pelo residuo".

RESULTADO (16/09/2026, 80 tarefas, teto 400, K=8):
  base 12/80 | tudo(85) 9/80 (-3) | barato(K) 12/80 (+0) | sorteado(K) 12/80 (+0)
  S0 . S1 . S2 FALHOU (+0, precisava +2) . S3 FALHOU
  propriedades em pilha vencedora: []   <- NENHUMA, nem as minhas

MINHA PREVISAO ERROU no que importava: eu apostei que S2 PASSARIA por pouco
(+2 a +4) porque "MDL do mapa e sinal forte e quase gratuito". Deu +0 exato.
Escolher por MDL do mapa == escolher AO ACASO. Todo o ganho e do CORTE, nenhum
do CRITERIO — e como o prior da LLM tambem e um criterio, ele cai junto, como
estava pre-registrado.

A CONCLUSAO REAL, mais funda que a pre-registrada: nao e que falta "vocabulario
CERTO" no eixo de propriedade. O EIXO INTEIRO E INERTE nestas 80 tarefas —
nenhuma propriedade, minha ou inventada, aparece em pilha vencedora. As 12
resolvidas sao resolvidas por codecs que nao usam propriedade. Ampliar dilui,
cortar devolve, e QUAL cortar nao importa porque nenhuma e usada.

Isto ENCERRA a linha "gerar mais propriedades" e "selecionar melhor entre elas".
O gargalo de vocabulario esta noutro eixo, nao neste.
"""

from __future__ import annotations

import argparse
import random
import time
from math import log2

from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D
from novaphys.descricao import objetos, fundo, BITS_COR
from novaphys.propriedade_inventada import (VOCAB, INVENTADAS, _valor_estendido,
                                            CorPorPropMais, ApagaPorPropMais)

SEMENTE = 20261023


def _mdl_do_mapa(prop, treino, alvo):
    """Bits do mapa propriedade->cor, SEM avaliar a pilha.

    Um mapa com `v` valores distintos custa v*(BITS_COR+4) para escrever e
    explica `n` objetos. O ganho e o que ele economiza sobre escrever a cor de
    cada objeto separadamente: n*BITS_COR - v*(BITS_COR+4). Propriedade que manda
    10 objetos em 10 valores tem ganho negativo e nao devia consumir avaliacao.
    Devolve None se a propriedade e inconsistente (mesmo valor, cores
    diferentes) — nesse caso ela nem e candidata.
    """
    mp, n = {}, 0
    for k, (a, b) in enumerate(treino):
        bg = fundo(a)
        al = None if alvo is None else alvo[k]
        for c, _kk in objetos(a, bg):
            if al is not None and not (c & al):
                continue
            cs = {b[y][x] for y, x in c}
            if len(cs) != 1:
                return None
            d = next(iter(cs))
            try:
                v = _valor_estendido(prop, c, a, bg)
            except Exception:
                return None
            if v in mp and mp[v] != d:
                return None
            mp[v] = d
            n += 1
    if not mp or n == 0:
        return None
    return n * BITS_COR - len(mp) * (BITS_COR + 4)


def _particao(prop, treino, alvo):
    """Que objetos esta propriedade JUNTA, na forma canonica (rotulos renomeados
    pela ordem de primeira aparicao). Duas propriedades com a mesma particao sao
    indistinguiveis para um codec que aprende mapa arbitrario valor->cor."""
    vis, prox, fora = {}, 0, []
    for k, (a, b) in enumerate(treino):
        bg = fundo(a)
        al = None if alvo is None else alvo[k]
        for c, _kk in objetos(a, bg):
            if al is not None and not (c & al):
                continue
            try:
                v = repr(_valor_estendido(prop, c, a, bg))
            except Exception:
                return None
            if v not in vis:
                vis[v] = prox
                prox += 1
            fora.append(vis[v])
        fora.append(-1)                       # separador entre pares
    return tuple(fora) if prox > 1 else None  # particao trivial nao conta


class _Corta:
    """Mistura que restringe PROPS as K melhores por um criterio, por tarefa."""

    def __init__(self, k, modo, semente=SEMENTE):
        self.k = k
        self.modo = modo
        self.rng = random.Random(semente)

    def _escolhe(self, treino, alvo):
        """Top-K por MDL do mapa, mas SEM duplicata de particao.

        Sem o dedup, o corte enchia as 8 vagas com quase-duplicatas —
        `difmax_cor`, `maximo_cor`, `maximo_n_cores`, `difmax_n_cores`... Todas
        com o mesmo ganho, desempate alfabetico, e K gasto em redundancia. Como
        o codec aprende um mapa arbitrario valor->cor, duas propriedades que
        induzem a MESMA PARTICAO dos objetos sao a mesma propriedade (foi assim
        que se descobriu que o meu `difmax` era o meu `posto`). Entao cada vaga
        do K tem de comprar uma particao DIFERENTE."""
        if self.modo == "sorteado":
            c = list(VOCAB)
            self.rng.shuffle(c)
            return tuple(c[: self.k])
        pont = []
        for p in VOCAB:
            g = _mdl_do_mapa(p, treino, alvo)
            if g is not None:
                pont.append((-g, p))
        pont.sort()
        fora, vistas = [], set()
        for _g, p in pont:
            q = _particao(p, treino, alvo)
            if q is None or q in vistas:
                continue
            vistas.add(q)
            fora.append(p)
            if len(fora) >= self.k:
                break
        return tuple(fora) or VOCAB[: self.k]


class CorPorPropSel(CorPorPropMais, _Corta):
    nome = "cor_por_prop_sel"

    def __init__(self, k, modo):
        _Corta.__init__(self, k, modo)

    def propor(self, treino, alvo=None):
        self.PROPS = self._escolhe(treino, alvo)
        return CorPorPropMais.propor(self, treino, alvo)


class ApagaPorPropSel(ApagaPorPropMais, _Corta):
    nome = "apaga_prop_sel"

    def __init__(self, k, modo):
        _Corta.__init__(self, k, modo)

    def propor(self, treino, alvo=None):
        self.PROPS = self._escolhe(treino, alvo)
        return ApagaPorPropMais.propor(self, treino, alvo)


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
                        if (not D._marca(it) and it[0].nome.endswith("_sel")
                                and isinstance(it[1], tuple) and it[1]):
                            usadas.append((tarefa, it[1][0]))
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
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    tarefas = carregar(PASTA_AVAL)[: a.tarefas]
    print("=" * 78)
    print(f"SELECAO — {len(tarefas)} tarefas, teto {a.teto}, vocabulario de "
          f"{len(VOCAB)} cortado em K={a.k}")
    print("=" * 78)

    base = roda(tarefas, D.CODECS, a.teto, "base")
    tudo = roda(tarefas, D.CODECS + [CorPorPropMais(), ApagaPorPropMais()],
                a.teto, "tudo")
    ub = []
    bar = roda(tarefas, D.CODECS + [CorPorPropSel(a.k, "barato"),
                                    ApagaPorPropSel(a.k, "barato")],
               a.teto, "barato", ub)
    sor = roda(tarefas, D.CODECS + [CorPorPropSel(a.k, "sorteado"),
                                    ApagaPorPropSel(a.k, "sorteado")],
               a.teto, "sorteado")

    nov = sorted({p for _t, p in ub if p in INVENTADAS})
    print("")
    print("=" * 78)
    print(f"  base       {len(base):3d}/{len(tarefas)}")
    print(f"  tudo (85)  {len(tudo):3d}/{len(tarefas)}   "
          f"{len(tudo)-len(base):+d}")
    print(f"  barato (K) {len(bar):3d}/{len(tarefas)}   "
          f"{len(bar)-len(base):+d}")
    print(f"  sorteado(K){len(sor):3d}/{len(tarefas)}   "
          f"{len(sor)-len(base):+d}")
    print(f"\n  propriedades em pilha vencedora: {sorted({p for _t, p in ub})}")
    print(f"  dessas, INVENTADAS: {nov}")
    print("")
    print(f"S0 {'.' if 8 <= len(tudo) <= 10 else 'FORA .'} tudo reproduz a perda: "
          f"{len(tudo)} (8-10)")
    print(f"S1 {'.' if len(bar) >= len(base) else 'FALHOU .'} o corte paga: "
          f"{len(bar)} >= {len(base)}")
    print(f"S2 {'.' if len(bar)-len(sor) >= 2 else 'FALHOU .'} o CRITERIO paga: "
          f"barato - sorteado = {len(bar)-len(sor):+d} (>= +2)")
    print(f"S3 {'.' if len(bar) > len(base) else 'FALHOU .'} e ABRE: "
          f"{len(bar)} > {len(base)}")


if __name__ == "__main__":
    main()
