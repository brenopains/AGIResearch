"""
novaphys.proponente — quem propoe a proxima tentativa: acaso, habito, ou analogia
====================================================================================

O GARGALO
-----------
O unico canal de estrutura nova neste projeto sou eu escrevendo primitivas. O
usuario apontou o ponto: enquanto for assim, cada conserto so desce o teto um
degrau. Um proponente automatico e o orgao que falta — e a pergunta dele foi se
da para faze-lo com HDC em vez de um LLM.

A RESPOSTA HONESTA, E O QUE ESTE EXPERIMENTO MEDE
---------------------------------------------------
HDC nao importa conhecimento de fora: ele RECOMBINA e RECUPERA o que ja passou.
Entao ele nao substitui o LLM como canal, mas e exatamente a peca da ANALOGIA: uma
memoria que se atualiza a cada experiencia e diz "este contexto se parece com
aquele, la funcionou isto".

Aqui o proponente nao inventa primitiva nova: ele decide a ORDEM em que os
candidatos ja existentes sao avaliados, com o teto de avaliacoes FIXO e igual para
todos os bracos. Com orcamento apertado, ordem boa = mais tarefas resolvidas. E a
medida limpa da qualidade da proposta, sem mudar mais nada no nucleo.

    fixo        a ordem historica dos codecs (linha de base)
    sorteado    ordem ao acaso (placebo)
    habito      ordena pelo que mais funcionou ANTES, ignorando o contexto
    hdc         analogia: liga o contexto atual com o candidato e pontua pela
                semelhanca com a memoria de sucessos passados

A memoria e ONLINE: as tarefas sao percorridas numa ordem fixa e a memoria so
contem o que veio ANTES daquela tarefa. Sem vazamento por construcao.

O braco HABITO e o controle que importa: ele separa "analogia" de "repetir o que
costuma dar certo". Contra o sorteado, qualquer memoria ganha.

PRE-REGISTRO, ANTES DE RODAR — 400 tarefas, teto de 400 avaliacoes por tarefa
--------------------------------------------------------------------------------
(teto reduzido de 1000 para 400 ANTES de rodar, declarado: no teste de fumaca com
24 tarefas o braco HDC levou 3x mais relogio que o fixo — nao por overhead do
vetor, mas porque ele prioriza codecs mais CAROS de avaliar. O orcamento e contado
em AVALIACOES, entao a comparacao segue justa; o teto menor traz a rodada de ~5 h
para ~2 h e aperta mais a ordem, que e o que se quer medir.)
  P0  sanidade: o braco FIXO resolve entre 25 e 40 tarefas com este teto apertado
  P1  hdc - sorteado >= +3 tarefas
  P2  hdc - habito >= +2 tarefas   (o teste que separa analogia de habito)
  P3  nenhum braco fica ABAIXO do sorteado (se ficar, o arcabouco esta errado)
PREVISAO: P0 passa. P1 passa. P2 FALHA — aposto que HDC empata com habito, porque
o contexto que eu sei descrever (dimensoes, cores, fracao nao explicada, tamanho
da pilha) e pobre demais para separar situacoes. Se P2 passar, a analogia esta
carregando informacao de verdade e vale investir no proponente.
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from novaphys.arc import carregar, dims, fundo
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D

TETO = 1000
DIM = 4096
SEMENTE = 20261018


# ─── contexto e codificacao ──────────────────────────────────────────────────

def _rotulo(it):
    """Nome curto e estavel do candidato (codec ou marcador)."""
    if D._marca(it):
        return str(it).split(":")[0]
    cd, m = it
    if cd.nome == "programa" and isinstance(m, tuple) and len(m) == 2:
        return f"programa:{m[1][0]}"       # o pintor distingue programas
    return cd.nome


class Base:
    """Proponente que nao faz nada: a ordem historica."""

    nome = "fixo"

    def contexto(self, treino, base, alvo, pilha):
        a0, b0 = base[0][0], treino[0][1]
        cels = sum(dims(b)[0] * dims(b)[1] for _a, b in treino)
        falta = sum(len(s) for s in alvo) / max(cels, 1) if alvo else 1.0
        return (
            ("dim_igual", dims(a0) == dims(b0)),
            ("cores", min(len({c for l in a0 for c in l}), 5)),
            ("falta", min(int(falta * 4), 3)),
            ("pilha", min(len(pilha), 3)),
            ("fundo", fundo(a0) == 0),
            ("grande", cels > 400),
        )

    def ordem(self, ctx, cand):
        return cand

    def registra(self, ctx, item, ganho):
        pass


class Sorteado(Base):
    nome = "sorteado"

    def __init__(self, semente):
        self.rng = random.Random(semente)

    def ordem(self, ctx, cand):
        c = list(cand)
        self.rng.shuffle(c)
        return c


class Habito(Base):
    """Ordena pelo que mais funcionou antes, sem olhar o contexto."""

    nome = "habito"

    def __init__(self, semente):
        self.peso = {}

    def ordem(self, ctx, cand):
        return sorted(cand, key=lambda it: -self.peso.get(_rotulo(it), 0.0))

    def registra(self, ctx, item, ganho):
        r = _rotulo(item)
        self.peso[r] = self.peso.get(r, 0.0) + 1.0


class HDC(Base):
    """Analogia: memoria hiperdimensional de (contexto ⊗ candidato) que funcionou."""

    nome = "hdc"

    def __init__(self, semente):
        self.semente = semente
        self.vet = {}
        self.M = np.zeros(DIM, dtype=np.float32)

    def _v(self, chave):
        v = self.vet.get(chave)
        if v is None:
            r = np.random.default_rng(abs(hash((self.semente, chave))) % (2 ** 32))
            v = r.integers(0, 2, DIM).astype(np.float32) * 2 - 1
            self.vet[chave] = v
        return v

    def _ctx(self, ctx):
        s = np.zeros(DIM, dtype=np.float32)
        for papel, valor in ctx:
            s += self._v(("papel", papel)) * self._v(("valor", papel, valor))
        n = np.linalg.norm(s)
        return s / n if n else s

    def ordem(self, ctx, cand):
        if not self.M.any():
            return cand
        c = self._ctx(ctx)
        nm = np.linalg.norm(self.M)
        alvo = self.M / nm
        pont = {}
        for it in cand:
            r = _rotulo(it)
            if r not in pont:
                pont[r] = float(np.dot(alvo, c * self._v(("cand", r))))
        return sorted(cand, key=lambda it: -pont[_rotulo(it)])

    def registra(self, ctx, item, ganho):
        c = self._ctx(ctx)
        self.M += c * self._v(("cand", _rotulo(item))) * float(min(ganho, 50.0))


BRACOS = {"fixo": Base, "sorteado": Sorteado, "habito": Habito, "hdc": HDC}


# ─── a corrida ───────────────────────────────────────────────────────────────

def roda(args):
    nome, tarefas, teto = args
    prop = BRACOS[nome](SEMENTE) if nome != "fixo" else Base()
    D.PROPONENTE = prop
    resolvidas, t0 = [], time.time()
    for k, (tarefa, tr, te) in enumerate(tarefas):
        try:
            fm, pilha, err = D.montar(tr, teto=teto)
            if pilha and err == 0 and all(D.aplicar((fm, pilha), x) == y for x, y in te):
                resolvidas.append(tarefa)
        except Exception:
            pass
        if (k + 1) % 100 == 0:
            print(f"    [{nome}] {k+1} tarefas, {len(resolvidas)} resolvidas, "
                  f"{time.time()-t0:.0f}s", flush=True)
    D.PROPONENTE = None
    return nome, resolvidas, time.time() - t0


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefas", type=int, default=400)
    ap.add_argument("--teto", type=int, default=TETO)
    ap.add_argument("--bracos", default="fixo,sorteado,habito,hdc")
    a = ap.parse_args()
    tarefas = carregar(PASTA_AVAL)[: a.tarefas]
    bracos = a.bracos.split(",")
    print("=" * 78)
    print(f"PROPONENTE — {len(tarefas)} tarefas, teto de {a.teto} avaliacoes, "
          f"memoria ONLINE (so o que veio antes)")
    print("=" * 78)
    t0 = time.time()
    with Pool(min(len(bracos), 4)) as pool:
        res = pool.map(roda, [(b, tarefas, a.teto) for b in bracos], chunksize=1)
    print(f"\n  ({time.time()-t0:.0f}s)\n")
    conta = {}
    for nome, resolvidas, dt in res:
        conta[nome] = set(resolvidas)
        print(f"  {nome:9s} {len(resolvidas):3d}/{len(tarefas)}   {dt:6.0f}s")
    print("")
    print("=" * 78)
    if {"fixo", "sorteado", "habito", "hdc"} <= set(conta):
        f, s = len(conta["fixo"]), len(conta["sorteado"])
        h, x = len(conta["habito"]), len(conta["hdc"])
        print(f"P0 {'.' if 30 <= f <= 44 else 'FORA .'} fixo resolveu {f} (esperado 30-44)")
        print(f"P1 {'.' if x - s >= 3 else 'FALHOU .'} hdc - sorteado = {x-s:+d} (>= +3)")
        print(f"P2 {'.' if x - h >= 2 else 'FALHOU .'} hdc - habito = {x-h:+d} (>= +2)")
        pior = min(f, h, x)
        print(f"P3 {'.' if pior >= s else 'FALHOU .'} ninguem abaixo do sorteado "
              f"(pior {pior} x sorteado {s})")
        print(f"\n  so o hdc resolve: {sorted(conta['hdc'] - conta['habito'] - conta['fixo'])}")
        print(f"  so o habito resolve: {sorted(conta['habito'] - conta['hdc'] - conta['fixo'])}")


if __name__ == "__main__":
    main()
