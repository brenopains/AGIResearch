"""
novaphys.experiencia — o Sistema 1 acumula experiencia entre problemas?
=========================================================================

A OUTRA METADE DA PAREDE
--------------------------
[[sistema1-transducao]] mostrou que um sistema que COMPLETA o padrao, aprendido
na propria tarefa, resolve tarefas que a linguagem explicita nao alcanca. Mas
ele aprende cada tarefa DO ZERO e esquece. Um humano nao: o Sistema 1 dele e
moldado por tudo que ele ja viu. A tarefa 400 fica mais facil que a 1.

Isto e o que o projeto nunca teve em lugar nenhum — [[arc-numero-oficial]]
(orcamento plano), [[ligar-nao-basta]], [[assinatura-nao-e-mecanismo]] (3 builds
de recuperacao por assinatura: 0/400): nada transforma experiencia passada em
intuicao para um problema novo. DreamCoder chama isso de modelo de
reconhecimento amortizado, e mostrou que e metade do que faz a biblioteca pagar
(arxiv 2306.07856 decompila justamente esse conhecimento).

O TESTE
---------
Duas metades DISJUNTAS das tarefas de mesma dimensao (A e B, por hash do nome).
Pre-treina-se UMA rede em todos os pares de A — treino E teste de A, porque A
nao e avaliada: e experiencia de vida. Depois cada tarefa de B e resolvida pelo
mesmo procedimento de [[sistema1-transducao]], so que partindo dos pesos
pre-treinados em vez do zero. Depois troca: pre-treina em B, resolve A. Assim
as 262 sao avaliadas e nenhuma tarefa ve a si mesma no pre-treino.

  DO ZERO      o Sistema 1 de hoje (resultado ja medido, semente 0)
  EXPERIENCIA  pre-treinado na outra metade
  PLACEBO      pre-treinado no MESMO volume da outra metade, mas com entrada de
               um par e saida de OUTRO par de outra tarefa — mesmas grades, mesmas
               cores, mesmo numero de passos, zero relacao entrada->saida. Isola
               "ver muitas grades" de "aprender como transformacoes funcionam".

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  E1  A EXPERIENCIA SE ACUMULA: EXPERIENCIA resolve >= DO ZERO + 5 nas 262 E o
      PLACEBO fica <= DO ZERO + 1
      -> o Sistema 1 carrega o que aprendeu em outros problemas; e a primeira
         vez que experiencia vira intuicao neste projeto.
  E2  EXPERIENCIA ~ DO ZERO (dentro de +-2) -> aprender por tarefa e tudo que
      ha nesta escala; a experiencia nao compoe.
  E3  PLACEBO tambem ganha (> DO ZERO + 1) -> o ganho e de "ver grades"
      (estatistica de pixel), nao de relacao; E1 fica invalidado mesmo se passar.
  E4  EXPERIENCIA < DO ZERO - 2 -> o pre-treino atrapalha (vies errado que o
      ajuste fino nao desfaz).

PREVISAO: E1 a 0,45, E2 a 0,30, E3 a 0,15, E4 a 0,10. Mecanismo de E1: as
tarefas do ARC compartilham PRIMITIVAS perceptuais (objeto, contorno, simetria,
preenchimento), e uma rede que ja viu 131 tarefas comeca o ajuste fino com
detectores dessas primitivas prontos. Risco: 131 tarefas e pouco — o que funciona
na literatura pre-treina em dezenas de milhares (re-ARC); e 600 passos de ajuste
fino podem apagar o pre-treino.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from novaphys.arc import carregar
from novaphys.promessa import PASTA
from novaphys import sistema1 as S

PRE_PASSOS = 8000


def _metade(nome: str) -> int:
    return int(hashlib.md5(nome.encode()).hexdigest(), 16) % 2


def pretreina(tarefas, placebo: bool, semente: int = 0):
    """Uma rede, todos os pares das tarefas dadas, aumento d4+cor."""
    rnd = random.Random(semente)
    torch.manual_seed(semente)
    pares = [(np.array(a), np.array(b)) for _n, tr, te in tarefas
             for a, b in tr + te]
    if placebo:
        # mesma forma de grade exigida (a rede e celula a celula): embaralha as
        # saidas DENTRO de cada grupo de mesma forma, cruzando tarefas
        por_forma: Dict[tuple, List[int]] = {}
        for i, (a, _b) in enumerate(pares):
            por_forma.setdefault(a.shape, []).append(i)
        novos = list(pares)
        for idx in por_forma.values():
            saidas = [pares[i][1] for i in idx]
            rnd.shuffle(saidas)
            for i, b in zip(idx, saidas):
                novos[i] = (pares[i][0], b)
        pares = novos
    rede = S.Rede().to(S.DEV)
    opt = torch.optim.Adam(rede.parameters(), lr=1e-3)
    rede.train()
    for _ in range(PRE_PASSOS):
        x, y, m = S._amostra(pares, rnd, True, True)
        L = (F.cross_entropy(rede(x), y, reduction="none") * m).sum() / m.sum()
        opt.zero_grad()
        L.backward()
        opt.step()
    return rede.state_dict()


def resolve_de(pesos, tr, te, semente: int = 0):
    """O procedimento de sistema1.resolve, partindo de pesos pre-treinados."""
    rnd = random.Random(semente)
    torch.manual_seed(semente)
    pares = [(np.array(a), np.array(b)) for a, b in tr]
    for nome, gira, cor in S.GRUPOS:
        rede = S.Rede().to(S.DEV)
        rede.load_state_dict(copy.deepcopy(pesos))
        opt = torch.optim.Adam(rede.parameters(), lr=2e-3)
        rede.train()
        for _ in range(S.PASSOS):
            x, y, m = S._amostra(pares, rnd, gira, cor)
            L = (F.cross_entropy(rede(x), y, reduction="none") * m).sum() / m.sum()
            opt.zero_grad()
            L.backward()
            opt.step()
        rede.eval()
        if all(np.array_equal(S._preve(rede, a, gira), b) for a, b in pares):
            ok = all(np.array_equal(S._preve(rede, np.array(a), gira), np.array(b))
                     for a, b in te)
            return ok, nome
    return False, "nenhum reproduz o treino"


def experimento() -> Dict:
    tarefas = [(n, tr, te) for n, tr, te in carregar(PASTA) if S._mesma(tr, te)]
    zero = set(json.load(open("resultados/sistema1.json"))["resolve"])
    A = [t for t in tarefas if _metade(t[0]) == 0]
    B = [t for t in tarefas if _metade(t[0]) == 1]
    print("=" * 78)
    print("EXPERIENCIA — o Sistema 1 acumula entre problemas?")
    print("=" * 78)
    print("  metades disjuntas: A %d, B %d tarefas. DO ZERO resolve %d (medido)"
          % (len(A), len(B), len(zero)))
    print("  pre-treino %d passos na OUTRA metade; ajuste fino %d passos\n"
          % (PRE_PASSOS, S.PASSOS), flush=True)
    res = {"EXPERIENCIA": [], "PLACEBO": []}
    t0 = time.time()
    for braco, placebo in (("EXPERIENCIA", False), ("PLACEBO", True)):
        for fonte, alvo in ((A, B), (B, A)):
            pesos = pretreina(fonte, placebo)
            for k, (n, tr, te) in enumerate(alvo):
                if resolve_de(pesos, tr, te)[0]:
                    res[braco].append(n)
            print("    %-11s metade feita | resolve %d | %.0f min"
                  % (braco, len(res[braco]), (time.time() - t0) / 60), flush=True)
    e, p, z = len(res["EXPERIENCIA"]), len(res["PLACEBO"]), len(zero)
    so_exp = sorted(set(res["EXPERIENCIA"]) - zero)
    perdeu = sorted(zero - set(res["EXPERIENCIA"]))
    nucleo = set(json.load(open("resultados/oficial.json")))
    print("\n" + "=" * 78)
    print("  DO ZERO %d | EXPERIENCIA %d | PLACEBO %d  (de %d)" % (z, e, p, len(tarefas)))
    print("  so a experiencia resolve: %d %s" % (len(so_exp), so_exp))
    print("  a experiencia perdeu:     %d %s" % (len(perdeu), perdeu))
    uniao = nucleo | zero | set(res["EXPERIENCIA"])
    print("  uniao nucleo + sistema1 (zero ou experiencia): %d/400" % len(uniao))
    if p > z + 1:
        ver = "E3"
        print("E3 . o PLACEBO tambem ganha (%+d): e estatistica de grade, nao relacao." % (p - z))
    elif e >= z + 5:
        ver = "E1"
        print("E1 . A EXPERIENCIA SE ACUMULA: %+d tarefas sobre aprender do zero," % (e - z))
        print("     placebo %+d. Experiencia de outros problemas vira intuicao." % (p - z))
    elif e < z - 2:
        ver = "E4"
        print("E4 . o pre-treino ATRAPALHA (%+d)." % (e - z))
    else:
        ver = "E2"
        print("E2 . experiencia nao compoe nesta escala (%+d)." % (e - z))
    r = {"veredito": ver, "zero": z, "experiencia": res["EXPERIENCIA"],
         "placebo": res["PLACEBO"], "so_experiencia": so_exp, "perdeu": perdeu,
         "uniao": len(uniao)}
    json.dump(r, open("resultados/experiencia.json", "w"), indent=1)
    return r


if __name__ == "__main__":
    experimento()
