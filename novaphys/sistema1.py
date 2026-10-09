"""
novaphys.sistema1 — a metade que falta: completar o padrao sem escrever a regra
=================================================================================

A PAREDE, COMO OS DADOS DE HOJE A DESENHAM
--------------------------------------------
Tres frentes independentes bateram no mesmo lugar no mesmo dia:

  R4 ([[gradiente-fora-da-amostra]])  admitir material novo que generaliza fora
        da amostra nao mudou UMA resolucao em 140 tarefas: base = tudo = gradiente
  [[protocolo-nao-percepcao]]  sem rotulo, TODA representacao para em ~0,70 no
        HAR; ver melhor nao ajuda
  [[mente-continua-v1]]  regimes semanticamente enormes quase nao mudam o preditor
        (6%, 0%, 5%): surpresa sozinha nao ve o que importa

E a velha: [[canal-de-importacao]] (40/60 falhas sao vocabulario AUSENTE),
[[ligar-nao-basta]] ("saber ONDE falhar nao ajuda quando falta O QUE procurar"),
[[arc-numero-oficial]] (curva de orcamento plana: buscar mais nao acha mais).

Tudo isso diz a mesma coisa sobre a arquitetura: ela e um SISTEMA 2 — linguagem
explicita, busca, verificacao exata. So conhece o que esta escrito na linguagem
dela, e a linguagem so cresce quando eu escrevo. Humanos tem tambem um SISTEMA 1:
reconhecimento e completamento de padrao aprendido da experiencia, que funciona
para conceitos que ainda NAO tem nome — e e justamente por isso que ele alcanca
o que a linguagem nao alcanca.

A literatura mede isso no ARC: [[literatura-arc-2025-2026]] — Li & Ellis, INDUZIR
o programa e PREVER a saida direto resolvem conjuntos DISJUNTOS, estaveis entre
sementes; todo topo de placar combina os dois. CompressARC faz 20% no eval com
ZERO pre-treino, treinando so na propria tarefa.

O QUE O PROJETO JA TENTOU E POR QUE NAO TESTA A HIPOTESE
----------------------------------------------------------
`transducao.py`: vizinho mais proximo sobre as features da gramatica, sem
aprender nada. 1/400 ([[credito-parcial-produz-composicao]]). Isso e um lookup,
nao um sistema que aprende a funcao da tarefa. A hipotese nunca foi testada.

A PECA
--------
Uma rede convolucional pequena, treinada DO ZERO em cada tarefa, so nos pares de
treino dela — sem pre-treino, sem dado externo, sem LLM. Aprende "celula de
saida = f(vizinhanca da entrada)" empilhando camadas, o que alcanca padroes
longos. Duas invariancias entram por AUMENTO DE DADO, e sao as que humanos tem
de nascenca ([[literatura-arc-2025-2026]], nucleo de conhecimento):
  - COR E ROTULO: permutacao consistente das cores 1..9 (fundo 0 fica)
  - ORIENTACAO NAO IMPORTA: as 8 simetrias do quadrado
Na previsao, a media das 8 orientacoes.

So tarefas em que entrada e saida tem as MESMAS dimensoes (262 de 400; 74 delas
ja resolvidas pelo nucleo). Mudar de tamanho e outra peca.

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  S0  PRE-CONDICAO. Resolve >= 10 das 262 (teste exato, todas as saidas de
      teste). Menos que isso e fraco demais para dizer qualquer coisa.
  S1  A METADE QUE FALTA E REAL: >= 1/3 do que ela resolve NAO esta nas 105 do
      nucleo -> sistemas complementares, e a uniao cresce exatamente por isso.
  S2  quase tudo que ela resolve o nucleo ja resolvia (< 1/3 novo) -> nao ha
      metade complementar nesta escala; a parede esta noutro lugar.
  PLACEBO: a mesma rede, mesmo treino, com pares EMBARALHADOS (entrada de um par
      com saida de outro). Tem de ficar <= 2; senao o que resolve e o vies da
      arquitetura e nao aprendizado da tarefa.

PREVISAO: S0 a 0,60, S1 a 0,50. Mecanismo: completar padrao local aprendido e o
tipo de conceito "perceptual difuso" que Li & Ellis mediram ser da transducao, e
que a linguagem de codecs cobre mal. Risco: sem pre-treino, 3 a 4 exemplos sao
pouco para uma rede — CompressARC compensa com uma arquitetura inteira desenhada
para ARC; aqui a rede e generica, e S0 pode falhar por fraqueza.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from novaphys.arc import carregar
from novaphys.promessa import PASTA

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LADO = 30
PASSOS = 600
LOTE = 16
CANAIS = 64


def _mesma(tr, te) -> bool:
    return all(len(a) == len(b) and len(a[0]) == len(b[0]) for a, b in tr + te)


def _um_quente(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    x = np.zeros((11, LADO, LADO), np.float32)
    x[10, :h, :w] = 1.0                           # canal de "dentro da grade"
    x[g, np.arange(h)[:, None], np.arange(w)[None, :]] = 1.0
    return x


def _d4(g: np.ndarray, k: int) -> np.ndarray:
    g = np.rot90(g, k % 4)
    return np.fliplr(g) if k >= 4 else g


def _d4_inv(g: np.ndarray, k: int) -> np.ndarray:
    if k >= 4:
        g = np.fliplr(g)
    return np.rot90(g, -(k % 4))


class Rede(nn.Module):
    """Convolucional residual com dilatacoes: ve de 1 a ~30 celulas de distancia."""

    def __init__(self, c: int = CANAIS) -> None:
        super().__init__()
        self.ent = nn.Conv2d(11, c, 3, padding=1)
        self.blocos = nn.ModuleList([
            nn.Conv2d(c, c, 3, padding=d, dilation=d) for d in (1, 2, 4, 8, 1, 2, 4, 1)])
        self.sai = nn.Conv2d(c, 10, 1)

    def forward(self, x):
        h = F.relu(self.ent(x))
        for b in self.blocos:
            h = h + F.relu(b(h))
        return self.sai(h)


# GRUPOS DE SIMETRIA, do mais restrito ao menos. DEFEITO DO PRIMEIRO PILOTO:
# eu IMPUNHA "cor e rotulo" e "orientacao nao importa" em toda tarefa, e isso
# CORROMPE tarefa que nao e simetrica — num mapa de cor ("vermelho vira azul")
# permutar cores muda a regra a cada exemplo; em "cai para baixo" girar muda a
# direcao. `0d3d703e`, um mapa de cor puro, falhou por isso. Humano nao assume
# simetria as cegas: percebe QUAIS valem. Aqui: a MAIOR simetria sob a qual os
# pares de treino ainda sao reproduzidos exatamente — hipotese mais restrita
# consistente com o dado, que e o principio do MDL do projeto.
GRUPOS = (("d4+cor", True, True), ("d4", True, False),
          ("cor", False, True), ("nenhuma", False, False))


def _amostra(pares, rnd, gira=True, cor=True
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lote aumentado dentro do grupo de simetria escolhido."""
    xs, ys, ms = [], [], []
    for _ in range(LOTE):
        a, b = pares[rnd.randrange(len(pares))]
        k = rnd.randrange(8) if gira else 0
        perm = np.arange(10)
        p = list(range(1, 10))
        if cor:
            rnd.shuffle(p)
        perm[1:] = p
        a2, b2 = perm[_d4(a, k)], perm[_d4(b, k)]
        xs.append(_um_quente(a2))
        y = np.zeros((LADO, LADO), np.int64)
        m = np.zeros((LADO, LADO), np.float32)
        y[:b2.shape[0], :b2.shape[1]] = b2
        m[:b2.shape[0], :b2.shape[1]] = 1.0
        ys.append(y)
        ms.append(m)
    return (torch.tensor(np.stack(xs), device=DEV),
            torch.tensor(np.stack(ys), device=DEV),
            torch.tensor(np.stack(ms), device=DEV))


def _treina(pares, rnd, gira, cor):
    rede = Rede().to(DEV)
    opt = torch.optim.Adam(rede.parameters(), lr=2e-3)
    rede.train()
    for _ in range(PASSOS):
        x, y, m = _amostra(pares, rnd, gira, cor)
        L = (F.cross_entropy(rede(x), y, reduction="none") * m).sum() / m.sum()
        opt.zero_grad()
        L.backward()
        opt.step()
    rede.eval()
    return rede


@torch.no_grad()
def _preve(rede, a: np.ndarray, gira: bool) -> np.ndarray:
    soma = np.zeros((10,) + a.shape, np.float32)
    for k in (range(8) if gira else (0,)):
        ak = _d4(a, k)
        lg = rede(torch.tensor(_um_quente(ak)[None], device=DEV))[0]
        lg = lg[:, :ak.shape[0], :ak.shape[1]].softmax(0).cpu().numpy()
        soma += np.stack([_d4_inv(lg[c], k) for c in range(10)])
    return soma.argmax(0)


def resolve(tr, te, semente: int = 0, embaralha: bool = False):
    """Devolve (resolveu, grupo usado). O TESTE so e olhado no fim."""
    rnd = random.Random(semente)
    torch.manual_seed(semente)
    pares = [(np.array(a), np.array(b)) for a, b in tr]
    if embaralha:
        # PLACEBO COMO DESARRANJO. A primeira versao usava `rnd.shuffle` + filtro
        # de forma + `or pares` — e os 15 "acertos do placebo" eram TODOS pares
        # verdadeiros: com 2 pares o embaralhamento e a identidade em 1/2 das
        # vezes, e o filtro descartava os trocados. Agora cada saida vem de
        # OUTRO par, com a mesma forma e DIFERENTE da verdadeira; se nao existir
        # desarranjo assim, a tarefa sai do placebo e isso e contado a parte.
        idx = list(range(len(pares)))
        for _ in range(500):
            rnd.shuffle(idx)
            if all(idx[i] != i
                   and pares[idx[i]][1].shape == pares[i][0].shape
                   and not np.array_equal(pares[idx[i]][1], pares[i][1])
                   for i in range(len(pares))):
                break
        else:
            return False, "placebo impossivel"
        pares = [(pares[i][0], pares[idx[i]][1]) for i in range(len(pares))]
    escolhido = None
    for nome, gira, cor in GRUPOS:
        rede = _treina(pares, rnd, gira, cor)
        if all(np.array_equal(_preve(rede, a, gira), b) for a, b in pares):
            escolhido = (nome, rede, gira)
            break
    if escolhido is None:
        return False, "nenhum reproduz o treino"
    nome, rede, gira = escolhido
    ok = all(np.array_equal(_preve(rede, np.array(a), gira), np.array(b))
             for a, b in te)
    return ok, nome


def experimento(placebo: bool = True) -> Dict:
    tarefas = [(n, tr, te) for n, tr, te in carregar(PASTA) if _mesma(tr, te)]
    nucleo = set(json.load(open("resultados/oficial.json")))
    print("=" * 78)
    print("SISTEMA 1 — completar o padrao sem escrever a regra")
    print("=" * 78)
    print("  %d tarefas de mesma dimensao; o nucleo (inducao) resolve %d delas"
          % (len(tarefas), len({n for n, _t, _e in tarefas} & nucleo)))
    print("  rede treinada DO ZERO em cada tarefa, %d passos, %s\n" % (PASSOS, DEV),
          flush=True)
    res, plc = [], []
    grupos = {}
    t0 = time.time()
    for k, (nome, tr, te) in enumerate(tarefas):
        ok, g = resolve(tr, te)
        grupos[nome] = g
        if ok:
            res.append(nome)
        if (k + 1) % 20 == 0:
            print("    %3d/%d | sistema1 %d (novas %d) | placebo %d | %.0f min"
                  % (k + 1, len(tarefas), len(res), len(set(res) - nucleo),
                     len(plc), (time.time() - t0) / 60), flush=True)
    if placebo:
        print("\n  PLACEBO (pares embaralhados), mesmas tarefas:", flush=True)
        for k, (nome, tr, te) in enumerate(tarefas):
            if resolve(tr, te, embaralha=True)[0]:
                plc.append(nome)
            if (k + 1) % 50 == 0:
                print("    %3d/%d | placebo %d" % (k + 1, len(tarefas), len(plc)),
                      flush=True)
    from collections import Counter
    print("  grupo de simetria escolhido: %s" % dict(Counter(grupos.values())))
    print("  grupo nas RESOLVIDAS: %s" % dict(Counter(grupos[n] for n in res)))
    novas = sorted(set(res) - nucleo)
    uniao = len(nucleo | set(res))
    print("\n" + "=" * 78)
    print("  SISTEMA 1 resolve %d/%d (placebo %d)" % (len(res), len(tarefas), len(plc)))
    print("  das que resolve, o nucleo NAO resolvia: %d" % len(novas))
    print("  uniao nucleo + sistema1 nas 400: %d (nucleo sozinho 105)" % uniao)
    print("  novas: %s" % novas)
    if len(res) < 10:
        ver = "S0 falhou"
        print("S0 FALHOU: %d < 10. Fraco demais para dizer algo." % len(res))
    elif len(plc) > 2:
        ver = "placebo"
        print("PLACEBO resolve %d: o que resolve e vies da arquitetura." % len(plc))
    elif len(novas) >= len(res) / 3:
        ver = "S1"
        print("S1 . A METADE QUE FALTA E REAL: %d de %d sao novas; a uniao vai"
              % (len(novas), len(res)))
        print("     de 105 para %d." % uniao)
    else:
        ver = "S2"
        print("S2 . so %d de %d sao novas: nao ha metade complementar aqui."
              % (len(novas), len(res)))
    r = {"veredito": ver, "resolve": res, "novas": novas, "placebo": plc,
         "uniao": uniao, "n": len(tarefas), "grupos": grupos}
    json.dump(r, open("resultados/sistema1.json", "w"), indent=1)
    return r


if __name__ == "__main__":
    experimento(placebo="--sem-placebo" not in sys.argv)


# ------------------------------------------------ SISTEMA 2 + SISTEMA 1 juntos
def resolve_com_forma(tr, te, max_formas: int = 4, semente: int = 0):
    """
    As tarefas em que a saida MUDA de tamanho (138 de 400) o Sistema 1 sozinho
    nao alcanca: ele e celula-a-celula. Mas o nucleo sabe geometria —
    `forma.propor` devolve modelos de forma (recorte, ladrilho, escala...) e
    `forma.base` leva a entrada para as dimensoes da saida.

    A divisao de trabalho e a que Li & Ellis mediram: o SISTEMA 2 (regra
    explicita) decide a geometria, que e computacao precisa; o SISTEMA 1
    preenche celula a celula, que e o conceito perceptual difuso. O mesmo
    principio de escolha: fica o primeiro modelo de forma sob o qual o Sistema 1
    reproduz o treino exatamente.
    """
    from novaphys import forma
    try:
        fms = list(forma.propor(tr))[:max_formas]
    except Exception:
        return False, "sem forma"
    for fm in fms:
        try:
            reb = [(forma.base(fm, a), b) for a, b in tr]
            rte = [(forma.base(fm, a), b) for a, b in te]
        except Exception:
            continue
        if any(g is None for g, _b in reb + rte):
            continue
        if not all(len(g) == len(b) and len(g[0]) == len(b[0])
                   for g, b in reb + rte):
            continue
        ok, grupo = resolve(reb, rte, semente=semente)
        if grupo != "nenhum reproduz o treino":
            return ok, "%s|%s" % (fm[0], grupo)
    return False, "nenhuma forma reproduz o treino"
