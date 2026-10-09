"""
novaphys.decompoe_har — a distancia no HAR e de PERCEPCAO ou de PROTOCOLO?
============================================================================

POR QUE ESTA MEDIDA EXISTE
----------------------------
[[escala-nao-e-o-gargalo]] concluiu "o buraco e de representacao" comparando
nosso pipeline (0,675) com sete linhas de estatistica (0,870). E
`percepcao.py` deu uma pista contra essa conclusao: com sonda LINEAR, ate uma
rede convolucional ALEATORIA bate o rival (0,909 x 0,870), mas no protocolo de
DESCOBERTA a mesma rede cai para 0,676 — o mesmo numero da nossa codificacao a
mao.

As duas comparacoes misturavam duas coisas:

    REPRESENTACAO   o que o vetor guarda sobre a janela
    PROTOCOLO       como o rotulo entra: sonda supervisionada (rival) contra
                    agrupar sem rotulo e so nomear pela maioria (nosso)

Esta tabela separa as duas: cada representacao avaliada nos DOIS protocolos.

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  H1  O BURACO E DE PROTOCOLO: as 7 linhas tambem caem no protocolo de
      DESCOBERTA para menos de 0,775 (menos da metade do caminho entre 0,675 e
      0,870). Entao a distancia de 0,195 e comparar agrupamento sem rotulo com
      classificador com rotulo, e [[escala-nao-e-o-gargalo]] atribuiu a
      percepcao o que era do protocolo.
  H2  O BURACO E DE REPRESENTACAO: as 7 linhas seguram >= 0,775 na descoberta
      E a nossa codificacao a mao fica abaixo delas tambem na SONDA. A
      conclusao antiga fica de pe.
  H3  MISTO: as 7 linhas seguram na descoberta, mas a nossa codificacao a mao
      empata com elas na sonda -> a representacao guarda a informacao e o que
      falha e o agrupamento em cima dela.

PREVISAO: H1 a 0,50. Mecanismo: [[motivo-e-pessoal]] mediu que a estrutura
dominante que o agrupamento acha sem rotulo e a PESSOA, nao a atividade; isso
vale para qualquer representacao, inclusive as 7 linhas. H3 a 0,30, H2 a 0,20.
"""

from __future__ import annotations

import json
import statistics
from typing import Dict

import numpy as np

from novaphys.ancora_externa import estatisticas
from novaphys.har import carrega
from novaphys.har_pessoas import K_GRUPOS, SEMENTE, PipelineCongelado
import novaphys.har_pessoas as HP
from novaphys.percepcao import (Codificador, DEV, descoberta, representa,
                                sonda, treina)


def experimento() -> Dict:
    import torch
    Xtr_raw, ytr = carrega("train", None)
    Xte_raw, yte = carrega("test", None)

    # --- 1. codificacao A MAO: os vetores HDC do pipeline, antes do agrupamento
    p = PipelineCongelado(HP.Espaco())
    p.ajusta(Xtr_raw, ytr, np.random.default_rng(SEMENTE), K_GRUPOS)
    Mtr = np.stack([p._vetor(s) for s in p._cadeias(Xtr_raw)]).astype(np.float32)
    Mte = np.stack([p._vetor(s) for s in p._cadeias(Xte_raw)]).astype(np.float32)
    proprio = float(np.mean(p.aplica(Xte_raw) == yte))

    # --- 2. as 7 linhas do rival
    Str, Ste = estatisticas(Xtr_raw), estatisticas(Xte_raw)

    # --- 3 e 4. convolucional aleatoria e aprendida (sem rotulo)
    Xtr = np.transpose(Xtr_raw, (0, 2, 1)).astype(np.float32)
    Xte = np.transpose(Xte_raw, (0, 2, 1)).astype(np.float32)
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    torch.manual_seed(0)
    alea = Codificador().to(DEV)
    enc, _ = treina(Xtr, 0)

    reps = {
        "A MAO (HDC, nosso)": (Mtr, Mte),
        "7 LINHAS (rival)": (Str, Ste),
        "CONV ALEATORIA": (representa(alea, Xtr), representa(alea, Xte)),
        "CONV APRENDIDA": (representa(enc, Xtr), representa(enc, Xte)),
    }

    print("=" * 78)
    print("DECOMPOSICAO — a distancia no HAR e de percepcao ou de protocolo?")
    print("=" * 78)
    print("  treino %d (21 pessoas), teste %d (9 NOVAS)" % (len(ytr), len(yte)))
    print("  pipeline a mao com o proprio agrupamento dele: %.3f\n" % proprio)
    print("  %-22s %14s %14s" % ("representacao", "DESCOBERTA", "SONDA"))
    print("  %-22s %14s %14s" % ("", "(sem rotulo)", "(com rotulo)"))
    print("  " + "-" * 52)
    tab = {}
    for nome, (Rtr, Rte) in reps.items():
        ds = [descoberta(Rtr, ytr, Rte, yte, s) for s in (0, 1, 2)]
        sn = sonda(Rtr, ytr, Rte, yte)
        tab[nome] = (statistics.mean(ds), statistics.pstdev(ds), sn)
        print("  %-22s %8.3f +-%.3f %14.3f"
              % (nome, statistics.mean(ds), statistics.pstdev(ds), sn),
              flush=True)

    d7 = tab["7 LINHAS (rival)"][0]
    s7 = tab["7 LINHAS (rival)"][2]
    sm = tab["A MAO (HDC, nosso)"][2]
    print("\n" + "=" * 78)
    if d7 < 0.775:
        ver = "H1"
        print("H1 . O BURACO E DE PROTOCOLO. As 7 linhas, sem rotulo, dao %.3f." % d7)
        print("     A distancia de 0,195 era agrupamento-sem-rotulo contra")
        print("     classificador-com-rotulo. [[escala-nao-e-o-gargalo]] atribuiu")
        print("     a percepcao o que era do protocolo.")
    elif sm >= s7 - 0.02:
        ver = "H3"
        print("H3 . MISTO: a codificacao a mao guarda a informacao (sonda %.3f x" % sm)
        print("     %.3f), mas o agrupamento sobre ela falha." % s7)
    else:
        ver = "H2"
        print("H2 . O BURACO E DE REPRESENTACAO: 7 linhas seguram %.3f sem rotulo" % d7)
        print("     e a mao fica abaixo tambem na sonda (%.3f x %.3f)." % (sm, s7))
    r = {"veredito": ver, "proprio_pipeline": proprio,
         "tabela": {k: {"descoberta": v[0], "desc_sd": v[1], "sonda": v[2]}
                    for k, v in tab.items()}}
    json.dump(r, open("resultados/decompoe_har.json", "w"), indent=1,
              ensure_ascii=False)
    return r


if __name__ == "__main__":
    experimento()
