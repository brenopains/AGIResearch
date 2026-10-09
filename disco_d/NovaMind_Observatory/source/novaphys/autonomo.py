"""
novaphys.autonomo — o laco sem mim: a propria duvida decide quando investigar
===============================================================================

O GAP
-------
No caderno vivo, QUEM decide quando o agente observa trajetorias (e pode inventar
um conceito) sou EU — nas tarefas 5, 15, 25 e 35. O agente aprende, esquece,
reanalisa, mas nao escolhe NO QUE TRABALHAR. O usuario pediu isso desde o comeco:
"um sistema que escolhe no que trabalhar".

O QUE TORNA POSSIVEL AGORA
----------------------------
`falseia2`: a duvida ESTRUTURAL — 1 menos o peso MDL da lei escolhida entre os
rivais de forma diferente — preve quando a lei esta errada com AUC 0,90. E um
sinal de CRISE que o proprio agente calcula, sem gabarito.

O MECANISMO
-------------
A cada tarefa de laboratorio (40 seguidas, 20 corpos cada, universo einstein ate
a 20 e oblato depois), o agente escreve a lei e calcula a propria duvida. Se a
duvida passa de 0,5 — a lei e mais provavelmente errada que certa, o limiar
natural, sem constante minha — ele declara CRISE: gasta uma observacao de
trajetorias (orcamento de 4, igual ao meu calendario), tenta inventar uma
grandeza, e se inventar, REANALISA todos os dados guardados com olhos frescos (a
regra que venceu no caderno: +36 pontos).

O mesmo aparato de aprendizagem nos quatro bracos (memoria com esquecimento 0,7,
reanalise com olhos frescos quando um conceito chega). Muda SO quando observar:
    nunca      nunca observa
    fixo       o meu calendario: antes das tarefas 5, 15, 25, 35
    aleatorio  4 momentos sorteados
    autonomo   quando a propria duvida passa de 0,5 (ate 4 vezes)
Uma correcao de honestidade: no `caderno`, a reanalise juntava so as tarefas "do
mundo atual" — o agente sabia onde o mundo mudou (vazamento meu, sem efeito la
porque a unica reanalise ocorreu antes da troca). Aqui ninguem sabe: a reanalise
junta TODAS as tarefas guardadas.

PRE-REGISTRO, ANTES DE RODAR — semente 20260929, 10 correntes por braco
-------------------------------------------------------------------------
Acerto na primeira tentativa (lei que extrapola), fase einstein = tarefas 0-19.
  AU1  autonomo - fixo >= 0 na fase einstein (o agente sozinho iguala ou bate o
       meu calendario)
  AU2  autonomo - aleatorio >= +10 na fase einstein
  AU3  a primeira observacao do autonomo acontece ate a tarefa 2 em >= 8/10
       correntes (ele percebe a crise logo)
  AU4  autonomo - nunca >= +30 na fase einstein
  (descritivo) quantas observacoes o autonomo gasta e quando; o que acontece
  depois da troca de lei, onde o conceito (h) continua valendo e a lei nao.

PREVISAO: fixo ~65% (0-4 sem conceito), autonomo ~85% (observa na tarefa 0 ou
1), aleatorio ~50%, nunca ~20%. As quatro passam. Depois da troca, o autonomo
deve gastar observacoes a toa: h e reinventado, e duplicado, e nao ha
reanalise — a crise sem conceito novo nao tem remedio neste desenho.
Calibragem: das ultimas 13 previsoes de magnitude, 8 erradas, quase todas para
cima.

RESULTADO DA RODADA 1 (176 s):
                    nunca   fixo   aleatorio   autonomo
  einstein (0-19)     0%     62%      52%        43%
  logo apos troca    62%     92%      86%        92%
  oblato (25-39)     92%     95%      95%        92%
  autonomo observou em: [12] [] [14,31,36] [2,33] [3,4,38] ...  (media 2,2)
  duvida da propria lei preve o erro: AUC 0,66 (era 0,90 na falseia2)
AU1 FALHOU (-18). AU2 FALHOU (-9). AU3 FALHOU (2/10). AU4 passou (+43).
DIAGNOSTICO: com 20 corpos e sem o conceito, a lei certa NAO e escrevivel como
rival de mesmo porte — sao dois termos de ~20 bits cada; a navalha da o peso a lei
simples e errada, e o agente fica CONFIANTE. A duvida estrutural mede o
desconhecido DENTRO da linguagem (qual expoente), nao o que falta FORA dela (um
conceito que ainda nao existe). Kuhn: crise nao e a duvida de um ajuste, e o
ACUMULO DE ANOMALIAS.

RODADA 2, PRE-REGISTRO — semente 20260930, 10 correntes
---------------------------------------------------------
Braco novo SURPRESA: antes de ajustar a tarefa k, o agente aplica a lei que ja
tem (a do passo anterior, ou a da reanalise) aos dados NOVOS e compara o erro com
o ruido do proprio instrumento (sigma conhecido — o agente sabe a precisao do que
mede; foi o conserto apontado em `inventa`: julgar contra o ruido esperado, nao
contra zero). Teste qui-quadrado: crise se SSE / (sigma^2 N) > 1 + 3 sqrt(2/N) (tres
desvios). Crise -> observa -> se inventar, reanalisa. Na tarefa 0 a "lei que ja
tem" e Newton, a teoria velha.
  S1  surpresa - fixo >= +10 na fase einstein
  S2  primeira crise da surpresa ate a tarefa 2 em >= 8/10
  S3  a surpresa DETECTA A TROCA DE LEI: uma crise nas tarefas 20-22 em >= 8/10
  (descritivo) quantas observacoes gasta; o braco autonomo (duvida) replica?
PREVISAO: S1 passa (~80% x ~60%), S2 passa (crise na tarefa 0, porque Newton nao
explica a anomalia), S3 passa (a lei de einstein erra o mundo oblato muito acima do
ruido).

RESULTADO DA RODADA 2 (183 s):
                    nunca   fixo   autonomo   surpresa
  einstein (0-19)     0%     62%      42%        69%
  logo apos troca    82%     92%      88%        92%
  surpresa observou em: [0,1,2,20] [0,1,2,4] [0,14,20,31] [0,1,20,33] [0,2,13]
S1 FALHOU (+7). S2 PASSOU (crise na tarefa 0 em 10/10). S3 PASSOU (troca de lei
detectada nas tarefas 20-22 em 9/10). Replica: duvida -20 de novo.
LEITURA: o DETECTOR de crise funciona — a surpresa contra o ruido e a
metacognicao certa para "minha teoria nao da conta". O que falhou foi a RESPOSTA:
gastou o orcamento em 0, 1, 2 (h reinventado, duplicado, desperdicado) e nunca
reanalisou os dados juntos, porque a reanalise so disparava com conceito NOVO.

RODADA 3, PRE-REGISTRO — semente 20261001, 10 correntes
---------------------------------------------------------
Braco CONSENSO: mesma surpresa como detector; a RESPOSTA muda.
  - toda crise dispara REANALISE com olhos frescos, mesmo sem conceito novo;
  - a reanalise e por CONSENSO: comeca pelos dados que surpreenderam, ajusta, e
    incorpora as tarefas guardadas que a lei nova explica (o MESMO qui-quadrado
    contra o ruido), reajusta, repete — depois da troca de lei nao mistura os dois
    mundos, e ninguem conta a ela onde o mundo mudou;
  - so observa de novo se a ultima observacao trouxe conceito novo.
  C1  consenso - fixo >= +10 na fase einstein
  C2  consenso >= fixo - 5 logo apos a troca (20-24)
  C3  consenso gasta <= 2 observacoes em media
PREVISAO: C1 passa (~85%), C2 passa, C3 passa (1 observacao util + no maximo 1
na troca). Calibragem ruim para magnitudes: 9 das ultimas 15 erradas para cima.

RESULTADO DA RODADA 3 (164 s):
                    fixo   surpresa   CONSENSO
  einstein (0-19)    68%      66%        83%
  logo apos troca    88%      88%        86%
  observacoes        4,0      3,9        2,0   (consenso: [0, 1] na maioria)
  crises do consenso: [0,8,14,20] [0,1,3,4,21,29] [0,1,2,3,18,20,25,34] ...
C1 PASSOU (+15), C2 PASSOU (-2), C3 PASSOU (2,0). Previsao certa, inclusive na
magnitude (85 previsto, 83 medido). A surpresa com a resposta velha deu -2: o +7
da rodada 2 era ruido; o que paga e a RESPOSTA por consenso.
O LACO FECHOU SEM MIM: o agente percebe a crise contra o ruido do instrumento
(tarefa 0), inventa o conceito, reanalisa por consenso, para de observar quando
observar nao rende, e percebe a troca de lei (tarefas 20-21) sem ninguem dizer —
e bate o calendario que eu desenhei. Ressalvas: um universo so, 10 correntes, o
alfabeto da invencao e meu, e o ruido do instrumento e conhecido.
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import random
import time

import numpy as np

from novaphys import orbita as O
from novaphys import inventa as I
from novaphys import caderno as C
from novaphys import falseia2 as F2
from novaphys import falseia as F

SEMENTE = 20260929
N_PASSOS = 40
ORCAMENTO = 4
LIMIAR = 0.5
FIXO = (5, 15, 25, 35)
BRACOS = ("nunca", "fixo", "aleatorio", "autonomo", "surpresa", "consenso")


def duvida(S, D, ac, mem):
    C._com_extras(mem)
    try:
        _rv, w = F2.rivais(S, D, ac)
        return float(1 - w[0])
    finally:
        O.EXTRAS.clear()


def observa(mem, mundo, base):
    rng = np.random.default_rng(base + 500)
    Ttr = I.trajetorias(mundo, rng, I.N_TRAJ)
    Tho = I.trajetorias(mundo, rng, I.N_TRAJ)
    aceito, top, _n = I.inventa(Ttr, Tho)
    if not aceito:
        return False
    _g, s1, w1, *_ = top[0]
    Q = I.grandeza(s1, w1, Ttr)
    Sh = O.estados(np.random.default_rng(base + 7), 500)
    if any(abs(np.corrcoef(Q(Sh), f(Sh))[0, 1]) > 0.99 for f in mem.conceitos.values()):
        return False
    mem.conceitos[f"q{len(mem.conceitos)}"] = Q
    return True


def reanalisa(mem, guardadas, base):
    Sa = np.concatenate([g[1] for g in guardadas])
    Da = np.concatenate([g[2] for g in guardadas])
    limpa = C.Memoria(mem.gama)
    limpa.conceitos = mem.conceitos
    ac, chs = C.descobre(Sa, Da, limpa, base + 11)
    mem.termos = {ch: float(len(guardadas)) for ch in chs}
    return ac


def reanalisa_consenso(mem, guardadas, S, D, base, limiar):
    """Olhos frescos, comecando pelos dados que surpreenderam (S, D) e incorporando
    as tarefas guardadas que a lei nova explica abaixo do limiar de ruido."""
    tarefas = [(g[1], g[2]) for g in guardadas] + [(S, D)]
    escolhidas = [len(tarefas) - 1]
    limpa = C.Memoria(mem.gama)
    limpa.conceitos = mem.conceitos
    ac, chs = [], []
    for _ in range(4):
        Sa = np.concatenate([tarefas[i][0] for i in escolhidas])
        Da = np.concatenate([tarefas[i][1] for i in escolhidas])
        ac, chs = C.descobre(Sa, Da, limpa, base + 11)
        novas = [i for i, (St, Dt) in enumerate(tarefas)
                 if surpresa(ac, St, Dt, mem) <= limiar] or escolhidas
        if set(novas) == set(escolhidas):
            break
        escolhidas = novas
    mem.termos = {ch: float(len(escolhidas)) for ch in chs}
    return ac


def surpresa(lei, S, D, mem):
    """Qui-quadrado da lei que o agente JA TEM nos dados novos, contra o ruido do
    instrumento. > 1 + 3 sqrt(2/N) = a lei nao explica o que acabou de chegar."""
    C._com_extras(mem)
    try:
        pred = O.preve(lei, S, False) if lei else np.zeros_like(D)
    finally:
        O.EXTRAS.clear()
    return float(((D - pred) ** 2).sum() / (O.SIGMA ** 2 * D.size))


def corrente(args):
    braco, s, semente = args
    mem = C.Memoria(0.7)
    aleat = set(random.Random(semente + 77 + 1000 * s).sample(range(N_PASSOS), ORCAMENTO))
    usadas, guardadas, primeiro, duvs, crises = [], [], {}, [], []
    lei = []                     # a teoria que o agente tem: comeca com Newton
    limiar_chi = 1 + 3 * np.sqrt(2 / (2 * C.N_LAB))
    pode_observar = True         # consenso: so observa de novo se a ultima rendeu
    for k in range(N_PASSOS):
        mundo = C.mundo_de(k)
        base = semente + 1000 * s + k
        rng = np.random.default_rng(base)
        S = O.estados(rng, C.N_LAB)
        D = O.mede(mundo, S, rng)
        quer = (braco == "fixo" and k in FIXO) or (braco == "aleatorio" and k in aleat)
        if braco in ("surpresa", "consenso"):
            z = surpresa(lei, S, D, mem)
            if z > limiar_chi:
                crises.append(k)
                if braco == "surpresa":
                    quer = True
                else:
                    if pode_observar and len(usadas) < ORCAMENTO:
                        usadas.append(k)
                        pode_observar = observa(mem, mundo, base)
                    lei = reanalisa_consenso(mem, guardadas, S, D, base, limiar_chi)
        if quer and len(usadas) < ORCAMENTO:
            usadas.append(k)
            if observa(mem, mundo, base) and guardadas:
                lei = reanalisa(mem, guardadas, base)
        ac, chaves = C.descobre(S, D, mem, base + 3)
        lei = ac
        primeiro[k] = C.acerta(ac, mem, mundo)
        guardadas.append((k, S, D, base + 3))
        mem.envelhece()
        for ch in chaves:
            mem.usa(ch)
        dv = duvida(S, D, ac, mem)
        duvs.append((k, dv, primeiro[k]))
        if braco == "autonomo" and dv > LIMIAR and len(usadas) < ORCAMENTO:
            usadas.append(k)            # CRISE: observa depois da tarefa k
            if observa(mem, mundo, base + 1):
                lei = reanalisa(mem, guardadas, base)
    # QUE conceito foi inventado: correlacao com a energia e com h
    Sh = O.estados(np.random.default_rng(semente + 999), 500)
    En = 0.5 * (Sh[:, 2] ** 2 + Sh[:, 3] ** 2) - 1 / np.hypot(Sh[:, 0], Sh[:, 1])
    Hn = Sh[:, 0] * Sh[:, 3] - Sh[:, 1] * Sh[:, 2]
    ids = [(abs(np.corrcoef(f(Sh), En)[0, 1]), abs(np.corrcoef(f(Sh), Hn)[0, 1]))
           for f in mem.conceitos.values()]
    return braco, s, primeiro, usadas, duvs, crises, ids


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--correntes", type=int, default=10)
    ap.add_argument("--nucleos", type=int, default=3)
    ap.add_argument("--semente", type=int, default=SEMENTE)
    ap.add_argument("--bracos", default=",".join(BRACOS))
    a = ap.parse_args()
    n = a.correntes
    BR = a.bracos.split(",")
    print("=" * 76)
    print("AUTONOMO — a propria duvida decide quando investigar")
    print("=" * 76)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(corrente, [(b, s, a.semente) for s in range(n) for b in BR],
                       chunksize=1)
    print(f"\n  semente {a.semente}   ({time.time()-t0:.0f}s)\n")
    fases = {"einstein (0-19)": range(0, 20), "logo apos a troca (20-24)": range(20, 25),
             "oblato (25-39)": range(25, 40)}
    tab = {}
    print("  acerto na primeira tentativa" + "".join(f"{b:>14s}" for b in BR))
    for nome, ks in fases.items():
        linha = ""
        for b in BR:
            v = [r[2][k] for r in res if r[0] == b for k in ks]
            tab[(nome, b)] = sum(v) / len(v)
            linha += f"{sum(v):6d}/{len(v):<3d}{100*sum(v)/len(v):3.0f}%"
        print(f"  {nome:28s}{linha}")
    print("\n  observacoes usadas (tarefa em que o agente foi olhar):")
    for b in [x for x in BR if x != "nunca"]:
        us = [r[3] for r in res if r[0] == b]
        print(f"    {b:10s} media {np.mean([len(u) for u in us]):.1f}   "
              + "  ".join(str(u) for u in us[:5]))
    dv = [x for r in res for x in r[4]]
    A = F.auc([d for _k, d, ok in dv if not ok], [d for _k, d, ok in dv if ok])
    print(f"\n  a duvida da propria lei preve o erro: AUC {A:.2f} ({len(dv)} leis)")
    E = "einstein (0-19)"
    if "consenso" in BR:
        co = [r for r in res if r[0] == "consenso"]
        print("  consenso observou em: " + "  ".join(str(r[3]) for r in co[:6]))
        print("  crises do consenso:   " + "  ".join(str(r[5][:8]) for r in co[:6]))
        T = "logo apos a troca (20-24)"
        c1 = tab[(E, "consenso")] - tab[(E, "fixo")]
        c2 = tab[(T, "consenso")] - tab[(T, "fixo")]
        c3 = np.mean([len(r[3]) for r in co])
        print("")
        print("=" * 76)
        print(f"C1 {'.' if c1 >= 0.10 else 'FALHOU .'} consenso - fixo na fase einstein = "
              f"{100*c1:+.0f} (limiar +10)")
        print(f"C2 {'.' if c2 >= -0.05 else 'FALHOU .'} consenso - fixo logo apos a troca = "
              f"{100*c2:+.0f} (limiar >= -5)")
        print(f"C3 {'.' if c3 <= 2 else 'FALHOU .'} observacoes do consenso: media {c3:.1f} (<= 2)")
        if "surpresa" in BR:
            print(f"   (replica) surpresa - fixo = "
                  f"{100*(tab[(E, 'surpresa')] - tab[(E, 'fixo')]):+.0f}")
        return
    if "surpresa" in BR:
        su = [r for r in res if r[0] == "surpresa"]
        print("  crises da surpresa (tarefas): " + "  ".join(str(r[5][:8]) for r in su[:5]))
        s1 = tab[(E, "surpresa")] - tab[(E, "fixo")]
        s2 = sum(1 for r in su if r[3] and r[3][0] <= 2)
        s3 = sum(1 for r in su if any(20 <= c <= 22 for c in r[5]))
        print("")
        print("=" * 76)
        print(f"S1 {'.' if s1 >= 0.10 else 'FALHOU .'} surpresa - fixo = {100*s1:+.0f} (limiar +10)")
        print(f"S2 {'.' if s2 >= 8 else 'FALHOU .'} primeira crise ate a tarefa 2: {s2}/{n}")
        print(f"S3 {'.' if s3 >= 8 else 'FALHOU .'} crise logo apos a troca (20-22): {s3}/{n}")
        if "autonomo" in BR:
            print(f"   (replica) autonomo - fixo = "
                  f"{100*(tab[(E, 'autonomo')] - tab[(E, 'fixo')]):+.0f}")
        return
    au = [r for r in res if r[0] == "autonomo"]
    cedo = sum(1 for r in au if r[3] and r[3][0] <= 2)
    print("")
    print("=" * 76)
    d1 = tab[(E, "autonomo")] - tab[(E, "fixo")]
    print(f"AU1 {'.' if d1 >= 0 else 'FALHOU .'} autonomo - fixo = {100*d1:+.0f} pontos (limiar >= 0)")
    d2 = tab[(E, "autonomo")] - tab[(E, "aleatorio")]
    print(f"AU2 {'.' if d2 >= 0.10 else 'FALHOU .'} autonomo - aleatorio = {100*d2:+.0f} (limiar +10)")
    print(f"AU3 {'.' if cedo >= 8 else 'FALHOU .'} primeira crise ate a tarefa 2: {cedo}/{n}")
    d4 = tab[(E, "autonomo")] - tab[(E, "nunca")]
    print(f"AU4 {'.' if d4 >= 0.30 else 'FALHOU .'} autonomo - nunca = {100*d4:+.0f} (limiar +30)")


if __name__ == "__main__":
    main()
