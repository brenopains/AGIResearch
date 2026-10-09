"""
novaphys.novidade — novidade no nivel do EVENTO, nao do simbolo de gene
=========================================================================

OS TRES DEFEITOS QUE `letalidade_sintetica.py` REVELOU
--------------------------------------------------------
A rodada no DepMap passou nos portoes mecanicos (nulo, controles, extrapolacao,
fora da SynLethDB) e deu 0/8 ineditos. Os 8 — e 36 dos 46 pares aceitos — eram
UM evento: a delecao 9p21. O filtro de novidade foi enganado de tres jeitos:

  1. CARONA        IFNA7, IFNA8, IFNB1, HACD4, DMRTA1, LINC01239... somem JUNTO com
                   MTAP/FOCAD na mesma delecao (Jaccard 0,36-0,64). Cada um virava
                   um "par novo".
  2. COMPLEXO      WDR77 e o parceiro obrigatorio do PRMT5 (dependencias com corr
                   +0,55); HBS1L e o do PELO (+0,44). MTAP->WDR77 parecia novo
                   porque a base so lista MTAP->PRMT5.
  3. EXTRAPOLACAO  FOCAD->PELO, o gene CAUSAL, pontuou 30,6 bits (3o maior) e foi
     POR SIMBOLO   excluido porque so 2 das 30 linhagens novas perderam FOCAD —
                   enquanto caronas com 3 passavam. Exigir n minimo POR GENE
                   premia o gene mais perdido, que tende a ser carona.

O CONSERTO
------------
  SEGMENTO   perdas que co-ocorrem (Jaccard >= 0,5) viram UM segmento
  MODULO     alvos cuja dependencia co-varia entre linhagens (|corr| >= 0,4)
             viram UM modulo
  EVENTO     (segmento, modulo). Representante = o par de maior bits.
  REPLICACAO no nivel do EVENTO: perda do evento = perda de QUALQUER membro do
             segmento; dependencia = media padronizada dos membros do modulo.
             Assim o FOCAD herda as linhagens novas dos caronas, e a exigencia de
             n minimo deixa de premiar o carona.
  NOVIDADE   um evento so e candidato se NENHUM par membro e controle nem esta na
             base; e o portao final continua sendo busca de literatura no nivel
             do evento (ex.: "9p21", "PRMT5 complex"), feita par a par a mao.

PRE-REGISTRO DESTE TESTE DO CONSERTO (dado ja visto: e teste da maquinaria, nao
descoberta)
---------------------------------------------------------------------------------
  N1  os 36 pares ligados a 9p21 colapsam em <= 3 eventos
  N2  o evento que contem FOCAD fica testavel nas linhagens novas (>= 3 com perda)
      e replica (mesmo sinal, >= metade do efeito)
  N3  sobram <= 6 eventos fora da 9p21
PREVISAO: N1, N2 e N3 passam. Se N1 falhar, o limiar de Jaccard esta frouxo demais
para delecoes grandes; se N2 falhar, a replicacao no nivel do evento nao resolveu
o defeito 3.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

import novaphys.letalidade_sintetica as S

REGIAO_9P21 = {"MTAP", "CDKN2A", "CDKN2B", "CDKN2B-AS1", "DMRTA1", "LINC01239",
               "FOCAD", "HACD4", "KLHL9", "MIR491", "IFNB1", "IFNW1", "IFNE",
               "MLLT3", "ELAVL2", "SNORA30B"} | {f"IFNA{k}" for k in range(1, 22)}


def agrupa(itens, parecido):
    """Agrupamento guloso: semente = primeiro livre; entra quem se parece com a
    semente. `parecido(i, j) -> bool`."""
    grupo, livres = {}, list(itens)
    gid = 0
    while livres:
        s = livres.pop(0)
        membros = [s] + [j for j in livres if parecido(s, j)]
        for m in membros:
            grupo[m] = gid
        livres = [j for j in livres if j not in membros]
        gid += 1
    return grupo


def segmentos_globais(Lb, colsA, alvos, jac=0.5, min_alt=10):
    """Para cada genotipo aceito, junta TODOS os genes do genoma que co-ocorrem
    com ele (Jaccard >= jac) — nao so os que estao na lista de aceitos.

    DEFEITO QUE ISTO CONSERTA: NEDD4L->VPS4A foi aceito enquanto VPS4B->VPS4A (o
    controle CAUSAL) nao passou. NEDD4L e carona da delecao 18q do VPS4B, e como
    VPS4B nao estava entre os aceitos, o colapso por segmento nao juntava os dois
    — o carona virava "evento proprio". Agrupar contra o genoma inteiro resolve.
    """
    ci = {g: i for i, g in enumerate(colsA)}
    n_alt = Lb.sum(0)
    grupos = []
    for g in alvos:
        if g not in ci:
            continue
        v = Lb[:, ci[g]]
        inter = (Lb & v[:, None]).sum(0)
        uni = n_alt + int(v.sum()) - inter
        j = inter / np.maximum(uni, 1)
        membros = {colsA[i] for i in np.where((j >= jac) & (n_alt >= min_alt))[0]}
        membros.add(g)
        grupos.append(membros)
    mudou = True
    while mudou:
        mudou = False
        for i in range(len(grupos)):
            for k in range(i + 1, len(grupos)):
                if grupos[i] & grupos[k]:
                    grupos[i] |= grupos[k]
                    grupos.pop(k)
                    mudou = True
                    break
            if mudou:
                break
    return {g: i for i, s in enumerate(grupos) for g in s}


def eventos(tab, L, Xdep, colsA, colsB, jac=0.5, cor=0.4):
    """Colapsa pares aceitos em eventos (segmento de perda, modulo de alvo)."""
    ci = {g: i for i, g in enumerate(colsA)}
    cb = {g: i for i, g in enumerate(colsB)}
    As = [a for a in pd.unique(tab.sort_values("bits", ascending=False).A) if a in ci]
    Bs = [b for b in pd.unique(tab.sort_values("bits", ascending=False).B) if b in cb]

    def jac_ok(a, b):
        u, v = L[:, ci[a]], L[:, ci[b]]
        return (u & v).sum() / max((u | v).sum(), 1) >= jac

    def cor_ok(a, b):
        return abs(np.corrcoef(Xdep[:, cb[a]], Xdep[:, cb[b]])[0, 1]) >= cor

    seg = agrupa(As, jac_ok)
    mod = agrupa(Bs, cor_ok)
    t = tab[tab.A.isin(seg) & tab.B.isin(mod)].copy()
    t["segmento"] = t.A.map(seg)
    t["modulo"] = t.B.map(mod)
    ev = []
    for (s, m), g in t.groupby(["segmento", "modulo"]):
        g = g.sort_values("bits", ascending=False)
        ev.append(dict(segmento=s, modulo=m, bits=g.bits.iloc[0],
                       rep_A=g.A.iloc[0], rep_B=g.B.iloc[0],
                       membros_A=sorted(set(g.A)), membros_B=sorted(set(g.B)),
                       n_pares=len(g), controle=bool(g.controle.any()),
                       synlethdb=bool(g.synlethdb.any())))
    return sorted(ev, key=lambda e: -e["bits"])


def replica_evento(e, L4, X4r, colsA, colsB, novas, velhas):
    ci = {g: i for i, g in enumerate(colsA)}
    cb = {g: i for i, g in enumerate(colsB)}
    ia = [ci[a] for a in e["membros_A"] if a in ci]
    ib = [cb[b] for b in e["membros_B"] if b in cb]
    if not ia or not ib:
        return None, 0
    perda = L4[:, ia].any(1)
    Z = X4r[:, ib]
    Z = (Z - Z[velhas].mean(0)) / (Z[velhas].std(0) + 1e-6)
    dep = Z.mean(1)
    alt, wt = perda & novas, (~perda) & novas
    if alt.sum() < 3 or wt.sum() < 3:
        return None, int(alt.sum())
    return float(dep[alt].mean() - dep[wt].mean()), int(alt.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabela", default=os.path.join(S.PASTA, "aceitos_24Q2.csv"))
    a = ap.parse_args()
    t0 = time.time()
    print("=" * 78)
    print("NOVIDADE NO NIVEL DO EVENTO — teste do conserto sobre o CRISPR ja rodado")
    print("=" * 78)
    tab = pd.read_csv(a.tabela)
    E, L, lin, _c = S.carrega("24Q2")
    Xr = S.residualiza(E.to_numpy(np.float32), lin.to_numpy(), np.ones(len(E), bool))
    ev = eventos(tab, L.to_numpy(), Xr, list(L.columns), list(E.columns))
    print(f"  {len(tab)} pares aceitos -> {len(ev)} eventos ({time.time()-t0:.0f}s)")

    E4, L4, lin4, _c4 = S.carrega("24Q4")
    vistas = set(E.index)
    novas = np.array([i not in vistas for i in E4.index])
    velhas = ~novas
    X4r = S.residualiza(E4.to_numpy(np.float32), lin4.to_numpy(), velhas)
    L4m = L4.to_numpy()

    print(f"\n  {'bits':>6s} {'rep A':9s} {'rep B':8s} {'pares':>5s} {'9p21':>5s} "
          f"{'n novas':>7s} {'d novas':>8s}  segmento / modulo")
    n9, foc, fora = 0, None, 0
    for e in ev:
        d, nn = replica_evento(e, L4m, X4r, list(L4.columns), list(E4.columns),
                               novas, velhas)
        e9 = bool(set(e["membros_A"]) & REGIAO_9P21)
        n9 += e9
        fora += (not e9)
        if "FOCAD" in e["membros_A"]:
            foc = (d, nn, e)
        marca = "C" if e["controle"] else ("DB" if e["synlethdb"] else "")
        print(f"  {e['bits']:6.1f} {e['rep_A']:9s} {e['rep_B']:8s} {e['n_pares']:5d} "
              f"{'sim' if e9 else '':>5s} {nn:7d} "
              f"{('%+.2f' % d) if d is not None else '     -':>8s}  "
              f"{','.join(e['membros_A'][:6])}{'...' if len(e['membros_A']) > 6 else ''}"
              f" / {','.join(e['membros_B'])} {marca}")

    pares9 = int(tab.A.isin(REGIAO_9P21).sum())
    print("")
    print("=" * 78)
    print(f"N1 {'.' if n9 <= 3 else 'FALHOU .'} {pares9} pares ligados a 9p21 -> "
          f"{n9} eventos (<= 3)")
    if foc:
        d, nn, e = foc
        rep = d is not None and d < 0
        print(f"N2 {'.' if nn >= 3 and rep else 'FALHOU .'} evento com FOCAD: "
              f"{nn} linhagens novas com perda, delta {d if d is None else round(d, 2)}")
    else:
        print("N2 FALHOU . FOCAD nao entrou em evento nenhum")
    print(f"N3 {'.' if fora <= 6 else 'FALHOU .'} eventos fora da 9p21: {fora} (<= 6)")
    print(f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
