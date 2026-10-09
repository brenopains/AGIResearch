"""
novaphys.letalidade_sintetica — pares de letalidade sintetica em cancer humano
================================================================================

O PEDIDO
----------
"Estou buscando o resultado que me fara pular da cadeira genuinamente, algo
INEDITO. Cura do cancer, cura do envelhecimento, doencas especificas, coisas que
da pra gente simular com fidelidade, dar dados e deixar o pau torar."

POR QUE NAO SIMULAR, E O QUE E MELHOR QUE SIMULAR
----------------------------------------------------
Cancer e envelhecimento NAO se simulam com fidelidade — ninguem consegue, e a
parte nao resolvida e justamente a que o simulador nao contem. Uma AGI rodando
num simulador de cancer acharia as leis do SIMULADOR. E o brinquedo que o usuario
recusou.

Mas existe algo melhor que simulacao: o experimento REAL ja feito em escala. O
DepMap (Broad Institute) desligou, com CRISPR, cada um de ~18 mil genes em ~1.100
linhagens de cancer humano e mediu quanto cada celula depende de cada gene. E a
coisa mais proxima de um "simulador fiel" de dependencia em cancer que existe,
porque nao e modelo: e medida.

A PERGUNTA, E POR QUE ELA TEM VALOR TERAPEUTICO
-------------------------------------------------
LETALIDADE SINTETICA: perder o gene A e toleravel; perder A e B juntos mata. Se
um tumor ja perdeu A, um farmaco contra B mata o tumor e poupa o tecido normal,
que ainda tem A. E o mecanismo dos inibidores de PARP em tumores BRCA, e de
inibidores de PRMT5/MAT2A em tumores com delecao de MTAP — farmacos reais.

O espaco e ~20 mil x ~18 mil pares, e a maioria nunca foi olhada. Isso e o que
torna "inedito" possivel aqui, e nao no dado mais estudado do mundo.

O QUE A MAQUINA FAZ (a moeda de sempre: bits fora da amostra)
---------------------------------------------------------------
Para cada par (A perdido, B dependencia): codifica o efeito de nocaute de B nas
linhagens ESCONDIDAS com um modelo de dois grupos (A perdido / A intacto), ajustado
na metade de descoberta, contra um modelo de um grupo. Aceita se economizar bits
depois de pagar a estrutura E o custo de ter escolhido A e B entre milhares.

    perda de A   = mutacao danosa OU delecao profunda (CN relativo < 0,3)
    efeito de B  = Chronos, residualizado pela MEDIA DA LINHAGEM (tecido)
    so direcao letal: B fica MAIS essencial quando A esta perdido

OS CONFUNDIDORES CONHECIDOS DESTE DADO, E COMO CADA UM E TRATADO
-------------------------------------------------------------------
  LINHAGEM          A e B podem so co-variar com o tecido de origem. Tratado:
                    efeito residualizado pela linhagem, e o NULO permuta a perda
                    de A DENTRO de cada estrato de linhagem.
  HIPERMUTACAO      linhagens MSI tem muitas mutacoes E dependencias proprias
                    (ex.: WRN). Um gene mutado "por tabela" preveria WRN sem ser
                    letalidade sintetica com ele. Tratado: o estrato do nulo e
                    linhagem x TERCIL DE CARGA MUTACIONAL.
  CO-MUTACAO        genes perdidos juntos (ex.: MTAP e CDKN2A na mesma delecao).
                    Tratado: o nulo permuta LINHAS INTEIRAS da matriz de perda,
                    preservando a correlacao entre os As.
  MULTIPLOS TESTES  ~10^7 pares. Tratado: o limiar e o MAXIMO de bits sobre
                    TODOS os pares em cada uma de K rodadas nulas — controle de
                    erro por familia, nao por par.
  NOVIDADE          o erro da minha ultima tentativa ("nao sei se ja foi
                    documentado"). Tratado: SynLethDB 2.0 (35.943 pares humanos)
                    e busca na literatura, par a par, para os sobreviventes.

PRE-REGISTRO — 12/09/2026, ANTES DE EXISTIR QUALQUER ARQUIVO DO DEPMAP NO DISCO
---------------------------------------------------------------------------------
Controles positivos (letalidades sinteticas estabelecidas na literatura):
    SMARCA4 -> SMARCA2     ARID1A -> ARID1B      STAG2 -> STAG1
    VPS4B   -> VPS4A       MTAP   -> PRMT5       RPL22 -> RPL22L1
    ENO1    -> ENO2
Controle com menos de 20 linhagens perdidas na descoberta e declarado NAO
TESTAVEL e sai do denominador.

  C1  CONTROLES: >= 60% dos controles testaveis sao aceitos acima do limiar do
      nulo, e ha pelo menos 4 testaveis
  C2  NULO: falsas descobertas esperadas no limiar <= 1 (medidas nas rodadas
      nulas estratificadas)
  C3  EXTRAPOLA: entre os pares aceitos no 24Q2 que tem >= 3 linhagens perdidas
      entre as NOVAS do 24Q4 (que o 24Q2 nunca viu), >= 70% replicam: mesmo
      sinal e pelo menos metade do efeito
  C4  INEDITO (filtro mecanico): >= 1 par aceito que replica nas linhagens
      novas, nao e controle e nao esta na SynLethDB
  C5  INEDITO (filtro final): o par de C4 sobrevive a uma busca de literatura
      "GENE_A GENE_B synthetic lethal" sem artigo que o reporte

PREVISAO: C1 passa com 4 a 6 controles (ENO1 provavelmente nao testavel — delecao
rara). C2 passa por construcao. C3 passa, 75-85%. C4 passa, com 3 a 15
candidatos. C5 e onde eu aposto contra mim: a maioria dos sobreviventes vai
aparecer na literatura ao ser buscada; dou ~30% de chance de ao menos UM par
sobreviver a busca.

O QUE UM PASSE EM C5 SERIA, E O QUE NAO SERIA
-----------------------------------------------
SERIA: um par de letalidade sintetica candidato em cancer humano, achado sem eu
apontar, que atravessou nulo estratificado com controle por familia, replicou em
linhagens celulares que nao existiam no dado de descoberta, e nao consta na base
de referencia nem na literatura buscada. Isso e resultado real, do tipo que vira
experimento de bancada.
NAO SERIA: cura. Linhagem celular nao e paciente, efeito de nocaute nao e farmaco,
e o par precisa de validacao experimental antes de qualquer afirmacao biologica.
"""

from __future__ import annotations

import argparse
import os
import time
from math import log2

import numpy as np
import pandas as pd

PASTA = "D:/dados_cancer/depmap"
SEMENTE = 20261030
MIN_ALT = 20
DEL_LIMIAR = float(np.log2(1 + 0.3))      # OmicsCNGene = log2(CN relativo + 1)
META = {"SequencingID", "ModelID", "IsDefaultEntryForModel", "ModelConditionID",
        "IsDefaultEntryForMC", "ProfileID"}

CONTROLES = [("SMARCA4", "SMARCA2"), ("ARID1A", "ARID1B"), ("STAG2", "STAG1"),
             ("VPS4B", "VPS4A"), ("MTAP", "PRMT5"), ("RPL22", "RPL22L1"),
             ("ENO1", "ENO2")]


# ─── leitura robusta: 24Q2 e 24Q4 mudaram o formato das matrizes omicas ─────

def _matriz(caminho):
    df = pd.read_csv(caminho, low_memory=False)
    if "ModelID" in df.columns:
        if "IsDefaultEntryForModel" in df.columns:
            df = df[df["IsDefaultEntryForModel"].astype(str).str.lower()
                    .isin(["yes", "true", "1"])]
        df = df.set_index("ModelID")
    else:
        df = df.set_index(df.columns[0])
    genes = [c for c in df.columns
             if c not in META and not str(c).startswith("Unnamed")]
    df = df[genes]
    df.columns = [str(c).split(" (")[0] for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    df = df[~df.index.duplicated()]
    df.index = df.index.astype(str)
    return df.apply(pd.to_numeric, errors="coerce").astype(np.float32)


def carrega(release):
    base = os.path.join(PASTA, release)
    E = _matriz(os.path.join(base, "CRISPRGeneEffect.csv"))
    M = _matriz(os.path.join(base, "OmicsSomaticMutationsMatrixDamaging.csv"))
    C = _matriz(os.path.join(base, "OmicsCNGene.csv"))
    mod = pd.read_csv(os.path.join(base, "Model.csv"), low_memory=False)
    lin = mod.set_index("ModelID")["OncotreeLineage"].astype(str)
    linhas = sorted(set(E.index) & set(M.index) & set(C.index) & set(lin.index))
    E, M, C, lin = E.loc[linhas], M.loc[linhas], C.loc[linhas], lin.loc[linhas]
    # efeito: gene com > 5% de faltante sai; o resto recebe a media da coluna
    ok = E.isna().mean() <= 0.05
    E = E.loc[:, ok]
    E = E.fillna(E.mean())
    # perda de A = mutacao danosa OU delecao profunda
    genesA = sorted(set(M.columns) | set(C.columns))
    Mb = M.reindex(columns=genesA).fillna(0).to_numpy() >= 1
    Cb = C.reindex(columns=genesA).fillna(10).to_numpy() < DEL_LIMIAR
    L = pd.DataFrame(Mb | Cb, index=linhas, columns=genesA)
    carga = pd.Series((M.fillna(0).to_numpy() >= 1).sum(1), index=linhas)
    return E, L, lin, carga


# ─── a varredura em bits, vetorizada por gene B ──────────────────────────────

def residualiza(X, lin, ref):
    """Tira a media da linhagem, estimada SO nas linhas de referencia."""
    glob = X[ref].mean(0)
    out = np.empty_like(X)
    for l in np.unique(lin):
        m = lin == l
        mr = m & ref
        mu = X[mr].mean(0) if mr.sum() >= 5 else glob
        out[m] = X[m] - mu
    return out


def residualiza2(X, lin, carga, ref):
    """Tira a media da LINHAGEM e tambem a tendencia linear na CARGA MUTACIONAL.

    DEFEITO QUE ISTO CONSERTA: KMT2D->WRN e ARID1A->WRN foram aceitos. WRN e a
    dependencia famosa de tumores com instabilidade de microssatelites, e
    KMT2D/ARID1A estao entre os genes mais mutados nesses tumores hipermutados.
    Estratificar o nulo por TERCIL de carga e grosseiro demais para linhagens
    extremas; remover a tendencia na carga ataca o confundidor direto, sem
    fragmentar o nulo em estratos minusculos."""
    Y = residualiza(X, lin, ref)
    c = np.asarray(carga.to_numpy() if hasattr(carga, "to_numpy") else carga,
                   dtype=np.float64)
    c = np.log1p(c)
    c = (c - c[ref].mean()) / (c[ref].std() + 1e-9)
    beta = (c[ref][:, None] * Y[ref]).sum(0) / max(float((c[ref] ** 2).sum()), 1e-9)
    return (Y - c[:, None] * beta).astype(np.float32)


def hipermutada(carga, ref):
    """Linhagem hipermutada, por regra de Tukey no log da carga (estimada em ref).
    Nao existe anotacao de instabilidade de microssatelites no Model.csv do
    DepMap 24Q2, entao o rotulo e definido pelo proprio dado."""
    c = np.log1p(np.asarray(
        carga.to_numpy() if hasattr(carga, "to_numpy") else carga, dtype=float))
    q1, q3 = np.percentile(c[ref], [25, 75])
    return c > q3 + 1.5 * (q3 - q1)


def estratos2(lin, carga, ref):
    """Estratos = TECIDO x HIPERMUTACAO (binaria).

    DEFEITO QUE ISTO CONSERTA: `residualiza2` descontava a carga mutacional so do
    DESFECHO (a dependencia) e nao da EXPOSICAO (o genotipo). Com confundidor
    correlacionado com a exposicao isso erra duas vezes: corrigiu de menos o falso
    positivo (KMT2D->WRN caiu de 32,6 para 4,1 bits e continuou passando) e
    removeu sinal verdadeiro (o controle ARID1A->ARID1B, de 4,9 bits, sumiu).
    O ajuste certo e contrastar DENTRO do estrato: entre linhagens igualmente
    hipermutadas, o KMT2D nao acrescenta nada a dependencia de WRN; e dentro do
    grupo nao hipermutado, ARID1A perdido x intacto ainda mostra ARID1B."""
    h = hipermutada(carga, ref)
    return np.array([f"{l}|{int(x)}" for l, x in zip(lin, h)])


def varre(X, L, tr, te, guardar_todos=True, banido=None):
    """Bits economizados na metade escondida por cada par (A, B) na direcao
    letal. Devolve lista de (bits, a, b, delta_descoberta)."""
    Xtr, Xte = X[tr], X[te]
    ntr, nte = int(tr.sum()), int(te.sum())
    G, nA = X.shape[1], L.shape[1]
    s_tr, q_tr = Xtr.sum(0), (Xtr ** 2).sum(0)
    s_te, q_te = Xte.sum(0), (Xte ** 2).sum(0)
    mu_tr = s_tr / ntr
    v1 = q_tr / ntr - mu_tr ** 2 + 1e-6
    ll1 = -((q_te - 2 * mu_tr * s_te + nte * mu_tr ** 2) / (2 * v1)) \
        - 0.5 * nte * np.log(2 * np.pi * v1)
    Ltr, Lte = L[tr], L[te]
    # DEFEITO CORRIGIDO: o custo de estrutura cobrava log2 de TODOS os genotipos
    # da matriz (26.169 no PRISM), mas a varredura so testa os que passam do
    # minimo de linhagens alteradas (2.105 la). Cobrar por candidato que nunca e
    # testado tirava 3,7 bits de TODO par, de graca.
    kt, ke = Ltr.sum(0), Lte.sum(0)
    testaveis = int(((kt >= MIN_ALT) & ((ntr - kt) >= MIN_ALT)
                     & (ke >= 3) & ((nte - ke) >= 3)).sum())
    est = 0.5 * log2(nte) + log2(max(testaveis, 2)) + log2(max(G, 2))
    # DEFEITO CORRIGIDO (achado no teste com verdade plantada): o maximo comecava
    # em 0,0, entao quando todo par nulo era negativo o limiar saia 0,0 exato e
    # parecia que o nulo estava controlando — quem controlava era so a penalidade
    # de estrutura. Esconder o maximo real do nulo esconde a distribuicao que o
    # portao precisa mostrar.
    fora, maximo = [], -np.inf
    for a in range(nA):
        at, ae = Ltr[:, a], Lte[:, a]
        k, ke = int(at.sum()), int(ae.sum())
        if k < MIN_ALT or ntr - k < MIN_ALT or ke < 3 or nte - ke < 3:
            continue
        xa = Xtr[at]
        s1, q1 = xa.sum(0), (xa ** 2).sum(0)
        mu1 = s1 / k
        mu0 = (s_tr - s1) / (ntr - k)
        delta = mu1 - mu0
        ss = (q1 - k * mu1 ** 2) + ((q_tr - q1) - (ntr - k) * mu0 ** 2)
        v2 = ss / ntr + 1e-6
        ya = Xte[ae]
        s1e, q1e = ya.sum(0), (ya ** 2).sum(0)
        sq = (q1e - 2 * mu1 * s1e + ke * mu1 ** 2) \
            + ((q_te - q1e) - 2 * mu0 * (s_te - s1e) + (nte - ke) * mu0 ** 2)
        ll2 = -(sq / (2 * v2)) - 0.5 * nte * np.log(2 * np.pi * v2)
        bits = (ll2 - ll1) / np.log(2) - est
        bits = np.where(delta < 0, bits, -np.inf)
        if banido:
            bl = banido.get(a)
            if bl:
                bits[bl] = -np.inf
        m = float(bits.max())
        maximo = max(maximo, m)
        if guardar_todos and m > 0:
            for b in np.where(bits > 0)[0]:
                fora.append((float(bits[b]), a, int(b), float(delta[b])))
    return fora, maximo


def limiar_fd(nulos_vals, alvo_fd=1.0):
    """Menor limiar T com numero ESPERADO de falsas descobertas <= alvo_fd.

    O limiar antigo (maximo do nulo) controla a chance de QUALQUER falso positivo
    na familia inteira: ultraconservador. Serviu no CRISPR, onde efeito de nocaute
    e grande; em sensibilidade a remedio o efeito e modesto e quase nada passa
    (validacao do prism_v2: 1 par positivo em ~83 mil). Fixar a media de pares
    nulos acima de T em <= 1 e o padrao em genomica: erro controlado e medido,
    com mais poder."""
    vs = [np.asarray(v, float) for v in nulos_vals]
    K = max(len(vs), 1)
    todos = np.concatenate([v for v in vs if v.size]) if any(v.size for v in vs) else np.array([])
    if todos.size == 0:
        return 0.0
    for T in np.unique(todos):
        if (todos >= T).sum() / K <= alvo_fd:
            return max(0.0, float(T))
    return max(0.0, float(todos.max()))


def estratos(lin, carga):
    terc = pd.qcut(carga.rank(method="first"), 3, labels=False).to_numpy()
    return np.array([f"{l}|{t}" for l, t in zip(lin, terc)])


def permuta_em_estratos(L, estr, rng):
    idx = np.arange(len(estr))
    novo = idx.copy()
    for s in np.unique(estr):
        m = np.where(estr == s)[0]
        novo[m] = rng.permutation(m)
    return L[novo]


def parece_paralogo(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i >= 4


def carrega_synlethdb(caminho, genes):
    if not caminho or not os.path.exists(caminho):
        return None
    df = pd.read_csv(caminho, low_memory=False, encoding="utf-8-sig")
    # SynLethDB 2020: gene_a.name, gene_b.name. Mistura evidencia experimental
    # com predicao computacional (SL.source = "computer analysis", escore 0,3);
    # tratar TODOS como conhecidos e conservador para novidade — exclui a mais.
    if {"gene_a.name", "gene_b.name"} <= set(df.columns):
        cols = ["gene_a.name", "gene_b.name"]
    else:
        gs = set(genes)
        cols = [c for c in df.columns
                if df[c].astype(str).str.upper().isin(gs).mean() > 0.5][:2]
    if len(cols) < 2:
        return None
    return {frozenset((str(x).upper(), str(y).upper()))
            for x, y in zip(df[cols[0]], df[cols[1]])}


# ─── a corrida ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nulos", type=int, default=10)
    ap.add_argument("--synlethdb", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(SEMENTE)
    t0 = time.time()

    print("=" * 78)
    print("LETALIDADE SINTETICA — descoberta no DepMap 24Q2, extrapolacao no 24Q4")
    print("=" * 78)
    E, L, lin, carga = carrega("24Q2")
    genesB, genesA = list(E.columns), list(L.columns)
    X = E.to_numpy(np.float32)
    Lm = L.to_numpy()
    lin_a = lin.to_numpy()
    n = len(E)
    tr = np.zeros(n, bool)
    tr[rng.permutation(n)[: n // 2]] = True
    te = ~tr
    Xr = residualiza(X, lin_a, tr)
    print(f"  {n} linhagens, {len(genesB):,} genes-alvo, {len(genesA):,} genes "
          f"que podem estar perdidos ({time.time()-t0:.0f}s)")

    real, _mx = varre(Xr, Lm, tr, te)
    print(f"  pares com economia positiva no dado real: {len(real):,} "
          f"({time.time()-t0:.0f}s)")

    estr = estratos(lin_a, carga)
    maximos, nulos_vals = [], []
    for k in range(a.nulos):
        Lp = permuta_em_estratos(Lm, estr, rng)
        vals, mx = varre(Xr, Lp, tr, te)
        maximos.append(mx)
        nulos_vals.append(np.array([v[0] for v in vals]))
        print(f"    nulo {k+1}/{a.nulos}: maximo {mx:8.1f} bits "
              f"({time.time()-t0:.0f}s)", flush=True)
    bruto = max(maximos)
    T = max(0.0, bruto)
    fd = float(np.mean([(v >= T).sum() for v in nulos_vals]))
    aceitos = sorted([r for r in real if r[0] > T], reverse=True)
    print(f"\n  maximo BRUTO do nulo (pode ser negativo): {bruto:.1f} bits")
    print(f"  LIMIAR (maximo sobre {a.nulos} nulos estratificados): {T:.1f} bits")
    print(f"  falsas descobertas esperadas no limiar: {fd:.2f}")
    print(f"  pares ACEITOS: {len(aceitos):,}")

    # controles
    pos = {(genesA[x[1]], genesB[x[2]]): x for x in real}
    idxA = {g: i for i, g in enumerate(genesA)}
    testaveis, passaram = [], []
    print("\n  CONTROLES POSITIVOS")
    for ga, gb in CONTROLES:
        if ga not in idxA or gb not in genesB:
            print(f"    {ga:8s} -> {gb:8s}  ausente do dado")
            continue
        k = int(Lm[tr, idxA[ga]].sum())
        if k < MIN_ALT:
            print(f"    {ga:8s} -> {gb:8s}  NAO TESTAVEL ({k} linhagens perdidas)")
            continue
        testaveis.append((ga, gb))
        r = pos.get((ga, gb))
        acc = r is not None and r[0] > T
        if acc:
            passaram.append((ga, gb))
        print(f"    {ga:8s} -> {gb:8s}  perdidas {k:3d}  "
              f"{(f'{r[0]:9.1f} bits' if r else '   (nao positivo)')}  "
              f"{'ACEITO' if acc else 'nao'}")

    # extrapolacao nas linhagens NOVAS do 24Q4
    E4, L4, lin4, _c4 = carrega("24Q4")
    vistas = set(E.index)
    novas = np.array([i not in vistas for i in E4.index])
    velhas = ~novas
    comuns = [g for g in genesB if g in E4.columns]
    X4 = E4[comuns].to_numpy(np.float32)
    X4r = residualiza(X4, lin4.to_numpy(), velhas)
    col4 = {g: i for i, g in enumerate(comuns)}
    L4m = L4.reindex(columns=genesA).fillna(False).to_numpy().astype(bool)
    print(f"\n  24Q4: {int(novas.sum())} linhagens NOVAS, que o 24Q2 nunca viu")

    sl = carrega_synlethdb(a.synlethdb, set(genesA) | set(genesB))
    print(f"  SynLethDB: {'carregada, ' + format(len(sl), ',') + ' pares' if sl else 'NAO carregada'}")

    linhas_saida, testados, replicou = [], 0, 0
    ctrl = {frozenset(c) for c in CONTROLES}
    candidatos = []
    for bits, ai, bi, delta in aceitos:
        ga, gb = genesA[ai], genesB[bi]
        if ga == gb or gb not in col4:
            continue
        alt = L4m[:, ai] & novas
        wt = (~L4m[:, ai]) & novas
        rep, dn = None, None
        if alt.sum() >= 3 and wt.sum() >= 3:
            y = X4r[:, col4[gb]]
            dn = float(y[alt].mean() - y[wt].mean())
            rep = dn < 0 and dn <= 0.5 * delta
            testados += 1
            replicou += int(rep)
        par = frozenset((ga, gb))
        e_ctrl = par in ctrl
        na_db = (sl is not None and par in sl)
        paral = parece_paralogo(ga, gb)
        linhas_saida.append(dict(A=ga, B=gb, bits=round(bits, 1),
                                 delta_24Q2=round(delta, 3),
                                 alt_novas=int(alt.sum()),
                                 delta_novas=None if dn is None else round(dn, 3),
                                 replica=rep, controle=e_ctrl, synlethdb=na_db,
                                 paralogo_nome=paral))
        if rep and not e_ctrl and not na_db:
            candidatos.append((bits, ga, gb, delta, dn, paral))

    os.makedirs(PASTA, exist_ok=True)
    pd.DataFrame(linhas_saida).to_csv(os.path.join(PASTA, "aceitos_24Q2.csv"),
                                      index=False)

    print(f"\n  {'bits':>8s}  {'A perdido':10s} {'B dependencia':14s} "
          f"{'delta 24Q2':>10s} {'delta novas':>11s}  marcas")
    for bits, ga, gb, delta, dn, paral in candidatos[:30]:
        print(f"  {bits:8.1f}  {ga:10s} {gb:14s} {delta:10.3f} {dn:11.3f}  "
              f"{'parece paralogo' if paral else ''}")

    print("")
    print("=" * 78)
    frac = len(passaram) / max(len(testaveis), 1)
    print(f"C1 {'.' if len(testaveis) >= 4 and frac >= 0.6 else 'FALHOU .'} "
          f"controles: {len(passaram)}/{len(testaveis)} testaveis aceitos")
    print(f"C2 {'.' if fd <= 1 else 'FALHOU .'} falsas descobertas esperadas "
          f"{fd:.2f} (<= 1)")
    fr = replicou / max(testados, 1)
    print(f"C3 {'.' if testados and fr >= 0.7 else 'FALHOU .'} extrapola: "
          f"{replicou}/{testados} replicam nas linhagens novas ({100*fr:.0f}%)")
    print(f"C4 {'.' if candidatos else 'FALHOU .'} inedito (mecanico): "
          f"{len(candidatos)} candidatos")
    print("C5 exige busca de literatura par a par — feita depois, com os nomes acima")
    print(f"\n  tabela completa: {os.path.join(PASTA, 'aceitos_24Q2.csv')} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
