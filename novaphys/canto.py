"""
novaphys.canto — as unidades do canto: chunk ou so Markov?
============================================================

DE ONDE VEM
-------------
Do usuario: "ela aprendeu o lexico da fala... talvez algo ali deveria ser usado
em problemas em aberto?"

O que [[mdl-descobre-palavras]] e [[limiar-de-dado-e-lei]] mediram: o criterio
de bits acha as UNIDADES de um fluxo sem separadores e generaliza para dado
nunca visto — fala (F1 0,557 x trivial 0,377) e codigo (0,560 x 0,358).

A PERGUNTA ABERTA, E ELA E CONTESTADA NA LITERATURA
-----------------------------------------------------
Canto de mandarim-do-japao (Bengalese finch) e uma sequencia de silabas. Uma
linha da literatura modela o canto com CHUNKS — sequencias fixas de silabas —
como unidade basica dos modelos de transicao. Outra argumenta que processos de
Markov ocultos simples explicam a mesma complexidade sem precisar de chunk.

A questao, posta de forma medivel: **existe estrutura alem da transicao par a
par?** Se existir, um lexico de chunks descoberto por MDL tem de prever canto
RETIDO melhor que um bigrama estimado do mesmo treino.

DADO
------
Bengalese Finch song repository (figshare 4805749), arquivo de anotacao
`sober.repo1.gy6or6.032612.cbin.not.mat.tar.gz` (38,5 MB, so rotulos, sem
audio). 39 cantos de UMA ave, 2.693 silabas, alfabeto de 11 tipos (a-k).
Exemplo real: `iiiiiiiabcdeefghjkiabcdeefghjk` — notas introdutorias `i` e o
motivo `abcdeefghjk` repetido.

O CONTROLE POSITIVO NAO VEM DA AVE
------------------------------------
Nao ha gabarito de chunk — e justamente o que esta em aberto. Entao o controle
positivo e a FALA, onde a verdade e conhecida e o mesmo codigo ja marcou 0,557
contra 0,377 trivial. Valida-se o instrumento onde ha gabarito e so depois se
aponta para onde nao ha. Foi a ausencia disso que tornou o PRISM ininterpretavel
([[letalidade-sintetica-depmap]]).

PRE-REGISTRO
--------------
Treino/retido por CANTO (nunca por silaba: silabas do mesmo canto vazariam).
Tudo em bits por silaba no canto RETIDO, com o modelo CONGELADO no treino.

  U0  PRE-CONDICAO. bigrama < unigrama em bits/silaba no retido. Se nem
      transicao par a par ajudar, nao ha estrutura sequencial e nada vale.
  U1  chunk < bigrama em >= 0,05 bits/silaba
      -> ha estrutura ALEM do par a par; o lado dos chunks ganha apoio
  U2  |chunk - bigrama| < 0,05  -> sem evidencia de chunk alem de Markov
  U3  o NULO (silabas embaralhadas dentro de cada canto, preservando a
      frequencia unigrama) mostra a mesma vantagem -> e duto, nao estrutura

  10 particoes treino/retido diferentes, com media e desvio. Nada de n=1 —
  quatro controles n=1 me enganaram em 17/09.

PREVISAO: U1, 0,6. Mecanismo: numa cadeia quase deterministica o bigrama ja
captura quase tudo, entao o ganho do chunk so pode vir dos PONTOS DE
RAMIFICACAO, onde a historia importa mais que o passo anterior — e e isso que a
literatura de chunk descreve. Risco nomeado: 2.693 silabas esta logo acima do
limiar de ~1000-1500, entao o lexico pode sair pobre e o chunk empatar por
falta de dado, nao por falta de estrutura.

LIMITES, escritos antes: UMA ave, 39 cantos. Um resultado aqui e sobre esta ave.
E segmentar nao e entender: achar as unidades nao diz o que elas significam.
"""

from __future__ import annotations

import glob
import math
import os
import random
import statistics
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from novaphys.lingua import _viterbi, segmenta

PASTA = r"D:\agi_dados\passaro"
N_PART = 10
SEMENTE = 20260917


def carrega_cantos(pasta: str = PASTA) -> List[str]:
    import scipy.io
    fs = sorted(glob.glob(os.path.join(pasta, "**", "*.not.mat"),
                          recursive=True))
    cantos = []
    for f in fs:
        try:
            m = scipy.io.loadmat(f)
            s = "".join(str(m["labels"][0]))
        except Exception:
            continue
        s = "".join(ch for ch in s if ch.isalpha())
        if len(s) >= 10:
            cantos.append(s)
    return cantos


# ─────────────────────────────────────────── os tres modelos
def bits_unigrama(treino: str, retido: Sequence[str]) -> float:
    c = Counter(treino)
    alf = sorted(set(treino) | {ch for s in retido for ch in s})
    tot = sum(c.values())
    p = {a: (c.get(a, 0) + 0.5) / (tot + 0.5 * len(alf)) for a in alf}
    bits = sum(-math.log2(p[ch]) for s in retido for ch in s)
    n = sum(len(s) for s in retido)
    return bits / max(n, 1)


def bits_ngrama(treino: Sequence[str], retido: Sequence[str],
                ordem: int = 2) -> float:
    """
    n-grama de ordem qualquer, com recuo (backoff) para ordens menores.

    ADICIONADO DEPOIS DA RODADA 1, e por honestidade e nao por resultado: a
    rodada 1 comparou chunk contra BIGRAMA e eu anunciei "rival forte". Bigrama
    e Markov de PRIMEIRA ORDEM; o rival serio da literatura tem mais estados.
    Ordem 3 e 4 sao o rival mais duro que cabe neste dado.
    """
    alf = sorted({ch for s in list(treino) + list(retido) for ch in s})
    V = len(alf)
    ctx = [Counter() for _ in range(ordem)]        # contagens por ordem
    for s in treino:
        for k in range(ordem):
            for i in range(len(s)):
                pre = s[max(0, i - k):i]
                if len(pre) == k:
                    ctx[k][(pre, s[i])] += 1
    tot = [Counter() for _ in range(ordem)]
    for k in range(ordem):
        for (pre, _c), v_ in ctx[k].items():
            tot[k][pre] += v_

    def prob(pre: str, ch: str) -> float:
        # recuo: usa a maior ordem com contexto visto, senao desce
        for k in range(min(len(pre), ordem - 1), -1, -1):
            pk = pre[len(pre) - k:] if k else ""
            t = tot[k].get(pk, 0)
            if t >= 2:
                return (ctx[k].get((pk, ch), 0) + 0.5) / (t + 0.5 * V)
        return 1.0 / V

    bits, n_ = 0.0, 0
    for s in retido:
        for i, ch in enumerate(s):
            pre = s[max(0, i - (ordem - 1)):i]
            bits += -math.log2(prob(pre, ch))
        n_ += len(s)
    return bits / max(n_, 1)


def bits_bigrama(treino: Sequence[str], retido: Sequence[str]) -> float:
    alf = sorted({ch for s in list(treino) + list(retido) for ch in s})
    ini = Counter()
    tr: Dict[str, Counter] = {a: Counter() for a in alf}
    for s in treino:
        ini[s[0]] += 1
        for a, b in zip(s, s[1:]):
            tr[a][b] += 1
    ti = sum(ini.values())
    pi = {a: (ini.get(a, 0) + 0.5) / (ti + 0.5 * len(alf)) for a in alf}
    bits, n = 0.0, 0
    for s in retido:
        bits += -math.log2(pi[s[0]])
        n += len(s)
        for a, b in zip(s, s[1:]):
            tt = sum(tr[a].values())
            bits += -math.log2((tr[a].get(b, 0) + 0.5) /
                               (tt + 0.5 * len(alf)))
    return bits / max(n, 1)


def bits_chunk(treino: Sequence[str], retido: Sequence[str],
               rodadas: int = 400, cand: int = 24) -> Tuple[float, int]:
    """
    Aprende o lexico de chunks do treino por MDL, CONGELA, e codifica o retido
    segmentando por viterbi. Modelo fixo no treino: nenhum olhar no retido.
    """
    corpus = "".join(treino)
    toks, lex, _, _ = segmenta(corpus, rodadas=rodadas, cand=cand)
    alf = {ch for s in list(treino) + list(retido) for ch in s}
    lex = set(lex) | alf
    c = Counter(toks)
    tot = sum(c.values())
    p = {w: (c.get(w, 0) + 0.5) / (tot + 0.5 * len(lex)) for w in lex}
    bits, n = 0.0, 0
    for s in retido:
        ped = _viterbi(s, lex, p)
        bits += sum(-math.log2(p[w]) for w in ped)
        n += len(s)
    return bits / max(n, 1), len([w for w in lex if len(w) > 1])


def bits_chunk_bigrama(treino: Sequence[str], retido: Sequence[str],
                       rodadas: int = 400, cand: int = 24) -> Tuple[float, int]:
    """
    Chunk COM TRANSICAO entre chunks — o modelo que a literatura de fato usa.

    ADICIONADO por honestidade: `bits_chunk` codifica cada chunk de forma
    INDEPENDENTE (unigrama sobre chunks) e perdeu do trigrama de silaba por
    0,066 bits. Mas isso comparava unigrama-de-chunk contra trigrama-de-silaba,
    o que handicapa o chunk. Aqui o chunk tem transicao, como o rival.

    A segmentacao do retido usa o lexico CONGELADO no treino e um viterbi de
    unigrama (a transicao entra so na codificacao, nao na escolha do corte);
    isso e conservador para o lado do chunk e esta declarado.
    """
    corpus = "".join(treino)
    toks, lex, _, _ = segmenta(corpus, rodadas=rodadas, cand=cand)
    alf = {ch for s in list(treino) + list(retido) for ch in s}
    lex = set(lex) | alf
    c = Counter(toks)
    tot = sum(c.values())
    p_uni = {w: (c.get(w, 0) + 0.5) / (tot + 0.5 * len(lex)) for w in lex}

    ini = Counter()
    trans: Dict[str, Counter] = {}
    for s in treino:
        ped = _viterbi(s, lex, p_uni)
        if not ped:
            continue
        ini[ped[0]] += 1
        for a, b in zip(ped, ped[1:]):
            trans.setdefault(a, Counter())[b] += 1
    L = len(lex)
    ti = sum(ini.values()) or 1

    bits, n = 0.0, 0
    for s in retido:
        ped = _viterbi(s, lex, p_uni)
        if not ped:
            continue
        bits += -math.log2((ini.get(ped[0], 0) + 0.5) / (ti + 0.5 * L))
        for a, b in zip(ped, ped[1:]):
            tc = trans.get(a)
            if tc and sum(tc.values()) >= 2:
                t = sum(tc.values())
                bits += -math.log2((tc.get(b, 0) + 0.5) / (t + 0.5 * L))
            else:
                bits += -math.log2(p_uni[b])      # recuo para o unigrama
        n += len(s)
    return bits / max(n, 1), len([w for w in lex if len(w) > 1])


def _embaralha(cantos: Sequence[str], rng) -> List[str]:
    """Nulo: embaralha as silabas DENTRO de cada canto. Preserva o unigrama
    e o comprimento, destroi a ordem — que e exatamente o que esta em teste."""
    out = []
    for s in cantos:
        l = list(s)
        rng.shuffle(l)
        out.append("".join(l))
    return out


def experimento() -> None:
    cantos = carrega_cantos()
    n_sil = sum(len(s) for s in cantos)
    print("=" * 76)
    print("CANTO — as unidades sao CHUNK, ou bigrama ja basta?")
    print("=" * 76)
    print(f"  {len(cantos)} cantos, {n_sil} silabas, "
          f"alfabeto {len(set(''.join(cantos)))}")
    print(f"  exemplo: {cantos[0][:60]}")
    print(f"  {N_PART} particoes treino/retido POR CANTO\n")

    rng = random.Random(SEMENTE)
    res = {"uni": [], "bi": [], "tri": [], "tetra": [], "chunk": [],
           "chunk_bi": [], "n_lex": [], "nulo_bi": [], "nulo_chunk": []}
    for _ in range(N_PART):
        idx = list(range(len(cantos)))
        rng.shuffle(idx)
        corte = int(0.75 * len(idx))
        tr = [cantos[i] for i in idx[:corte]]
        ho = [cantos[i] for i in idx[corte:]]
        res["uni"].append(bits_unigrama("".join(tr), ho))
        res["bi"].append(bits_bigrama(tr, ho))
        res["tri"].append(bits_ngrama(tr, ho, 3))
        res["tetra"].append(bits_ngrama(tr, ho, 4))
        b, nl = bits_chunk(tr, ho)
        res["chunk"].append(b)
        res["chunk_bi"].append(bits_chunk_bigrama(tr, ho)[0])
        res["n_lex"].append(nl)
        tre, hoe = _embaralha(tr, rng), _embaralha(ho, rng)
        res["nulo_bi"].append(bits_bigrama(tre, hoe))
        res["nulo_chunk"].append(bits_chunk(tre, hoe)[0])

    def mm(k):
        return statistics.mean(res[k]), statistics.pstdev(res[k])

    print("  modelo          bits/silaba no retido      nulo (embaralhado)")
    print("  ------------    ----------------------     ------------------")
    u, su = mm("uni")
    b, sb = mm("bi")
    c, sc = mm("chunk")
    nb, _ = mm("nulo_bi")
    nc, _ = mm("nulo_chunk")
    print(f"  unigrama        {u:.3f} +- {su:.3f}")
    t3, st3 = mm("tri")
    t4, st4 = mm("tetra")
    print(f"  bigrama         {b:.3f} +- {sb:.3f}              {nb:.3f}")
    print(f"  trigrama        {t3:.3f} +- {st3:.3f}")
    print(f"  4-grama         {t4:.3f} +- {st4:.3f}")
    print(f"  chunk (MDL)     {c:.3f} +- {sc:.3f}              {nc:.3f}")
    cb, scb = mm("chunk_bi")
    print(f"  chunk + transicao {cb:.3f} +- {scb:.3f}")
    print(f"  lexico medio: {statistics.mean(res['n_lex']):.0f} chunks")
    print("")

    print("=" * 76)
    if b >= u:
        print(f"U0 FALHOU: bigrama ({b:.3f}) nao bate unigrama ({u:.3f}).")
        print("   Nao ha nem estrutura sequencial; nada acima e interpretavel.")
        return
    print(f"U0 . bigrama bate unigrama ({b:.3f} < {u:.3f}): ha estrutura "
          f"sequencial.")
    ganho_nulo = nb - nc
    ganho = b - c
    if ganho_nulo >= 0.05:
        print(f"U3 . o NULO tambem mostra ganho de chunk ({ganho_nulo:+.3f} "
              f"bits). E DUTO, nao estrutura do canto.")
        return
    print(f"  no nulo o ganho do chunk e {ganho_nulo:+.3f} bits (sem efeito).")
    c = min(c, cb)            # o MELHOR modelo de chunk, nao o pior
    melhor_ng = min(b, t3, t4)
    nome_ng = {b: "bigrama", t3: "trigrama", t4: "4-grama"}[melhor_ng]
    g_forte = melhor_ng - c
    print(f"  melhor n-grama: {nome_ng} ({melhor_ng:.3f}); "
          f"chunk - melhor = {g_forte:+.3f}")
    if ganho >= 0.05 and g_forte >= 0.05:
        print(f"U1 . HA ESTRUTURA ALEM DE N-GRAMA: chunk poupa "
              f"{g_forte:.3f} bits/silaba sobre o MELHOR n-grama ({nome_ng}).")
        print("     O lado dos CHUNKS ganha apoio nesta ave.")
    elif ganho >= 0.05:
        print(f"U1 PARCIAL: chunk bate o bigrama ({ganho:+.3f}) mas NAO o")
        print(f"     melhor n-grama ({nome_ng}, {g_forte:+.3f}). O que ha alem")
        print("     do par a par e capturado por contexto maior, sem precisar")
        print("     de chunk.")
    else:
        print(f"U2 . SEM EVIDENCIA DE CHUNK: chunk x bigrama = {ganho:+.3f} "
              f"bits/silaba.")
        print("     Nesta ave, transicao par a par ja explica o que o lexico")
        print("     de chunks explica.")


if __name__ == "__main__":
    experimento()
