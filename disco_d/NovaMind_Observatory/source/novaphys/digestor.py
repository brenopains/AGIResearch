"""
novaphys.digestor — a maquina que digere antes de armazenar
=============================================================

DE ONDE VEM
-------------
Do usuario: "uma maquina pra digerir... ela digere e extrai os 'nutrientes'
pra inteligencia enquanto do outro lado ela armazena em hdc que pode ser
reconsultado depois". E da pergunta que veio junto: da pra ir atras de datasets
variados e multimodais agora?

A medicao de 17/09 diz que NAO da, e diz por que. Ver
[[integrar-exige-dose-e-afinidade]]: ingerir so paga com AFINIDADE alta E DOSE
>= 1:1. Fonte variada = baixa afinidade cada uma. Coleta continua = dose
pequena cada uma. "Varios datasets variados" viola as duas ao mesmo tempo, e o
material sem estrutura nao fica inerte: derruba o F1 de 0,506 para 0,379 e
colapsa o lexico para o alfabeto.

Entao o portao vem ANTES da coleta. Este modulo e o portao.

A AFINIDADE SEM GABARITO
--------------------------
O medidor nao pode saber de que corpus veio o bloco — num uso real nao sabe.
Entao mede o que o proprio projeto sempre mede: QUANTO O MODELO ATUAL COMPRIME
O MATERIAL NOVO. Segmenta o bloco com o lexico corrente e olha o comprimento
medio dos pedacos. Se o que eu ja sei explica o que chegou, os pedacos sao
longos; se nao explica, tudo desanda em caracteres soltos.

    afinidade = (comprimento medio do token sob o lexico atual - 1)
                / (comprimento medio sob um lexico perfeito - 1)

E autorreferente de proposito: e a unica informacao que a maquina tem.

O RISCO REAL DO DESENHO — ARRANQUE FRIO
-----------------------------------------
A semente e de 500 tokens, ABAIXO do limiar de ~700 em que o lexico comeca a
existir. Nesse ponto o lexico e quase o alfabeto e a afinidade de QUALQUER
coisa parece baixa e parecida. Se o portao aceitar no escuro, ele engole ruido
antes de saber recusar, e ruido envenena. Este e o modo de falha que o teste
tem que poder pegar.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
Quatro bracos, mesmo held-out de 500 palavras de fala, nunca visto por ninguem:

  A  INGENUO    aceita todo bloco que chega (a "coleta de datasets variados")
  B  DIGESTOR   quarentena; integra um grupo so com afinidade >= limiar E
                massa acumulada >= 1:1 contra o corpus corrente
  C  PLACEBO    MESMA regra de dose e MESMA TAXA DE ACEITACAO de B, mas
                escolhendo os blocos por SORTEIO — afinidade ignorada
  D  ORACULO    so os blocos do dominio-alvo (teto de referencia)

  G0  PRE-CONDICAO. D reproduz a curva conhecida da fala (>= 0,48).
  G1  B - A >= 0,03   -> a digestora paga
  G2  B - C >= 0,03   -> a AFINIDADE paga alem do mero enquadramento em lote
  G3  |B - C| < 0,03  -> quem trabalha e o LOTE; o medidor e decoracao
  G4  B <= A          -> o desenho inteiro falha

C e o controle que importa: sem ele, um B melhor que A so provaria que agrupar
ajuda, o que a curva de dose ja tinha dito. A TAXA e casada de proposito.

PREVISAO: G1 e G2 passam. Mas registro que errei TRES previsoes seguidas hoje,
todas para o lado de "o sistema e mais simples do que e", e que o arranque frio
e um modo de falha que eu nao sei precificar. 0,55 em G1+G2.
"""

from __future__ import annotations

import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from novaphys.lingua import _carrega, segmenta, f1, _viterbi

ALFA = set("abcdefghijklmnopqrstuvwxyz")
STD = r"C:\Program Files\Python312\Lib"
SEMENTE = 20260917


# ─────────────────────────────────────────────────────── fontes
def _tokens_py(pasta: str, n: int, recursivo: bool = False) -> List[str]:
    out: List[str] = []
    arqs: List[str] = []
    if recursivo:
        for raiz, _d, fs in os.walk(pasta):
            arqs += [os.path.join(raiz, f) for f in sorted(fs)
                     if f.endswith(".py")]
            if len(arqs) > 400:
                break
    else:
        arqs = [os.path.join(pasta, f) for f in sorted(os.listdir(pasta))
                if f.endswith(".py")]
    for a in arqs:
        try:
            t = open(a, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        t = unicodedata.normalize("NFKD", t.lower())
        t = "".join(c for c in t if not unicodedata.combining(c))
        out += [w for w in re.sub(r"[^a-z]", " ", t).split() if 1 <= len(w) <= 20]
        if len(out) >= n:
            break
    return out[:n]


# ─────────────────────────────────────────────── modelo corrente
def _prob(toks: Sequence[str], lex) -> Dict[str, float]:
    c = Counter(toks)
    tot = sum(c.values())
    p = {w: (c[w] + 0.5) / (tot + 0.5 * len(lex)) for w in lex}
    piso = 0.5 / (tot + 0.5 * len(lex))
    for ch in ALFA:
        p.setdefault(ch, piso)
    return p


def aprende(corpus: str, rodadas: int = 400, cand: int = 24):
    toks, lex, _, _ = segmenta(corpus, rodadas=rodadas, cand=cand)
    return _prob(toks, lex)


def afinidade(bloco: str, prob: Dict[str, float], teto: float = 4.5) -> float:
    """Quanto o modelo ATUAL comprime o material novo. Sem gabarito."""
    if not prob:
        return 0.0
    pedacos = _viterbi(bloco, set(prob), prob)
    if not pedacos:
        return 0.0
    medio = sum(len(t) for t in pedacos) / len(pedacos)
    return max(0.0, min(1.0, (medio - 1.0) / (teto - 1.0)))


# ─────────────────────────────────────────────────────── o laco
def digere(semente: str, blocos: List[Tuple[str, str]], limiar: float = 0.0,
           razao: float = 1.0, verbose: bool = True):
    """
    Quarentena com dois portoes. Devolve (corpus aceito, diario).

    O portao de afinidade e RELATIVO, nao absoluto — conserto ditado pela
    primeira rodada, em que um limiar fixo de 0,45 recusou 16 de 16 blocos.
    No arranque frio o lexico e quase o alfabeto e TODA afinidade cai em
    0,03-0,07; nao ha escala absoluta. Mas a ORDEM estava certa (fala 0,065 >
    stdlib 0,050 ~ novaphys 0,049 > ruido 0,046), entao o portao usa a ordem:
    um bloco passa se sua afinidade >= MEDIANA de tudo que ja foi visto. O
    limiar se autocalibra e nao precisa de numero magico.

    E o mesmo padrao do HDC pela terceira vez no projeto: ordenador sim,
    juiz absoluto nao.
    """
    corpus = semente
    prob = aprende(corpus)
    quar: List[Tuple[str, str, float]] = []
    vistas: List[float] = []
    diario = []
    for rot, b in blocos:
        a = afinidade(b, prob)
        vistas.append(a)
        corte = sorted(vistas)[len(vistas) // 2]
        if a < corte:
            diario.append((rot, a, f"DESCARTA (< mediana {corte:.3f})",
                           len(quar)))
            continue
        quar.append((rot, b, a))
        massa = sum(len(x[1]) for x in quar)
        if massa < razao * len(corpus):
            diario.append((rot, a, "quarentena", massa))
            continue
        corpus += "".join(x[1] for x in quar)
        prob = aprende(corpus)
        diario.append((rot, a, f"INTEGRA {len(quar)} blocos", len(corpus)))
        quar = []
    if quar:                       # o que sobrou nao atingiu massa: nao entra
        diario.append(("-", 0.0, f"sobrou {len(quar)} em quarentena", 0))
    if verbose:
        for r, a, ac, m in diario:
            print(f"      {r:12s} afin {a:.3f}  {ac:28s} {m}")
    return corpus, diario


# ─────────────────────────────────────────────────── o experimento
def _heldout(ho: Sequence[str]):
    s = "".join(ho)
    fr, i = set(), 0
    for w in ho[:-1]:
        i += len(w)
        fr.add(i)
    return s, fr


def _avalia(corpus: str, s_ho: str, fr_ho) -> Tuple[float, int]:
    prob = aprende(corpus)
    F, _P, _R = f1(_viterbi(s_ho, set(prob), prob), fr_ho)
    return F, len(prob)


def experimento(n_sem: int = 500, tam: int = 250, limiar: float = 0.45,
                nucleos: int = 1) -> None:
    rng = random.Random(SEMENTE)
    pal = _carrega().split()
    ho, resto = pal[-500:], pal[:-500]
    s_ho, fr_ho = _heldout(ho)

    semente = "".join(resto[:n_sem])
    fala = resto[n_sem:]
    pt = _tokens_py(os.path.dirname(os.path.abspath(__file__)), 1200)
    en = _tokens_py(STD, 1200, recursivo=True)
    ruido = list("".join(pt[:1200]))
    rng.shuffle(ruido)

    blocos: List[Tuple[str, str]] = []
    for k in range(4):
        blocos.append(("fala", "".join(fala[k * tam:(k + 1) * tam])))
        blocos.append(("novaphys", "".join(pt[k * tam:(k + 1) * tam])))
        blocos.append(("stdlib", "".join(en[k * tam:(k + 1) * tam])))
        blocos.append(("ruido", "".join(ruido[k * 1100:(k + 1) * 1100])))
    blocos = [b for b in blocos if len(b[1]) > 50]
    rng.shuffle(blocos)

    print("=" * 80)
    print("DIGESTOR — digerir antes de armazenar paga?")
    print("=" * 80)
    print(f"  semente {n_sem} tokens de fala (ABAIXO do limiar: arranque frio)")
    print(f"  {len(blocos)} blocos chegando em ordem sorteada: "
          f"{Counter(r for r, _ in blocos)}")
    print(f"  held-out {len(ho)} palavras de fala, nunca vistas\n")

    t0 = time.time()
    print("  B  DIGESTOR (afinidade + dose)")
    corpB, diario = digere(semente, blocos, limiar=limiar)
    aceitos = sum(1 for _r, _a, ac, _m in diario if ac.startswith("INTEGRA"))
    n_bloco_aceito = sum(int(ac.split()[1]) for _r, _a, ac, _m in diario
                         if ac.startswith("INTEGRA"))
    print(f"     -> {n_bloco_aceito} de {len(blocos)} blocos integrados em "
          f"{aceitos} lotes, corpus {len(corpB)} chars\n")

    corpA = semente + "".join(b for _r, b in blocos)
    corpD = semente + "".join(b for r, b in blocos if r == "fala")

    # C: MESMA taxa de aceitacao de B, blocos escolhidos por SORTEIO
    idx = list(range(len(blocos)))
    rng.shuffle(idx)
    corpC = semente + "".join(blocos[i][1] for i in sorted(idx[:n_bloco_aceito]))

    print("  braco                          lexico   F1 held-out")
    print("  ----------------------------   ------   -----------")
    res = {}
    for rot, c in (("A  ingenuo (aceita tudo)", corpA),
                   ("B  digestor", corpB),
                   ("C  placebo (lote sorteado)", corpC),
                   ("D  oraculo (so o alvo)", corpD)):
        F, L = _avalia(c, s_ho, fr_ho)
        res[rot[0]] = F
        print(f"  {rot:28s}   {L:6d}   {F:11.3f}   ({len(c)} chars)", flush=True)

    dA, dC = res["B"] - res["A"], res["B"] - res["C"]
    print(f"\n  B-A {dA:+.3f}   B-C {dC:+.3f}   D {res['D']:.3f}   "
          f"({time.time()-t0:.0f}s)\n")
    if res["D"] < 0.48:
        print("G0 FALHOU: o oraculo nao reproduz a curva da fala. Nada a medir.")
        return
    if dA <= 0:
        print("G4 · o desenho falha: digerir nao bate aceitar tudo.")
    elif dA >= 0.03 and dC >= 0.03:
        print("G1+G2 · A DIGESTORA PAGA, e a AFINIDADE paga alem do lote.")
        print("     O portao por afinidade sobrevive ao arranque frio: o")
        print("     medidor decide sem saber de que fonte veio o bloco.")
    elif dA >= 0.03:
        print("G3 · paga, mas quem trabalha e o LOTE: o placebo com a mesma")
        print("     taxa de aceitacao empata. O medidor de afinidade e")
        print("     decoracao — a dose explica tudo.")
    else:
        print(f"G1 FALHOU (B-A {dA:+.3f}): digerir nao pagou o suficiente.")


if __name__ == "__main__":
    experimento()
