"""
novaphys.lingua — o mesmo criterio que resolve grades descobre PALAVRAS?
=========================================================================

A INTUICAO DO USUARIO
-----------------------
"Talvez conseguiriamos fazer ela usar os proprios mecanismos para internalizar
aprendizado sobre escrita, lingua, gramatica e assim formular seus pensamentos."

Nao via LLM. Via o MESMO criterio que ja esta no nucleo: comprimento de
descricao. Se o principio for geral, ele tem que achar estrutura em lingua do
mesmo jeito que acha em grade.

E ha precedente forte, de trinta anos: de Marcken (1995/96) mostrou que MDL
constroi segmentadores de texto para ingles e para chines — construindo um
lexico avaliado por quanto ele COMPRIME o texto. Goldsmith (2001) fez o mesmo
com morfologia de linguas europeias. O principio e Solomonoff (1964) e
Rissanen (1978), o mesmo que fundou o nosso nucleo.

O EXPERIMENTO
---------------
Pegar texto REAL, apagar todos os espacos, e perguntar: minimizando

    bits(lexico) + bits(texto codificado com o lexico)

o sistema recupera onde estavam as fronteiras de palavra?

O CORPUS, E POR QUE ESTE
--------------------------
A transcricao da gravacao que o proprio usuario fez resolvendo o ARC e narrando
o que pensava — 455 segmentos de portugues falado. Tres razoes: e texto real e
nao escrito por mim (eu escrever o corpus contaminaria o teste); e portugues
coloquial, nao prosa editada, o que e mais dificil; e e o mesmo projeto.

O CONTROLE PAREADO
--------------------
Embaralhar os CARACTERES do corpus: preserva exatamente a frequencia de cada
letra e destroi toda estrutura de palavra. Se o MDL "achar palavras" ai, esta
achando artefato de frequencia, nao lingua.

A medida e F1 de fronteira contra os espacos verdadeiros.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  G0  PRE-CONDICAO. O MDL tem que REDUZIR a descricao em relacao ao lexico de
      caracteres isolados. Senao o algoritmo nao esta funcionando.

  G1  F1 real bem acima do controle -> o mesmo criterio que resolve grades
      descobre palavras; o principio e geral
  G2  F1 real ~ controle -> MDL acha frequencia, nao lingua

MINHA CALIBRAGEM: cinquenta e oito previsoes, VINTE E DUAS certas.

PREVISAO: G1, com F1 de fronteira entre 0,45 e 0,65 no real e perto de 0,15 no
controle. Isto e o teste com MAIS precedente na literatura de toda a sessao —
de Marcken ja fez funcionar — entao aposto que funciona. O que ele NAO prova, e
digo antes de ver: segmentar nao e entender. E o primeiro degrau, e o unico que
se consegue sem dado massivo.
"""

from __future__ import annotations

import io
import os
import random
import re
import time
import unicodedata
from collections import Counter
from math import log2

SEMENTE = 20260910
CAMINHOS = (
    ('C:\\Users\\usuario\\Documents\\AGI 2.0\\transcricao_introspeccao.txt').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')),
    ('C:\\Users\\usuario\\Documents\\Simulador de realidade\\transcricao_introspeccao.txt').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')),
)


def _limpa(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"\[[^\]]*\]", " ", t)          # carimbos de tempo
    t = re.sub(r"[^a-z ]", " ", t)
    return re.sub(r" +", " ", t).strip()


def _carrega() -> str:
    for c in CAMINHOS:
        if os.path.exists(c):
            return _limpa(io.open(c, encoding="utf-8", errors="ignore").read())
    raise FileNotFoundError("transcricao nao encontrada em " + str(CAMINHOS))


def _dl(tokens, lex):
    """bits(lexico) + bits(corpus | lexico), codigo de unigrama."""
    alfa = log2(27)
    blex = sum((len(w) + 1) * alfa for w in lex)
    c = Counter(tokens)
    n = sum(c.values())
    bcor = -sum(v * log2(v / n) for v in c.values())
    return blex + bcor


def _viterbi(s, lex, prob):
    """Segmentacao de custo minimo dado um lexico com probabilidades."""
    L = max((len(w) for w in lex), default=1)
    inf = float("inf")
    best = [0.0] + [inf] * len(s)
    back = [0] * (len(s) + 1)
    for i in range(1, len(s) + 1):
        for k in range(1, min(L, i) + 1):
            w = s[i - k:i]
            if w in prob:
                c = best[i - k] - log2(prob[w])
                if c < best[i]:
                    best[i], back[i] = c, i - k
    out, i = [], len(s)
    while i > 0:
        out.append(s[back[i]:i])
        i = back[i]
    return out[::-1]


def segmenta(s: str, rodadas: int = 60, max_len: int = 12, cand: int = 8):
    """
    MDL guloso a la de Marcken: parte do lexico de caracteres, propoe fundir o
    par adjacente mais frequente, ACEITA so se a descricao total cai. O lexico
    cresce enquanto comprimir, e para quando uma entrada nova custa mais do que
    economiza — Ockham como aritmetica, o mesmo criterio do nucleo das grades.
    """
    toks = list(s)
    lex = set(toks)
    dl0 = _dl(toks, lex)
    dl = dl0
    for _ in range(rodadas):
        pares = Counter(zip(toks, toks[1:]))
        aceitou = False
        for (x, y), _n in pares.most_common(cand):
            novo = x + y
            if len(novo) > max_len or novo in lex:
                continue
            lex2 = lex | {novo}
            c = Counter(toks)
            c[novo] = _n
            tot = sum(c.values())
            prob = {w: (c[w] + 0.5) / (tot + 0.5 * len(lex2)) for w in lex2}
            t2 = _viterbi(s, lex2, prob)
            d2 = _dl(t2, lex2)
            if d2 < dl:
                toks, lex, dl, aceitou = t2, lex2, d2, True
                break
        if not aceitou:
            break
    return toks, lex, dl0, dl


def f1(toks, verdade_pos):
    pos, i = set(), 0
    for t in toks[:-1]:
        i += len(t)
        pos.add(i)
    tp = len(pos & verdade_pos)
    p = tp / max(len(pos), 1)
    r = tp / max(len(verdade_pos), 1)
    return 2 * p * r / max(p + r, 1e-9), p, r


def main() -> None:
    texto = _carrega()
    palavras = texto.split()[:1400]
    s = "".join(palavras)
    fronteiras, i = set(), 0
    for w in palavras[:-1]:
        i += len(w)
        fronteiras.add(i)

    rng = random.Random(SEMENTE)
    emb = list(s)
    rng.shuffle(emb)
    emb = "".join(emb)

    print("=" * 76)
    print("LINGUA — o criterio que resolve grades descobre PALAVRAS?")
    print("=" * 76)
    print(f"  corpus: transcricao da gravacao do usuario, {len(palavras)} "
          f"palavras, {len(s)} caracteres sem espaco")

    t0 = time.time()
    tr, lr, a0, a1 = segmenta(s)
    te, le, b0, b1 = segmenta(emb)
    Fr, Pr, Rr = f1(tr, fronteiras)
    Fe, Pe, Re = f1(te, fronteiras)

    print(f"\n  ({time.time()-t0:.0f}s)\n")
    print("  corpus        lexico   descricao (bits)        F1 fronteira")
    print("  -----------   ------   ---------------------   ------------")
    print(f"  real          {len(lr):6d}   {a0:9.0f} -> {a1:9.0f}   "
          f"{Fr:.3f}  (p {Pr:.2f}, r {Rr:.2f})")
    print(f"  embaralhado   {len(le):6d}   {b0:9.0f} -> {b1:9.0f}   "
          f"{Fe:.3f}  (p {Pe:.2f}, r {Re:.2f})")
    novas = sorted((w for w in lr if len(w) >= 3), key=len, reverse=True)
    print(f"\n  unidades de 3+ letras que o MDL criou no real: {novas[:24]}")

    print("")
    print("=" * 76)
    if a1 >= a0:
        print("G0 FALHOU: o MDL nao reduziu a descricao; algoritmo quebrado.")
    elif Fr > Fe + 0.15:
        print("G1 . o mesmo criterio que resolve grades descobre palavras.")
        print("     O principio e geral — e segmentar ainda nao e entender.")
    else:
        print("G2 . MDL acha frequencia de letra, nao estrutura de lingua.")


if __name__ == "__main__":
    main()
