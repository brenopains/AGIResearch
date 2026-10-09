"""
novaphys.acervo — o elo que faltava: o extraido VAI para a loja
=================================================================

DE ONDE VEM
-------------
Do usuario, corrigindo minha leitura da digestora: ela nao e porteiro, e "o que
fez ela funcionar e extrair os nutrientes pra AGI enquanto manda dado
organizado pro hdc pra ela digerir mais sob demanda depois".

Checado por grep: a seta do meio NAO existe. `minerar` e chamado em quatro
modulos e cada um guarda a biblioteca numa variavel LOCAL; o lexico de
`lingua.segmenta` nem sai da funcao. A digestora extrai nutriente e ele evapora
no escopo. [[sob-demanda-funciona]] mediu a loja guardando dado BRUTO; ninguem
nunca guardou o EXTRAIDO.

A PERGUNTA DE DESENHO QUE ISSO ABRE
-------------------------------------
Guardar o nutriente extraido e melhor do que guardar o dado bruto? Nao e
obvio nos dois sentidos:

  a favor  o lexico extraido de um corpus se aplica DIRETO a uma tarefa nova;
           o bloco bruto precisa ser re-aprendido junto com a sonda. A extracao
           e a parte cara e reaproveitavel.
  contra   o lexico e um resumo com perda. O bloco bruto tem a distribuicao
           inteira, e [[memoria-entrincheira]] mediu que guardar o BRUTO paga
           (+36 em reanalisar sem as leis velhas, contra +15 de so juntar).

PRE-REGISTRO
--------------
  T0  PRE-CONDICAO. O braco BRUTO reproduz a vantagem de [[sob-demanda-funciona]]
      sobre o sorteio (>= 2 desvios). Senao o instrumento nao resolve.
  T1  EXTRAIDO > BRUTO em >= 0,02 de F1  -> extrair antes de guardar paga
  T2  |EXTRAIDO - BRUTO| < 0,02          -> extracao neutra; quem trabalha e a
                                            loja, e o elo e conveniencia
  T3  EXTRAIDO < BRUTO - 0,02            -> extrair PERDE informacao do bruto

  Orcamento casado: os dois bracos recuperam o MESMO numero de itens (K) e
  sao avaliados no MESMO held-out. O placebo sorteia K itens, com >= 10
  sorteios e desvio reportado — a regra que eu quebrei quatro vezes hoje.

PREVISAO: T1, 0,65, pelo mecanismo "a extracao e a parte cara e reaproveitavel".
Risco nomeado: o lexico extraido de um corpus pequeno esta ABAIXO do limiar de
[[limiar-de-dado-e-lei]] e nesse caso e o alfabeto — guardar isso nao pode
ajudar, e o braco extraido empataria com nada.
Placar do dia: palpite sem mecanismo 0/4; com mecanismo 3/7, e os 4 erros foram
todos por nao perguntar qual mecanismo era o DOMINANTE.
"""

from __future__ import annotations

import os
import random
import re
import statistics
import unicodedata
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from novaphys.digestor import _heldout, aprende
from novaphys.hdc_indice import Espaco, de_texto, sim
from novaphys.lingua import _carrega, f1, _viterbi, segmenta

ALFA = set("abcdefghijklmnopqrstuvwxyz")
STD = r"C:\Program Files\Python312\Lib"
K_TOPO = 4
N_SORTEIOS = 10
SEMENTE = 20260917


# ───────────────────────────────────────────────── o acervo
class Acervo:
    """
    Guarda tudo, em duas formas, e ordena sob demanda.

    Cada item leva o BRUTO e o EXTRAIDO. O vetor de busca e sempre do bruto,
    para que a recuperacao seja a MESMA nos dois bracos e a unica diferenca
    seja o que se USA depois de recuperar — senao o teste compararia duas
    coisas ao mesmo tempo.
    """

    def __init__(self, e: Espaco) -> None:
        self.e = e
        self.itens: List[dict] = []

    def guarda(self, rotulo: str, bruto: str, rodadas: int = 400,
               cand: int = 24) -> None:
        toks, lex, _, _ = segmenta(bruto, rodadas=rodadas, cand=cand)
        self.itens.append({
            "rotulo": rotulo,
            "bruto": bruto,
            "lexico": {w for w in lex if len(w) > 1},   # o NUTRIENTE
            "conta": Counter(toks),
            "v": de_texto(self.e, bruto),
        })

    def ordena(self, sonda: str) -> List[dict]:
        sv = de_texto(self.e, sonda)
        r = sorted(self.itens, key=lambda it: -sim(sv, it["v"]))
        return r


def _prob_de(conta: Counter, lex: set) -> Dict[str, float]:
    tot = sum(conta.values()) or 1
    p = {w: (conta.get(w, 0) + 0.5) / (tot + 0.5 * len(lex)) for w in lex}
    piso = 0.5 / (tot + 0.5 * max(len(lex), 1))
    for ch in ALFA:
        p.setdefault(ch, piso)
    return p


def usa_bruto(sonda: str, itens: Sequence[dict], s_ho, fr_ho) -> float:
    """Concatena os brutos recuperados com a sonda e RE-APRENDE. O caro."""
    corpus = sonda + "".join(it["bruto"] for it in itens)
    prob = aprende(corpus)
    return f1(_viterbi(s_ho, set(prob), prob), fr_ho)[0]


def usa_extraido(sonda: str, itens: Sequence[dict], s_ho, fr_ho) -> float:
    """
    Usa o LEXICO ja extraido: aprende so da sonda e ACRESCENTA as entradas
    recuperadas, com as contagens que vieram junto. Nao re-aprende o corpus
    inteiro — e esse o ponto.
    """
    toks, lex, _, _ = segmenta(sonda, rodadas=400, cand=24)
    conta = Counter(toks)
    lex = set(lex)
    for it in itens:
        lex |= it["lexico"]
        for w, c in it["conta"].items():
            if len(w) > 1:
                conta[w] += c
    prob = _prob_de(conta, lex)
    return f1(_viterbi(s_ho, set(prob), prob), fr_ho)[0]


# ───────────────────────────────────────────────── as fontes
def _tokens_py(pasta: str, n: int, recursivo: bool = False) -> List[str]:
    out, arqs = [], []
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


def experimento(tam_bloco: int = 2000) -> None:
    """
    RODADA 2. A rodada 1 tinha blocos de 250-400 tokens e o lexico extraido
    saiu com 16 ENTRADAS — abaixo do limiar de [[limiar-de-dado-e-lei]], onde
    o lexico E o alfabeto. Era o risco que o pre-registro nomeou, e deu T3
    (-0,065) medindo o limiar em vez de medir extraido-contra-bruto.

    O alvo muda de FALA para CODIGO porque a transcricao tem 2.070 palavras no
    total e, tirando held-out e sonda, sobram 1.370: da para UM bloco acima do
    limiar, nao para dezesseis. Em codigo ha material de sobra, e cada bloco
    entra com 2.000 tokens — na curva medida, lexico de ~230 entradas.
    """
    e = Espaco()
    rng = random.Random(SEMENTE)
    aqui = os.path.dirname(os.path.abspath(__file__))
    pal = _carrega().split()

    en_todo = _tokens_py(STD, 24000, recursivo=True)
    ho = en_todo[:400]                       # held-out de CODIGO
    sonda = "".join(en_todo[400:400 + 600])  # sonda de codigo
    en = en_todo[1000:]
    pt = _tokens_py(aqui, 4 * tam_bloco)
    ruido = list("".join(pt[:2 * tam_bloco]))
    rng.shuffle(ruido)
    s_ho, fr_ho = _heldout(ho)

    print("=" * 78)
    print("ACERVO — guardar o EXTRAIDO bate guardar o BRUTO? (rodada 2)")
    print("=" * 78)
    print(f"  blocos de {tam_bloco} tokens — ACIMA do limiar, para que haja")
    print("  nutriente de verdade a extrair (a rodada 1 tinha 16 entradas)")
    print("")
    ac = Acervo(e)
    nb = tam_bloco
    for k in range(4):
        ac.guarda(f"stdlib{k}", "".join(en[k * nb:(k + 1) * nb]))
        ac.guarda(f"novaphys{k}", "".join(pt[k * nb:(k + 1) * nb]))
        ac.guarda(f"fala{k}", "".join(pal[k * 400:(k + 1) * 400]))
        ac.guarda(f"ruido{k}", "".join(ruido[k * 7000:(k + 1) * 7000]))
    print(f"  acervo: {len(ac.itens)} itens, cada um com BRUTO e LEXICO extraido")
    med = statistics.mean(len(it["lexico"]) for it in ac.itens)
    print(f"  tamanho medio do lexico extraido: {med:.0f} entradas")
    print(f"  tarefa: segmentar {len(ho)} palavras de fala, sonda de 300\n")

    topo = ac.ordena(sonda)[:K_TOPO]
    print(f"  recuperado (K={K_TOPO}): {[it['rotulo'] for it in topo]}")

    f_bruto = usa_bruto(sonda, topo, s_ho, fr_ho)
    f_extra = usa_extraido(sonda, topo, s_ho, fr_ho)
    f_so_sonda = usa_extraido(sonda, [], s_ho, fr_ho)

    pb, pe = [], []
    for _ in range(N_SORTEIOS):
        pick = rng.sample(ac.itens, K_TOPO)
        pb.append(usa_bruto(sonda, pick, s_ho, fr_ho))
        pe.append(usa_extraido(sonda, pick, s_ho, fr_ho))
    mb, sb = statistics.mean(pb), statistics.pstdev(pb)
    me, se = statistics.mean(pe), statistics.pstdev(pe)

    print("")
    print("  braco                          F1 held-out")
    print("  ---------------------------    -----------")
    print(f"  so a sonda (nada recuperado)   {f_so_sonda:.3f}")
    print(f"  BRUTO, recuperado              {f_bruto:.3f}")
    print(f"  BRUTO, sorteado ({N_SORTEIOS}x)         {mb:.3f} +- {sb:.3f}")
    print(f"  EXTRAIDO, recuperado           {f_extra:.3f}")
    print(f"  EXTRAIDO, sorteado ({N_SORTEIOS}x)      {me:.3f} +- {se:.3f}")

    print("\n" + "=" * 78)
    if sb > 1e-9 and (f_bruto - mb) / sb < 2.0:
        print(f"T0 FALHOU: o bruto recuperado esta a "
              f"{(f_bruto-mb)/max(sb,1e-9):.1f} desvios do sorteio. O")
        print("   instrumento nao reproduz a vantagem conhecida; nada abaixo")
        print("   e interpretavel.")
        return
    print(f"T0 . bruto recuperado a {(f_bruto-mb)/max(sb,1e-9):.1f} desvios "
          f"do sorteio.")
    d = f_extra - f_bruto
    if d >= 0.02:
        print(f"T1 . GUARDAR O EXTRAIDO PAGA: {f_extra:.3f} contra "
              f"{f_bruto:.3f} ({d:+.3f}).")
        print("     A extracao e a parte cara e reaproveitavel, e guarda-la")
        print("     pronta rende. O elo extrator->loja se justifica sozinho.")
    elif abs(d) < 0.02:
        print(f"T2 . NEUTRO ({d:+.3f}). Quem trabalha e a loja; o elo e")
        print("     conveniencia e economia de computo, nao ganho de acerto.")
    else:
        print(f"T3 . EXTRAIR PERDE ({d:+.3f}). O resumo joga fora informacao")
        print("     que o bruto tinha — guardar o bruto e obrigatorio, e isso")
        print("     concorda com o +36 da reanalise em [[memoria-entrincheira]].")


if __name__ == "__main__":
    experimento()
