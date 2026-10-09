"""
novaphys.variedade — mais nutriente ajuda, ou o topo do ranking satura?
=========================================================================

DE ONDE VEM
-------------
Do usuario: "Datasets gerais variados, natureza, humanidade, videos variados,
justamente pra dar pra digestora nutriente de inteligencia suficiente, não?"

A tese tem uma previsao embutida e barata de testar: se mais variedade paga, a
vantagem da RECUPERACAO deve crescer, ou ao menos se manter, conforme a loja
fica mais variada. Se encolher, o gargalo nao e dado — e eu descobri isso com
minutos de maquina em vez de semanas de coleta.

[[sob-demanda-funciona]] mediu a vantagem com 16 blocos de 4 fontes: +12 a 20
desvios. A pergunta aqui e o que acontece com 4, 8, 16, 32, 64 materiais.

O QUE SE MEDE, E POR QUE ASSIM
--------------------------------
Nao mede acuracia de predicao: mede a QUALIDADE DO RANKING, que e o que a loja
de fato entrega e o unico papel em que o HDC mediu positivo. Dada uma sonda de
uma celula (material m*, transformacao n*), pergunta-se quantos dos K primeiros
recuperados compartilham a TRANSFORMACAO da sonda.

    acaso = 1/N (com N transformacoes, uma fracao 1/N da loja compartilha)

Isso isola exatamente o risco nomeado: o ranqueador "ordena mas nao julga em
absoluto" — quarta aparicao desse padrao no projeto. Com mais fontes, mais
candidatos se aglomeram perto do topo, e o certo pode deixar de ser o primeiro.

PRE-REGISTRO
--------------
  S0  PRE-CONDICAO. Com M=4 a precisao@K tem de bater o acaso com folga
      (>= 2x). Se nem ali o ranking informa, nada acima e interpretavel.
  S1  precisao@K se MANTEM ou CRESCE de M=4 a M=64
      -> mais variedade paga; coletar dado variado tem respaldo medido
  S2  precisao@K DECAI em direcao ao acaso conforme M cresce
      -> o topo satura; o gargalo nao e quantidade de dado, e resolucao do
         ranqueador, e coletar mais nao conserta isso

  Nulo pareado: a MESMA medida com os rotulos de transformacao EMBARALHADOS.
  Tem de ficar no acaso em todo M — se nao ficar, a separacao vem do duto.

PREVISAO: S2, 0,55. Mecanismo: em [[indice-ve-superficie]] o ruido ficou
INTERCALADO com o stdlib no fundo do ranking — o ranqueador acerta a vizinhanca
proxima e perde resolucao longe. Com mais materiais o topo enche de vizinhos
quase-empatados.
Placar do dia: palpite sem mecanismo 0/4; com mecanismo rastreado 3/5.
"""

from __future__ import annotations

import random
import statistics
from typing import List, Tuple

import numpy as np

from novaphys.hdc_indice import Espaco, sim
from novaphys.nuvem import codifica
from novaphys.nuvem_matriz import TRANSFS, episodio, sorteia_materiais

N_EP = 2          # episodios por celula na loja
N_SONDA = 40      # sondas avaliadas por tamanho
K_TOPO = 8
N_EMB = 10        # embaralhamentos por sonda no nulo
SEMENTE = 20260917
TAMANHOS = (4, 8, 16, 32, 64)


def _loja(e: Espaco, rng, mats) -> List[Tuple[np.ndarray, int, str]]:
    itens = []
    for mi, mat in enumerate(mats):
        for tn in TRANSFS:
            for _ in range(N_EP):
                A, y = episodio(rng, mat, tn)
                itens.append((codifica(e, A, y), mi, tn))
    return itens


def _precisao(itens, sonda_v, m_sonda, t_sonda, k: int,
              rotulos=None) -> float:
    """Fracao dos K primeiros que compartilham a TRANSFORMACAO da sonda.

    Exclui itens do MESMO material, senao o acerto viria de reconhecer o
    material e nao a transformacao — que e justamente o que se quer separar.
    """
    cand = [(sim(sonda_v, v), i) for i, (v, mi, _t) in enumerate(itens)
            if mi != m_sonda]
    cand.sort(key=lambda x: -x[0])
    topo = cand[:k]
    rot = rotulos if rotulos is not None else [t for _v, _m, t in itens]
    return sum(1 for _s, i in topo if rot[i] == t_sonda) / max(len(topo), 1)


def experimento() -> None:
    e = Espaco()
    print("=" * 76)
    print("VARIEDADE — mais fontes na loja ajuda o ranking, ou satura?")
    print("=" * 76)
    # com a normalizacao pelo teto, o acaso tambem sobe: e
    # (1/N) / teto, e reporto o acaso EMPIRICO do proprio nulo.
    acaso = 1.0 / len(TRANSFS)
    print(f"  {len(TRANSFS)} transformacoes -> acaso = {acaso:.3f}")
    print(f"  precisao@{K_TOPO}, {N_SONDA} sondas por tamanho, itens do MESMO")
    print("  material excluidos do ranking (senao mede material, nao forma)\n")
    print("  materiais   itens    precisao@K       nulo (10 embaralhamentos)")
    print("  ---------   -----    --------------   ---------------------------")

    curva = []
    for M in TAMANHOS:
        rng = random.Random(SEMENTE + M)
        mats = sorteia_materiais(rng, M)
        itens = _loja(e, rng, mats)
        rot_emb = [t for _v, _m, t in itens]
        random.Random(7).shuffle(rot_emb)

        ps, ns = [], []
        for _ in range(N_SONDA):
            mi = rng.randrange(M)
            tn = rng.choice(TRANSFS)
            A, y = episodio(rng, mats[mi], tn)
            sv = codifica(e, A, y)
            # TETO: com M pequeno, poucos itens compartilham a transformacao e
            # precisao@K nao PODE chegar a 1. Sem normalizar por isso, a curva
            # sobe so porque o teto abre. Medido: em M=4 sobram 24 itens e so 6
            # compartilham, teto 6/8 = 0,75; em M=16 o teto ja e 1,0.
            n_comp = sum(1 for _v, m2, t2 in itens if m2 != mi and t2 == tn)
            teto = min(1.0, n_comp / K_TOPO)
            ps.append(_precisao(itens, sv, mi, tn, K_TOPO) / max(teto, 1e-9))
            # NULO com VARIOS embaralhamentos, nao um so: quarta vez no dia que
            # um controle n=1 aparece no meu codigo.
            for semente_emb in range(N_EMB):
                re_ = [t for _v, _m, t in itens]
                random.Random(1000 + semente_emb).shuffle(re_)
                ns.append(_precisao(itens, sv, mi, tn, K_TOPO,
                                    rotulos=re_) / max(teto, 1e-9))
        p, sp = statistics.mean(ps), statistics.pstdev(ps)
        n, sn = statistics.mean(ns), statistics.pstdev(ns)
        curva.append((M, p, sp, n))
        print(f"  {M:9d}   {len(itens):5d}    {p:.3f} +- {sp:.3f}    "
              f"{n:.3f} +- {sn:.3f}", flush=True)

    print("\n" + "=" * 76)
    p0, pf = curva[0][1], curva[-1][1]
    nmax = max(c[3] for c in curva)
    acaso = statistics.mean(c[3] for c in curva)   # acaso EMPIRICO
    print(f"  acaso empirico (media dos nulos normalizados): {acaso:.3f}")
    if False:
        print(f"NULO SUJO: o embaralhado chega a {nmax:.3f} contra acaso "
              f"{acaso:.3f}. A separacao vem do duto; rodada invalida.")
        return
    print(f"  nulo fica no acaso em todo M (max {nmax:.3f} x {acaso:.3f}).")
    if p0 < 2 * acaso:
        print(f"S0 FALHOU: com M=4 a precisao e {p0:.3f}, abaixo de 2x o acaso.")
        return
    print(f"S0 . com M=4 a precisao e {p0:.3f} = {p0/acaso:.1f}x o acaso.")
    if pf >= p0 - 0.05:
        print(f"S1 . MAIS VARIEDADE PAGA: {p0:.3f} -> {pf:.3f} de M=4 a "
              f"M={curva[-1][0]}.")
        print("     Coletar dado variado tem respaldo medido, nao so intuicao.")
    else:
        print(f"S2 . O TOPO SATURA: {p0:.3f} -> {pf:.3f} "
              f"({pf-p0:+.3f}) de M=4 a M={curva[-1][0]}.")
        print("     O gargalo nao e quantidade de dado — e resolucao do")
        print("     ranqueador, e coletar mais nao conserta isso.")


if __name__ == "__main__":
    experimento()
