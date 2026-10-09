"""
novaphys.nuvem_matriz — a nuvem estilhacando, com MAIS materiais e transformacoes
==================================================================================

`nuvem.py` fechou o caso minimo: 2 materiais x 2 transformacoes, composto 81,2%
contra transformacao-errada em 50,3% (ver [[nuvem-estilhaca]]). O limite que eu
mesmo registrei era o tamanho: duas celulas de cada lado nao distinguem
"compoe" de "so tem duas opcoes e acertou a moeda".

Aqui a matriz e M x N com UMA celula retirada por vez, e a media roda sobre
TODAS as celulas retiradas.

MATERIAIS — gerados, nao escritos a mao
-----------------------------------------
Cada material e uma codificacao propria do MESMO latente em 3 canais, sorteando
tipo, coeficiente, sinal e deslocamento:

    linear      a*s + b          absoluto    a*|s| + b
    quadratico  a*s^2 + b        saturado    a*tanh(1,5s) + b
    ruido       so ruido (canal sem informacao)

Garante-se >= 2 canais com informacao para o latente ser recuperavel, mas o
SINAL e a ESCALA sao sorteados — que e o que quebrou o encoder ingenuo — e a
POSICAO do canal forte varia entre materiais, o que testa a convencao de fase.

TRANSFORMACOES — quatro, e nao sao ortogonais de proposito
------------------------------------------------------------
    regime      s > limiar            (nivel)
    deriva      s subindo             (primeira diferenca)
    extremo     |s| > limiar          (magnitude, ignora sinal)
    acelera     a diferenca crescendo (segunda diferenca)

`regime` e `extremo` compartilham dependencia de nivel; `deriva` e `acelera`
compartilham dependencia de mudanca. Se a composicao so funcionava por as duas
opcoes serem opostas, isto derruba.

A PONTE, igual a de `nuvem.py`
--------------------------------
Para a celula retirada (m*, n*): a REGRA vem de (m_origem, n*) e a FASE vem de
completar a analogia com uma transformacao-ponte n_b != n* que existe nos DOIS
materiais e esta ROTULADA na loja. Nada toca o alvo de teste.

PRE-REGISTRO
--------------
  P0  PRE-CONDICAO. Teto medio > 60%. Abaixo disso o recall nao expressa as
      regras e nada e interpretavel.
  P1  composto medio > controle medio + 10 pontos, sobre TODAS as celulas
      -> a composicao sobrevive ao aumento de materiais e transformacoes
  P2  composto medio <= controle + 10 -> nao sobrevive; o resultado 2x2 era
      artefato de so haver duas opcoes
  P3  relatorio por celula: QUAIS pares falham, e se a falha acompanha o tipo
      do canal dominante do material (o suspeito nomeado antes de rodar e o
      material cujo canal forte seja `quadratico` ou `absoluto`, porque os dois
      destroem o sinal do latente e a convencao de fase perde referencia)

PREVISAO: P1 passa, mas com queda em relacao ao 2x2, e as falhas concentradas
nos materiais de canal dominante nao-linear. 0,6.
Placar do dia: palpite sem mecanismo 0/4; com mecanismo rastreado 3/4 (a que
errou tinha o mecanismo certo e a magnitude errada, por eu nao ter feito a
conta). Esta e do segundo tipo, e eu fiz a conta: |s| e s^2 sao pares em s,
entao correlacionam ~0 com s e o PC1 perde a referencia de sinal.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Dict, List, Sequence, Tuple

import numpy as np

from novaphys.hdc_indice import Espaco, feixe, sim
from novaphys.nuvem import (_niv, _tabela, orienta_pc1, preve,
                            termos_do_instante)

M_MAT = 5
T = 150
N_LOJA = 10
N_TESTE = 10
SEMENTE = 20260917

TRANSFS = ("regime", "deriva", "extremo", "acelera")
TIPOS = ("linear", "linear", "quadratico", "absoluto", "saturado", "ruido")


def _latente(rng: random.Random, n: int) -> np.ndarray:
    s, out = 0.0, []
    for _ in range(n):
        s = 0.86 * s + rng.gauss(0, 0.45)
        out.append(s)
    return np.array(out)


def sorteia_materiais(rng: random.Random, m: int) -> List[List[dict]]:
    """Cada material: 3 canais, com >= 2 informativos. Sinal/escala sorteados."""
    mats = []
    for _ in range(m):
        canais = []
        infor = 0
        while len(canais) < 3:
            tipo = rng.choice(TIPOS)
            if tipo == "ruido" and infor < 2 and len(canais) >= 2:
                continue
            if tipo == "ruido" and len(canais) < 2:
                continue
            a = rng.choice([-1, 1]) * rng.uniform(0.4, 2.5)
            b = rng.uniform(-3.0, 3.0)
            canais.append({"tipo": tipo, "a": a, "b": b})
            if tipo != "ruido":
                infor += 1
        rng.shuffle(canais)
        mats.append(canais)
    return mats


def _canal(c: dict, s: np.ndarray, rng: random.Random) -> np.ndarray:
    ruido = np.array([rng.gauss(0, 0.05) for _ in range(len(s))])
    t = c["tipo"]
    if t == "linear":
        v = c["a"] * s
    elif t == "quadratico":
        v = c["a"] * s * s
    elif t == "absoluto":
        v = c["a"] * np.abs(s)
    elif t == "saturado":
        v = c["a"] * np.tanh(1.5 * s)
    else:
        return np.array([rng.gauss(0, 1.0) for _ in range(len(s))])
    return v + c["b"] + ruido


def _alvo(s: np.ndarray, transf: str, rng: random.Random) -> np.ndarray:
    lim = rng.uniform(-0.3, 0.3)
    y = np.ones(len(s), dtype=int)
    d = np.diff(s, prepend=s[0])
    dd = np.diff(d, prepend=d[0])
    if transf == "regime":
        y = np.where(s > lim, 1, -1)
    elif transf == "deriva":
        y = np.where(d > 0, 1, -1)
    elif transf == "extremo":
        y = np.where(np.abs(s) > abs(lim) + 0.5, 1, -1)
    elif transf == "acelera":
        y = np.where(dd > 0, 1, -1)
    else:
        y = np.array([rng.choice([1, -1]) for _ in range(len(s))])
    return y


def episodio(rng, material: List[dict], transf: str):
    s = _latente(rng, T)
    A = np.stack([_canal(c, s, rng) for c in material], axis=1)
    return A, _alvo(s, transf, rng)


def codifica(e: Espaco, A: np.ndarray, y: np.ndarray) -> np.ndarray:
    Zo, papel = orienta_pc1(A)
    termos: List[np.ndarray] = []
    for t in range(A.shape[0]):
        termos += termos_do_instante(e, Zo, papel, t, int(y[t]))
    return feixe(termos, e.dim)


def regra(e, rng, material, transf, n=N_LOJA) -> np.ndarray:
    return feixe([codifica(e, *episodio(rng, material, transf))
                  for _ in range(n)], e.dim)


def calibra(e, R_ponte, rotulados) -> Tuple[float, float]:
    """Devolve (fase, confianca). Confianca = |acerto(+1) - acerto(-1)|."""
    ac = {}
    for flip in (1.0, -1.0):
        ac[flip] = statistics.mean(float(np.mean(preve(e, R_ponte, A, flip) == y))
                                   for A, y in rotulados)
    fase = 1.0 if ac[1.0] >= ac[-1.0] else -1.0
    return fase, abs(ac[1.0] - ac[-1.0])


def calibra_voto(e, rng, mats, m_src, mi, tn) -> Tuple[float, str]:
    """
    A FASE POR VOTO DE VARIAS PONTES.

    O defeito medido ([[nuvem-matriz-encoder]]): com UMA ponte, 2 celulas em 20
    sairam ~94% INVERTIDAS (mat1|acelera 5,6%, mat3|acelera 12,2%, com tetos
    93,5 e 88,2). A informacao atravessava e so a fase vinha trocada — 1 bit.

    Cada transformacao-ponte estima o MESMO bit de forma independente. Com taxa
    de acerto ~0,9 por ponte (2 erros em 20), a maioria de tres acerta ~0,97.
    Vota-se pesando pela confianca de cada ponte, para que uma ponte indecisa
    nao cancele uma decidida.

    Isto e uma operacao sobre TRANSFORMACOES — segunda ordem. O usuario pediu
    "a massinha que transforma ate as transformacoes"; a ponte ja era uma
    instancia disso, e votar entre pontes e a mesma ideia aplicada de novo.
    """
    voto = 0.0
    detalhe = []
    for n_b in [x for x in TRANSFS if x != tn]:
        R_b = regra(e, rng, mats[m_src], n_b)
        rot = [episodio(rng, mats[mi], n_b) for _ in range(N_LOJA)]
        f, conf = calibra(e, R_b, rot)
        voto += f * conf
        detalhe.append(f"{n_b[:3]}{'+' if f > 0 else '-'}{conf:.2f}")
    return (1.0 if voto >= 0 else -1.0), " ".join(detalhe)


def experimento() -> None:
    e = Espaco()
    rng = random.Random(SEMENTE)
    mats = sorteia_materiais(rng, M_MAT)

    print("=" * 84)
    print("NUVEM MATRIZ — a composicao sobrevive a mais materiais e transformacoes?")
    print("=" * 84)
    print(f"  {M_MAT} materiais x {len(TRANSFS)} transformacoes = "
          f"{M_MAT*len(TRANSFS)} celulas retiradas, uma por vez")
    for i, mm in enumerate(mats):
        d = " | ".join(f"{c['tipo'][:4]} a={c['a']:+.1f}" for c in mm)
        print(f"    material {i}: {d}")
    print("")
    print("  celula retirada          teto   composto   ctrl(transf errada)   nulo")
    print("  ----------------------   ----   --------   -------------------   ----")

    linhas = []
    for mi in range(M_MAT):
        for tn in TRANSFS:
            m_src = rng.choice([x for x in range(M_MAT) if x != mi])
            n_b = rng.choice([x for x in TRANSFS if x != tn])

            R_comp = regra(e, rng, mats[m_src], tn)
            R_teto = regra(e, rng, mats[mi], tn)
            R_ctrl = regra(e, rng, mats[m_src], n_b)
            R_nulo = regra(e, rng, mats[m_src], "ruido")

            fase, det = calibra_voto(e, rng, mats, m_src, mi, tn)

            testes = [episodio(rng, mats[mi], tn) for _ in range(N_TESTE)]
            def ac(R, f):
                return statistics.mean(float(np.mean(preve(e, R, A, f) == y))
                                       for A, y in testes)
            teto = ac(R_teto, 1.0)
            comp = ac(R_comp, fase)
            ctrl = ac(R_ctrl, fase)
            nulo = ac(R_nulo, fase)
            dom = mats[mi][0]["tipo"]
            marca = "  <-" if comp < ctrl + 0.10 else ""
            print(f"  mat{mi}|{tn:8s} ({dom[:4]})   {100*teto:4.1f}   "
                  f"{100*comp:6.1f}     {100*ctrl:13.1f}     {100*nulo:4.1f}{marca}",
                  flush=True)
            linhas.append((mi, tn, dom, teto, comp, ctrl, nulo))

    mt = statistics.mean(l[3] for l in linhas)
    mc = statistics.mean(l[4] for l in linhas)
    mk = statistics.mean(l[5] for l in linhas)
    mn = statistics.mean(l[6] for l in linhas)
    print("\n" + "=" * 84)
    print(f"  MEDIA   teto {100*mt:.1f}   composto {100*mc:.1f}   "
          f"controle {100*mk:.1f}   nulo {100*mn:.1f}")
    falhas = [l for l in linhas if l[4] < l[5] + 0.10]
    print(f"  celulas em que o composto NAO bate o controle+10: "
          f"{len(falhas)}/{len(linhas)}")
    if falhas:
        from collections import Counter
        print(f"    por tipo do canal 0 do material: "
              f"{dict(Counter(l[2] for l in falhas))}")
        print(f"    por transformacao: {dict(Counter(l[1] for l in falhas))}")
    print("")
    if mt <= 0.60:
        print(f"P0 FALHOU: teto medio {100*mt:.1f}%. O recall nao expressa as")
        print("   regras nesta escala; nada abaixo e interpretavel.")
        return
    print(f"P0 . teto medio {100*mt:.1f}% > 60%.")
    if mc > mk + 0.10:
        print(f"P1 . A COMPOSICAO SOBREVIVE: {100*mc:.1f}% contra "
              f"{100*mk:.1f}% do controle,")
        print(f"     sobre {len(linhas)} celulas retiradas, com nulo "
              f"{100*mn:.1f}%.")
    else:
        print(f"P2 . NAO SOBREVIVE: composto {100*mc:.1f}% contra controle "
              f"{100*mk:.1f}%.")
        print("     O 2x2 era artefato de so haver duas opcoes.")


if __name__ == "__main__":
    experimento()
