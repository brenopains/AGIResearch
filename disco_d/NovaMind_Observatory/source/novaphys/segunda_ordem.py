"""
novaphys.segunda_ordem — operadores que recebem TRANSFORMACOES
================================================================

DE ONDE VEM
-------------
Do usuario: "a 'massinha' de modelar que transforma até as transformações, a
escolha da AGI para simulação mental? A meu ver é o que falta."

Checado por grep antes de escrever: o nucleo tem combinadores de PRIMEIRA ordem
(`ENTAO`, `ATE`, `VEZES:k` em `descricao.py`, usados em 42% das pilhas
vencedoras) e NENHUM operador que receba operador. A unica excecao era a ponte
de `nuvem.py` — `calibra` recebe duas transformacoes e delas computa um
mapeamento entre materiais. Instancia unica, escrita a mao.

O DEGRAU QUE ESTE MODULO ATACA
--------------------------------
[[nuvem-estilhaca]] aplicou uma transformacao GUARDADA a um material novo. Aqui
a transformacao em si NUNCA esteve na loja: ela e CONSTRUIDA compondo duas que
estavam.

  loja:   A|regime, A|deriva, B|regime, B|deriva     (as primitivas)
  tarefa: prever B | (regime E deriva)               (o composto, inedito)

O operador de segunda ordem `E(T1, T2)` recebe dois PREDITORES e devolve um
preditor novo. Nada e re-treinado; a composicao acontece no nivel da regra.

POR QUE ISSO E DIFERENTE DE SO COMBINAR SAIDAS
------------------------------------------------
Porque cada primitiva precisa primeiro ser TRANSPORTADA para o material B, e o
transporte usa a ponte (que ja e segunda ordem) com voto de varias pontes (que
consertou o bit de fase: `mat1|acelera` saiu de 5,6% para 92,1%). Entao a cadeia
inteira e de segunda ordem: transporta duas transformacoes, compoe as duas
transportadas, e o resultado nunca existiu em lugar nenhum.

PRE-REGISTRO
--------------
Duas composicoes, de propriedades opostas de propositio:
  E    (AND)  alvo +1 so quando as DUAS valem  -> classe rara, majoritaria alta
  XOU  (XOR)  alvo +1 quando UMA vale          -> classe balanceada

Bracos, no MESMO retido de material B:
  TETO        regra aprendida direto de B|composto (a tarefa nao pode ter)
  COMPOSTO    E(transporta(A|T1), transporta(A|T2))     <- o que se testa
  PRIM1/PRIM2 cada primitiva transportada, sozinha
  CONTROLE    E das primitivas ERRADAS (as outras duas transformacoes)
  NULO        E de regras com alvo SORTEADO
  MAIORIA     classe majoritaria do episodio (a linha de base que AND infla)

  X0  PRE-CONDICAO. teto > maioria + 5 pontos. Se nem a regra do proprio
      material bate chutar a classe comum, a tarefa nao e mensuravel.
  X1  composto > max(prim1, prim2, controle) + 5 E composto > maioria + 5
      -> o sistema CONSTRUIU uma transformacao que nunca teve
  X2  composto <= melhor primitiva -> a composicao nao acrescenta; o que
      parece composto e uma primitiva sozinha
  X3  nulo > maioria + 5 -> duto

PREVISAO: X1 para XOU, 0,6. Para E, 0,45 — mecanismo: AND torna a classe +1
rara (~25%), a maioria vira ~75%, e bater isso exige acertar justamente os
instantes raros, onde os dois preditores tem de estar certos ao mesmo tempo;
erro de qualquer um estraga. XOU e balanceado e mais justo com o mecanismo.
Placar do dia anterior: palpite sem mecanismo 0/4; com mecanismo 4/8 — e os
erros foram por nao perguntar qual mecanismo era o DOMINANTE.

LIMITE declarado: as primitivas sao DADAS por mim. O sistema compoe o que tem;
nao inventa a primitiva. Isso continua sendo o gargalo de importacao.
"""

from __future__ import annotations

import random
import statistics
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from novaphys.hdc_indice import Espaco, feixe
from novaphys.nuvem import preve
from novaphys.nuvem_matriz import (N_LOJA, TRANSFS, _alvo, _canal, _latente,
                                   calibra_voto, codifica, episodio, regra,
                                   sorteia_materiais)

M_MAT = 4
T = 150
N_TESTE = 12
SEMENTE = 20260918


# ────────────────────────────────── as composicoes
def alvo_composto(s: np.ndarray, t1: str, t2: str, modo: str,
                  rng: random.Random) -> np.ndarray:
    """O alvo do COMPOSTO, calculado do latente — o gabarito da tarefa."""
    r1 = _alvo(s, t1, random.Random(rng.randrange(1 << 30)))
    r2 = _alvo(s, t2, random.Random(rng.randrange(1 << 30)))
    if modo == "E":
        return np.where((r1 > 0) & (r2 > 0), 1, -1)
    return np.where((r1 > 0) ^ (r2 > 0), 1, -1)          # XOU


def episodio_composto(rng, material, t1: str, t2: str, modo: str):
    s = _latente(rng, T)
    A = np.stack([_canal(c, s, rng) for c in material], axis=1)
    return A, alvo_composto(s, t1, t2, modo, rng)


# ────────────────────────── o operador de SEGUNDA ORDEM
def compoe(p1: Callable, p2: Callable, modo: str) -> Callable:
    """
    Recebe DOIS PREDITORES e devolve um preditor NOVO.

    Este e o objeto que faltava: nada aqui e re-treinado, e o preditor que sai
    nunca existiu na loja. `p1` e `p2` ja vieram TRANSPORTADOS para o material
    de destino pela ponte com voto — entao a cadeia inteira e de segunda ordem.
    """
    def novo(A: np.ndarray) -> np.ndarray:
        a, b = p1(A), p2(A)
        if modo == "E":
            return np.where((a > 0) & (b > 0), 1, -1)
        return np.where((a > 0) ^ (b > 0), 1, -1)
    return novo


def transporta(e: Espaco, rng, mats, m_src: int, m_dst: int,
               tn: str) -> Callable:
    """
    Leva a transformacao `tn` do material de origem para o de destino.

    A fase vem de `calibra_voto` — varias pontes votando, que e o conserto
    medido do bit que fazia celulas sairem 94% invertidas.
    """
    R = regra(e, rng, mats[m_src], tn)
    fase, _det = calibra_voto(e, rng, mats, m_src, m_dst, tn)
    return lambda A: preve(e, R, A, fase)


def _ac(p: Callable, testes) -> float:
    """
    Acuracia BALANCEADA: media da revocacao por classe.

    CONSERTO. A 1a versao usava acuracia bruta — e eu mesmo escrevi no
    pre-registro que "AND torna a classe +1 rara (~25%) e a maioria vira ~75%",
    e entao medi com a metrica que premia exatamente chutar a classe comum. Com
    28/72, prever sempre -1 ja entrega 72% e o teto "falha" por artefato.
    Balanceada: chutar a classe comum vale 0,50 e so habilidade real sobe.
    """
    fs = []
    for A, y in testes:
        pr = p(A)
        rec = []
        for cls in (1, -1):
            m = (y == cls)
            if m.sum() > 0:
                rec.append(float(np.mean(pr[m] == cls)))
        if rec:
            fs.append(statistics.mean(rec))
    return statistics.mean(fs) if fs else 0.5


def experimento(modo: str = "XOU") -> Dict[str, float]:
    e = Espaco()
    rng = random.Random(SEMENTE)
    # RESTRICAO DECLARADA, e ela nao e escolha de resultado: a matriz mediu que
    # materiais de canal dominante QUADRATICO ou ABSOLUTO quebram o transporte,
    # porque s^2 e |s| sao PARES em s, correlacionam ~0 com o latente e o PC1
    # perde a referencia de sinal. Isso e limite do ENCODER, ja registrado em
    # [[indice-ve-superficie]], e e independente da pergunta de composicao.
    # Testar composicao sobre material onde o transporte nao funciona nao mede
    # composicao — mede o encoder de novo. Entao aqui so entram materiais cujo
    # canal dominante e linear ou saturado.
    mats = [m for m in sorteia_materiais(rng, 6 * M_MAT)
            if m[0]["tipo"] in ("linear", "saturado")][:M_MAT]
    print(f"  materiais aceitos: "
          f"{[m[0]['tipo'][:4] for m in mats]}")
    t1, t2 = "regime", "deriva"
    outras = [x for x in TRANSFS if x not in (t1, t2)]
    m_src, m_dst = 0, 1

    testes = [episodio_composto(rng, mats[m_dst], t1, t2, modo)
              for _ in range(N_TESTE)]
    # em acuracia BALANCEADA, chutar a classe comum vale exatamente 0,50
    maioria = 0.50
    desbal = statistics.mean(max(float(np.mean(y == 1)),
                                 float(np.mean(y == -1))) for _A, y in testes)

    print("=" * 78)
    print(f"SEGUNDA ORDEM — construir a transformacao '{t1} {modo} {t2}',")
    print("                que NUNCA esteve na loja")
    print("=" * 78)
    print(f"  loja: material {m_src} e {m_dst}, com as PRIMITIVAS "
          f"{t1}/{t2}/{'/'.join(outras)}")
    print(f"  tarefa: prever o COMPOSTO no material {m_dst}, "
          f"{N_TESTE} episodios novos\n")

    p1 = transporta(e, rng, mats, m_src, m_dst, t1)
    p2 = transporta(e, rng, mats, m_src, m_dst, t2)
    comp = compoe(p1, p2, modo)

    # XPRE: o TRANSPORTE funciona neste par de materiais?
    # Faltava isto na 1a versao, e foi por isso que o composto deu 54,3% sem eu
    # saber se a culpa era da composicao ou do transporte. Cada primitiva e
    # medida contra o ALVO DELA antes de qualquer composicao. E a regra que eu
    # ja tinha aprendido no PRISM e violei de novo: checar se o instrumento
    # recupera a linha de base conhecida ANTES de interpretar o resto.
    tp1 = [episodio(rng, mats[m_dst], t1) for _ in range(N_TESTE)]
    tp2 = [episodio(rng, mats[m_dst], t2) for _ in range(N_TESTE)]
    a1, a2 = _ac(p1, tp1), _ac(p2, tp2)
    mj1 = statistics.mean(max(float(np.mean(y == 1)), float(np.mean(y == -1)))
                          for _A, y in tp1)
    mj2 = statistics.mean(max(float(np.mean(y == 1)), float(np.mean(y == -1)))
                          for _A, y in tp2)
    print(f"  XPRE transporte: {t1} {100*a1:.1f}% (maioria {100*mj1:.1f}%), "
          f"{t2} {100*a2:.1f}% (maioria {100*mj2:.1f}%)")
    esperado = a1 * a2 + (1 - a1) * (1 - a2) if modo == "E" else         a1 * (1 - a2) + (1 - a1) * a2
    print(f"  se a composicao for exata, o composto deveria dar ~"
          f"{100*max(esperado, 1-esperado):.0f}%")
    if a1 < mj1 + 0.10 or a2 < mj2 + 0.10:
        print("  XPRE FALHOU: o transporte nao bate a maioria+10 nas primitivas.")
        print("  O par de materiais nao serve; a composicao nao e testavel aqui.")
        print("")

    c1 = transporta(e, rng, mats, m_src, m_dst, outras[0])
    c2 = transporta(e, rng, mats, m_src, m_dst, outras[1])
    ctrl = compoe(c1, c2, modo)

    Rn1 = regra(e, rng, mats[m_src], "ruido")
    Rn2 = regra(e, rng, mats[m_src], "ruido")
    nulo = compoe(lambda A: preve(e, Rn1, A, 1.0),
                  lambda A: preve(e, Rn2, A, 1.0), modo)

    R_teto = feixe([codifica(e, *episodio_composto(rng, mats[m_dst], t1, t2,
                                                   modo))
                    for _ in range(N_LOJA)], e.dim)
    teto = lambda A: preve(e, R_teto, A, 1.0)

    res = {
        "TETO (regra do proprio composto)": _ac(teto, testes),
        "COMPOSTO (2a ordem)": _ac(comp, testes),
        f"so {t1} transportado": _ac(p1, testes),
        f"so {t2} transportado": _ac(p2, testes),
        "CONTROLE (primitivas erradas)": _ac(ctrl, testes),
        "NULO (alvo sorteado)": _ac(nulo, testes),
    }
    print("  braco                              acerto")
    print("  --------------------------------   ------")
    for k, v in res.items():
        print(f"  {k:32s}   {100*v:5.1f}%")
    print(f"  {'chutar a classe comum':32s}   {100*maioria:5.1f}%"
          f"   (a classe comum ocupa {100*desbal:.0f}% dos instantes)")

    print("\n" + "=" * 78)
    tt, cc = res["TETO (regra do proprio composto)"], res["COMPOSTO (2a ordem)"]
    melhor_prim = max(res[f"so {t1} transportado"],
                      res[f"so {t2} transportado"],
                      res["CONTROLE (primitivas erradas)"])
    nn = res["NULO (alvo sorteado)"]
    if tt <= maioria + 0.05:
        print(f"X0 FALHOU: teto {100*tt:.1f}% nao bate maioria "
              f"{100*maioria:.1f}%+5. Tarefa nao mensuravel neste modo.")
        return res
    print(f"X0 . teto {100*tt:.1f}% > maioria {100*maioria:.1f}%+5.")
    if nn > maioria + 0.05:
        print(f"X3 . o NULO passa da maioria ({100*nn:.1f}%): e duto.")
        return res
    if cc > melhor_prim + 0.05 and cc > maioria + 0.05:
        print(f"X1 . O SISTEMA CONSTRUIU A TRANSFORMACAO: composto "
              f"{100*cc:.1f}% contra melhor primitiva {100*melhor_prim:.1f}% "
              f"e maioria {100*maioria:.1f}%.")
        print("     Um operador de SEGUNDA ORDEM produziu um preditor que")
        print("     nunca esteve na loja, a partir de dois que estavam.")
    else:
        print(f"X2 . composto {100*cc:.1f}% nao supera a melhor primitiva "
              f"{100*melhor_prim:.1f}% + 5.")
        print("     A composicao nao acrescenta acima do que uma peca ja dava.")
    return res


if __name__ == "__main__":
    for modo in ("XOU", "E"):
        experimento(modo)
        print("\n")
