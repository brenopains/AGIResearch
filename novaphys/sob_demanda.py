"""
novaphys.sob_demanda — extrair na hora da tarefa, nao na hora de guardar
==========================================================================

A CORRECAO DO USUARIO
-----------------------
Eu tinha construido um PORTEIRO (`digestor.py`): decide o que entra, descarta o
resto, funde tudo num corpus. Falhou — ver [[digestor-sem-resolucao]].

O usuario corrigiu: a digestora nao e porteiro, e ATIVADOR. Nada e descartado.
Uma copia vai para o indice (HDC, multimodal, enderecavel) e fica disponivel
sob demanda; a extracao acontece NA HORA DA TAREFA. E entao "a digestora
consegue extrair, sob demanda, algo que antes havia achado inutil" — porque
utilidade e relativa a TAREFA, e no momento de guardar nao ha tarefa.

Duas medicoes minhas, que eu tinha e nao conectei, sustentam isso:

  1. o medidor de afinidade falhou como JUIZ ABSOLUTO e acertou a ORDEM
     (fala 0,065 > codigo 0,049 > ruido 0,046). Porteiro precisa de juiz;
     recuperacao so precisa de ordenador. O desenho dele pede a capacidade
     que o medidor TEM.
  2. o envenenamento (-0,127, lexico colapsa ao alfabeto) acontece quando tudo
     e FUNDIDO num modelo so. Aqui nada funde: recupera-se um subconjunto por
     tarefa. O resultado nao contraria o desenho dele — explica por que corpus
     unico e o recipiente errado.

E [[memoria-entrincheira]] ja mediu o nucleo da ideia: reanalisar o bruto
guardado quando o conceito chega recupera 50/50 das falhas antigas (90% x 30%),
e separar os ingredientes da +15 por juntar contra +36 por reanalisar SEM as
leis velhas.

O DESENHO — UMA LOJA SO, DUAS TAREFAS
---------------------------------------
Loja: 16 blocos, 4 de cada fonte (fala pt, codigo pt, codigo en, ruido).
Nenhum e descartado. Cada tarefa traz uma SONDA (amostra pequena do alvo,
disjunta do held-out) — e o que existe de verdade quando uma tarefa chega.

  afinidade(bloco | sonda) = quanto um modelo aprendido DA SONDA comprime o
  bloco. Ordena a loja para AQUELA tarefa. Nao ha limiar, nao ha descarte.

  T1  segmentar FALA      (held-out de fala)
  T2  segmentar CODIGO    (held-out de codigo em ingles)

  A  fundido     sonda + TODOS os blocos          (o que eu fiz antes)
  B  sob demanda sonda + os k melhores pela sonda
  C  placebo     sonda + k blocos sorteados, MEDIA de 12 sorteios + desvio
  D  oraculo     sonda + os k blocos da fonte certa

PRE-REGISTRO
--------------
  H0  PRE-CONDICAO. D > A nas DUAS tarefas. Se a loja nao tem sinal
      recuperavel, nada a medir.
  H1  B > media(C) + 1 desvio nas DUAS tarefas -> recuperar por tarefa paga
  H2  o CONJUNTO recuperado MUDA entre T1 e T2, e o stdlib que T1 ranqueia no
      fundo sobe ao topo em T2 -> "inutil vira nutriente" demonstrado, que e a
      afirmacao do usuario, nao uma proxy dela
  H3  B dentro do ruido de C -> a ordenacao nao informa e a loja precisa de
      indice melhor que compressao

CONTRA O DEFEITO DA RODADA ANTERIOR: la o placebo era n=1 e a variancia entre
blocos (0,389 x 0,464 com composicao quase igual) engolia o efeito. Aqui C e
media de 12 sorteios com desvio reportado, e o veredito usa o desvio.

PREVISAO: H2 passa (e quase mecanico — a sonda de codigo vai ranquear codigo no
topo). H1 e a duvida real: recuperar os 6 melhores pode nao bater sortear 6,
porque a variancia entre blocos e grande. 0,5 em H1, 0,8 em H2.
Placar do dia: 4 previsoes, 4 erradas.
"""

from __future__ import annotations

import os
import random
import statistics
import time
from collections import Counter
from typing import Dict, List, Tuple

from novaphys.digestor import _tokens_py, _heldout, aprende, STD
from novaphys.lingua import _carrega, f1, _viterbi

SEMENTE = 20260917
K = 6
N_SORTEIOS = 12


def afinidade_sonda(bloco: str, prob_sonda: Dict[str, float]) -> float:
    """Quanto o modelo da SONDA comprime este bloco. Ordena, nao julga."""
    ped = _viterbi(bloco, set(prob_sonda), prob_sonda)
    return sum(len(t) for t in ped) / max(len(ped), 1)


def _avalia(corpus: str, s_ho: str, fr_ho) -> Tuple[float, int]:
    prob = aprende(corpus)
    F, _P, _R = f1(_viterbi(s_ho, set(prob), prob), fr_ho)
    return F, len(prob)


def _posicao(notas, rotulo: str) -> int:
    """Melhor posicao (1-indexada) em que `rotulo` aparece no ranking."""
    for i, (_n, r, _b) in enumerate(notas):
        if r == rotulo:
            return i + 1
    return len(notas) + 1


def tarefa(nome, sonda_t, ho_t, blocos, fonte_certa, rng) -> Dict:
    sonda = "".join(sonda_t)
    s_ho, fr_ho = _heldout(ho_t)
    prob_s = aprende(sonda)

    notas = [(afinidade_sonda(b, prob_s), rot, b) for rot, b in blocos]
    notas.sort(key=lambda x: -x[0])
    topo = notas[:K]

    print("\n  -- " + nome + " " + "-" * max(4, 58 - len(nome)))
    print("     sonda de " + str(len(sonda_t)) + " tokens; loja ordenada por ela:")
    for i, (n, rot, _b) in enumerate(notas):
        marca = "  <- recuperado" if i < K else ""
        print("       {0:2d}. {1:10s} {2:.3f}{3}".format(i + 1, rot, n, marca))

    corpA = sonda + "".join(b for _r, b in blocos)
    corpB = sonda + "".join(b for _n, _r, b in topo)
    certos = "".join(b for r, b in blocos if r == fonte_certa)
    corpD = sonda + certos[:max(1, len(corpB) - len(sonda))]

    fs = []
    for _ in range(N_SORTEIOS):
        pick = rng.sample(range(len(blocos)), K)
        corp = sonda + "".join(blocos[i][1] for i in sorted(pick))
        fs.append(_avalia(corp, s_ho, fr_ho)[0])
    mC, sC = statistics.mean(fs), statistics.pstdev(fs)

    FA, LA = _avalia(corpA, s_ho, fr_ho)
    FB, LB = _avalia(corpB, s_ho, fr_ho)
    FD, LD = _avalia(corpD, s_ho, fr_ho)
    print("")
    print("     A fundido      {0:.3f} (lex {1})".format(FA, LA))
    print("     B sob demanda  {0:.3f} (lex {1})".format(FB, LB))
    print("     C placebo      {0:.3f} +- {1:.3f}  ({2} sorteios)".format(
        mC, sC, N_SORTEIOS))
    print("     D oraculo      {0:.3f} (lex {1})".format(FD, LD))
    return {"nome": nome, "A": FA, "B": FB, "Cm": mC, "Cs": sC, "D": FD,
            "topo": [r for _n, r, _b in topo], "notas": notas}


def experimento() -> None:
    rng = random.Random(SEMENTE)
    pal = _carrega().split()
    ho_fala, resto = pal[-400:], pal[:-400]
    sonda_fala = resto[:300]
    fala_bloco = resto[300:]

    aqui = os.path.dirname(os.path.abspath(__file__))
    pt = _tokens_py(aqui, 1200)
    en_todo = _tokens_py(STD, 2400, recursivo=True)
    en = en_todo[:1200]
    ho_cod = en_todo[1200:1600]
    sonda_cod = en_todo[1600:1900]
    ruido = list("".join(pt[:1200]))
    rng.shuffle(ruido)

    blocos: List[Tuple[str, str]] = []
    for k in range(4):
        blocos.append(("fala", "".join(fala_bloco[k * 250:(k + 1) * 250])))
        blocos.append(("novaphys", "".join(pt[k * 250:(k + 1) * 250])))
        blocos.append(("stdlib", "".join(en[k * 250:(k + 1) * 250])))
        blocos.append(("ruido", "".join(ruido[k * 1100:(k + 1) * 1100])))
    blocos = [b for b in blocos if len(b[1]) > 50]
    rng.shuffle(blocos)

    print("=" * 78)
    print("SOB DEMANDA — extrair na hora da tarefa, nao na hora de guardar")
    print("=" * 78)
    print("  loja: {0} blocos, nada descartado {1}".format(
        len(blocos), dict(Counter(r for r, _ in blocos))))
    print("  a MESMA loja serve as duas tarefas; so a SONDA muda")

    t0 = time.time()
    r1 = tarefa("T1 segmentar FALA", sonda_fala, ho_fala, blocos, "fala", rng)
    r2 = tarefa("T2 segmentar CODIGO", sonda_cod, ho_cod, blocos, "stdlib", rng)

    print("\n" + "=" * 78)
    p1 = _posicao(r1["notas"], "stdlib")
    p2 = _posicao(r2["notas"], "stdlib")
    print("  stdlib: melhor posicao {0} em T1, {1} em T2".format(p1, p2))
    print("  recuperado T1 {0}".format(dict(Counter(r1["topo"]))))
    print("  recuperado T2 {0}".format(dict(Counter(r2["topo"]))))
    print("  ({0:.0f}s)\n".format(time.time() - t0))

    if not (r1["D"] > r1["A"] and r2["D"] > r2["A"]):
        print("H0 FALHOU: o oraculo nao bate o fundido. A loja nao tem sinal")
        print("   recuperavel nessa escala; nada a medir.")
        return
    if all(r["B"] > r["Cm"] + r["Cs"] for r in (r1, r2)):
        print("H1 . RECUPERAR POR TAREFA PAGA nas duas tarefas, acima de um")
        print("     desvio do sorteio. A mesma loja, sondas diferentes.")
    else:
        print("H3 . B cai dentro do ruido do sorteio em alguma tarefa:")
        print("     a ordenacao por compressao nao informa o suficiente.")
    if Counter(r1["topo"]) != Counter(r2["topo"]) and p2 < p1:
        print("H2 . INUTIL VIRA NUTRIENTE: o stdlib sobe de {0} para {1}".format(
            p1, p2))
        print("     no ranking quando a tarefa muda, com a loja INTACTA.")
    else:
        print("H2 FALHOU: o conjunto recuperado nao mudou com a tarefa.")


if __name__ == "__main__":
    experimento()
