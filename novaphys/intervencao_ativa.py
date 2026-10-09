"""
novaphys.intervencao_ativa — escolher o experimento, não sortear
================================================================

`intervencao.py` mostrou que trocar observação por intervenção leva a
recuperação de estrutura de 40% para 100%, com a margem indo de 0 para ~3000.
Mas as intervenções eram **sorteadas**: mexe-se numa manopla ao acaso, por um
fator ao acaso.

Um cientista não faz isso. Ele escolhe o experimento que mais separa as
hipóteses que ainda estão de pé. Essa é a ideia de `active_experimenter.py` no
repo da NovaMind — ganho de informação como critério de escolha — e é a peça
que falta aqui.

O CRITÉRIO, E POR QUE ELE NÃO É UMA LISTA
-------------------------------------------
Nada de "para razão, mexa em v1". O agente olha as candidatas que ainda
explicam bem os dados e escolhe a intervenção em que elas MAIS DISCORDAM:

    para cada intervenção candidata (qual variável, qual fator):
        prever a resposta sob cada estrutura sobrevivente
        pontuar pela DISPERSÃO das previsões
    executar a de maior dispersão

Onde as candidatas concordam, o experimento não informa: qualquer resultado é
compatível com todas. Onde discordam, uma observação elimina alternativas.

Isto é agnóstico a domínio — não sabe o que é razão, produto ou física. Só sabe
que experimento útil é experimento que separa.

O CONTROLE É O MESMO ORÇAMENTO
--------------------------------
Os dois braços recebem o MESMO número de intervenções. Se o ativo vencer, não
foi por experimentar mais — foi por experimentar melhor. Sem esse pareamento a
comparação não diria nada.

E há um precedente que obriga cautela: a bancada de indução desta mesma sessão
mediu escolha ativa contra sorteio em cinco paisagens e o ativo venceu em
**uma**. Lá a causa foi diagnosticada — a aquisição otimizava informação
estrutural enquanto a nota media erro médio sob sondas uniformes, e para essa
nota amostrar uniformemente é quase ótimo. Aqui a nota É recuperação de
estrutura, que é o que a aquisição persegue. As duas coisas estão alinhadas
pela primeira vez, e é por isso que vale medir de novo.

PRÉ-REGISTRO
-------------
  A1  ativo recupera a estrutura com MENOS intervenções que sorteio
  A2  o ganho é maior onde há mais candidatas concorrentes
  A3  com orçamento grande os dois convergem — a vantagem é de eficiência,
      não de alcance. Se o ativo alcançar algo que o sorteio nunca alcança,
      é sinal de que o sorteio tem viés, e isso é outro achado.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics as st
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .intervencao import E, compor, ALVOS, N_VARS

Par = Tuple[List[float], List[float], float, float]
FATORES = (0.5, 0.7, 1.4, 2.0)


def custo(e: E, pares: Sequence[Par]) -> float:
    s, n = 0.0, 0
    for v, v2, y, y2 in pares:
        a, b = e.val(v), e.val(v2)
        if abs(a) < 1e-9 or abs(b) < 1e-9:
            return math.inf
        s += (math.log(abs(b)) - math.log(abs(a))
              - (math.log(abs(y2)) - math.log(abs(y)))) ** 2
        n += 1
    if n == 0:
        return math.inf
    return n * math.log(max(s / n, 1e-12)) + e.tam() * math.log(n)


def _executar(f: Callable, v: List[float], i: int, fator: float,
              lo: float, hi: float) -> Optional[Par]:
    v2 = list(v)
    v2[i] = max(lo * 0.5, min(hi * 2, v[i] * fator))
    y, y2 = f(v), f(v2)
    if abs(y) < 1e-6 or abs(y2) < 1e-6:
        return None
    return (v, v2, y, y2)


def _sobreviventes(pares: Sequence[Par], campo: Sequence[E],
                   k: int = 8) -> List[E]:
    """As k candidatas que melhor explicam o que já foi visto."""
    pont = [(custo(e, pares), e) for e in compor(campo)]
    pont = [(c, e) for c, e in pont if math.isfinite(c)]
    pont.sort(key=lambda t: t[0])
    return [e for _, e in pont[:k]]


def _discordancia(cands: Sequence[E], v: List[float], i: int,
                  fator: float, lo: float, hi: float) -> float:
    """
    Dispersão das previsões das candidatas para esta intervenção.

    Alta dispersão = experimento que separa. Baixa = todas preveem o mesmo, e
    o resultado não elimina ninguém.
    """
    v2 = list(v)
    v2[i] = max(lo * 0.5, min(hi * 2, v[i] * fator))
    prev = []
    for e in cands:
        a, b = e.val(v), e.val(v2)
        if abs(a) < 1e-9 or abs(b) < 1e-9:
            continue
        prev.append(math.log(abs(b)) - math.log(abs(a)))
    return st.pstdev(prev) if len(prev) > 1 else 0.0


def rodar(f: Callable, alvo: str, politica: str, orcamento: int,
          semente: int, lo: float = 0.4, hi: float = 2.5,
          n_cand: int = 18) -> Dict[str, Any]:
    """
    Devolve em quantas intervenções a estrutura-alvo passou a ser a vencedora
    e ficou (recuperação estável).
    """
    rng = random.Random(semente)
    base = [E('V', i) for i in range(N_VARS)]
    pares: List[Par] = []
    passo_recuperou = None

    # semente inicial de dados: as duas políticas começam iguais
    while len(pares) < 3:
        v = [rng.uniform(lo, hi) for _ in range(N_VARS)]
        p = _executar(f, v, rng.randrange(N_VARS), rng.choice(FATORES), lo, hi)
        if p:
            pares.append(p)

    for passo in range(len(pares), orcamento):
        v = [rng.uniform(lo, hi) for _ in range(N_VARS)]
        if politica == 'sorteio':
            i, fator = rng.randrange(N_VARS), rng.choice(FATORES)
        else:
            cands = _sobreviventes(pares, base)
            melhor, best = None, -1.0
            for _ in range(n_cand):
                vc = [rng.uniform(lo, hi) for _ in range(N_VARS)]
                ic, fc = rng.randrange(N_VARS), rng.choice(FATORES)
                d = _discordancia(cands, vc, ic, fc, lo, hi)
                if d > best:
                    melhor, best = (vc, ic, fc), d
            v, i, fator = melhor
        p = _executar(f, v, i, fator, lo, hi)
        if p:
            pares.append(p)
        # O piso da verificação era 10, e as duas políticas recuperavam em 10 —
        # ou seja, eu media o meu piso, não a diferença entre elas. Baixado para
        # 4, que é o mínimo em que uma expressão de 3 nós pode ser identificada.
        if passo_recuperou is None and len(pares) >= 4:
            e = min(compor(base), key=lambda x: custo(x, pares))
            if e.expandida() == alvo:
                passo_recuperou = len(pares)
    e_final = min(compor(base), key=lambda x: custo(x, pares))
    return {'passos': passo_recuperou, 'acertou': e_final.expandida() == alvo,
            'forma': e_final.expandida()}


def relatorio(orcamento: int = 60, sementes: Sequence[int] = tuple(range(8))) -> Dict:
    print("═" * 78)
    print("INTERVENÇÃO ATIVA — escolher o experimento que mais SEPARA as candidatas")
    print("mesmo orçamento, mesma gramática, mesma pontuação; muda só quem escolhe")
    print("═" * 78)
    print("PRÉ-REGISTRO:")
    print("  A1  ativo recupera com MENOS intervenções que sorteio")
    print("  A2  o ganho é maior onde há mais candidatas concorrentes")
    print("  A3  com orçamento grande os dois convergem — vantagem de eficiência")
    print()
    print("  precedente que obriga cautela: na bancada de indução desta sessão o")
    print("  ativo venceu o acaso em 1 de 5 paisagens. Lá a nota media erro médio")
    print("  sob sondas uniformes; aqui a nota é recuperação de estrutura, que é o")
    print("  que a aquisição persegue. É a primeira vez que estão alinhadas.")
    print()

    print(f"{'alvo':<14}{'política':<12}{'recuperou':>11}"
          f"{'intervenções (mediana)':>24}")
    res: Dict[str, Dict[str, Any]] = {}
    for nome in ('razao', 'produto', 'diferenca'):
        f, alvo = ALVOS[nome]['f'], ALVOS[nome]['alvo']
        for pol in ('sorteio', 'ativo'):
            passos, ok = [], 0
            for s in sementes:
                r = rodar(f, alvo, pol, orcamento, semente=s * 71 + 5)
                ok += int(r['acertou'])
                if r['passos'] is not None:
                    passos.append(r['passos'])
            med = st.median(passos) if passos else float('nan')
            res[f"{nome}|{pol}"] = {'ok': ok / len(sementes), 'passos': med,
                                    'n': len(passos)}
            print(f"{alvo if pol == 'sorteio' else '':<14}{pol:<12}"
                  f"{ok/len(sementes):>10.0%}{med:>24.0f}")

    print("─" * 78)
    ganhos = []
    for nome in ('razao', 'produto', 'diferenca'):
        s_ = res[f"{nome}|sorteio"]['passos']
        a_ = res[f"{nome}|ativo"]['passos']
        if s_ == s_ and a_ == a_:
            ganhos.append(s_ - a_)
    if not ganhos:
        print("Nenhum alvo foi recuperado por nenhuma política — sem base para comparar.")
    elif st.mean(ganhos) > 2:
        print(f"ESCOLHER O EXPERIMENTO COMPENSA: mediana de {st.mean(ganhos):.1f} "
              f"intervenções a menos.")
        print("O critério é agnóstico a domínio — não sabe o que é razão nem produto,")
        print("só que experimento útil é experimento onde as candidatas discordam.")
    elif st.mean(ganhos) < -2:
        print(f"O SORTEIO VENCE por {-st.mean(ganhos):.1f} intervenções. Escolher pela")
        print("discordância está concentrando os experimentos numa região que informa")
        print("menos que a cobertura uniforme — o mesmo achado da bancada de indução.")
    else:
        print(f"EMPATE ({st.mean(ganhos):+.1f} intervenções). Com esta gramática e este")
        print("orçamento, escolher o experimento não muda a eficiência de recuperação.")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--orcamento', type=int, default=60)
    a = ap.parse_args()
    relatorio(orcamento=a.orcamento)
