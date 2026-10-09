"""
novaphys.invencao — expandir o espaço, não escolher dentro dele
================================================================

O elo que faltava, formulado com precisão:

    seleção   H_novo ∈ H_substrato      medido em descoberta.py (I = 1,585 bits)
    INVENÇÃO  H_novo ∉ H_substrato      aqui

`mutacao.py` tropeçou porque uma primitiva nova só pode ser COMPOSIÇÃO dos
átomos existentes — e composição nunca sai do span do substrato. Sob aquela
definição, invenção é impossível por construção, não difícil.

A saída é outra operação, e ela é geral:

    GENERALIZAR UM ÁTOMO — trocar um encaixe de constante por um de variável

    substrato:  MENOR(VAR, NUM)     só cortes alinhados aos eixos
    inventado:  MENOR(VAR, VAR)     comparação entre duas grandezas

`MENOR(VAR,VAR)` não é composição de `MENOR(VAR,NUM)` com nada. É um átomo
NOVO, obtido levantando uma constante a parâmetro. Nenhuma árvore construída
com o substrato original o contém, em profundidade nenhuma. Isso é expansão do
espaço de hipóteses, e é verificável: basta enumerar.

A PRÉ-CONDIÇÃO, QUE VEM ANTES DO MECANISMO
--------------------------------------------
`mutacao.py` foi montada três vezes e as três não criavam a pressão que
prometiam — a estrutura profunda era mais CARA que a rasa em 2 de 3 sementes.
Só descobri isso depois de quatro correções, cada uma feita já sabendo que
direção fazia o número subir.

Então aqui a ordem é invertida e a regra é dura:

    `precondicao()` roda PRIMEIRO e por força bruta. Ela compara o melhor
    programa possível SEM o átomo novo contra o melhor COM ele. Se o átomo
    novo não compensar em pelo menos 90% das sementes, o experimento NÃO É
    EXECUTADO — a bancada não mede o que promete, e rodar o mecanismo em cima
    dela só produziria um número.

O que a escadaria custa: aproximar `v0 < v1` com cortes alinhados exige uma
escada de degraus, e cada degrau é um nó. O erro cai devagar e o tamanho sobe
rápido. É essa assimetria que faz a invenção valer a pena — e é ela que a
pré-condição verifica em vez de supor.

O MECANISMO
------------
Sem lista de átomos candidatos. Uma operação só, guiada pelo que as soluções
já mostram:

    1. resolve tarefas com o substrato original
    2. olha as soluções que compartilham esqueleto e diferem só nas CONSTANTES
    3. se a constante de um encaixe covaria com o valor de outra variável, esse
       encaixe é candidato a virar variável
    4. propõe o átomo generalizado
    5. aceita só se reduzir descrição, reaparecer, e NÃO ser usado no controle

O passo 3 é o coração e não menciona domínio: é a observação de que um valor
que eu tratava como constante está, na verdade, seguindo outra coisa.

OS CONTROLES
-------------
    · tarefas alinhadas ao eixo    o átomo novo NÃO pode ser inventado nem usado
    · regra de capacidade          o inventado tem de estar FORA de G(substrato),
                                   senão não é invenção, é enumeração
"""

from __future__ import annotations

import itertools
import math
import random
import statistics as st
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

Obs = Tuple[List[float], int]
N_VARS = 3


# ══════════════════════════════════════════════ programas
@dataclass(frozen=True)
class No:
    op: str
    args: Tuple[Any, ...] = ()

    def aval(self, v: Sequence[float]) -> Any:
        o = self.op
        if o == 'VAR':
            return v[self.args[0]]
        if o == 'NUM':
            return self.args[0]
        if o == 'SINAL':
            return self.args[0]
        if o == 'MENOR':
            return self.args[0].aval(v) < self.args[1].aval(v)
        if o == 'SE':
            return (self.args[1] if self.args[0].aval(v) else self.args[2]).aval(v)
        raise ValueError(o)

    def tamanho(self) -> int:
        return 1 + sum(a.tamanho() for a in self.args if isinstance(a, No))

    def esqueleto(self) -> str:
        o = self.op
        if o == 'VAR':
            return 'v'
        if o == 'NUM':
            return 'k'
        if o == 'SINAL':
            return 's'
        return f"{o}({','.join(a.esqueleto() for a in self.args if isinstance(a, No))})"

    def usa_var_var(self) -> bool:
        if self.op == 'MENOR' and all(isinstance(a, No) and a.op == 'VAR'
                                      for a in self.args):
            return True
        return any(a.usa_var_var() for a in self.args if isinstance(a, No))

    def __str__(self) -> str:
        o = self.op
        if o == 'VAR':
            return f"v{self.args[0]}"
        if o == 'NUM':
            return f"{self.args[0]:.2f}"
        if o == 'SINAL':
            return '+' if self.args[0] > 0 else '−'
        if o == 'MENOR':
            return f"{self.args[0]}<{self.args[1]}"
        return f"se({self.args[0]}){{{self.args[1]}}}senão{{{self.args[2]}}}"


MAIS, MEN = No('SINAL', (1,)), No('SINAL', (-1,))
LAM = 1.0


# ══════════════════════════════════════════════ tarefas
def tarefa(tipo: str, rng: random.Random, n: int = 80,
           par: Tuple[int, int] = (0, 1)) -> List[Obs]:
    """
    'relacional'  o sinal depende de qual das duas variáveis é maior
    'alinhada'    o sinal depende de uma variável contra um limiar (controle)
    """
    i, j = par
    c = rng.uniform(0.3, 0.7)
    obs: List[Obs] = []
    for _ in range(n):
        v = [rng.random() for _ in range(N_VARS)]
        if tipo == 'relacional':
            s = 1 if v[i] < v[j] else -1
        elif tipo == 'alinhada':
            s = 1 if v[i] < c else -1
        else:
            raise ValueError(tipo)
        obs.append((v, s))
    return obs


# ══════════════════════════════════════════════ busca
def _testes(obs: Sequence[Obs], n_cortes: int, var_var: bool) -> List[No]:
    ct = []
    for i in range(N_VARS):
        vals = sorted(o[0][i] for o in obs)
        ct.append([vals[int((k + 1) * len(vals) / (n_cortes + 1))]
                   for k in range(n_cortes)])
    testes = [No('MENOR', (No('VAR', (i,)), No('NUM', (c,))))
              for i in range(N_VARS) for c in ct[i]]
    if var_var:
        testes += [No('MENOR', (No('VAR', (i,)), No('VAR', (j,))))
                   for i in range(N_VARS) for j in range(N_VARS) if i != j]
    return testes


def _escadas(testes: Sequence[No], profundidade: int) -> List[No]:
    """Árvores de decisão até `profundidade`, aninhando sempre pelo ramo falso."""
    progs: List[No] = [MAIS, MEN]
    nivel: List[No] = [MAIS, MEN]
    for _ in range(profundidade):
        novo: List[No] = []
        for t in testes:
            for folha in nivel:
                novo.append(No('SE', (t, MAIS, folha)))
                novo.append(No('SE', (t, MEN, folha)))
        progs.extend(novo)
        nivel = novo[:90]           # teto para a enumeração não explodir
    return progs


def melhor(obs: Sequence[Obs], var_var: bool, profundidade: int = 3,
           n_cortes: int = 4) -> Dict[str, Any]:
    testes = _testes(obs, n_cortes, var_var)
    bp, bc = None, math.inf
    for p in _escadas(testes, profundidade):
        e = sum(1 for (v, s) in obs if p.aval(v) != s)
        c = e + LAM * p.tamanho()
        if c < bc:
            bp, bc = p, c
    ac = sum(1 for (v, s) in obs if bp.aval(v) == s) / max(1, len(obs))
    return {'programa': bp, 'custo': bc, 'ajuste': ac, 'tamanho': bp.tamanho()}


# ══════════════════════════════════════════════ PRÉ-CONDIÇÃO
def precondicao(sementes: Sequence[int] = tuple(range(8)),
                verboso: bool = True) -> Dict[str, Any]:
    """
    O átomo novo COMPENSA? Força bruta, antes de qualquer mecanismo.

    Sem esta verificação, medir se o sistema inventa é medir se ele responde a
    uma pressão que talvez não exista. Foi exatamente o que arruinou
    `mutacao.py`: a estrutura profunda era mais cara que a rasa em 2 de 3
    sementes, e eu só descobri depois de quatro correções.
    """
    linhas = []
    for s in sementes:
        rng = random.Random(s * 977 + 13)
        obs = tarefa('relacional', rng)
        sem = melhor(obs, var_var=False)
        com = melhor(obs, var_var=True)
        linhas.append({'semente': s, 'custo_sem': sem['custo'], 'custo_com': com['custo'],
                       'ajuste_sem': sem['ajuste'], 'ajuste_com': com['ajuste'],
                       'tam_sem': sem['tamanho'], 'tam_com': com['tamanho'],
                       'compensa': com['custo'] < sem['custo'] - 1.0})
    taxa = sum(l['compensa'] for l in linhas) / len(linhas)
    if verboso:
        print("PRÉ-CONDIÇÃO — o átomo novo compensa? (força bruta, sem mecanismo)")
        print(f"{'semente':>8}{'custo sem':>11}{'custo com':>11}"
              f"{'ajuste sem':>12}{'ajuste com':>12}{'':>4}")
        for l in linhas:
            print(f"{l['semente']:>8}{l['custo_sem']:>11.1f}{l['custo_com']:>11.1f}"
                  f"{l['ajuste_sem']:>11.0%}{l['ajuste_com']:>12.0%}"
                  f"{'  ✓' if l['compensa'] else '  ✗':>4}")
        print(f"\ncompensa em {taxa:.0%} das sementes  (exigido ≥ 90%)")
    return {'taxa': taxa, 'linhas': linhas, 'aprovada': taxa >= 0.9}


# ══════════════════════════════════════════════ o mecanismo
def generalizar(solucoes: Sequence[No], obs_por_sol: Sequence[Sequence[Obs]],
                limiar_corr: float = 0.6) -> List[Tuple[int, int]]:
    """
    Onde uma CONSTANTE está, na verdade, seguindo outra variável.

    Para cada teste `v_i < k` que aparece nas soluções, junta os pares
    (k usado, valor típico de v_j nas observações daquela tarefa). Se k
    acompanha v_j de tarefa para tarefa, o encaixe de constante é candidato a
    virar encaixe de variável — isto é, `MENOR(VAR_i, VAR_j)`.

    Não há lista de átomos candidatos. A operação é uma só: levantar uma
    constante a parâmetro. O que decide QUAL levantar é a covariância medida,
    e ela não menciona domínio nenhum.
    """
    registros: Dict[int, List[Tuple[float, List[float]]]] = {}
    for sol, obs in zip(solucoes, obs_por_sol):
        pilha = [sol]
        while pilha:
            n = pilha.pop()
            if (n.op == 'MENOR' and n.args[0].op == 'VAR'
                    and n.args[1].op == 'NUM'):
                i = n.args[0].args[0]
                k = n.args[1].args[0]
                medias = [st.mean(o[0][j] for o in obs) for j in range(N_VARS)]
                registros.setdefault(i, []).append((k, medias))
            for a in n.args:
                if isinstance(a, No):
                    pilha.append(a)

    candidatos: List[Tuple[int, int]] = []
    for i, regs in registros.items():
        if len(regs) < 4:
            continue
        ks = [k for k, _ in regs]
        if st.pstdev(ks) < 1e-9:
            continue
        for j in range(N_VARS):
            if j == i:
                continue
            mj = [m[j] for _, m in regs]
            if st.pstdev(mj) < 1e-9:
                continue
            try:
                r = st.correlation(ks, mj)
            except Exception:
                continue
            if abs(r) >= limiar_corr:
                candidatos.append((i, j))
    return candidatos


def experimento(n_tarefas: int = 8, sementes: Sequence[int] = (0, 1, 2, 3, 4),
                n_teste: int = 6) -> Dict[str, Any]:
    res = {'inventou': 0, 'usou_relacional': 0, 'usou_alinhada': 0,
           'ganho': [], 'n': 0, 'fora_do_span': 0}
    for s in sementes:
        rng = random.Random(s * 331 + 5)
        sols, obss = [], []
        for k in range(n_tarefas):
            par = [(0, 1), (1, 2), (2, 0)][k % 3]
            ob = tarefa('relacional', rng, par=par)
            sols.append(melhor(ob, var_var=False)['programa'])
            obss.append(ob)
        cands = generalizar(sols, obss)
        res['n'] += 1
        if not cands:
            continue
        res['inventou'] += 1
        # o átomo proposto está FORA do span do substrato original?
        span = {p.esqueleto() for p in _escadas(_testes(obss[0], 4, False), 2)}
        novo = No('MENOR', (No('VAR', (cands[0][0],)), No('VAR', (cands[0][1],))))
        prog_novo = No('SE', (novo, MAIS, MEN))
        if prog_novo.esqueleto() not in span:
            res['fora_do_span'] += 1

        for _ in range(n_teste):
            ob = tarefa('relacional', rng)
            a = melhor(ob, var_var=False)
            b = melhor(ob, var_var=True)
            res['ganho'].append(a['custo'] - b['custo'])
            res['usou_relacional'] += int(b['programa'].usa_var_var())
            oc = tarefa('alinhada', rng)
            bc = melhor(oc, var_var=True)
            res['usou_alinhada'] += int(bc['programa'].usa_var_var())
    tot = max(1, res['inventou'] * n_teste)
    return {**res, 'uso_rel': res['usou_relacional'] / tot,
            'uso_ali': res['usou_alinhada'] / tot,
            'ganho_medio': st.mean(res['ganho']) if res['ganho'] else 0.0}


def relatorio() -> Dict[str, Any]:
    print("═" * 78)
    print("INVENÇÃO — H_novo ∉ H_substrato")
    print("substrato: SE, MENOR(VAR,NUM), VAR, NUM — só cortes alinhados aos eixos")
    print("átomo possível de inventar: MENOR(VAR,VAR), que NÃO é composição de nada")
    print("═" * 78)

    pc = precondicao()
    print("─" * 78)
    if not pc['aprovada']:
        print("PRÉ-CONDIÇÃO REPROVADA. A bancada não cria a pressão que promete:")
        print("o átomo novo não compensa em sementes suficientes. O mecanismo NÃO")
        print("será executado — rodá-lo aqui só produziria um número sobre uma")
        print("pressão que não existe, que foi exatamente o erro de `mutacao.py`.")
        return {'precondicao': pc, 'experimento': None}

    print("Pré-condição APROVADA: a invenção compensa. O mecanismo pode rodar.")
    print()
    r = experimento()
    print("MECANISMO — generalização de constante para variável")
    print(f"  inventou o átomo em ................. {r['inventou']}/{r['n']} rodadas")
    print(f"  o inventado está FORA do span ....... {r['fora_do_span']}/{max(1,r['inventou'])}")
    print(f"  usado em tarefas relacionais ........ {r['uso_rel']:.0%}")
    print(f"  usado no controle alinhado .......... {r['uso_ali']:.0%}   (ideal 0%)")
    print(f"  ganho médio de descrição ............ {r['ganho_medio']:+.1f}")
    print("─" * 78)
    if r['inventou'] == 0:
        print("Não inventou. A pressão existe (pré-condição aprovada) e o mecanismo")
        print("não a converteu em átomo novo — o resultado é sobre o MECANISMO, que")
        print("é o que se queria medir.")
    elif r['uso_ali'] > 0.2:
        print(f"CONTROLE REPROVADO: o átomo novo é usado em {r['uso_ali']:.0%} das tarefas")
        print("alinhadas, onde não ajuda. Isso é inventar por inventar.")
    elif r['uso_rel'] > 0.8 and r['ganho_medio'] > 0:
        print("ATRAVESSOU. O sistema levantou uma constante a parâmetro, produzindo um")
        print("átomo que NÃO está no span do substrato original — não é composição, é")
        print("expansão. Ele reduz a descrição, é usado onde serve e não é usado onde")
        print("não serve.")
        print()
        print("A ressalva que estava escrita antes de rodar: a OPERAÇÃO de generalizar")
        print("foi escolhida por mim. O que o sistema decidiu foi QUANDO aplicá-la e a")
        print("QUE encaixe — guiado pela covariância medida, não por uma lista.")
    else:
        print(f"Parcial: inventou, mas uso em relacional {r['uso_rel']:.0%} e ganho")
        print(f"{r['ganho_medio']:+.1f}. Não fecha o critério.")
    return {'precondicao': pc, 'experimento': r}


if __name__ == "__main__":
    relatorio()
