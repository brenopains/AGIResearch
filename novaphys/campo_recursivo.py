"""
novaphys.campo_recursivo — cada descoberta vira material da seguinte
=====================================================================

A propriedade perseguida desde `crencas.py`, agora formulada de um jeito que dá
para medir sem ambiguidade:

    o espaço EFETIVO de representações que o sistema consegue usar cresce
    progressivamente com o que ele já descobriu?

Não é "achou uma estrutura". É cascata:

    átomos → E1 → E2 → E3 → ...

onde E2 só é alcançável porque E1 existe, e E3 só porque E2 existe. Se cada
descoberta vira ingrediente da próxima, o alcançável cresce em cadeia. Se não,
o que existe é busca mais funda com outro nome.

A MÉTRICA MUDA, E ESSA É A CORREÇÃO MAIS IMPORTANTE
-----------------------------------------------------
Os dois experimentos anteriores mediram ACERTO DE CLASSIFICAÇÃO e ficaram sem
margem: listas de decisão com cortes bem postos aproximam fronteiras
complicadas bem demais, então o teto ficava em 90% mesmo sem a estrutura certa,
e o efeito real cabia em três pontos percentuais.

Aqui a métrica é RECUPERAÇÃO DE ESTRUTURA: a expressão encontrada, expandida
até os átomos, é a forma-alvo? É binária, não tem teto artificial, e foi com
ela que `descoberta.py` mediu 2,455 de 2,585 bits com o efeito nítido. Eu a
tinha abandonado ao mudar de pergunta, sem notar.

O DESENHO — DOIS BRAÇOS, MESMO ORÇAMENTO
------------------------------------------
    CUMULATIVO   o campo cresce: o que se consolida no estágio k vira terminal
                 no estágio k+1
    PLANO        o campo nunca cresce; mesmo orçamento de composição

Três estágios, cada um exigindo a descoberta do anterior como ingrediente:

    E1  (v/v)                    alcançável a partir dos átomos
    E2  |(v/v)−v|                alcançável a partir de E1
    E3  (|(v/v)−v|·v)            alcançável a partir de E2

A expansão de composição é fixa e IDÊNTICA nos dois braços — binárias sobre o
campo, mais módulo sobre campo e binárias. O que muda é só o que está no campo.

PRÉ-CONDIÇÃO É PORTÃO
----------------------
Antes do mecanismo, força bruta verifica que E2 e E3 estão FORA do alcance do
braço plano. Se estiverem dentro, a comparação não mede nada e não se roda.

PRÉ-REGISTRO
-------------
  T1  braço PLANO recupera E1, e falha em E2 e E3
  T2  braço CUMULATIVO recupera os três, com o MESMO orçamento
  T3  no controle raso, nenhum braço usa terminal consolidado — se usar, a
      consolidação está inflando a busca em vez de ampliar o alcance
  T4  a consolidação tem de ser AUTOMÁTICA: entra o que recorre e comprime,
      sem eu escolher. Os rótulos são por índice, nunca semânticos.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics as st
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

Obs = Tuple[List[float], int]
N_VARS = 4
LAM = 0.6
EPS = 1e-6
OPS2 = ('+', '-', '*', '/')


# ══════════════════════════════════════════════ expressões
@dataclass(frozen=True)
class E:
    op: str                 # 'V' | 'C' | '+' '-' '*' '/' 'abs'
    a: Any = None
    b: Any = None
    rot: str = ''

    def val(self, v: Sequence[float]) -> float:
        o = self.op
        if o == 'V':
            return v[self.a]
        if o == 'C':
            return self.a.val(v)
        x = self.a.val(v)
        if o == 'abs':
            return abs(x)
        y = self.b.val(v)
        if o == '+':
            return x + y
        if o == '-':
            return x - y
        if o == '*':
            return x * y
        return x / y if abs(y) > EPS else math.copysign(1e6, x or 1.0)

    def tam(self) -> int:
        if self.op in ('V', 'C'):
            return 1
        if self.op == 'abs':
            return 1 + self.a.tam()
        return 1 + self.a.tam() + self.b.tam()

    def expandida(self) -> str:
        """Forma com as consolidadas ABERTAS — é o que se compara com o alvo."""
        if self.op == 'V':
            return 'v'
        if self.op == 'C':
            return self.a.expandida()
        if self.op == 'abs':
            return f"|{self.a.expandida()}|"
        return f"({self.a.expandida()}{self.op}{self.b.expandida()})"

    def usa_C(self) -> bool:
        if self.op == 'C':
            return True
        return any(x.usa_C() for x in (self.a, self.b) if isinstance(x, E))

    def __str__(self) -> str:
        if self.op == 'V':
            return f"v{self.a}"
        if self.op == 'C':
            return self.rot
        if self.op == 'abs':
            return f"|{self.a}|"
        return f"({self.a}{self.op}{self.b})"


@dataclass(frozen=True)
class Molde:
    """
    Uma consolidada com BURACOS no lugar das variáveis.

    A primeira versão guardava a estrutura com os índices congelados — `E1` era
    literalmente `v2/v3`, não "a razão entre duas grandezas". No estágio
    seguinte, cuja tarefa usa `v0/v1`, o terminal era inútil e a busca caía numa
    expressão qualquer. A cascata reprovou por isso, não por falta do mecanismo:
    consolidou `E2 = (v+(v/v))` em vez de `|((v/v)-v)|`.
    
    Abstrair as variáveis é o que "abstração" significa, e `abstracao.py` já
    fazia isso com `Primitiva.instanciar`. Perder essa peça aqui foi regressão.
    """
    corpo: E
    n_slots: int
    rot: str

    def instancias(self, n_vars: int) -> List[E]:
        import itertools

        def preencher(e: E, atrib: Sequence[int]) -> E:
            if e.op == 'V':
                return E('V', atrib[e.a])
            if e.op == 'C':
                return e
            if e.op == 'abs':
                return E('abs', preencher(e.a, atrib))
            return E(e.op, preencher(e.a, atrib), preencher(e.b, atrib))

        out = []
        for atrib in itertools.permutations(range(n_vars), self.n_slots):
            out.append(E('C', preencher(self.corpo, atrib), rot=self.rot))
        return out


def abstrair(e: E) -> Molde:
    """Troca cada variável concreta por um slot, na ordem em que aparece."""
    mapa: Dict[int, int] = {}

    def rec(x: E) -> E:
        if x.op == 'V':
            if x.a not in mapa:
                mapa[x.a] = len(mapa)
            return E('V', mapa[x.a])
        if x.op == 'C':
            return rec(x.a)
        if x.op == 'abs':
            return E('abs', rec(x.a))
        return E(x.op, rec(x.a), rec(x.b))

    corpo = rec(e)
    return Molde(corpo, len(mapa), '')


def compor(campo: Sequence[E]) -> List[E]:
    """
    UM passo de composição, idêntico nos dois braços: binárias sobre o campo,
    mais módulo sobre o campo e sobre as binárias recém-formadas.
    """
    bin_: List[E] = []
    for x in campo:
        for y in campo:
            if x is y:
                continue
            for o in OPS2:
                bin_.append(E(o, x, y))
    return list(campo) + bin_ + [E('abs', z) for z in list(campo) + bin_]


# ══════════════════════════════════════════════ programa
@dataclass(frozen=True)
class Prog:
    e: E
    k: float
    s: int

    def aval(self, v: Sequence[float]) -> int:
        return self.s if self.e.val(v) < self.k else -self.s

    def custo(self, obs: Sequence[Obs]) -> float:
        return (sum(1 for (v, y) in obs if self.aval(v) != y)
                + LAM * self.e.tam())


def resolver(obs: Sequence[Obs], campo: Sequence[E],
             n_cortes: int = 7) -> Prog:
    best, bc = None, math.inf
    for e in compor(campo):
        vals = sorted(e.val(v) for v, _ in obs)
        for i in range(n_cortes):
            k = vals[int((i + 1) * len(vals) / (n_cortes + 1))]
            for s in (1, -1):
                p = Prog(e, k, s)
                c = p.custo(obs)
                if c < bc:
                    best, bc = p, c
    return best


# ══════════════════════════════════════════════ estágios
ESTAGIOS = [
    {'nome': 'E1', 'alvo': '(v/v)',
     'f': lambda v, c: 1 if v[0] / v[1] < c else -1, 'c': (0.7, 1.5)},
    {'nome': 'E2', 'alvo': '|((v/v)-v)|',
     'f': lambda v, c: 1 if abs(v[0] / v[1] - v[2]) < c else -1, 'c': (0.25, 0.6)},
    {'nome': 'E3', 'alvo': '(|((v/v)-v)|*v)',
     'f': lambda v, c: 1 if abs(v[0] / v[1] - v[2]) * v[3] < c else -1,
     'c': (0.15, 0.45)},
]


def tarefa(est: Dict[str, Any], rng: random.Random, n: int = 260) -> List[Obs]:
    c = rng.uniform(*est['c'])
    obs = []
    for _ in range(n):
        v = [rng.uniform(0.25, 1.0) for _ in range(N_VARS)]
        obs.append((v, est['f'](v, c)))
    return obs


def tarefa_rasa(rng: random.Random, n: int = 260) -> List[Obs]:
    c = rng.uniform(0.3, 0.7)
    return [([rng.uniform(0.25, 1.0) for _ in range(N_VARS)], 0) for _ in range(0)] or \
           [(v, 1 if v[0] < c else -1)
            for v in ([rng.uniform(0.25, 1.0) for _ in range(N_VARS)]
                      for _ in range(n))]


# ══════════════════════════════════════════════ consolidação automática
def consolidar(descobertas: Sequence[E], ja: int,
               min_vezes: int = 2) -> Optional[Molde]:
    """
    Entra o que RECORRE e COMPRIME. Sem eu escolher, rótulo por índice.
    Chamar de "razão" recolocaria o cardápio pela porta dos fundos.
    """
    cont = Counter(e.expandida() for e in descobertas if e.op not in ('V', 'C'))
    for forma, n in cont.most_common():
        if n < min_vezes:
            break
        modelo = next(e for e in descobertas if e.expandida() == forma)
        if modelo.tam() <= 2:
            continue
        m = abstrair(modelo)
        return Molde(m.corpo, m.n_slots, f"E{ja + 1}")
    return None


# ══════════════════════════════════════════════ pré-condição
def discriminabilidade(est: Dict[str, Any], campo: Sequence[E],
                       sementes: Sequence[int], n: int = 260) -> Dict[str, Any]:
    """
    O alvo é CLARAMENTE o melhor entre os alcançáveis, ou há empate?

    Este teste faltava, e a falta derrubou duas rodadas. A pré-condição
    verificava ALCANÇABILIDADE — o alvo está no espaço? — e não
    DISCRIMINABILIDADE — o alvo vence com folga?

    O estrago concreto: `v0/v1 < c` é bem aproximado por `v0 − v1 < k` quando as
    variáveis vivem num intervalo estreito, porque ali a hipérbole é quase uma
    reta. As duas formas têm o mesmo tamanho, o MDL não as separa, e o erro
    decide por uma margem mínima. Resultado: o estágio 1 recuperava a razão em
    40% das sementes e consolidava `(v-v)`, e a cascata morria no primeiro elo.

    Uma tarefa em que duas estruturas diferentes custam quase o mesmo não mede
    estrutura — mede fronteira. Sem esta checagem, o portão aprovava assim mesmo.
    """
    margens = []
    for sem in sementes:
        rng = random.Random(sem * 577 + hash(est['nome']) % 67)
        obs = tarefa(est, rng, n=n)
        c_alvo, c_outro = math.inf, math.inf
        for e in compor(campo):
            vals = sorted(e.val(v) for v, _ in obs)
            melhor_c = math.inf
            for i in range(7):
                k = vals[int((i + 1) * len(vals) / 8)]
                for sg in (1, -1):
                    melhor_c = min(melhor_c, Prog(e, k, sg).custo(obs))
            if e.expandida() == est['alvo']:
                c_alvo = min(c_alvo, melhor_c)
            else:
                c_outro = min(c_outro, melhor_c)
        margens.append(c_outro - c_alvo)
    return {'margem': st.mean(margens), 'min': min(margens)}


def precondicao(sementes: Sequence[int] = (0, 1, 2), n: int = 260,
                margem_min: float = 4.0, verboso: bool = True) -> Dict[str, Any]:
    base = [E('V', i) for i in range(N_VARS)]
    formas = {e.expandida() for e in compor(base)}
    fora = {est['nome']: est['alvo'] not in formas for est in ESTAGIOS}
    if verboso:
        print("PRÉ-CONDIÇÃO (1/2) — ALCANCE: o braço PLANO chega ao alvo em UM passo?")
        for est in ESTAGIOS:
            d = est['alvo'] in formas
            print(f"  {est['nome']}  {est['alvo']:<20} "
                  f"{'DENTRO do alcance' if d else 'fora do alcance'}")
    alcance_ok = (not fora['E1']) and fora['E2'] and fora['E3']

    # discriminabilidade só faz sentido onde o alvo é alcançável
    disc = discriminabilidade(ESTAGIOS[0], base, sementes, n)
    if verboso:
        print()
        print("PRÉ-CONDIÇÃO (2/2) — DISCRIMINAÇÃO: o alvo vence com folga?")
        print(f"  E1  margem de custo sobre a melhor alternativa: "
              f"{disc['margem']:.1f}  (mínima {disc['min']:.1f}, exigido ≥ {margem_min})")
    disc_ok = disc['min'] >= margem_min
    return {'aprovada': alcance_ok and disc_ok, 'fora': fora, 'disc': disc}


# ══════════════════════════════════════════════ experimento
def rodar_braco(cumulativo: bool, sementes: Sequence[int],
                n: int = 260, verboso: bool = False) -> Dict[str, Any]:
    base = [E('V', i) for i in range(N_VARS)]
    campo = list(base)
    consolidadas: List[Molde] = []
    recuperou: Dict[str, float] = {}

    for est in ESTAGIOS:
        achou, descobertas = 0, []
        for s in sementes:
            rng = random.Random(s * 577 + hash(est['nome']) % 67)
            obs = tarefa(est, rng, n=n)
            p = resolver(obs, campo)
            descobertas.append(p.e)
            if p.e.expandida() == est['alvo']:
                achou += 1
        recuperou[est['nome']] = achou / len(sementes)
        if cumulativo:
            nova = consolidar(descobertas, len(consolidadas))
            if nova is not None:
                consolidadas.append(nova)
                campo = list(base)
                for md in consolidadas:
                    campo.extend(md.instancias(N_VARS))
                if verboso:
                    print(f"  consolidou {nova.rot} = {nova.corpo.expandida()}"
                          f"  ({nova.n_slots} slots, "
                          f"{len(nova.instancias(N_VARS))} instâncias)")
    # controle raso
    usou = 0
    for s in sementes:
        rng = random.Random(s * 31 + 7)
        p = resolver(tarefa_rasa(rng, n=n), campo)
        usou += int(p.e.usa_C())
    return {'recuperou': recuperou, 'consolidadas': consolidadas,
            'usou_ctrl': usou / len(sementes)}


def relatorio(sementes: Sequence[int] = (0, 1, 2, 3, 4), n: int = 260) -> Dict[str, Any]:
    print("═" * 78)
    print("CAMPO RECURSIVO — cada descoberta vira material da seguinte")
    print("métrica: RECUPERAÇÃO DE ESTRUTURA (a expressão expandida é o alvo?)")
    print("═" * 78)
    print("PRÉ-REGISTRO:")
    print("  T1  braço PLANO recupera E1 e falha em E2 e E3")
    print("  T2  braço CUMULATIVO recupera os três, com o MESMO orçamento")
    print("  T3  controle raso não usa terminal consolidado")
    print("  T4  a consolidação é automática, rótulo por índice")
    print()

    pc = precondicao()
    print("─" * 78)
    if not pc['aprovada']:
        print("PRÉ-CONDIÇÃO REPROVADA: os alvos não têm a relação de alcance que o")
        print("desenho exige. Não rodo — a comparação não mediria nada.")
        return {'precondicao': pc}
    print("Pré-condição APROVADA: E1 ao alcance, E2 e E3 fora. O mecanismo pode rodar.")
    print()

    print("braço CUMULATIVO:")
    cum = rodar_braco(True, sementes, n=n, verboso=True)
    pla = rodar_braco(False, sementes, n=n)

    print()
    print(f"{'estágio':<10}{'alvo':<22}{'plano':>9}{'cumulativo':>13}")
    for est in ESTAGIOS:
        nm = est['nome']
        print(f"{nm:<10}{est['alvo']:<22}{pla['recuperou'][nm]:>8.0%}"
              f"{cum['recuperou'][nm]:>13.0%}")
    print()
    print(f"controle raso — usou consolidada: {cum['usou_ctrl']:.0%}   (ideal 0%)")

    print("─" * 78)
    p2, p3 = pla['recuperou']['E2'], pla['recuperou']['E3']
    c2, c3 = cum['recuperou']['E2'], cum['recuperou']['E3']
    if cum['usou_ctrl'] > 0.2:
        print(f"CONTROLE REPROVADO: consolidada usada em {cum['usou_ctrl']:.0%} do controle")
        print("raso. A consolidação inflou a busca — os ganhos ficam ilegíveis.")
    elif p2 < 0.2 and p3 < 0.2 and c2 > 0.8 and c3 > 0.8:
        print("CASCATA. O braço plano não alcança E2 nem E3 em nenhum orçamento deste")
        print("desenho. O cumulativo alcança os dois — porque E1 virou ingrediente de")
        print("E2, e E2 virou ingrediente de E3.")
        print()
        print("O espaço EFETIVO de representações cresceu com o que já havia sido")
        print("descoberto, e não por busca mais funda: a expansão de composição é")
        print("idêntica nos dois braços.")
        print()
        print("A ressalva de sempre: os átomos iniciais e a gramática são meus. O que")
        print("cresceu foi o alcançável a partir deles.")
    else:
        print(f"Parcial — plano E2 {p2:.0%} E3 {p3:.0%} · "
              f"cumulativo E2 {c2:.0%} E3 {c3:.0%}.")
    return {'cumulativo': cum, 'plano': pla}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=260)
    a = ap.parse_args()
    relatorio(n=a.n)
