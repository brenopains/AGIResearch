"""
novaphys.intervencao — descobrir estrutura intervindo, não observando
======================================================================

`campo_recursivo.py` bateu num limite que não é bug e não tem conserto dentro
do desenho:

    duas estruturas que induzem a MESMA partição do espaço são
    indistinguíveis por qualquer quantidade de rótulos

Medido: o alvo `(v/v)` teve margem de custo **0,0** sobre a melhor alternativa,
e o alvo não-linear `|((v/v)-v)|` também **0,0**. `v0/v1 < c` é uma reta pela
origem; `v0−v1 < k` é outra reta; sobre uma caixa limitada elas coincidem o
suficiente para o MDL não as separar. Ampliar o domínio não muda nada — a
fronteira de uma razão é linear por construção.

Classificação binária é um canal estreito demais para recuperar estrutura
simbólica. O dado não contém a resposta.

O QUE INTERVENÇÃO ACRESCENTA
------------------------------
Estruturas que compartilham fronteira NÃO compartilham resposta a intervenção:

    y = x/z     dobrar x → y dobra        dobrar z → y cai pela metade
    y = x−z     dobrar x → y soma x       dobrar z → y subtrai z
    y = x·z     dobrar x → y dobra        dobrar z → y dobra

A assinatura de escalonamento separa o que a partição funde. É por isso que
física se faz mexendo no aparelho, e não só olhando.

COMO A PONTUAÇÃO MUDA
----------------------
Não se pontua acerto de classe. Pontua-se quão bem a candidata prevê a
RESPOSTA a uma intervenção, em escala logarítmica relativa:

    para cada par (v, v') onde v' difere de v numa coordenada:
        prevista  = log|E(v')| − log|E(v)|
        observada = log|y'|    − log|y|
        erro      = (prevista − observada)²

Trabalhar com a variação relativa torna a pontuação invariante a `E → k·E`.
Isso é deliberado: não interessa a constante, interessa a FORMA. Uma candidata
que acerta a forma e erra a escala é a resposta certa.

O PORTÃO VEM PRIMEIRO, E É O TESTE DO REDESENHO
-------------------------------------------------
Antes de qualquer mecanismo, a mesma pergunta que reprovou a bancada anterior:
o alvo vence com folga? Se a margem sob intervenção continuar em 0,0, o
redesenho não resolveu nada e não se roda o resto.

Os mesmos alvos, a mesma gramática, a mesma métrica de margem — só o canal de
informação muda. É a comparação limpa entre observar e intervir.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import statistics as st
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

N_VARS = 4
LAM = 0.35
EPS = 1e-9
OPS2 = ('+', '-', '*', '/')

# (v, v', y, y') — um par de intervenção: v' difere de v numa coordenada
Par = Tuple[List[float], List[float], float, float]


# ══════════════════════════════════════════════ expressões
@dataclass(frozen=True)
class E:
    op: str
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
        return x / y if abs(y) > EPS else math.copysign(1e9, x or 1.0)

    def tam(self) -> int:
        if self.op in ('V', 'C'):
            return 1
        if self.op == 'abs':
            return 1 + self.a.tam()
        return 1 + self.a.tam() + self.b.tam()

    def expandida(self) -> str:
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


def compor(campo: Sequence[E]) -> List[E]:
    bin_: List[E] = []
    for x in campo:
        for y in campo:
            if x is y:
                continue
            for o in OPS2:
                bin_.append(E(o, x, y))
    return list(campo) + bin_ + [E('abs', z) for z in list(campo) + bin_]


@dataclass(frozen=True)
class Molde:
    """Consolidada com buracos — abstrai as variáveis, como em `abstracao.py`."""
    corpo: E
    n_slots: int
    rot: str

    def instancias(self, n_vars: int) -> List[E]:
        def preencher(e: E, at: Sequence[int]) -> E:
            if e.op == 'V':
                return E('V', at[e.a])
            if e.op == 'C':
                return e
            if e.op == 'abs':
                return E('abs', preencher(e.a, at))
            return E(e.op, preencher(e.a, at), preencher(e.b, at))
        return [E('C', preencher(self.corpo, at), rot=self.rot)
                for at in itertools.permutations(range(n_vars), self.n_slots)]


def abstrair(e: E, rot: str) -> Molde:
    mapa: Dict[int, int] = {}

    def rec(x: E) -> E:
        if x.op == 'V':
            mapa.setdefault(x.a, len(mapa))
            return E('V', mapa[x.a])
        if x.op == 'C':
            return rec(x.a)
        if x.op == 'abs':
            return E('abs', rec(x.a))
        return E(x.op, rec(x.a), rec(x.b))

    corpo = rec(e)
    return Molde(corpo, len(mapa), rot)


# ══════════════════════════════════════════════ o mundo, com intervenção
ALVOS = {
    'razao':      {'f': lambda v: v[0] / v[1],            'alvo': '(v/v)'},
    'diferenca':  {'f': lambda v: v[0] - v[1],            'alvo': '(v-v)'},
    'produto':    {'f': lambda v: v[0] * v[1],            'alvo': '(v*v)'},
    'razao_menos':{'f': lambda v: abs(v[0] / v[1] - v[2]),'alvo': '|((v/v)-v)|'},
}


def coletar(f: Callable[[Sequence[float]], float], rng: random.Random,
            n: int = 120, lo: float = 0.4, hi: float = 2.5) -> List[Par]:
    """
    Cada amostra é uma INTERVENÇÃO: escolhe-se um estado, mexe-se numa
    coordenada, e observa-se como a saída respondeu. É mexer no aparelho.
    """
    pares: List[Par] = []
    for _ in range(n):
        v = [rng.uniform(lo, hi) for _ in range(N_VARS)]
        i = rng.randrange(N_VARS)
        fator = rng.choice([0.5, 0.7, 1.4, 2.0])
        v2 = list(v)
        v2[i] = max(lo * 0.5, min(hi * 2, v[i] * fator))
        y, y2 = f(v), f(v2)
        if abs(y) < 1e-6 or abs(y2) < 1e-6:
            continue
        pares.append((v, v2, y, y2))
    return pares


def custo(e: E, pares: Sequence[Par]) -> float:
    """
    Comprimento de descrição do resíduo, em bits, mais o custo estrutural.

        custo = n·log(EQM) + tamanho·log(n)

    A forma anterior era `EQM + λ·tamanho`, e ela é ERRADA por um motivo que
    aparece na unidade: o erro em log-resposta é da ordem de 0,1–1, e λ=0,35
    dava vantagem de 0,7 a um átomo solitário sobre uma expressão de três nós.
    Resultado medido: `v0` batia `(v0/v1)` no próprio mundo gerado por
    `v0/v1`, e a margem do alvo saiu em −0,641 — negativa, o que é impossível
    quando a pontuação está certa.

    Somar erro quadrático com contagem de nós é somar coisas sem unidade comum.
    A forma BIC põe as duas em bits e dispensa a constante arbitrária.

    A escala logarítmica relativa continua: torna a pontuação invariante a
    `E → k·E`, porque o que se cobra é a FORMA da resposta, não a constante.
    """
    s = 0.0
    n = 0
    for v, v2, y, y2 in pares:
        a, b = e.val(v), e.val(v2)
        if abs(a) < 1e-9 or abs(b) < 1e-9:
            return math.inf
        try:
            prev = math.log(abs(b)) - math.log(abs(a))
            obs = math.log(abs(y2)) - math.log(abs(y))
        except ValueError:
            return math.inf
        s += (prev - obs) ** 2
        n += 1
    if n == 0:
        return math.inf
    eqm = max(s / n, 1e-12)
    return n * math.log(eqm) + e.tam() * math.log(n)


def resolver(pares: Sequence[Par], campo: Sequence[E]) -> Tuple[E, float]:
    best, bc = None, math.inf
    for e in compor(campo):
        c = custo(e, pares)
        if c < bc:
            best, bc = e, c
    return best, bc


# ══════════════════════════════════════════════ PORTÃO
def discriminabilidade(nome: str, campo: Sequence[E], sementes: Sequence[int],
                       n: int = 120) -> Dict[str, Any]:
    alvo = ALVOS[nome]['alvo']
    f = ALVOS[nome]['f']
    margens = []
    rivais = []
    for s in sementes:
        pares = coletar(f, random.Random(s * 331 + 11), n=n)

        def assinatura(e: E) -> Optional[Tuple[int, ...]]:
            """Resposta prevista, arredondada — identifica DUPLICATAS funcionais."""
            out = []
            for v, v2, _, _ in pares[:40]:
                a, b = e.val(v), e.val(v2)
                if abs(a) < 1e-9 or abs(b) < 1e-9:
                    return None
                out.append(round((math.log(abs(b)) - math.log(abs(a))) * 1e6))
            return tuple(out)

        c_alvo, e_alvo = math.inf, None
        for e in compor(campo):
            if e.expandida() == alvo:
                c = custo(e, pares)
                if c < c_alvo:
                    c_alvo, e_alvo = c, e
        assin_alvo = assinatura(e_alvo) if e_alvo is not None else None

        # O rival tem de ser FUNCIONALMENTE DISTINTO. `|(v/v)|` é o mesmo que
        # `(v/v)` em dados positivos; compará-los mede um nó de custo, não
        # discriminação. Foi o que produziu margem 4,787 = log(n) idêntica nos
        # três mundos — número igual a três casas em problemas diferentes é
        # sinal de artefato estrutural, não de medida.
        c_outro, e_outro = math.inf, None
        for e in compor(campo):
            if e.expandida() == alvo or assinatura(e) == assin_alvo:
                continue
            c = custo(e, pares)
            if c < c_outro:
                c_outro, e_outro = c, e
        margens.append(c_outro - c_alvo)
        if e_outro is not None:
            rivais.append(e_outro.expandida())
    return {'margem': st.mean(margens), 'min': min(margens),
            'rival': Counter(rivais).most_common(1)[0][0] if rivais else '—'}


def portao(sementes: Sequence[int] = (0, 1, 2), n: int = 120,
           margem_min: float = 20.0, verboso: bool = True) -> Dict[str, Any]:
    """
    A MESMA pergunta que reprovou a bancada por observação, agora sob
    intervenção. Se a margem continuar ~0, o redesenho não resolveu nada.
    """
    base = [E('V', i) for i in range(N_VARS)]
    res = {}
    if verboso:
        print("PORTÃO — o alvo vence com folga, agora sob INTERVENÇÃO?")
        print(f"{'alvo':<16}{'alcançável':>12}{'margem média':>15}{'margem mín':>13}"
              f"   melhor rival DISTINTO")
    for nome in ('razao', 'diferenca', 'produto'):
        formas = {e.expandida() for e in compor(base)}
        d = discriminabilidade(nome, base, sementes, n)
        res[nome] = d
        if verboso:
            print(f"{ALVOS[nome]['alvo']:<16}"
                  f"{'sim' if ALVOS[nome]['alvo'] in formas else 'NÃO':>12}"
                  f"{d['margem']:>15.1f}{d['min']:>13.1f}   rival: {d['rival']}")
    ok = all(r['min'] >= margem_min for r in res.values())
    return {'aprovado': ok, 'res': res}


# ══════════════════════════════════════════════ recuperação
def recuperacao(sementes: Sequence[int] = (0, 1, 2, 3, 4),
                n: int = 120) -> Dict[str, float]:
    base = [E('V', i) for i in range(N_VARS)]
    out = {}
    for nome, d in ALVOS.items():
        if nome == 'razao_menos':
            continue
        ok = 0
        for s in sementes:
            pares = coletar(d['f'], random.Random(s * 331 + 11), n=n)
            e, _ = resolver(pares, base)
            ok += int(e.expandida() == d['alvo'])
        out[nome] = ok / len(sementes)
    return out


def _equivalente(e: E, alvo_e: E, pares: Sequence[Par], tol: float = 1e-6) -> bool:
    """
    Mesma FUNÇÃO sob a métrica, não mesma string.

    A comparação por string reprovou a cascata em 0% contra um resultado que
    estava certo: o vencedor foi `(v-(v/v))`, que é o NEGATIVO de
    `|((v/v)-v)|`. Como a pontuação usa `log|·|`, o sinal é invisível — as duas
    são a mesma função aqui, e a que venceu é MENOR, por não carregar o nó
    `abs` que a métrica não cobra.

    O sistema achou o equivalente mais parcimonioso. Quem estava errada era a
    checagem — mesma classe do rival duplicado no portão.
    """
    for v, v2, _, _ in pares[:60]:
        a, b = e.val(v), e.val(v2)
        c, d = alvo_e.val(v), alvo_e.val(v2)
        if min(abs(a), abs(b), abs(c), abs(d)) < 1e-9:
            return False
        r1 = math.log(abs(b)) - math.log(abs(a))
        r2 = math.log(abs(d)) - math.log(abs(c))
        if abs(r1 - r2) > tol:
            return False
    return True


def cascata(sementes: Sequence[int] = (0, 1, 2, 3, 4),
            n: int = 120, verboso: bool = True) -> Dict[str, Any]:
    """
    Estágio 1 descobre `(v/v)`; consolida; estágio 2 usa a consolidada para
    alcançar `|((v/v)-v)|`, que está FORA do alcance do campo plano.
    """
    base = [E('V', i) for i in range(N_VARS)]
    plano = {e.expandida() for e in compor(base)}
    fora = ALVOS['razao_menos']['alvo'] not in plano

    descobertas = []
    for s in sementes:
        pares = coletar(ALVOS['razao']['f'], random.Random(s * 331 + 11), n=n)
        e, _ = resolver(pares, base)
        descobertas.append(e)
    cont = Counter(x.expandida() for x in descobertas)
    forma, vezes = cont.most_common(1)[0]
    modelo = next(x for x in descobertas if x.expandida() == forma)
    molde = abstrair(modelo, 'E1')
    campo2 = base + molde.instancias(N_VARS)
    if verboso:
        print(f"  consolidou E1 = {molde.corpo.expandida()} "
              f"({vezes}/{len(sementes)} das descobertas, {molde.n_slots} slots)")

    alvo_e = E('abs', E('-', E('/', E('V', 0), E('V', 1)), E('V', 2)))
    rec_plano = rec_cum = 0
    formas_cum = []
    for s in sementes:
        pares = coletar(ALVOS['razao_menos']['f'], random.Random(s * 331 + 11), n=n)
        ep, _ = resolver(pares, base)
        ec, _ = resolver(pares, campo2)
        rec_plano += int(_equivalente(ep, alvo_e, pares))
        rec_cum += int(_equivalente(ec, alvo_e, pares))
        formas_cum.append(ec.expandida())
    return {'fora_do_plano': fora, 'plano': rec_plano / len(sementes),
            'cumulativo': rec_cum / len(sementes), 'E1': molde.corpo.expandida(),
            'forma_cum': Counter(formas_cum).most_common(1)[0][0]}


def relatorio(n: int = 120) -> Dict[str, Any]:
    print("═" * 78)
    print("INTERVENÇÃO — mexer no aparelho, em vez de só olhar")
    print("pontuação: erro na RESPOSTA relativa à intervenção + λ·tamanho")
    print("═" * 78)
    print("O que motivou o redesenho, medido na bancada anterior:")
    print("  por classificação, a margem do alvo sobre a alternativa foi 0,000")
    print("  — duas estruturas com a mesma fronteira são indistinguíveis por rótulo")
    print()

    p = portao(n=n)
    print("─" * 78)
    if not p['aprovado']:
        print("PORTÃO REPROVADO: a margem continua baixa mesmo sob intervenção.")
        print("O redesenho não resolveu o problema — não rodo o resto.")
        return {'portao': p}
    print("PORTÃO APROVADO: sob intervenção o alvo vence com folga. O canal de")
    print("informação mudou de fato, e o resto do experimento faz sentido.")
    print()

    r = recuperacao(n=n)
    print("RECUPERAÇÃO DE ESTRUTURA (a expressão encontrada É o alvo?)")
    for k, v in r.items():
        print(f"  {ALVOS[k]['alvo']:<16}{v:>6.0%}")

    print()
    print("CASCATA — E1 vira ingrediente de E2")
    c = cascata(n=n)
    print(f"  alvo |((v/v)-v)| fora do alcance do campo plano: "
          f"{'sim' if c['fora_do_plano'] else 'NÃO'}")
    print(f"  recuperação (equivalência FUNCIONAL) — plano {c['plano']:.0%}  ·  "
          f"cumulativo {c['cumulativo']:.0%}")
    print(f"  forma encontrada pelo cumulativo: {c['forma_cum']}")

    print("─" * 78)
    if c['fora_do_plano'] and c['plano'] < 0.2 and c['cumulativo'] > 0.8:
        print("CASCATA FECHADA. Com intervenção como canal, o estágio 1 recupera a")
        print("razão; ela é abstraída em molde com buracos e volta ao campo; e o")
        print("estágio 2 alcança uma estrutura que o campo plano não alcança.")
        print()
        print("O que mudou não foi o mecanismo — foi a informação disponível. A mesma")
        print("busca, a mesma gramática e a mesma consolidação falhavam por")
        print("observação e funcionam por intervenção.")
    else:
        print(f"Parcial — plano {c['plano']:.0%}, cumulativo {c['cumulativo']:.0%}.")
    return {'portao': p, 'recuperacao': r, 'cascata': c}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=120)
    a = ap.parse_args()
    relatorio(n=a.n)
