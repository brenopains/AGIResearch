"""
novaphys.operadores — quais transformações são necessárias para este mundo?
============================================================================

`hipoteses.py` resolveu uma pergunta: **como gerar hipóteses recursivamente**.
Um operador, PARTIR, aplicado até o comprimento de descrição parar de melhorar.
Geração 95%, contenção 99%.

Não resolveu a seguinte, e a diferença entre as duas é a parede atual:

    resolvido      como compor hipóteses dentro de um espaço de operadores
    não resolvido  como descobrir QUAIS operadores esse espaço precisa ter

Porque fui eu que escolhi PARTIR. O sistema constrói estruturas que não estavam
em lugar nenhum — mas dentro do espaço que aquela escolha define. Trocar "lista
de formas" por "lista de operadores" é progresso real e é a mesma armadilha um
nível acima, se ninguém disser isso em voz alta.

O DESENHO
----------
Cinco primitivas, e NENHUMA indicação de qual serve para qual problema:

    CONST         a direção é a mesma em todo lugar
    PARTIR        a direção muda conforme a POSIÇÃO em x
    CONDICIONAR   a direção muda conforme uma VARIÁVEL DE CONTEXTO z
    INVERTER      troca o sinal do que estiver dentro
    COMBINAR      soma os efeitos de duas subestruturas

PARTIR e CONDICIONAR são o par que interessa. Os dois criam regimes, e por
motivos diferentes: um parte o eixo que está sendo variado, o outro depende de
algo que não é esse eixo. Num mundo onde a direção depende de z, PARTIR não
resolve com corte nenhum — pode picar x em vinte pedaços que a resposta
continua errada, porque a variável certa não é x.

O CUSTO, QUE É O QUE IMPEDE A MÁQUINA DE DESCULPAS
----------------------------------------------------
Deixar o sistema tentar todas as combinações até uma funcionar não é descoberta,
é busca exaustiva com relatório bonito. Então cada programa paga:

    score = −2·log-verossimilhança          (quão bem explica)
          + λ_nós    · nº de nós            (penalidade estrutural)
          + λ_ops    · nº de operadores DISTINTOS usados   (custo de vocabulário)
          + λ_prof   · profundidade         (custo de encadeamento)

O termo de operadores distintos é o que difere isto de um BIC comum: usar uma
transformação NOVA custa mais caro que reusar uma que já está no programa.
É a formalização de "só introduza um operador se ele for necessário" — e é o
que faz a resposta certa em mundo monotônico ser CONST, e não CONST embrulhado
em INVERTER duas vezes.

O QUE ESTA BANCADA MEDE
------------------------
    1. SELEÇÃO      no mundo feito por um operador, ele escolhe esse operador?
    2. PARCIMÔNIA   no mundo monotônico, ele volta para CONST?
    3. TRANSFERÊNCIA no mundo onde PARTIR é inútil, ele acha CONDICIONAR?
    4. FRONTEIRA    no mundo que NENHUMA composição explica, ele sinaliza?

A prova 3 é a que o experimento existe para responder. A prova 4 é a que mede
a parede seguinte: se nem a melhor composição explica, o sistema percebe — ou
entrega o menos ruim com cara de resposta?

O LIMITE, DITO ANTES DO RESULTADO
----------------------------------
A busca é ENUMERAÇÃO LIMITADA: profundidade 2, e no máximo um filho composto
por nó. Então "nenhuma composição explica" quer dizer "nenhuma dentro deste
teto". Um espaço de programas ilimitado precisaria de busca de verdade, e isso
é outro problema — mas o teto é declarado aqui para que a frase não seja lida
como mais forte do que é.
"""

from __future__ import annotations

import math
import random
import statistics as st
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# (x1, x2, z, subiu) — z é a variável de contexto, constante durante a medição
Obs = Tuple[float, float, float, bool]

LAMBDA_NOS = 2.0
LAMBDA_OPS = 4.0        # operador NOVO custa mais que reusar um já presente
LAMBDA_PROF = 1.5


# ══════════════════════════════════════════════ os programas
@dataclass(frozen=True)
class Prog:
    op: str
    args: Tuple[Any, ...] = ()

    # ── semântica ────────────────────────────────────────────────────────
    def delta(self, x1: float, x2: float, z: float) -> float:
        """Variação prevista de y entre x1 e x2, no contexto z."""
        o = self.op
        if o == 'CONST':
            return self.args[0] * (x2 - x1)
        if o == 'INVERTER':
            return -self.args[0].delta(x1, x2, z)
        if o == 'COMBINAR':
            return self.args[0].delta(x1, x2, z) + self.args[1].delta(x1, x2, z)
        if o == 'CONDICIONAR':
            c, A, B = self.args
            return (A if z < c else B).delta(x1, x2, z)
        if o == 'PARTIR':
            c, A, B = self.args
            e = 0.0
            if x1 < c:
                e += A.delta(x1, min(x2, c), z)
            if x2 > c:
                e += B.delta(max(x1, c), x2, z)
            return e
        raise ValueError(o)

    # ── custo ────────────────────────────────────────────────────────────
    def nos(self) -> int:
        return 1 + sum(a.nos() for a in self.args if isinstance(a, Prog))

    def profundidade(self) -> int:
        f = [a.profundidade() for a in self.args if isinstance(a, Prog)]
        return 1 + (max(f) if f else 0)

    def operadores(self) -> set:
        s = {self.op}
        for a in self.args:
            if isinstance(a, Prog):
                s |= a.operadores()
        return s

    # ── ajuste ───────────────────────────────────────────────────────────
    def log_veros(self, obs: Sequence[Obs], q: float = 0.8) -> float:
        ll = 0.0
        for (x1, x2, z, subiu) in obs:
            d = self.delta(x1, x2, z)
            if abs(d) < 1e-12:
                ll += math.log(0.5)
            else:
                ll += math.log(q if (d > 0) == subiu else 1 - q)
        return ll

    def score(self, obs: Sequence[Obs], q: float = 0.8) -> float:
        return (-2.0 * self.log_veros(obs, q)
                + LAMBDA_NOS * self.nos()
                + LAMBDA_OPS * len(self.operadores())
                + LAMBDA_PROF * self.profundidade())

    def __str__(self) -> str:
        o = self.op
        if o == 'CONST':
            return '+' if self.args[0] > 0 else '−'
        if o == 'INVERTER':
            return f"INV({self.args[0]})"
        if o == 'COMBINAR':
            return f"COMB({self.args[0]},{self.args[1]})"
        if o == 'CONDICIONAR':
            return f"COND(z<{self.args[0]:.2g} ? {self.args[1]} : {self.args[2]})"
        return f"PART(x<{self.args[0]:.3g} ? {self.args[1]} : {self.args[2]})"


MAIS, MENOS = Prog('CONST', (1,)), Prog('CONST', (-1,))


# ══════════════════════════════════════════════ a busca
def _candidatos(obs: Sequence[Obs], n_cortes: int = 7,
                n_ctx: int = 5) -> Tuple[List[float], List[float]]:
    xs = sorted({x for (a, b, _, _) in obs for x in (a, b)})
    lo, hi = xs[0], xs[-1]
    cortes = [lo + (hi - lo) * (i + 1) / (n_cortes + 1) for i in range(n_cortes)]
    zs = sorted({z for (_, _, z, _) in obs})
    zlo, zhi = (zs[0], zs[-1]) if zs else (0.0, 1.0)
    ctx = [zlo + (zhi - zlo) * (i + 1) / (n_ctx + 1) for i in range(n_ctx)]
    return cortes, ctx


def _expandir(base: Sequence[Prog], cortes: List[float],
              ctx: List[float]) -> List[Prog]:
    """Um passo de composição. Todos os cinco operadores, sem preferência."""
    out: List[Prog] = []
    for A in base:
        out.append(Prog('INVERTER', (A,)))
        for B in base:
            out.append(Prog('COMBINAR', (A, B)))
            for c in cortes:
                out.append(Prog('PARTIR', (c, A, B)))
            for c in ctx:
                out.append(Prog('CONDICIONAR', (c, A, B)))
    return out


def buscar(obs: Sequence[Obs], q: float = 0.8,
           profundidade: int = 2) -> Dict[str, Any]:
    """
    Enumeração limitada com poda por score.

    O teto é declarado: profundidade 2, e na segunda camada só as melhores
    folhas seguem como filhos. Sem o teto isto vira busca exaustiva, que
    encontra explicação para qualquer coisa e não é evidência de nada.
    """
    cortes, ctx = _candidatos(obs)
    folhas = [MAIS, MENOS]
    todos: List[Prog] = list(folhas)
    camada = list(folhas)
    for _ in range(profundidade):
        nova = _expandir(camada, cortes, ctx)
        todos.extend(nova)
        # poda: só as 12 melhores da camada seguem, senão a camada seguinte
        # explode e a busca deixa de ser limitada de fato
        nova.sort(key=lambda p: p.score(obs, q))
        camada = nova[:12]
    melhor = min(todos, key=lambda p: p.score(obs, q))
    ll = melhor.log_veros(obs, q)
    n = max(1, len(obs))
    # acerto do melhor programa nas próprias observações — o piso é q, porque
    # nem o programa verdadeiro acerta mais que a confiabilidade do sensor
    acertos = sum(1 for (x1, x2, z, s) in obs
                  if (melhor.delta(x1, x2, z) > 0) == s)
    return {'programa': melhor, 'score': melhor.score(obs, q),
            'log_veros': ll, 'ajuste': acertos / n,
            'operadores': melhor.operadores(), 'n_avaliados': len(todos)}


# ══════════════════════════════════════════════ suficiência do vocabulário
def suficiente(resultado: Dict[str, Any], n_obs: int, q: float,
               z_critico: float = 2.5) -> Dict[str, Any]:
    """
    "Alguma composição minha explica estes dados?"

    O teto é conhecido e não precisa ser estimado: se o programa estivesse
    CERTO, ele ainda erraria a fração (1−q) das observações, porque é o sensor
    que erra. Então o ajuste esperado de um programa correto é exatamente q, e
    ajuste MUITO abaixo de q é evidência de que nenhum programa disponível está
    certo — não de que faltam dados.

    Isso é diferente de tudo o que veio antes nesta sessão: `crencas.py` sabia
    quando não sabia LER a relação; aqui o sistema sabe quando o próprio
    VOCABULÁRIO DE OPERADORES não alcança o mundo. É o sinal que a bancada
    mostrou existir nos dados (ajuste 57% contra 82%) e que ninguém estava
    lendo.

    Teste binomial, unilateral. `z_critico` alto de propósito: o custo de
    gritar "meu vocabulário é insuficiente" à toa é alto — manda construir
    operador novo quando bastava mais dado.
    """
    a = resultado['ajuste']
    n = max(1, n_obs)
    ep = math.sqrt(max(1e-12, q * (1 - q) / n))
    z = (a - q) / ep
    return {'suficiente': z > -z_critico, 'z': z,
            'ajuste': a, 'teto': q,
            'veredito': ('o vocabulário alcança' if z > -z_critico
                         else 'NENHUMA composição disponível explica')}


# ══════════════════════════════════════════════ os mundos
def mundo(tipo: str, rng: random.Random) -> Callable[[float, float, float], bool]:
    """Devolve a verdade: y subiu de x1 para x2 no contexto z?"""
    lo, hi = 0.0, 10.0
    c = lo + (hi - lo) * rng.uniform(0.35, 0.65)
    zc = rng.uniform(0.35, 0.65)

    def f_monotonico(x):     return x
    def f_pico(x):           return -abs(x - c)

    if tipo == 'monotonico':
        return lambda x1, x2, z: f_monotonico(x2) > f_monotonico(x1)
    if tipo == 'parte-em-x':
        return lambda x1, x2, z: f_pico(x2) > f_pico(x1)
    if tipo == 'condiciona-em-z':
        # A DIREÇÃO DEPENDE DO CONTEXTO, não da posição. PARTIR não alcança:
        # não existe corte em x que separe estes casos, porque o que separa
        # não é x.
        return lambda x1, x2, z: (x2 > x1) if z < zc else (x2 < x1)
    if tipo == 'periodico':
        # NENHUM dos cinco operadores representa isto com custo razoável: são
        # muitos regimes alternados, e cada um custa um nó. Está aqui para
        # medir se o sistema SINALIZA insuficiência ou entrega o menos ruim
        # com cara de resposta.
        k = 2.0 * math.pi / (hi - lo) * 3.0
        return lambda x1, x2, z: math.sin(k * x2) > math.sin(k * x1)
    raise ValueError(tipo)


def amostrar(v: Callable[[float, float, float], bool], n: int, q: float,
             rng: random.Random) -> List[Obs]:
    obs = []
    for _ in range(n):
        x1, x2 = sorted((rng.uniform(0, 10), rng.uniform(0, 10)))
        if x2 - x1 < 0.8:
            x2 = min(10.0, x1 + 2.5)
        z = rng.random()
        real = v(x1, x2, z)
        obs.append((x1, x2, z, real if rng.random() < q else (not real)))
    return obs


# ══════════════════════════════════════════════ avaliação
def avaliar(n_casos: int = 25, n_obs: int = 70, q: float = 0.85,
            semente: int = 0) -> Dict[str, Any]:
    tipos = ['monotonico', 'parte-em-x', 'condiciona-em-z', 'periodico']
    esperado = {'monotonico': 'CONST', 'parte-em-x': 'PARTIR',
                'condiciona-em-z': 'CONDICIONAR', 'periodico': None}
    r: Dict[str, Any] = {}
    for t in tipos:
        acertos, ajustes, nos, insuf = 0, [], [], 0
        for i in range(n_casos):
            rng = random.Random(semente * 977 + i)
            obs = amostrar(mundo(t, rng), n_obs, q, rng)
            b = buscar(obs, q=q)
            if not suficiente(b, len(obs), q)['suficiente']:
                insuf += 1
            ops = b['operadores']
            alvo = esperado[t]
            if alvo == 'CONST':
                ok = ops == {'CONST'}
            elif alvo is not None:
                ok = alvo in ops
            else:
                ok = False
            acertos += int(ok)
            ajustes.append(b['ajuste'])
            nos.append(b['programa'].nos())
        r[t] = {'selecao': acertos / n_casos, 'ajuste': st.mean(ajustes),
                'nos': st.mean(nos), 'insuficiencia': insuf / n_casos}
    return r


def relatorio(n_casos: int = 25, sementes: Tuple[int, ...] = (0, 1),
              q: float = 0.85) -> Dict[str, Any]:
    ls = [avaliar(n_casos=n_casos, semente=s, q=q) for s in sementes]
    med = lambda t, k: st.mean([l[t][k] for l in ls])

    print("═" * 78)
    print("SELEÇÃO E COMPOSIÇÃO DE OPERADORES")
    print("cinco primitivas, nenhuma indicação de qual serve para qual problema")
    print(f"custo: {LAMBDA_NOS}·nós + {LAMBDA_OPS}·operadores distintos + "
          f"{LAMBDA_PROF}·profundidade")
    print("═" * 78)
    print(f"{'mundo':<20}{'operador certo':>16}{'ajuste':>9}{'nós':>6}"
          f"{'sinal insuf.':>14}   o que testa")
    legenda = {
        'monotonico': 'parcimônia: volta para CONST?',
        'parte-em-x': 'seleção: regime por posição',
        'condiciona-em-z': 'TRANSFERÊNCIA: PARTIR é inútil aqui',
        'periodico': 'FRONTEIRA: nenhum operador alcança',
    }
    for t in ('monotonico', 'parte-em-x', 'condiciona-em-z', 'periodico'):
        s = med(t, 'selecao')
        s_txt = f"{s:.0%}" if t != 'periodico' else "—"
        print(f"{t:<20}{s_txt:>16}{med(t,'ajuste'):>9.0%}{med(t,'nos'):>6.1f}"
              f"{med(t,'insuficiencia'):>13.0%}   {legenda[t]}")

    print("─" * 78)
    mono, part, cond = (med('monotonico', 'selecao'), med('parte-em-x', 'selecao'),
                        med('condiciona-em-z', 'selecao'))
    aj_per, nos_per = med('periodico', 'ajuste'), med('periodico', 'nos')
    aj_cond = med('condiciona-em-z', 'ajuste')

    if cond > 0.8 and part > 0.8 and mono > 0.8:
        print("SELEÇÃO E TRANSFERÊNCIA: o sistema escolhe o operador que o mundo pede,")
        print("inclusive num mundo em que PARTIR não resolve com corte nenhum — porque")
        print("o que separa os regimes não é a variável que está sendo variada.")
        print("Ninguém disse qual operador usar; o custo de vocabulário é que decidiu.")
    elif cond <= 0.5:
        print(f"TRANSFERÊNCIA FALHOU ({cond:.0%}): no mundo em que a direção depende do")
        print("contexto, o sistema não chegou a CONDICIONAR. Ele generaliza dentro do")
        print("operador que domina, e não entre operadores.")
    else:
        print(f"Resultado misto — mono {mono:.0%} · partir {part:.0%} · cond {cond:.0%}.")

    print()
    print("A FRONTEIRA (mundo periódico):")
    print(f"  ajuste {aj_per:.0%} com {nos_per:.1f} nós, contra {aj_cond:.0%} no mundo")
    print("  que os operadores alcançam.")
    det = med('periodico', 'insuficiencia')
    fa = max(med('monotonico', 'insuficiencia'), med('parte-em-x', 'insuficiencia'),
             med('condiciona-em-z', 'insuficiencia'))
    if det > 0.8 and fa < 0.1:
        print(f"  E o sistema SINALIZA: {det:.0%} de detecção de insuficiência aqui,")
        print(f"  contra {fa:.0%} de falso alarme nos mundos que ele alcança.")
        print()
        print("  O teto não precisou ser estimado: se o programa estivesse certo, ele")
        print("  ainda erraria (1−q) por culpa do sensor. Ajuste muito abaixo de q é")
        print("  evidência de que NENHUM programa disponível está certo — e não de que")
        print("  faltam dados. São diagnósticos com ações opostas.")
        print()
        print("  O que isto NÃO faz: construir o operador que falta. Ele sabe que o")
        print("  vocabulário não alcança; não sabe o que acrescentar. Essa é a parede")
        print("  seguinte, e ela é qualitativamente diferente — exige gerar uma")
        print("  transformação, não escolher entre as que existem.")
    elif det <= 0.8:
        print(f"  Detecção de insuficiência {det:.0%}: o sinal existe nos dados e o")
        print("  mecanismo não o está lendo de forma confiável.")
    elif fa >= 0.1:
        print(f"  Falso alarme {fa:.0%} nos mundos alcançáveis: o sinal dispara onde")
        print("  não devia, e isso anula a detecção.")
    else:
        print("  O ajuste NÃO cai — vale checar se o mundo periódico está mesmo fora")
        print("  do alcance dos operadores antes de tratar como fronteira.")
    return {'linhas': ls}


if __name__ == "__main__":
    relatorio()
