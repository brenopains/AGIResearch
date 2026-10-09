"""
novaphys.semantica — detectar mudança de mecanismo sem símbolo novo
===================================================================

A bancada de aquisição terminou com um diagnóstico limpo: na célula `trocado`
o sinal de insuficiência ficou em 12%, igual à célula de controle. `causa`
continuava sendo uma palavra conhecida, e o comportamento dela tinha mudado —
então a consulta ao vocabulário devolvia competência 1,0 dentro de um mundo
que o sistema estava lendo errado.

    vocabulário → competente
    modelo      → errado
    observação  → contradiz a previsão

O conserto não é uma lista maior. É trocar a definição de "desconhecido":

    ANTES   desconhecido = o nome não está no meu vocabulário
    AGORA   desconhecido = minha previsão vem falhando neste canal

A segunda não precisa de catálogo. Ela descobre a limitação pela interação, e
é por isso que o teste decisivo é este: **mudar o comportamento sem introduzir
nenhum símbolo novo**, e medir quanto tempo o sistema leva para perceber.

DUAS HIPÓTESES CONCORRENTES, NÃO UMA CONFIANÇA QUE CAI
-------------------------------------------------------
A surpresa não derruba a crença na aresta. Ela redistribui massa entre duas
leituras do MECANISMO:

    H1  o tipo `causa` tem a semântica que eu conheço
    H2  o tipo `causa` se comporta de outro jeito NESTE mundo

Isso preserva a versão anterior em vez de sobrescrevê-la, que é a diferença
entre revisar e esquecer. O histórico guarda a trajetória inteira, e é ele que
permite perguntar se a transição foi progressiva ou um degrau.

A COMPETÊNCIA MUDA DE SENTIDO, E É O PONTO
-------------------------------------------
    competência = 2 · |P(H1) − 0,5|

Não é "acredito na semântica conhecida". É **sei qual das duas vale** — e fica
alta mesmo quando a resposta é *é o oposto do que eu supunha*. Descobrir que
`causa` não compõe neste mundo é conhecimento, não ignorância.

A consequência aparece sozinha na trajetória: durante a virada, P(H1) atravessa
0,5 e a competência **cai a zero**. É o intervalo em que as duas versões
coexistem e o sistema honestamente não sabe qual usar. Não foi programado; sai
da forma da medida.

O QUE ESTA BANCADA MEDE
------------------------
    1. DIAGNOSTICIDADE  observações que não distinguem H1 de H2 não movem nada
    2. LATÊNCIA         quantas observações até perceber a troca
    3. FALSO ALARME     num mundo estável, ele fica quieto?
    4. GRADUAL          com a mudança acontecendo devagar, ele acompanha ou pula?
"""

from __future__ import annotations

import math
import random
import statistics as st
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .crencas import _logit, _sig


@dataclass
class ModeloSemantico:
    """
    O que o sistema acredita sobre COMO um tipo de relação se comporta aqui.

    O prior favorece a semântica conhecida (0,9) porque é isso que "conhecer o
    vocabulário" deveria significar: uma expectativa forte e revisável, não uma
    certeza. A versão anterior tratava como certeza — e era exatamente por isso
    que nada podia derrubá-la.
    """
    tipo: str
    p_h1_inicial: float = 0.9
    # Probabilidade de o MECANISMO mudar entre duas observações.
    #
    # Sem isto o filtro é um acumulador: as log-odds crescem sem limite com o
    # número de observações, e o modelo passa a afirmar, implicitamente, que a
    # semântica é uma coisa FIXA — o que contradiz exatamente o fenômeno que
    # ele existe para detectar. Medido na prova 4: correlação 0,503 com a rampa
    # e só 6 observações de coexistência, porque um acumulador atravessa o meio
    # em quatro passos e nunca mais volta. A prova 5 (reversão) nasceu desse
    # defeito: sem deriva, o modelo chega a log-odds de −40 e nem centenas de
    # observações contrárias o trazem de volta. Reter a versão anterior no
    # histórico não basta — ela precisa continuar ALCANÇÁVEL.
    #
    # O valor limita as log-odds em log((1−ε)/ε), isto é, nenhuma quantidade de
    # evidência passada fecha a porta para evidência futura. Com 0,002 o teto é
    # ±6,2 e P(H1) fica em [0,002; 0,998].
    #
    # 0,002 NÃO foi escolhido por fazer as provas passarem — esse é o modo de
    # ajustar ao gabarito que já apareceu três vezes nesta sessão. Ele saiu de
    # `curva_compromisso()`, que varre ε e imprime latência, detecção e falso
    # alarme lado a lado. A curva mostra que rastrear e ficar quieto são
    # objetivos opostos e que ε é o botão entre eles:
    #
    #     ε=0,000 (acumulador)  latência 30 · detecção 100% · falso alarme 0%
    #                           — e reprova em gradual e reversão: nunca revisa
    #     ε=0,002               latência 13 · detecção  98% · falso alarme 0%
    #     ε=0,030               latência  9 · detecção  76% · falso alarme 14%
    #     ε=0,080               nunca detecta: o filtro esquece mais rápido do
    #                           que aprende
    #
    # A curva fica no módulo justamente para que este número seja uma DECISÃO
    # declarada e auditável, e não um valor que passou no teste.
    deriva: float = 0.002
    # Observações consecutivas que uma leitura nova precisa sustentar antes de
    # virar estado. Sem isto o detector CHIA: a deriva limita as log-odds, então
    # duas observações ruidosas seguidas cruzam o limiar e voltam. Medido com
    # persistência 1: 11% de falso alarme em mundo estável e 10% de alarme antes
    # mesmo de a mudança acontecer. Histerese é a resposta padrão a isso, e o
    # preço em latência aparece na curva em vez de ser suposto.
    persistencia: int = 5
    _log_odds: float = 0.0
    n_obs: int = 0
    n_diagnosticas: int = 0
    _estado_conf: str = 'estavel'
    _candidato: str = ''
    _conta: int = 0
    historico: List[Dict[str, Any]] = field(default_factory=list)
    versoes: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self._log_odds = _logit(self.p_h1_inicial)
        self._estado_conf = self._estado_instantaneo()
        self.historico.append({'n': 0, 'p_h1': self.p_h1_inicial, 'estado': self.estado})
        self.versoes.append({'desde': 0, 'leitura': 'H1 (semântica conhecida)',
                             'p': self.p_h1_inicial})

    @property
    def p_h1(self) -> float:
        return _sig(self._log_odds)

    @property
    def competencia(self) -> float:
        """Sei QUAL semântica vale — alta em 0,0 e em 1,0, zero em 0,5."""
        return 2.0 * abs(self.p_h1 - 0.5)

    def _estado_instantaneo(self) -> str:
        p = self.p_h1
        if p > 0.9:
            return 'estavel'          # a semântica conhecida se sustenta
        if p < 0.1:
            return 'revisada'         # H2 venceu; H1 fica no histórico
        return 'suspensa'             # as duas versões coexistem

    @property
    def estado(self) -> str:
        """O estado CONFIRMADO — o instantâneo sustentado por `persistencia` obs."""
        return self._estado_conf

    def _confirmar(self) -> None:
        inst = self._estado_instantaneo()
        if inst == self._estado_conf:
            self._candidato, self._conta = '', 0
            return
        if inst == self._candidato:
            self._conta += 1
        else:
            self._candidato, self._conta = inst, 1
        if self._conta >= self.persistencia:
            self._estado_conf = inst
            self._candidato, self._conta = '', 0

    def observar(self, prev_h1: bool, prev_h2: bool, obs: bool, q: float = 0.8) -> None:
        """
        Uma observação só informa se as duas hipóteses PREVEEM COISAS
        DIFERENTES. Onde H1 e H2 concordam, o resultado é compatível com ambas
        e não é evidência sobre nenhuma — contá-lo seria inventar informação a
        partir de concordância, que é o modo mais silencioso de um detector
        virar gerador de alarme.
        """
        self.n_obs += 1
        if prev_h1 == prev_h2:
            self.historico.append({'n': self.n_obs, 'p_h1': self.p_h1,
                                   'estado': self.estado, 'diagnostica': False})
            return
        self.n_diagnosticas += 1
        lr = (q / (1 - q)) if obs == prev_h1 else ((1 - q) / q)
        antes = self.estado
        self._log_odds += math.log(lr)
        # Passo de transição: o mundo pode ter mudado desde a última observação.
        # É o que transforma o acumulador num rastreador — e o que permite ao
        # sistema VOLTAR se o mecanismo voltar, coisa que evidência puramente
        # acumulada torna impossível.
        if self.deriva > 0:
            p = _sig(self._log_odds)
            p = (1 - self.deriva) * p + self.deriva * (1 - p)
            self._log_odds = _logit(p)
        self._confirmar()
        depois = self.estado
        self.historico.append({'n': self.n_obs, 'p_h1': self.p_h1,
                               'estado': depois, 'diagnostica': True})
        if depois != antes:
            self.versoes.append({
                'desde': self.n_obs, 'p': self.p_h1,
                'leitura': {'estavel': 'H1 (semântica conhecida)',
                            'suspensa': 'H1 e H2 concorrentes',
                            'revisada': 'H2 (comportamento diferente neste mundo)'}[depois]})

    def decodificar(self) -> str:
        linhas = [
            f"Mecanismo  '{self.tipo}'",
            f"  P(H1: semântica conhecida) ..... {self.p_h1:.3f}",
            f"  P(H2: outro comportamento) ..... {1 - self.p_h1:.3f}",
            f"  competência (sei QUAL vale) .... {self.competencia:.3f}",
            f"  observações .................... {self.n_obs} "
            f"({self.n_diagnosticas} diagnósticas)",
            f"  ESTADO ......................... {self.estado.upper()}",
        ]
        if len(self.versoes) > 1:
            linhas.append("  versões (a anterior é preservada, não sobrescrita):")
            for v in self.versoes:
                linhas.append(f"     obs {v['desde']:>4}  p={v['p']:.3f}  {v['leitura']}")
        return "\n".join(linhas)


# ══════════════════════════════════════════════ o mundo mínimo diagnóstico
def _observacao(compoe_agora: bool, diagnostica: bool, q: float,
                rng: random.Random) -> Tuple[bool, bool, bool]:
    """
    Um par sondado. Devolve (previsão sob H1, previsão sob H2, observação).

    H1 diz que o tipo compõe; H2 diz que não. Num par diagnóstico as previsões
    divergem, e a verdade depende de o tipo compor NESTE momento do mundo.
    Num par não-diagnóstico as duas preveem a mesma coisa — está aqui porque
    conhecimento real é assim, e a prova 1 mede se o sistema os ignora.
    """
    prev_h1 = True
    prev_h2 = False if diagnostica else True
    verdade = compoe_agora if diagnostica else True
    obs = verdade if rng.random() < q else (not verdade)
    return prev_h1, prev_h2, obs


# ══════════════════════════════════════════════ 1. diagnosticidade
def prova_diagnosticidade(n: int = 500, semente: int = 0) -> Dict[str, Any]:
    """Só observações onde H1 e H2 concordam. O posterior não pode se mover."""
    rng = random.Random(semente)
    m = ModeloSemantico(tipo='causa')
    p0 = m.p_h1
    for _ in range(n):
        ph1, ph2, obs = _observacao(True, diagnostica=False, q=0.8, rng=rng)
        m.observar(ph1, ph2, obs)
    return {'prova': 'diagnosticidade', 'deriva': abs(m.p_h1 - p0),
            'diagnosticas': m.n_diagnosticas, 'passou': abs(m.p_h1 - p0) < 1e-12}


# ══════════════════════════════════════════════ 2. latência
def prova_latencia(n_execucoes: int = 300, antes: int = 20, teto: int = 200,
                   q: float = 0.8, semente: int = 0,
                   deriva: float = 0.002, persistencia: int = 5) -> Dict[str, Any]:
    """
    O mundo compõe normalmente por `antes` observações; depois PARA de compor.
    Nenhum símbolo novo aparece — o nome do tipo é o mesmo o tempo todo.

    Mede quantas observações até o estado sair de 'estavel'. É a latência de
    perceber que o próprio modelo preditivo está errado.
    """
    rng = random.Random(semente)
    lat_susp, lat_rev, falhou = [], [], 0
    for _ in range(n_execucoes):
        m = ModeloSemantico(tipo='causa', deriva=deriva, persistencia=persistencia)
        for _ in range(antes):
            ph1, ph2, obs = _observacao(True, True, q, rng)
            m.observar(ph1, ph2, obs, q)
        if m.estado != 'estavel':
            falhou += 1     # alarmou antes de a mudança acontecer
            continue
        s = r = None
        for k in range(1, teto + 1):
            ph1, ph2, obs = _observacao(False, True, q, rng)
            m.observar(ph1, ph2, obs, q)
            if s is None and m.estado == 'suspensa':
                s = k
            if r is None and m.estado == 'revisada':
                r = k; break
        if s:
            lat_susp.append(s)
        if r:
            lat_rev.append(r)
    return {'prova': 'latência',
            'ate_suspensa': st.median(lat_susp) if lat_susp else float('nan'),
            'ate_revisada': st.median(lat_rev) if lat_rev else float('nan'),
            'taxa_deteccao': len(lat_rev) / max(1, n_execucoes),
            'alarme_precoce': falhou / max(1, n_execucoes),
            'passou': len(lat_rev) / max(1, n_execucoes) > 0.95}


# ══════════════════════════════════════════════ 3. falso alarme
def prova_falso_alarme(n_execucoes: int = 300, n_obs: int = 220,
                       q: float = 0.8, semente: int = 0,
                       deriva: float = 0.002, persistencia: int = 5) -> Dict[str, Any]:
    """
    Mundo ESTÁVEL: o tipo compõe do começo ao fim. O sistema tem de ficar
    quieto por 220 observações — mais do que a latência da prova 2, senão o
    teste não separa detectar de simplesmente derivar com o tempo.
    """
    rng = random.Random(semente)
    alarmou = 0
    for _ in range(n_execucoes):
        m = ModeloSemantico(tipo='causa', deriva=deriva, persistencia=persistencia)
        for _ in range(n_obs):
            ph1, ph2, obs = _observacao(True, True, q, rng)
            m.observar(ph1, ph2, obs, q)
        if m.estado != 'estavel':
            alarmou += 1
    return {'prova': 'falso alarme', 'taxa': alarmou / n_execucoes,
            'passou': alarmou / n_execucoes < 0.05}


# ══════════════════════════════════════════════ 4. mudança gradual
def prova_gradual(n_execucoes: int = 200, estavel: int = 20, rampa: int = 120,
                  depois: int = 60, q: float = 0.8, semente: int = 0) -> Dict[str, Any]:
    """
    A composição não vira de uma vez: a probabilidade de compor desce de 1,0 a
    0,0 ao longo de `rampa` observações.

    A pergunta é se a representação ACOMPANHA ou PULA. Duas medidas:

      correlação      P(H2) segue a fração real de composição ao longo do tempo?
      permanência     quantas observações o sistema passa em 'suspensa' — isto é,
                      com as duas versões coexistindo em vez de trocar uma pela
                      outra num degrau?

    Um sistema que substitui a aresta passa ~0 observações em 'suspensa'.
    """
    rng = random.Random(semente)
    perman, corrs, ordenado = [], [], 0
    for _ in range(n_execucoes):
        m = ModeloSemantico(tipo='causa')
        serie_p, serie_real = [], []
        for _ in range(estavel):
            ph1, ph2, obs = _observacao(True, True, q, rng)
            m.observar(ph1, ph2, obs, q)
            serie_p.append(1 - m.p_h1); serie_real.append(0.0)
        for k in range(rampa):
            frac = 1.0 - k / max(1, rampa - 1)     # prob. de compor
            ph1, ph2, obs = _observacao(rng.random() < frac, True, q, rng)
            m.observar(ph1, ph2, obs, q)
            serie_p.append(1 - m.p_h1); serie_real.append(1 - frac)
        for _ in range(depois):
            ph1, ph2, obs = _observacao(False, True, q, rng)
            m.observar(ph1, ph2, obs, q)
            serie_p.append(1 - m.p_h1); serie_real.append(1.0)
        perman.append(sum(1 for h in m.historico if h['estado'] == 'suspensa'))
        try:
            corrs.append(st.correlation(serie_p, serie_real))
        except Exception:
            pass
        if m.estado == 'revisada':
            ordenado += 1
    return {'prova': 'gradual',
            'permanencia_suspensa': st.median(perman),
            'correlacao': st.mean(corrs) if corrs else float('nan'),
            'convergiu': ordenado / n_execucoes,
            'passou': st.median(perman) > 5 and (st.mean(corrs) if corrs else 0) > 0.7}


# ══════════════════════════════════════════════ 5. reversão
def prova_reversao(n_execucoes: int = 200, fase: int = 45, teto: int = 200,
                   q: float = 0.8, semente: int = 0) -> Dict[str, Any]:
    """
    O mundo compõe, PARA de compor, e volta a compor. O sistema volta com ele?

    Esta prova não estava no plano — ela apareceu como corolário do defeito da
    prova 4. Um filtro que só acumula chega a log-odds de −40 e nada o traz de
    volta: ele teria "aprendido" que `causa` não compõe e ficaria preso nisso
    para sempre, mesmo diante de centenas de observações contrárias.

    Reter a versão anterior no histórico não basta: é preciso que ela continue
    ALCANÇÁVEL. Revisão que só anda para um lado é esquecimento com carimbo.
    """
    rng = random.Random(semente)
    voltou, custo = 0, []
    for _ in range(n_execucoes):
        m = ModeloSemantico(tipo='causa')
        for _ in range(fase):                       # compõe
            m.observar(*_observacao(True, True, q, rng), q=q)
        for _ in range(fase):                       # para de compor
            m.observar(*_observacao(False, True, q, rng), q=q)
        if m.estado != 'revisada':
            continue
        for k in range(1, teto + 1):                # volta a compor
            m.observar(*_observacao(True, True, q, rng), q=q)
            if m.estado == 'estavel':
                voltou += 1; custo.append(k); break
    return {'prova': 'reversão', 'taxa': voltou / n_execucoes,
            'custo_mediano': st.median(custo) if custo else float('nan'),
            'passou': voltou / n_execucoes > 0.9}


# ══════════════════════════════════════════════ a curva do compromisso
def curva_compromisso(derivas=(0.0, 0.002, 0.005, 0.01, 0.03, 0.08),
                      persistencias=(1, 3, 5)) -> List[Dict[str, Any]]:
    """
    Rastrear e ficar quieto são objetivos OPOSTOS, e `deriva` é o botão entre
    os dois. Escolher o valor que faz as quatro provas passarem seria ajustar
    ao gabarito — a mesma armadilha que já apareceu três vezes nesta sessão.

    Então a saída é a curva inteira. Quem lê escolhe o ponto de operação
    sabendo o que está comprando, e o valor usado no resto do módulo passa a
    ser uma decisão declarada em vez de um número que passou no teste.
    """
    linhas = []
    for d in derivas:
        for pers in persistencias:
            l = prova_latencia(n_execucoes=200, deriva=d, persistencia=pers)
            f = prova_falso_alarme(n_execucoes=200, deriva=d, persistencia=pers)
            linhas.append({'deriva': d, 'persistencia': pers,
                           'latencia': l['ate_revisada'], 'deteccao': l['taxa_deteccao'],
                           'falso_alarme': f['taxa']})
    return linhas


def relatorio_compromisso() -> List[Dict[str, Any]]:
    print("═" * 78)
    print("CURVA DO COMPROMISSO — rastrear vs. ficar quieto")
    print("deriva alta segue o mundo e treme; deriva zero é firme e nunca revisa")
    print("═" * 78)
    print(f"{'deriva':>8}{'persist.':>10}{'latência':>11}{'detecção':>11}{'falso alarme':>15}")
    linhas = curva_compromisso()
    for l in linhas:
        lat = f"{l['latencia']:.0f}" if l['latencia'] == l['latencia'] else "nunca"
        print(f"{l['deriva']:>8.3f}{l['persistencia']:>10}{lat:>11}"
              f"{l['deteccao']:>10.0%}{l['falso_alarme']:>14.0%}")
    print("─" * 78)
    bons = [l for l in linhas if l['falso_alarme'] < 0.05 and l['deteccao'] > 0.95]
    if bons:
        b = min(bons, key=lambda x: x['latencia'])
        print(f"melhor ponto com falso alarme < 5% e detecção > 95%:")
        print(f"  deriva {b['deriva']:.3f} · persistência {b['persistencia']} → "
              f"latência {b['latencia']:.0f} obs")
    else:
        print("NENHUM ponto satisfaz falso alarme < 5% E detecção > 95% ao mesmo tempo.")
        print("O compromisso é real e não há configuração que o dissolva — o que é uma")
        print("informação melhor que um ponto escolhido a dedo.")
    return linhas


# ══════════════════════════════════════════════ relatório
def relatorio() -> Dict[str, Any]:
    print("═" * 78)
    print("SEMÂNTICA DE MECANISMO — detectar a mudança SEM símbolo novo")
    print("═" * 78)

    d = prova_diagnosticidade()
    print(f"1. diagnosticidade ......... deriva {d['deriva']:.2e} em 500 obs não-informativas"
          f"   {'OK' if d['passou'] else 'FALHOU'}")

    l = prova_latencia()
    print(f"2. latência da detecção .... suspensa em {l['ate_suspensa']:.0f} obs · "
          f"revisada em {l['ate_revisada']:.0f}   {'OK' if l['passou'] else 'FALHOU'}")
    print(f"   detectou em {l['taxa_deteccao']:.0%} das execuções · "
          f"alarme antes da mudança {l['alarme_precoce']:.0%}")

    f = prova_falso_alarme()
    print(f"3. falso alarme (mundo estável, 220 obs) ..... {f['taxa']:.0%}   "
          f"{'OK' if f['passou'] else 'FALHOU'}    ideal < 5%")

    g = prova_gradual()
    print(f"4. mudança gradual ......... correlação {g['correlacao']:.3f} · "
          f"{g['permanencia_suspensa']:.0f} obs com as duas versões coexistindo")
    print(f"   convergiu ao final em {g['convergiu']:.0%} das execuções   "
          f"{'OK' if g['passou'] else 'FALHOU'}")

    rv = prova_reversao()
    print(f"5. reversão (o mundo volta) . {rv['taxa']:.0%} voltam em "
          f"{rv['custo_mediano']:.0f} obs   {'OK' if rv['passou'] else 'FALHOU'}")

    print("─" * 78)
    if l['passou'] and f['passou']:
        print(f"O sistema percebe que o próprio modelo preditivo está errado em "
              f"{l['ate_revisada']:.0f} observações,")
        print(f"sem nenhum símbolo novo e sem regra dizendo 'procure mudança de "
              f"semântica' —")
        print(f"e fica quieto por 220 observações num mundo estável ({f['taxa']:.0%} de falso alarme).")
        print()
        print("A diferença com a bancada anterior é de definição, não de ajuste:")
        print("  ANTES  desconhecido = o nome não está no vocabulário")
        print("  AGORA  desconhecido = minha previsão vem falhando neste canal")
        print("A segunda não precisa de catálogo, então não tem como ficar desatualizada.")
    else:
        print("Alguma das duas metades falhou. Detecção sem o controle de falso alarme")
        print("não vale nada: um detector que dispara sempre acerta toda mudança.")

    if g['passou']:
        print()
        print(f"Na mudança gradual a representação ACOMPANHA (correlação "
              f"{g['correlacao']:.2f}) e passa")
        print(f"{g['permanencia_suspensa']:.0f} observações com as duas versões concorrentes, em vez de")
        print("trocar uma aresta pela outra num degrau. A versão anterior fica no")
        print("histórico — o sistema revisa sem esquecer.")
    if rv['passou']:
        print()
        print(f"E a revisão anda para os DOIS lados: quando o mecanismo volta, o")
        print(f"sistema volta em {rv['custo_mediano']:.0f} observações. Revisão que só anda para um")
        print("lado é esquecimento com carimbo — a versão antiga tem de continuar")
        print("alcançável, não só arquivada.")
    return {'diagnosticidade': d, 'latencia': l, 'falso_alarme': f,
            'gradual': g, 'reversao': rv}


def demonstrar(semente: int = 3) -> None:
    """A trajetória de uma execução: a competência caindo a zero na virada."""
    rng = random.Random(semente)
    m = ModeloSemantico(tipo='causa')
    print("═" * 78)
    print("UMA EXECUÇÃO — 'causa' deixa de compor na observação 20")
    print("═" * 78)
    print(f"{'obs':>5}{'P(H1)':>9}{'P(H2)':>9}{'competência':>14}   estado")
    for k in range(1, 61):
        compoe = k <= 20
        ph1, ph2, obs = _observacao(compoe, True, 0.8, rng)
        m.observar(ph1, ph2, obs, 0.8)
        if k % 5 == 0 or k in (21, 22):
            marca = "  ← mecanismo muda aqui" if k == 20 else ""
            print(f"{k:>5}{m.p_h1:>9.3f}{1-m.p_h1:>9.3f}{m.competencia:>14.3f}   "
                  f"{m.estado}{marca}")
    print()
    print(m.decodificar())


if __name__ == "__main__":
    relatorio()
    print()
    demonstrar()
