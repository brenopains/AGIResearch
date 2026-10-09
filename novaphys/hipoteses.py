"""
novaphys.hipoteses — construir a explicação, não escolher entre as prontas
==========================================================================

Diagnóstico que trouxe até aqui, medido três vezes em condições independentes
(gerador sintético meu, gerador com semente externa, física real do simulador):

    O sistema pesa hipóteses. Não as constrói.

Diante de uma dependência com pico, o modelo tinha duas gavetas — "cresce" e
"decresce" — e a resposta certa não estava em nenhuma. Superconfiança de 89%
em física real.

A ARMADILHA QUE ESTE MÓDULO PRECISA EVITAR
-------------------------------------------
Dar ao gerador uma biblioteca de formas possíveis não resolve nada: troca
"escolha entre H1 e H2" por "escolha entre H1, H2, H3, H4" e o sistema continua
sem construir coisa alguma. Só empurra a parede.

Então aqui não há catálogo de formas. Há **um operador**, aplicado
recursivamente:

    PARTIR — o domínio pode ter mais de um regime

Só isso. O mecanismo não sabe o que é um pico, um limiar, uma ressonância ou um
ótimo. Ele sabe que a estrutura atual erra e que a menor mudança concebível é
admitir que a regra não vale no domínio inteiro. Aplicado a uma curva com pico,
o resultado que EMERGE é

    região 1: cresce   ·   região 2: decresce

que é "existe um regime ótimo" — sem que ninguém tenha escrito "ótimo" em lugar
nenhum. Aplicado duas vezes, dá três regimes. O espaço é gerado, não enumerado,
e é infinito: é essa a diferença entre construir e escolher.

O QUE IMPEDE DE VIRAR MÁQUINA DE DESCULPAS
-------------------------------------------
Um mecanismo que parte o domínio sempre que erra explica qualquer coisa: com
cortes suficientes, ruído puro vira dez regimes. Isso não é hipótese, é
decoração de dados.

O critério de aceitação é COMPRIMENTO DE DESCRIÇÃO (BIC): a extensão só entra
se a melhora na verossimilhança pagar o parâmetro extra que ela custa. É a
formalização de "menor extensão estrutural necessária" — necessária tem de ser
uma conta, não uma opinião.

E DEPOIS A HIPÓTESE TEM DE SER ATACADA
---------------------------------------
Gerar não é acertar. Depois de proposta, a estrutura nova é usada para prever
uma consequência que as antigas NÃO preveem, e o sistema escolhe a intervenção
que mais separa as duas — não a que mais a confirma.

    anomalia → gerar → prever → experimento discriminante → confirmar ou matar

A métrica é **taxa de hipóteses novas que sobrevivem a teste adversarial**.
Gerar quinhentas hipóteses absurdas não é inteligência; gerar uma, desenhar o
experimento que poderia matá-la, e ela sobreviver, é.

O CONTROLE NEGATIVO, QUE É O QUE FAZ O NÚMERO VALER
-----------------------------------------------------
Metade dos casos de teste é monotônica de verdade, com ruído suficiente para
PARECER que tem pico. Ali a estrutura nova tem de ser rejeitada. Um gerador que
propõe regime novo em tudo passa na métrica de geração e reprova aqui.
"""

from __future__ import annotations

import math
import random
import statistics as st
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Observacao = Tuple[float, float, bool]      # (x1, x2, y subiu de x1 para x2)


# ══════════════════════════════════════════════ a estrutura
@dataclass
class Estrutura:
    """
    Uma explicação: onde o domínio se parte, e o sentido em cada parte.

    `cortes=[]` com `direcoes=[+1]` é "cresce sempre" — a hipótese original.
    `cortes=[x*]` com `direcoes=[+1,-1]` é "sobe até x*, depois desce", que
    ninguém escreveu: saiu de aplicar PARTIR uma vez.
    """
    cortes: List[float] = field(default_factory=list)
    direcoes: List[int] = field(default_factory=lambda: [1])

    @property
    def n_parametros(self) -> int:
        return len(self.direcoes) + len(self.cortes)

    def delta_previsto(self, x1: float, x2: float) -> float:
        """
        Soma dos comprimentos com sinal em cada região entre x1 e x2.

        Trata naturalmente a observação que ATRAVESSA um corte, que é o caso
        mais informativo e o que uma classificação global não consegue
        representar: subir 3 e descer 3 dá zero, e o sinal disso é real.
        """
        bordas = [-math.inf] + list(self.cortes) + [math.inf]
        total = 0.0
        for i, d in enumerate(self.direcoes):
            lo, hi = bordas[i], bordas[i + 1]
            a, b = max(x1, lo), min(x2, hi)
            if b > a:
                total += d * (b - a)
        return total

    def preve_subida(self, x1: float, x2: float) -> Optional[bool]:
        dl = self.delta_previsto(x1, x2)
        if abs(dl) < 1e-12:
            return None                      # a estrutura não se compromete
        return dl > 0

    def log_verossimilhanca(self, obs: Sequence[Observacao], q: float = 0.8) -> float:
        ll = 0.0
        for (x1, x2, subiu) in obs:
            p = self.preve_subida(x1, x2)
            if p is None:
                ll += math.log(0.5)          # não prevê: nem crédito, nem culpa
            else:
                ll += math.log(q if p == subiu else 1 - q)
        return ll

    def bic(self, obs: Sequence[Observacao], q: float = 0.8) -> float:
        n = max(1, len(obs))
        return -2.0 * self.log_verossimilhanca(obs, q) + self.n_parametros * math.log(n)

    def descrever(self) -> str:
        if not self.cortes:
            return "cresce em todo o domínio" if self.direcoes[0] > 0 \
                   else "decresce em todo o domínio"
        partes = []
        bordas = ['—'] + [f"{c:.3g}" for c in self.cortes] + ['—']
        for i, d in enumerate(self.direcoes):
            partes.append(f"[{bordas[i]}, {bordas[i+1]}] {'sobe' if d > 0 else 'desce'}")
        return " · ".join(partes)

    def resumo_fenomeno(self) -> str:
        """
        Como um humano leria a estrutura. NÃO é usado no mecanismo — é só
        tradução para a saída. A distinção importa: se as palavras "pico" ou
        "limiar" participassem da busca, elas seriam a biblioteca de formas que
        este módulo existe para não ter.
        """
        d = self.direcoes
        if len(d) == 1:
            return "monotônica"
        if len(d) == 2 and d[0] > 0 > d[1]:
            return f"há um regime ótimo perto de {self.cortes[0]:.3g}"
        if len(d) == 2 and d[0] < 0 < d[1]:
            return f"há um mínimo perto de {self.cortes[0]:.3g}"
        return f"{len(d)} regimes distintos"


# ══════════════════════════════════════════════ a geração
def _melhor_com_k_cortes(obs: Sequence[Observacao], k: int, q: float,
                         candidatos: Sequence[float]) -> Optional[Estrutura]:
    """
    Busca exaustiva sobre posições de corte tiradas DOS DADOS e sobre os sinais
    das regiões. Nada aqui vem de uma lista de formas conhecidas: os cortes são
    valores de x que foram observados, e os sinais são ±1.
    """
    if k == 0:
        return min((Estrutura([], [s]) for s in (1, -1)),
                   key=lambda e: e.bic(obs, q))
    melhor, melhor_bic = None, math.inf
    if k == 1:
        for c in candidatos:
            for s1 in (1, -1):
                e = Estrutura([c], [s1, -s1])
                b = e.bic(obs, q)
                if b < melhor_bic:
                    melhor, melhor_bic = e, b
    elif k == 2:
        cs = sorted(set(candidatos))
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                for s1 in (1, -1):
                    e = Estrutura([cs[i], cs[j]], [s1, -s1, s1])
                    b = e.bic(obs, q)
                    if b < melhor_bic:
                        melhor, melhor_bic = e, b
    return melhor


def gerar(obs: Sequence[Observacao], q: float = 0.8,
          max_cortes: int = 2, n_candidatos: int = 24) -> Dict[str, Any]:
    """
    Aplica PARTIR enquanto o comprimento de descrição melhorar.

    Devolve a estrutura escolhida, a melhor estrutura SEM extensão (para o
    experimento discriminante) e o ganho de BIC — que é a medida de "a extensão
    foi necessária", em vez de "a extensão ajustou melhor". Qualquer extensão
    ajusta melhor; é para isso que serve a penalidade.
    """
    xs = sorted({x for (a, b, _) in obs for x in (a, b)})
    if len(xs) < 3:
        base = _melhor_com_k_cortes(obs, 0, q, xs)
        return {'estrutura': base, 'base': base, 'ganho_bic': 0.0, 'cortes': 0}
    lo, hi = xs[0], xs[-1]
    cands = [lo + (hi - lo) * (i + 1) / (n_candidatos + 1) for i in range(n_candidatos)]

    base = _melhor_com_k_cortes(obs, 0, q, cands)
    melhor, bic_melhor, k_melhor = base, base.bic(obs, q), 0
    for k in range(1, max_cortes + 1):
        e = _melhor_com_k_cortes(obs, k, q, cands)
        if e is None:
            continue
        b = e.bic(obs, q)
        if b < bic_melhor - 1e-9:
            melhor, bic_melhor, k_melhor = e, b, k
    return {'estrutura': melhor, 'base': base, 'cortes': k_melhor,
            'ganho_bic': base.bic(obs, q) - bic_melhor}


# ══════════════════════════════════════════════ a falsificação
def experimento_discriminante(nova: Estrutura, base: Estrutura,
                              lo: float, hi: float, n: int = 400,
                              rng: Optional[random.Random] = None) -> Optional[Tuple[float, float]]:
    """
    A intervenção em que as duas estruturas preveem COISAS OPOSTAS.

    Não é o experimento que mais confirma a hipótese nova — é o que mais a
    expõe. Se `nova` estiver errada, é aqui que ela morre, e é por isso que a
    métrica que interessa é quantas SOBREVIVEM a isto.
    """
    rng = rng or random.Random(0)
    for _ in range(n):
        x1, x2 = sorted((rng.uniform(lo, hi), rng.uniform(lo, hi)))
        if x2 - x1 < 0.05 * (hi - lo):
            continue
        pn, pb = nova.preve_subida(x1, x2), base.preve_subida(x1, x2)
        if pn is not None and pb is not None and pn != pb:
            return (x1, x2)
    return None


def falsificar(nova: Estrutura, base: Estrutura, lo: float, hi: float,
               oraculo: Callable[[float, float], bool], n_testes: int = 8,
               rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """
    Roda os experimentos discriminantes e conta de que lado a natureza ficou.

    Sobrevive se a maioria dos testes apontar para a estrutura nova. Note que o
    critério é sobre experimentos ESCOLHIDOS PARA SEPARAR — acertar num
    experimento qualquer é fácil; acertar onde as hipóteses divergem, não.
    """
    rng = rng or random.Random(0)
    a_favor = contra = 0
    usados = []
    for _ in range(n_testes):
        ex = experimento_discriminante(nova, base, lo, hi, rng=rng)
        if ex is None:
            break
        x1, x2 = ex
        real = oraculo(x1, x2)
        if nova.preve_subida(x1, x2) == real:
            a_favor += 1
        else:
            contra += 1
        usados.append((x1, x2, real))
    total = a_favor + contra
    return {'sobreviveu': total > 0 and a_favor > contra,
            'a_favor': a_favor, 'contra': contra, 'n': total, 'testes': usados}


# ══════════════════════════════════════════════ o ciclo completo
def ciclo(obs: Sequence[Observacao], lo: float, hi: float,
          oraculo: Callable[[float, float], bool], q: float = 0.8,
          semente: int = 0) -> Dict[str, Any]:
    """
    anomalia → gerar → prever → experimento discriminante → confirmar ou matar
    """
    rng = random.Random(semente)
    g = gerar(obs, q=q)
    nova, base = g['estrutura'], g['base']
    if g['cortes'] == 0:
        return {'propos': False, 'estrutura': nova, 'ganho_bic': 0.0,
                'sobreviveu': None, 'fenomeno': nova.resumo_fenomeno()}
    f = falsificar(nova, base, lo, hi, oraculo, rng=rng)
    return {'propos': True, 'estrutura': nova, 'base': base,
            'ganho_bic': g['ganho_bic'], 'cortes': g['cortes'],
            'sobreviveu': f['sobreviveu'], 'falsificacao': f,
            'fenomeno': nova.resumo_fenomeno()}
