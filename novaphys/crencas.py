"""
novaphys.crencas — campo de crenças com três incertezas separadas
==================================================================

A bancada de calibração terminou com um caso limpo: a relação
`varia_junto_com` produziu 0% de acerto com confiança 0,850, sem truncamento e
sem regra ilegível. As duas explicações anteriores estavam eliminadas — o
buraco era de REPRESENTAÇÃO. O motor não sabia que não sabia.

Este módulo não tenta melhorar a confiança. Tenta uma coisa anterior e mais
difícil de fingir: **representar corretamente a própria incerteza**.

TRÊS COISAS QUE UM NÚMERO SÓ NÃO CONSEGUE DIZER
------------------------------------------------
Um campo escalar "confiança 0,85" mistura três perguntas diferentes, e as
decisões que se seguem a cada uma são opostas:

  1. p(H)          A hipótese é verdadeira?
                   "A → B provavelmente vale."
                   Pouca evidência ⟹ COLETE MAIS EVIDÊNCIA.

  2. competência   Eu sei avaliar este TIPO de relação?
                   "Sei o que fazer com `causa`; nunca vi `varia_junto_com`."
                   Baixa ⟹ NÃO CONFIE NA CONTA, teste o mecanismo.

  3. abertura      Existe explicação que eu nem estou representando?
                   Alta ⟹ EXPANDA O ESPAÇO DE HIPÓTESES.

O `varia_junto_com` era (3) disfarçado de (1). O sistema tinha evidência
abundante e a conta interna estava certa — errado era o espaço em que a conta
rodava. Nenhum volume de evidência conserta isso, e é por isso que a terceira
tem de ser um número separado: ela pede uma AÇÃO diferente.

A ATUALIZAÇÃO É BAYESIANA, E ISSO É O QUE A TORNA VERIFICÁVEL
--------------------------------------------------------------
    prior → observação → verossimilhança → posterior → decisão

Em log-odds, para não perder precisão em cadeias longas de evidência. A
competência entra como EXPOENTE da razão de verossimilhança:

    log_odds += competencia · log(LR)

Isto é a verossimilhança temperada (power likelihood), e a escolha tem
conteúdo: com competência 1 a atualização é a de Bayes exata, e com
competência 0 a evidência não move nada. A leitura é direta — evidência lida
por um mecanismo que talvez esteja errado vale menos que evidência lida por um
mecanismo que se sabe correto, e vale zero se o mecanismo é desconhecido.

A prova 1 desta bancada confere esse cálculo contra a forma fechada. Se a
aritmética estiver errada, nenhuma das outras significa coisa alguma — e essa
ordem, arnês antes de capacidade, é a que se manteve na sessão inteira.

O QUE ESTA BANCADA MEDE
------------------------
  1. ARITMÉTICA     o posterior bate com o produto analítico das razões?
  2. CALIBRAÇÃO     posterior 0,7 quer dizer verdade em 70% dos casos?
  3. CONVERGÊNCIA   com evidência consistente, ele fecha?
  4. ABERTURA       com evidência ambígua, ele NÃO fecha?
  5. RECUPERAÇÃO    depois de evidência enganosa, a hipótese certa volta?
  6. INSUFICIÊNCIA  detecta representação insuficiente SEM gritar em tudo?

A prova 6 é a que interessa, e é de dois lados de propósito. Um sistema que
marca "não sei" em toda pergunta acerta 100% da detecção e é inútil. Por isso
a nota é o par (detecção, falso alarme), nunca a detecção sozinha — e há uma
terceira célula que separa o que costuma ser confundido: evidência fraca com
vocabulário conhecido deve dar ABERTA (colete mais dados), não SUSPENSA (o
mecanismo está furado). São diagnósticos diferentes que pedem ações diferentes.
"""

from __future__ import annotations

import math
import random
import statistics as st
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# ══════════════════════════════════════════════ vocabulário de relações
# O que o motor SABE avaliar. A divisão não é cosmética: é a lista contra a
# qual a competência é medida, e qualquer coisa fora dela é novidade
# epistemológica, não um caso a mais.
TRANSITIVAS = {'causa', 'leva_a', 'implica', 'requer', 'produz', 'precede',
               'causes', 'leads_to', 'implies', 'requires', 'produces'}
NAO_TRANSITIVAS = {'correlaciona_com', 'correlates_with', 'associado_a',
                   'associated_with', 'similar_a', 'similar_to', 'co_ocorre',
                   'proximo_de', 'contrasta_com'}
VOCABULARIO = TRANSITIVAS | NAO_TRANSITIVAS


def _sig(x: float) -> float:
    if x > 40:
        return 1.0
    if x < -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


@dataclass
class Observacao:
    """Uma evidência. `lr` é a razão de verossimilhança P(obs|H)/P(obs|¬H)."""
    lr: float
    relacao: str = 'causa'
    rotulo: str = ''
    experimento: int = -1


@dataclass
class Hipotese:
    """
    Uma hipótese e tudo o que sustenta ou mina a crença nela.

    O histórico de versões não é luxo de auditoria: sem ele não dá para provar
    RECUPERAÇÃO (prova 5), porque a pergunta lá é sobre a trajetória do
    posterior, e não sobre onde ele parou.
    """
    afirmacao: Tuple[str, str]
    prior: float = 0.5
    _log_odds: float = 0.0
    evid_favor: List[Observacao] = field(default_factory=list)
    evid_contra: List[Observacao] = field(default_factory=list)
    relacoes_usadas: Set[str] = field(default_factory=set)
    relacoes_desconhecidas: Set[str] = field(default_factory=set)
    dependencias: List[Tuple[str, str]] = field(default_factory=list)
    historico: List[Dict[str, Any]] = field(default_factory=list)
    dominio: str = ''
    _n_obs: int = 0
    _n_conhecidas: int = 0

    def __post_init__(self):
        self._log_odds = _logit(self.prior)
        self.historico.append({'n': 0, 'posterior': self.prior, 'evento': 'prior'})

    # ── as três incertezas ────────────────────────────────────────────────
    @property
    def posterior(self) -> float:
        return _sig(self._log_odds)

    @property
    def competencia(self) -> float:
        """
        Fração das OBSERVAÇÕES que chegaram por uma relação do vocabulário.

        Contam-se observações, não tipos distintos. A primeira versão contava
        tipos e suavizava com pseudo-contagem — o que travava a competência em
        0,75 mesmo com tudo conhecido, e portanto o campo NUNCA fazia Bayes
        exato. A prova 1 pegou: erro máximo de 1,86e-01 contra a forma fechada.
        Um posterior sistematicamente amortecido não é um posterior, e aí
        "calibrado" deixa de querer dizer alguma coisa.

        Sem observação nenhuma: 0,5. Não se presume competência por omissão —
        foi exatamente essa presunção (tipo desconhecido = transitivo) que
        produziu 0,850 de confiança sobre 0% de acerto na bancada anterior.

        Competência 0 (tudo desconhecido) significa que a evidência não move o
        posterior. É o comportamento certo, e é forte de propósito: se não sei
        ler a relação, não sei sequer em que DIREÇÃO a observação aponta.
        Muita evidência ilegível não é muita informação — é nenhuma, e o campo
        tem de terminar dizendo isso em vez de terminar confiante.
        """
        if self._n_obs == 0:
            return 0.5
        return self._n_conhecidas / self._n_obs

    @property
    def abertura(self) -> float:
        """
        Massa de probabilidade em "a explicação certa não está representada".

        Sobe com relação desconhecida (o mecanismo pode estar lendo o mundo
        errado) e com conflito não resolvido (a evidência briga entre si, o que
        costuma indicar uma variável que ninguém modelou). Os dois canais são
        combinados como probabilidades independentes de haver buraco.
        """
        p_vocab = 1.0 - math.prod(
            [0.35] * len(self.relacoes_desconhecidas)) if self.relacoes_desconhecidas else 0.0
        n_f, n_c = len(self.evid_favor), len(self.evid_contra)
        if n_f + n_c >= 4 and min(n_f, n_c) / max(1, n_f + n_c) > 0.35:
            p_conf = min(0.6, min(n_f, n_c) / (n_f + n_c))
        else:
            p_conf = 0.0
        return 1.0 - (1.0 - p_vocab) * (1.0 - p_conf)

    # ── atualização ───────────────────────────────────────────────────────
    def observar(self, obs: Observacao) -> None:
        self.relacoes_usadas.add(obs.relacao)
        self._n_obs += 1
        if obs.relacao in VOCABULARIO:
            self._n_conhecidas += 1
        else:
            self.relacoes_desconhecidas.add(obs.relacao)
        (self.evid_favor if obs.lr > 1.0 else self.evid_contra).append(obs)
        # A competência é recalculada ANTES do uso: uma relação desconhecida
        # tem de descontar a própria observação que a introduziu, senão a
        # primeira evidência de um mecanismo novo entra com peso cheio.
        self._log_odds += self.competencia * math.log(max(obs.lr, 1e-12))
        self.historico.append({'n': len(self.historico), 'posterior': self.posterior,
                               'evento': obs.rotulo or f'lr={obs.lr:.3g}',
                               'competencia': self.competencia})

    # ── leitura ───────────────────────────────────────────────────────────
    @property
    def estado(self) -> str:
        """
        O diagnóstico, e cada um pede uma AÇÃO diferente:

          SUSPENSA      o mecanismo pode estar errado  → teste o MECANISMO
          CRISTALIZADA  evidência forte, vocabulário conhecido → pode usar
          REFUTADA      evidência forte contra → descarte
          ABERTA        falta evidência → colete MAIS DADOS

        A ordem importa: SUSPENSA vem primeiro porque um posterior alto obtido
        por um mecanismo suspeito não é conhecimento, é um número.
        """
        if self.abertura > 0.4:
            return 'suspensa'
        if self.competencia < 0.6:
            return 'suspensa'
        p = self.posterior
        if p > 0.9:
            return 'cristalizada'
        if p < 0.1:
            return 'refutada'
        return 'aberta'

    def decodificar(self) -> str:
        """
        O decoder não decodifica só conteúdo — decodifica o ESTADO da própria
        representação. É esta string que alimentaria a escolha do próximo
        experimento.
        """
        a, b = self.afirmacao
        linhas = [
            f"Hipótese  {a} ⇝ {b}",
            f"  posterior ............... {self.posterior:.3f}",
            f"  competência ............. {self.competencia:.3f}",
            f"  abertura ................ {self.abertura:.3f}",
            f"  evidência ............... {len(self.evid_favor)} a favor, "
            f"{len(self.evid_contra)} contra",
            f"  relações usadas ......... {sorted(self.relacoes_usadas) or '—'}",
        ]
        if self.relacoes_desconhecidas:
            linhas.append(f"  RELAÇÃO DESCONHECIDA .... {sorted(self.relacoes_desconhecidas)}")
            linhas.append(f"     ↳ encontrei relação fora do vocabulário causal conhecido;")
            linhas.append(f"       a conta rodou num espaço que pode ser o errado.")
        if self.dependencias:
            cadeia = [self.dependencias[0][0]]
            for (o, d) in self.dependencias:
                if cadeia[-1] != o:
                    cadeia.append(o)
                cadeia.append(d)
            linhas.append(f"  dependências ............ {' → '.join(cadeia)}")
        if self.dominio:
            linhas.append(f"  domínio ................. {self.dominio}")
        linhas.append(f"  ESTADO .................. {self.estado.upper()}")
        linhas.append(f"  ação sugerida ........... {ACOES[self.estado]}")
        return "\n".join(linhas)


ACOES = {
    'suspensa': 'testar o MECANISMO (evidência não resolve isto)',
    'aberta': 'coletar mais evidência',
    'cristalizada': 'pode usar; reabrir se surgir contra-evidência',
    'refutada': 'descartar',
}


class CampoDeCrencas:
    """Um conjunto de hipóteses concorrentes sobre a mesma pergunta."""

    def __init__(self, priors: Optional[Dict[Tuple[str, str], float]] = None):
        self.hipoteses: Dict[Tuple[str, str], Hipotese] = {}
        for af, p in (priors or {}).items():
            self.hipoteses[af] = Hipotese(afirmacao=af, prior=p)

    def adicionar(self, h: Hipotese) -> None:
        self.hipoteses[h.afirmacao] = h

    def observar(self, afirmacao: Tuple[str, str], obs: Observacao) -> None:
        if afirmacao not in self.hipoteses:
            self.hipoteses[afirmacao] = Hipotese(afirmacao=afirmacao)
        self.hipoteses[afirmacao].observar(obs)

    def campo(self) -> List[Tuple[Tuple[str, str], float]]:
        """As hipóteses normalizadas — o 'campo' propriamente dito."""
        ps = {k: h.posterior for k, h in self.hipoteses.items()}
        s = sum(ps.values()) or 1.0
        return sorted(((k, v / s) for k, v in ps.items()), key=lambda x: -x[1])

    def desenhar(self, largura: int = 40) -> str:
        out = []
        for (af, p) in self.campo():
            barra = "─" * max(1, int(round(p * largura)))
            out.append(f"  {af[0]}⇝{af[1]:<10} {barra} {p:.2f}")
        return "\n".join(out)

    @property
    def pergunta_aberta(self) -> Optional[Hipotese]:
        """A hipótese cujo estado mais pede um experimento — o gancho do loop."""
        susp = [h for h in self.hipoteses.values() if h.estado == 'suspensa']
        if susp:
            return max(susp, key=lambda h: h.abertura)
        abertas = [h for h in self.hipoteses.values() if h.estado == 'aberta']
        if abertas:
            return min(abertas, key=lambda h: abs(h.posterior - 0.5))
        return None
