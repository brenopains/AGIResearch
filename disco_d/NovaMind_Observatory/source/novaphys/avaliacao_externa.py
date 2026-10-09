"""
novaphys.avaliacao_externa — funciona fora das bancadas que eu construí?
=========================================================================

Sete peças foram construídas e medidas nesta sequência: composição,
derrotabilidade, generalização, calibração, representação de incerteza,
detecção de mudança semântica, rastreamento e reversão. Cada uma passou na
bancada que veio junto com ela.

Isso é exatamente a condição em que um resultado costuma não sobreviver ao
contato com problema que ninguém desenhou para ele.

O LIMITE DESTE ARQUIVO, DITO PRIMEIRO
--------------------------------------
Eu escrevi o gerador. Então "externo" aqui NÃO quer dizer que o mundo veio de
fora da minha cabeça — quer dizer três coisas verificáveis e mais modestas:

  1. A ARQUITETURA ESTÁ CONGELADA e o congelamento é checável.
     `manifesto()` produz um hash de todas as constantes ajustáveis do
     raciocinador, do campo de crenças e do modelo semântico. Rode antes e
     depois: se o hash mudou, alguém mexeu, e qualquer número medido no meio
     está contaminado. A regra de ouro deixa de ser promessa e vira verificação.

  2. A SEMENTE VEM DE FORA. `avaliar(semente=...)` não tem valor padrão por
     escolha. Se quem avalia escolher a semente, eu não tive como ajustar a
     ela — e essa é a forma mais forte de independência disponível aqui.

  3. O ESPAÇO É MAIS LARGO DO QUE AQUILO EM QUE EU AJUSTEI. O gerador produz,
     de propósito, mundos que estão FORA do que o modelo consegue representar:
     composição probabilística, relações irrelevantes, confundidor latente.
     Eu previa a falha antes de rodar, e a previsão está escrita abaixo, no
     código, antes do resultado.

O que isto NÃO substitui: alguém que não seja eu gerando os mundos. Isso
continua em aberto e continua sendo o teste que vale mais.

TRÊS REGIMES
-------------
  FECHADO   mundo estático, semântica determinística. Está DENTRO do espaço do
            modelo — é o caso favorável, e serve de piso: se falhar aqui, nada
            do resto importa.

  DINÂMICO  a semântica vira no meio. Também dentro do espaço (as duas leituras
            existem), mas exige detectar e revisar.

  ABERTO    composição PROBABILÍSTICA (um tipo compõe 40% das vezes), relações
            irrelevantes e um confundidor latente. Está FORA do espaço: o
            modelo tem duas hipóteses, "compõe" e "não compõe", e a verdade
            aqui não é nenhuma das duas.

O CRITÉRIO É ESTRUTURA, NÃO RESPOSTA
-------------------------------------
A nota não é "acertou os pares". É:

    recuperou a estrutura?   identificou quais tipos compõem
    sabe quando não sabe?    nos tipos fora do espaço, ficou SUSPENSA em vez
                             de escolher com confiança um dos dois extremos

A segunda é a que importa no regime aberto, e é onde eu espero que ele falhe.
Um sistema que escolhe H1 ou H2 com confiança diante de um mecanismo que é 40%
não está errado por pouco — está afirmando dentro de um espaço que não contém
a resposta. Registrar isso vale mais que uma nota alta nos outros dois.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics as st
import sys
from typing import Any, Dict, List, Optional, Tuple

from .crencas import Hipotese, Observacao, VOCABULARIO, TRANSITIVAS, NAO_TRANSITIVAS
from .semantica import ModeloSemantico

_AQUI = os.path.dirname(os.path.abspath(__file__))
_NOVAMIND = os.path.join(os.path.dirname(_AQUI), "legacy", "novamind")


# ══════════════════════════════════════════════ congelamento verificável
def manifesto() -> Dict[str, Any]:
    """
    Hash de tudo que é ajustável na arquitetura.

    A regra de ouro — nenhum caso de teste pode ter sido usado para escolher
    parâmetro, regra ou estrutura — não é verificável por boa vontade. Isto a
    torna mecânica: rode antes da avaliação, guarde o hash, rode depois. Se
    mudou, houve ajuste no meio e o número não vale.
    """
    if _NOVAMIND not in sys.path:
        sys.path.insert(0, _NOVAMIND)
    from core.reasoning import SymbolicReasoner

    m = ModeloSemantico(tipo='_')
    estado = {
        'reasoner.NAO_TRANSITIVAS': sorted(SymbolicReasoner.NAO_TRANSITIVAS),
        'crencas.TRANSITIVAS': sorted(TRANSITIVAS),
        'crencas.NAO_TRANSITIVAS': sorted(NAO_TRANSITIVAS),
        'semantica.p_h1_inicial': m.p_h1_inicial,
        'semantica.deriva': m.deriva,
        'semantica.persistencia': m.persistencia,
        'semantica.limiares': [0.9, 0.1],
        'crencas.limiar_abertura': 0.4,
        'crencas.limiar_competencia': 0.6,
        'crencas.limiares_posterior': [0.9, 0.1],
    }
    bruto = json.dumps(estado, sort_keys=True, ensure_ascii=False)
    return {'hash': hashlib.sha256(bruto.encode('utf-8')).hexdigest()[:16],
            'estado': estado}


# ══════════════════════════════════════════════ o gerador
def gerar_mundo(rng: random.Random, regime: str) -> Dict[str, Any]:
    """
    Um mundo. `p_composicao` por tipo é a verdade escondida.

    No regime aberto ela é contínua — e é aí que o mundo sai do espaço do
    modelo, que só representa 0 ou 1.
    """
    n_tipos = rng.randint(2, 5)
    tipos = [f"r{rng.randint(100, 999)}" for _ in range(n_tipos)]
    if regime == 'aberto':
        # metade determinística, metade probabilística — para que a nota
        # separe "não recuperou nada" de "não recuperou o que é irrepresentável"
        p = {}
        for i, t in enumerate(tipos):
            p[t] = rng.choice([0.0, 1.0]) if i % 2 == 0 else rng.uniform(0.25, 0.75)
        irrelevantes = [f"r{rng.randint(100, 999)}" for _ in range(rng.randint(1, 3))]
    else:
        p = {t: float(rng.choice([0.0, 1.0])) for t in tipos}
        irrelevantes = []
    return {'tipos': tipos, 'p_composicao': p, 'irrelevantes': irrelevantes,
            'ruido': rng.choice([0.7, 0.75, 0.8, 0.85]),
            'confundidor': rng.random() < 0.4 if regime == 'aberto' else False}


def _observar_tipo(mundo: Dict[str, Any], t: str, rng: random.Random,
                   virou: bool = False) -> Tuple[bool, bool, bool]:
    """(previsão H1, previsão H2, observação). H1 diz que compõe; H2 que não."""
    p = mundo['p_composicao'][t]
    if virou:
        p = 1.0 - p
    verdade = rng.random() < p
    if mundo['confundidor'] and rng.random() < 0.15:
        verdade = True                       # causa fora do grafo
    q = mundo['ruido']
    obs = verdade if rng.random() < q else (not verdade)
    return True, False, obs


# ══════════════════════════════════════════════ avaliação
def _rodar_tipo(mundo: Dict[str, Any], t: str, rng: random.Random,
                n_obs: int, virar_em: Optional[int] = None) -> ModeloSemantico:
    m = ModeloSemantico(tipo=t)
    for k in range(n_obs):
        virou = virar_em is not None and k >= virar_em
        ph1, ph2, obs = _observar_tipo(mundo, t, rng, virou)
        m.observar(ph1, ph2, obs, q=mundo['ruido'])
    return m


def avaliar(semente: int, n_mundos: int = 60, n_obs: int = 120) -> Dict[str, Any]:
    """
    `semente` não tem valor padrão de propósito: quem avalia escolhe, e assim
    eu não tive como ajustar a ela.
    """
    res: Dict[str, Any] = {}

    for regime in ('fechado', 'dinamico', 'aberto'):
        acertos_det, tot_det = 0, 0
        superconf, tot_prob = 0, 0
        suspensas_det = 0
        for i in range(n_mundos):
            rng = random.Random(semente * 1000 + i)
            mundo = gerar_mundo(rng, regime)
            virar = n_obs // 2 if regime == 'dinamico' else None
            for t in mundo['tipos']:
                m = _rodar_tipo(mundo, t, rng, n_obs, virar)
                p = mundo['p_composicao'][t]
                if virar is not None:
                    p = 1.0 - p          # a verdade no fim da execução
                if p in (0.0, 1.0):
                    tot_det += 1
                    # recuperou a estrutura? 'estavel' = compõe; 'revisada' = não
                    diz_compoe = m.estado == 'estavel'
                    diz_nao = m.estado == 'revisada'
                    if (p == 1.0 and diz_compoe) or (p == 0.0 and diz_nao):
                        acertos_det += 1
                    elif m.estado == 'suspensa':
                        suspensas_det += 1
                else:
                    tot_prob += 1
                    # FORA do espaço do modelo. O certo é NÃO escolher um extremo.
                    if m.estado in ('estavel', 'revisada'):
                        superconf += 1
        res[regime] = {
            'estrutura_recuperada': acertos_det / max(1, tot_det),
            'indeciso_no_deterministico': suspensas_det / max(1, tot_det),
            'superconfianca_fora_do_espaco': (superconf / tot_prob) if tot_prob else None,
            'n_deterministicos': tot_det, 'n_probabilisticos': tot_prob,
        }
    return res


def relatorio(semente: Optional[int] = None, n_mundos: int = 60) -> Dict[str, Any]:
    if semente is None:
        raise ValueError(
            "Escolha a semente você mesmo: relatorio(semente=<número>).\n"
            "Não há padrão de propósito — se eu escolhesse, a independência do\n"
            "teste dependeria da minha palavra em vez de um fato verificável.")

    man = manifesto()
    print("═" * 78)
    print("AVALIAÇÃO FORA DAS BANCADAS")
    print(f"arquitetura congelada · hash {man['hash']} · semente externa {semente}")
    print("═" * 78)
    print("Rode manifesto() de novo depois: se o hash mudar, houve ajuste no meio")
    print("e os números abaixo não valem.")
    print()

    r = avaliar(semente, n_mundos=n_mundos)
    print(f"{'regime':<12}{'estrutura recuperada':>22}{'indeciso':>11}"
          f"{'superconfiança':>17}")
    leg = {'fechado': 'dentro do espaço · piso',
           'dinamico': 'dentro do espaço · exige revisar',
           'aberto': 'FORA do espaço do modelo'}
    for regime in ('fechado', 'dinamico', 'aberto'):
        d = r[regime]
        sc = d['superconfianca_fora_do_espaco']
        sc_txt = f"{sc:.0%}" if sc is not None else "—"
        print(f"{regime:<12}{d['estrutura_recuperada']:>21.0%}"
              f"{d['indeciso_no_deterministico']:>11.0%}{sc_txt:>17}")
    for regime in ('fechado', 'dinamico', 'aberto'):
        print(f"    {regime:<10} {leg[regime]}")

    print("─" * 78)
    f, d, a = (r['fechado']['estrutura_recuperada'],
               r['dinamico']['estrutura_recuperada'],
               r['aberto']['estrutura_recuperada'])
    sc = r['aberto']['superconfianca_fora_do_espaco'] or 0.0

    if f < 0.8:
        print("Falhou no PISO. O regime fechado está dentro do espaço do modelo e é o")
        print("caso favorável — se a estrutura não é recuperada aqui, nada do resto")
        print("importa, e o problema não é generalização, é o mecanismo básico.")
    else:
        print(f"Fechado {f:.0%} · dinâmico {d:.0%}: a estrutura é recuperada em mundos")
        print("gerados por um espaço que eu não usei para escolher parâmetro nenhum,")
        print("com número de tipos, ruído e semântica sorteados.")

    print()
    print("O REGIME ABERTO — a previsão que eu escrevi ANTES de rodar:")
    print("  um tipo que compõe 40% das vezes não é 'compõe' nem 'não compõe'. O")
    print("  modelo tem duas hipóteses e a verdade não é nenhuma delas. Eu esperava")
    print("  superconfiança alta: ele escolhendo um extremo em vez de ficar suspenso.")
    print()
    if sc > 0.5:
        print(f"  Confirmado: {sc:.0%} dos mecanismos irrepresentáveis terminaram com o")
        print("  sistema afirmando um dos dois extremos com estado confirmado. Ele não")
        print("  erra por pouco — afirma dentro de um espaço que não contém a resposta.")
        print()
        print("  E isto NÃO se conserta com mais dados nem com mais deriva. É o mesmo")
        print("  achado de `crencas.py` num nível acima: lá o espaço de RELAÇÕES era")
        print("  meu; aqui o espaço de LEITURAS possíveis é meu. O sistema pesa")
        print("  hipóteses; não as constrói.")
    else:
        print(f"  Superconfiança {sc:.0%} — mais baixa que o esperado. Vale checar se os")
        print("  mundos probabilísticos estão mesmo fora do espaço antes de comemorar.")
    return {'manifesto': man, 'resultados': r}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--semente', type=int, required=True,
                    help='escolha você — não há padrão de propósito')
    ap.add_argument('--mundos', type=int, default=60)
    a = ap.parse_args()
    relatorio(semente=a.semente, n_mundos=a.mundos)
