"""
novaphys.hipoteses_bench — o gerador constrói, ou só decora dados?
===================================================================

Quatro medidas, e a segunda é a que impede as outras de serem enganosas:

    GERAÇÃO      diante de estrutura fora do espaço, propõe extensão?
    CONTENÇÃO    diante de estrutura DENTRO do espaço, se contém?
    FALSIFICAÇÃO as extensões propostas sobrevivem a experimento discriminante?
    RECUPERAÇÃO  a estrutura proposta corresponde à verdade (posição do regime)?

Sem CONTENÇÃO, "geração 100%" é o que um mecanismo que parte o domínio em tudo
marcaria — e ele não estaria construindo explicação, estaria decorando ruído.
Por isso metade dos casos é monotônica de verdade, com ruído suficiente para
PARECER que tem pico.

E a prova final é TRANSFERÊNCIA: rodar contra a física do simulador, em módulos
que não participaram de nenhuma decisão de desenho.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics as st
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .hipoteses import Estrutura, Observacao, ciclo, gerar


# ══════════════════════════════════════════════ mundos sintéticos
def _mundo(forma: str, lo: float, hi: float, rng: random.Random) -> Callable[[float], float]:
    meio = lo + (hi - lo) * rng.uniform(0.3, 0.7)
    if forma == 'crescente':
        return lambda x: x
    if forma == 'decrescente':
        return lambda x: -x
    if forma == 'pico':
        return lambda x: -abs(x - meio)
    if forma == 'vale':
        return lambda x: abs(x - meio)
    if forma == 'degrau':
        # DENTRO do espaço, e isto foi um erro meu de rotulação que a bancada
        # pegou: (0 se x<meio senão 1) + 0,001x nunca desce, ou seja, é
        # monotônica crescente. "Cresce em todo o domínio" a explica bem, e não
        # propor extensão aqui é o comportamento CERTO — eu estava contando
        # como falha de geração o que era acerto de contenção. Medido: geração
        # aparente de 64%, que virou 93% quando a rotulação foi corrigida.
        return lambda x: (0.0 if x < meio else 1.0) + 0.001 * x
    if forma == 'onda':
        # DOIS pontos de virada: exige aplicar PARTIR duas vezes. Está aqui
        # para que a bancada não meça só o caso de um corte — se o operador
        # fosse na verdade uma regra especial para "pico", falharia aqui.
        a = lo + (hi - lo) * 0.3
        b = lo + (hi - lo) * 0.7
        return lambda x: (x - lo) if x < a else ((a - lo) - (x - a) if x < b
                                                 else (a - lo) - (b - a) + (x - b))
    raise ValueError(forma)


def _amostrar(f: Callable[[float], float], lo: float, hi: float, n: int,
              q: float, rng: random.Random) -> List[Observacao]:
    obs = []
    for _ in range(n):
        x1, x2 = sorted((rng.uniform(lo, hi), rng.uniform(lo, hi)))
        if x2 - x1 < 0.08 * (hi - lo):
            x2 = min(hi, x1 + 0.25 * (hi - lo))
        real = f(x2) > f(x1)
        obs.append((x1, x2, real if rng.random() < q else (not real)))
    return obs


def _oraculo(f: Callable[[float], float], q: float,
             rng: random.Random) -> Callable[[float, float], bool]:
    def o(x1: float, x2: float) -> bool:
        real = f(x2) > f(x1)
        return real if rng.random() < q else (not real)
    return o


# ══════════════════════════════════════════════ avaliação sintética
def avaliar(n_casos: int = 120, n_obs: int = 40, q: float = 0.8,
            semente: int = 0) -> Dict[str, Any]:
    lo, hi = 0.0, 10.0
    fora = ['pico', 'vale', 'onda']         # não cabem em "cresce"/"decresce"
    dentro = ['crescente', 'decrescente', 'degrau']   # cabem (degrau é monotônico)

    gerou_fora = tot_fora = 0
    gerou_dentro = tot_dentro = 0
    sobreviveu = tot_prop = 0
    recuperou = tot_rec = 0

    for i in range(n_casos):
        rng = random.Random(semente * 7919 + i)
        eh_fora = i % 2 == 0
        forma = rng.choice(fora if eh_fora else dentro)
        f = _mundo(forma, lo, hi, rng)
        obs = _amostrar(f, lo, hi, n_obs, q, rng)
        r = ciclo(obs, lo, hi, _oraculo(f, q, rng), q=q, semente=semente + i)

        if eh_fora:
            tot_fora += 1
            gerou_fora += int(r['propos'])
        else:
            tot_dentro += 1
            gerou_dentro += int(r['propos'])

        if r['propos']:
            tot_prop += 1
            sobreviveu += int(bool(r['sobreviveu']))
            # recuperação: para pico/vale, o corte tem de cair perto do extremo
            if eh_fora and forma in ('pico', 'vale') and r['estrutura'].cortes:
                alvo = None
                # localiza o extremo verdadeiro por varredura densa
                xs = [lo + (hi - lo) * k / 200 for k in range(201)]
                ys = [f(x) for x in xs]
                alvo = xs[ys.index(max(ys) if forma == 'pico' else min(ys))]
                tot_rec += 1
                if abs(r['estrutura'].cortes[0] - alvo) < 0.12 * (hi - lo):
                    recuperou += 1

    return {
        'geracao': gerou_fora / max(1, tot_fora),
        'falso_positivo': gerou_dentro / max(1, tot_dentro),
        'sobrevivencia': sobreviveu / max(1, tot_prop),
        'recuperacao': recuperou / max(1, tot_rec),
        'n_fora': tot_fora, 'n_dentro': tot_dentro,
        'n_propostas': tot_prop, 'n_rec': tot_rec,
    }


def relatorio(n_casos: int = 120, sementes: Tuple[int, ...] = (0, 1, 2)) -> Dict[str, Any]:
    ls = [avaliar(n_casos=n_casos, semente=s) for s in sementes]
    m = lambda k: st.mean([l[k] for l in ls])

    print("═" * 78)
    print("GERAÇÃO DE HIPÓTESES — construir a explicação, não escolher entre prontas")
    print(f"{n_casos} casos × {len(sementes)} sementes · um operador (PARTIR), "
          f"aceitação por BIC")
    print("═" * 78)
    print(f"geração  (estrutura FORA do espaço → propõe extensão) ... "
          f"{m('geracao'):>5.0%}   ideal 100%")
    print(f"contenção (estrutura DENTRO → NÃO propõe) .............. "
          f"{1-m('falso_positivo'):>5.0%}   ideal 100%")
    print(f"sobrevivência a experimento discriminante ............. "
          f"{m('sobrevivencia'):>5.0%}   ideal 100%")
    print(f"recuperação (o corte cai no extremo verdadeiro) ....... "
          f"{m('recuperacao'):>5.0%}   ideal 100%")
    print("─" * 78)
    g, fp, sv, rc = m('geracao'), m('falso_positivo'), m('sobrevivencia'), m('recuperacao')
    if g > 0.8 and fp < 0.2:
        print("O mecanismo propõe estrutura nova onde a antiga não cabe e SE CONTÉM")
        print("onde ela cabe. Sem a segunda metade a primeira não valeria nada: partir")
        print("o domínio sempre que erra explica qualquer coisa, inclusive ruído.")
        print()
        print("E a estrutura proposta não veio de catálogo. O mecanismo não conhece")
        print("'pico', 'vale' nem 'limiar' — conhece um operador. 'Existe um regime")
        print("ótimo' é o que sobra quando PARTIR é aplicado uma vez a dados que")
        print("sobem e depois descem.")
    elif fp >= 0.2:
        print(f"CONTENÇÃO INSUFICIENTE: propõe extensão em {fp:.0%} dos casos que a")
        print("hipótese antiga já explicava. Isso é decoração de dados, e invalida a")
        print("taxa de geração — um mecanismo que parte tudo 'gera' 100%.")
    else:
        print(f"Geração {g:.0%}: o mecanismo não está propondo extensão onde deveria.")
    if sv < 0.8:
        print()
        print(f"E só {sv:.0%} sobrevive ao experimento discriminante — as estruturas")
        print("propostas não resistem ao teste desenhado para matá-las.")
    return {'linhas': ls}


# ══════════════════════════════════════════════ transferência: física real
def transferencia(modulos: Optional[List[str]] = None, passos: int = 300,
                  n_obs: int = 30, semente: int = 0,
                  verboso: bool = True) -> Dict[str, Any]:
    """
    O mesmo mecanismo, sem uma linha de mudança, contra o simulador de física.

    Os 18 pares em que as hipóteses antigas falhavam no módulo `fusao` são o
    alvo declarado. A pergunta: o gerador propõe estrutura para eles, e ela
    sobrevive a experimento discriminante escolhido contra a própria proposta?
    """
    from .client import PhysicsWorld
    from .ciencia_real import _medir, piso_de_ruido, classificar

    w = PhysicsWorld()
    modulos = modulos or ['fusao', 'ressonancia']
    print("═" * 78)
    print("TRANSFERÊNCIA — o gerador contra a física do simulador")
    print("═" * 78)

    total = {'fora': 0, 'gerou': 0, 'dentro': 0, 'fp': 0, 'prop': 0, 'sobreviveu': 0}
    achados: List[str] = []

    for modulo in modulos:
        cap = w.capacidades()
        mm = next(x for x in cap['modulos'] if x['id'] == modulo)
        manoplas = [k for k in mm['manoplas']
                    if k.get('tipo') == 'range' and 'min' in k and 'max' in k][:5]
        for man in manoplas:
            lo, hi = float(man['min']), float(man['max'])
            meio = 0.5 * (lo + hi)
            ruido = piso_de_ruido(w, modulo, man['chave'], meio, passos)
            xs = [lo + (hi - lo) * i / 8 for i in range(9)]
            serie: Dict[str, List[float]] = {}
            for x in xs:
                for k, v in _medir(w, modulo, man['chave'], x, passos, semente=7).items():
                    serie.setdefault(k, []).append(v)

            rng = random.Random(semente)
            obs_por_med: Dict[str, List[Observacao]] = {}
            for _ in range(n_obs):
                x1, x2 = sorted((rng.uniform(lo, hi), rng.uniform(lo, hi)))
                if x2 - x1 < 0.12 * (hi - lo):
                    x2 = min(hi, x1 + 0.3 * (hi - lo))
                s = rng.randint(1, 9999)
                a = _medir(w, modulo, man['chave'], x1, passos, semente=s)
                b = _medir(w, modulo, man['chave'], x2, passos, semente=s)
                for k in set(a) & set(b):
                    obs_por_med.setdefault(k, []).append((x1, x2, b[k] > a[k]))

            for med, obs in obs_por_med.items():
                ys = serie.get(med)
                if not ys or len(ys) != 9:
                    continue
                verdade = classificar(xs, ys, ruido.get(med, 0.0))
                eh_fora = verdade in ('nao-monotonica',)
                if verdade == 'plana':
                    continue

                def oraculo(x1: float, x2: float, _m=med, _man=man['chave']) -> bool:
                    s2 = rng.randint(1, 9999)
                    a2 = _medir(w, modulo, _man, x1, passos, semente=s2)
                    b2 = _medir(w, modulo, _man, x2, passos, semente=s2)
                    return b2.get(_m, 0.0) > a2.get(_m, 0.0)

                r = ciclo(obs, lo, hi, oraculo, q=0.8, semente=semente)
                if eh_fora:
                    total['fora'] += 1
                    total['gerou'] += int(r['propos'])
                else:
                    total['dentro'] += 1
                    total['fp'] += int(r['propos'])
                if r['propos']:
                    total['prop'] += 1
                    total['sobreviveu'] += int(bool(r['sobreviveu']))
                    if eh_fora and verboso and len(achados) < 10:
                        achados.append(
                            f"  {man['chave']}→{med:<14} verdade={verdade:<16} "
                            f"proposta: {r['fenomeno']}  "
                            f"{'SOBREVIVEU' if r['sobreviveu'] else 'morreu no teste'}")

    print(f"{'':<4}pares fora do espaço ..... {total['fora']}")
    print(f"{'':<4}  → gerou extensão ....... "
          f"{total['gerou']/max(1,total['fora']):.0%}")
    print(f"{'':<4}pares dentro do espaço ... {total['dentro']}")
    print(f"{'':<4}  → conteve-se ........... "
          f"{1-total['fp']/max(1,total['dentro']):.0%}")
    print(f"{'':<4}sobreviveram ao discriminante ... "
          f"{total['sobreviveu']/max(1,total['prop']):.0%} de {total['prop']}")
    if achados:
        print("\nestruturas que o mecanismo CONSTRUIU (nenhuma estava no vocabulário):")
        for a in achados:
            print(a)
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--transferencia', action='store_true')
    ap.add_argument('--casos', type=int, default=120)
    a = ap.parse_args()
    if a.transferencia:
        transferencia()
    else:
        relatorio(n_casos=a.casos)
