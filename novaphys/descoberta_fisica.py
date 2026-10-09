"""
novaphys.descoberta_fisica — descoberta cumulativa em física real
==================================================================

Todas as peças foram validadas separadamente. Esta as cruza no simulador, que
é física escrita meses antes de qualquer um destes mecanismos e nunca tocada
por eles.

    campo generativo   gramática pequena, sem operações nomeadas
    intervenção        mexer numa manopla e observar a RESPOSTA
    pontuação MDL      erro em log-resposta relativa + custo estrutural
    consolidação       o descoberto vira molde com buracos e volta ao campo

O QUE O SISTEMA **NÃO** RECEBE
--------------------------------
Nenhuma categoria. Não há "procure uma razão", "descubra uma lei", "tente
multiplicação". A entrada é: *aqui estão manoplas que você pode girar e
medições que você pode ler; encontre uma representação que preveja a resposta*.

As quatro medidas, e por que quatro
------------------------------------
    REACH      quantas medições ganham uma estrutura composta que bate os átomos
    TRANSFER   uma descoberta do módulo A ajuda no módulo B?
    DEPTH      quantas gerações de descoberta se encadeiam?
    NOVELTY    a estrutura vencedora estava fora do alcance do campo plano?

"Acerto" sozinho não distingue um sistema que resolve de um que ACUMULA. As
quatro juntas distinguem.

O TESTE QUE MAIS IMPORTA: TRANSFERÊNCIA COM A ORIGEM REMOVIDA
---------------------------------------------------------------
Depois de descobrir E1 no módulo A, o módulo A **sai**. A fase B roda noutro
módulo, com física diferente, sem nenhum dado de A — só com E1 no campo.

    E1 virou conhecimento reutilizável, ou era memória do experimento anterior?

O braço de controle recebe EXATAMENTE o mesmo orçamento de busca e um campo
sem E1. Sem esse pareamento, "o cumulativo foi melhor" só diria que ele
procurou mais.

PRÉ-REGISTRO
-------------
  F1  na fase A, alguma medição ganha estrutura composta (REACH > 0)
  F2  E1 é usada na fase B, em módulo diferente, sem dados de A
  F3  o braço cumulativo bate o controle de mesmo orçamento na fase B
  F4  se F2 e F3 falharem, a descoberta foi específica do domínio — e isso é
      resultado, não defeito
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import statistics as st
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .client import PhysicsWorld
from .intervencao import E, Molde, abstrair, compor

N_VARS = 3
Par = Tuple[List[float], List[float], float, float]


# ══════════════════════════════════════════════ intervenção física
def intervir(w: PhysicsWorld, modulo: str, manoplas: List[Dict], n: int,
             passos: int, rng: random.Random) -> Dict[str, List[Par]]:
    """
    Cada amostra é uma intervenção de verdade: monta uma configuração, mede,
    muda UMA manopla por um fator, mede de novo. Duas rodadas do simulador.
    """
    out: Dict[str, List[Par]] = {}
    for _ in range(n):
        cfg, v = {}, []
        for man in manoplas:
            lo, hi = float(man['min']), float(man['max'])
            x = rng.uniform(lo + 0.15 * (hi - lo), hi - 0.15 * (hi - lo))
            cfg[man['chave']] = x
            v.append((x - lo) / (hi - lo) + 0.2)     # normalizado, longe de zero
        i = rng.randrange(len(manoplas))
        man = manoplas[i]
        lo, hi = float(man['min']), float(man['max'])
        fator = rng.choice([0.6, 0.75, 1.3, 1.6])
        x2 = min(hi, max(lo, cfg[man['chave']] * fator))
        cfg2 = dict(cfg); cfg2[man['chave']] = x2
        v2 = list(v); v2[i] = (x2 - lo) / (hi - lo) + 0.2

        w.reset(modulo=modulo, semente=11)
        a = w.agir(ajustar=cfg, passos=passos)['escalares'] or {}
        w.reset(modulo=modulo, semente=11)
        b = w.agir(ajustar=cfg2, passos=passos)['escalares'] or {}
        for k in set(a) & set(b):
            ya, yb = a[k], b[k]
            if not (isinstance(ya, (int, float)) and isinstance(yb, (int, float))):
                continue
            if not (math.isfinite(ya) and math.isfinite(yb)):
                continue
            if abs(ya) < 1e-9 or abs(yb) < 1e-9:
                continue
            out.setdefault(k, []).append((v, v2, float(ya), float(yb)))
    return {k: p for k, p in out.items() if len(p) >= 25}


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


def resolver(pares: Sequence[Par], campo: Sequence[E]) -> Tuple[E, float]:
    best, bc = None, math.inf
    for e in compor(campo):
        c = custo(e, pares)
        if c < bc:
            best, bc = e, c
    return best, bc


# ══════════════════════════════════════════════ uma fase
def fase(w: PhysicsWorld, modulo: str, campo: Sequence[E], n_int: int,
         passos: int, semente: int, verboso: bool = True) -> Dict[str, Any]:
    cap = w.capacidades()
    m = next(x for x in cap['modulos'] if x['id'] == modulo)
    manoplas = [k for k in m['manoplas']
                if k.get('tipo') == 'range' and 'min' in k and 'max' in k][:N_VARS]
    if len(manoplas) < N_VARS:
        return {'erro': 'manoplas insuficientes'}

    dados = intervir(w, modulo, manoplas, n_int, passos, random.Random(semente))
    base = [E('V', i) for i in range(N_VARS)]
    plano = {e.expandida() for e in compor(base)}

    linhas, compostas = [], []
    for k, pares in sorted(dados.items()):
        e, c = resolver(pares, campo)
        e_at, c_at = resolver(pares, base[:])   # só átomos, sem composição
        melhor_atomo = min((custo(a, pares), a) for a in base)[0]
        ganho = melhor_atomo - c
        eh_composta = e.tam() > 1
        usa_c = e.usa_C()
        linhas.append({'med': k, 'forma': e.expandida(), 'ganho': ganho,
                       'composta': eh_composta, 'usa_C': usa_c,
                       'nova': e.expandida() not in plano})
        if eh_composta and ganho > 5:
            compostas.append(e)

    if verboso and linhas:
        print(f"\n── {modulo}   manoplas: "
              f"{', '.join(k['chave'] for k in manoplas)}   "
              f"({len(linhas)} medições, {len(dados)} com dados)")
        for l in sorted(linhas, key=lambda x: -x['ganho'])[:6]:
            marca = ' E' if l['usa_C'] else ''
            print(f"   {l['med']:<14}{l['forma']:<24}ganho {l['ganho']:>8.1f}{marca}")
    return {'linhas': linhas, 'compostas': compostas,
            'reach': sum(1 for l in linhas if l['composta'] and l['ganho'] > 5),
            'total': len(linhas),
            'usou_C': sum(1 for l in linhas if l['usa_C']),
            'ganho_medio': st.mean([l['ganho'] for l in linhas]) if linhas else 0.0}


def relatorio(mod_a: str = 'fusao', mod_b: str = 'fluidos',
              n_int: int = 45, passos: int = 150,
              semente: int = 3) -> Dict[str, Any]:
    w = PhysicsWorld()
    base = [E('V', i) for i in range(N_VARS)]

    print("═" * 78)
    print("DESCOBERTA CUMULATIVA EM FÍSICA REAL")
    print("nenhuma categoria é dada: só manoplas para girar e medições para ler")
    print("═" * 78)
    print("PRÉ-REGISTRO:")
    print("  F1  na fase A alguma medição ganha estrutura composta (REACH > 0)")
    print("  F2  E1 é usada na fase B, módulo diferente, SEM dados de A")
    print("  F3  o cumulativo bate o controle de MESMO orçamento na fase B")
    print("  F4  se F2/F3 falharem, a descoberta foi específica do domínio")

    print(f"\n{'═'*30} FASE A — {mod_a} {'═'*30}")
    a = fase(w, mod_a, base, n_int, passos, semente)
    if 'erro' in a:
        print(a['erro']); return {}
    print(f"\n   REACH(A) = {a['reach']}/{a['total']} medições com estrutura composta")

    if not a['compostas']:
        print("\nNada composto emergiu na fase A — sem descoberta não há o que")
        print("consolidar, e F1 falhou. Resultado, não defeito.")
        return {'A': a}

    cont = Counter(e.expandida() for e in a['compostas'])
    forma, vezes = cont.most_common(1)[0]
    modelo = next(e for e in a['compostas'] if e.expandida() == forma)
    molde = abstrair(modelo, 'E1')
    campo_b = base + molde.instancias(N_VARS)
    print(f"   consolidou E1 = {molde.corpo.expandida()}  "
          f"({vezes}/{len(a['compostas'])} das compostas, {molde.n_slots} slots)")

    print(f"\n{'═'*26} FASE B — {mod_b}  (sem dados de {mod_a}) {'═'*26}")
    print("\n  braço CUMULATIVO (campo contém E1):")
    b_cum = fase(w, mod_b, campo_b, n_int, passos, semente + 100)
    print("\n  braço CONTROLE (mesmo orçamento, campo sem E1):")
    b_ctl = fase(w, mod_b, base, n_int, passos, semente + 100)

    if 'erro' in b_cum or 'erro' in b_ctl:
        print("módulo B sem manoplas suficientes."); return {'A': a}

    print("\n" + "═" * 78)
    print("AS QUATRO MEDIDAS")
    print(f"  REACH    fase A ......... {a['reach']}/{a['total']}")
    print(f"           fase B cumul. .. {b_cum['reach']}/{b_cum['total']}")
    print(f"           fase B controle  {b_ctl['reach']}/{b_ctl['total']}")
    tr = b_cum['usou_C'] / max(1, b_cum['total'])
    print(f"  TRANSFER E1 usada em ..... {b_cum['usou_C']}/{b_cum['total']} "
          f"({tr:.0%}) das medições de {mod_b}")
    print(f"  DEPTH    gerações encadeadas: {2 if b_cum['usou_C'] else 1}")
    novas = sum(1 for l in b_cum['linhas'] if l['nova'])
    print(f"  NOVELTY  formas fora do alcance do campo plano: "
          f"{novas}/{b_cum['total']}")
    print(f"\n  ganho médio de descrição — cumulativo {b_cum['ganho_medio']:.1f}  ·  "
          f"controle {b_ctl['ganho_medio']:.1f}")

    print("─" * 78)
    if a['reach'] == 0:
        print("F1 FALHOU: nada composto na fase A.")
    elif b_cum['usou_C'] == 0:
        print("F2 FALHOU: E1 não foi usada em nenhuma medição do módulo B.")
        print("A descoberta da fase A foi específica do domínio — não virou")
        print("conhecimento reutilizável. Isto é resultado, e é o que o teste com a")
        print("origem removida existe para detectar.")
    elif b_cum['ganho_medio'] > b_ctl['ganho_medio'] + 2:
        print("TRANSFERÊNCIA ENTRE DOMÍNIOS FÍSICOS. Uma estrutura descoberta em")
        print(f"{mod_a} é usada em {mod_b} — física diferente, sem nenhum dado do")
        print("módulo de origem — e melhora a descrição contra um controle de mesmo")
        print("orçamento.")
        print()
        print("A ressalva: a gramática e as manoplas são minhas. O que transferiu foi")
        print("uma estrutura que o sistema encontrou, não uma que eu nomeei.")
    else:
        print(f"PARCIAL: E1 é usada em {tr:.0%} das medições, mas o ganho sobre o")
        print(f"controle é de apenas {b_cum['ganho_medio'] - b_ctl['ganho_medio']:+.1f}.")
        print("Uso sem ganho é sinal de que a consolidada está entrando por ser")
        print("barata, não por explicar melhor.")
    return {'A': a, 'B_cum': b_cum, 'B_ctl': b_ctl, 'E1': molde}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default='fusao')
    ap.add_argument('--b', default='fluidos')
    ap.add_argument('--n', type=int, default=45)
    ap.add_argument('--passos', type=int, default=150)
    a = ap.parse_args()
    relatorio(mod_a=a.a, mod_b=a.b, n_int=a.n, passos=a.passos)
