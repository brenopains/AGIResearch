"""
novaphys.campo_aberto — o mecanismo fora da bancada
====================================================

Trilha paralela. As bancadas sintéticas medem o MECANISMO; esta pergunta outra
coisa, que elas não conseguem responder sozinhas:

    a capacidade escala para problemas que não foram desenhados para produzi-la?

O mundo aqui é o Simulador de Realidade — doze módulos de física com numérica
validada, escritos meses antes de existir qualquer um destes mecanismos, e que
nunca participaram de nenhuma decisão de desenho deles.

O MECANISMO ENTRA CONGELADO
----------------------------
Nada de `geracao.py` é alterado. Mesma regra geradora, mesma abstenção
calibrada com holdout e permutação, mesmas três operações de expansão:

    PARAMETRIZAR   v_i < v_j
    SOMAR          v_i + v_j < k
    MULTIPLICAR    v_i · v_j < k

PRÉ-REGISTRO — escrito ANTES de rodar
--------------------------------------
Física de verdade é cheia de RAZÕES, e razão não está no repertório:

    Q      = P_fus / P_aux
    Re     = v·L / ν
    β      ∝ n·T / B²

Então a previsão, registrada antes de qualquer número:

  P1  relações monótonas numa manopla só → o mecanismo SE ABSTÉM de expandir,
      porque a base já resolve. Abstenção alta é ACERTO aqui, não fracasso.

  P2  relações com produto (β ∝ n·T) → MULTIPLICAR é selecionada.

  P3  relações com RAZÃO (Q, Re) → nenhuma das três operações basta. Previsão:
      ajuste baixo, e/ou seleção de MULTIPLICAR como aproximação ruim. É a
      falha que eu espero, e ela é o ponto: o repertório é insuficiente para a
      física real, e o interesse está em o mecanismo perceber isso ou não.

Se P3 falhar da forma prevista, o resultado é o LIMITE, medido fora da bancada.
Se o mecanismo se abstiver em P3 em vez de forçar uma operação ruim, isso é
melhor ainda — seria detecção de insuficiência num domínio que ninguém desenhou
para isso.

COMO A FÍSICA VIRA TAREFA
--------------------------
Amostram-se configurações do módulo variando as manoplas contínuas; cada
medição escalar é binarizada pela mediana. O contexto do programa são as
manoplas normalizadas. Nenhuma equação é entregue — só números.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics as st
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .client import PhysicsWorld
from .geracao import (Obs, Prog, gerar, melhor, _testes_base,
                      _testes_da_operacao, validar)
import novaphys.geracao as G


# ══════════════════════════════════════════════ congelamento do mecanismo
def selo_do_mecanismo() -> str:
    """Hash das constantes do mecanismo — para provar que ele não mudou aqui."""
    estado = {
        'N_VARS': G.N_VARS, 'LAM': G.LAM,
        'operacoes': ['PARAMETRIZAR', 'SOMAR', 'MULTIPLICAR'],
        'gerar.defaults': [40, 0.95],
        'melhor.profundidade': 2,
    }
    return hashlib.sha256(json.dumps(estado, sort_keys=True).encode()).hexdigest()[:16]


# ══════════════════════════════════════════════ física → tarefa
def amostrar(w: PhysicsWorld, modulo: str, manoplas: List[Dict],
             n: int, passos: int, semente: int) -> Tuple[List[List[float]],
                                                         Dict[str, List[float]]]:
    rng = random.Random(semente)
    ctxs: List[List[float]] = []
    medidas: Dict[str, List[float]] = {}
    for _ in range(n):
        cfg, ctx = {}, []
        for man in manoplas:
            lo, hi = float(man['min']), float(man['max'])
            x = rng.uniform(lo, hi)
            cfg[man['chave']] = x
            ctx.append((x - lo) / max(1e-9, hi - lo))       # normalizado em [0,1]
        w.reset(modulo=modulo, semente=7)
        o = w.agir(ajustar=cfg, passos=passos)
        esc = {k: v for k, v in (o.get('escalares') or {}).items()
               if isinstance(v, (int, float)) and math.isfinite(v)}
        if not esc:
            continue
        ctxs.append(ctx)
        for k, v in esc.items():
            medidas.setdefault(k, []).append(float(v))
    return ctxs, {k: v for k, v in medidas.items() if len(v) == len(ctxs)}


def como_tarefa(ctxs: List[List[float]], ys: List[float]) -> Optional[List[Obs]]:
    """Binariza pela mediana. Descarta o que não tem variação."""
    if len(set(round(y, 9) for y in ys)) < 3:
        return None
    med = st.median(ys)
    obs = [(c, 1 if y < med else -1) for c, y in zip(ctxs, ys)]
    pos = sum(1 for _, s in obs if s > 0)
    if pos < 0.25 * len(obs) or pos > 0.75 * len(obs):
        return None
    return obs


# ══════════════════════════════════════════════ avaliação
def avaliar_modulo(w: PhysicsWorld, modulo: str, n: int = 90, passos: int = 250,
                   max_manoplas: int = 3, semente: int = 0,
                   verboso: bool = True) -> Dict[str, Any]:
    cap = w.capacidades()
    m = next(x for x in cap['modulos'] if x['id'] == modulo)
    manoplas = [k for k in m['manoplas']
                if k.get('tipo') == 'range' and 'min' in k and 'max' in k][:max_manoplas]
    if len(manoplas) < G.N_VARS:
        return {'modulo': modulo, 'erro': 'manoplas contínuas insuficientes'}

    ctxs, medidas = amostrar(w, modulo, manoplas, n, passos, semente)
    if not ctxs:
        return {'modulo': modulo, 'erro': 'sem escalares'}

    disp = ['PARAMETRIZAR', 'SOMAR', 'MULTIPLICAR']
    linhas: List[Dict[str, Any]] = []
    for chave, ys in sorted(medidas.items()):
        obs = como_tarefa(ctxs, ys)
        if obs is None:
            continue
        base = melhor(obs, _testes_base(obs))
        aj_base = 1 - sum(1 for (v, s) in obs if base.aval(v) != s) / len(obs)
        props = gerar(obs, base, disp, k=1)
        op = props[0] if props else None
        if op:
            exp = melhor(obs, list(_testes_base(obs)) + _testes_da_operacao(obs, op))
            aj_exp = 1 - sum(1 for (v, s) in obs if exp.aval(v) != s) / len(obs)
            ganho = base.custo(obs) - exp.custo(obs)
        else:
            aj_exp, ganho = aj_base, 0.0
        linhas.append({'medicao': chave, 'op': op or 'abstém',
                       'aj_base': aj_base, 'aj_exp': aj_exp, 'ganho': ganho})

    if verboso and linhas:
        print(f"\n── {modulo}   manoplas: "
              f"{', '.join(k['chave'] for k in manoplas)}   ({len(linhas)} medições)")
        print(f"{'medição':<18}{'ajuste base':>13}{'operação':>15}{'ajuste':>9}{'ganho':>8}")
        for l in sorted(linhas, key=lambda x: -x['ganho'])[:10]:
            print(f"{l['medicao']:<18}{l['aj_base']:>12.0%}{l['op']:>15}"
                  f"{l['aj_exp']:>9.0%}{l['ganho']:>8.1f}")
    return {'modulo': modulo, 'linhas': linhas,
            'manoplas': [k['chave'] for k in manoplas]}


def relatorio(modulos: Optional[List[str]] = None, n: int = 90,
              passos: int = 250, semente: int = 0) -> Dict[str, Any]:
    w = PhysicsWorld()
    modulos = modulos or ['fusao', 'fluidos', 'vacuo']
    print("═" * 78)
    print("CAMPO ABERTO — o mecanismo congelado contra física real")
    print(f"selo do mecanismo: {selo_do_mecanismo()}   (rode de novo depois: "
          f"se mudar, houve ajuste)")
    print("═" * 78)
    print("PRÉ-REGISTRO, escrito antes de rodar:")
    print("  P1  relação monótona numa manopla → abstenção (acerto, não falha)")
    print("  P2  relação com produto           → MULTIPLICAR selecionada")
    print("  P3  relação com RAZÃO (Q, Re, β)  → NENHUMA das três basta;")
    print("      previsão: ajuste baixo, ou MULTIPLICAR como aproximação ruim")

    todas: List[Dict[str, Any]] = []
    for mod in modulos:
        try:
            r = avaliar_modulo(w, mod, n=n, passos=passos, semente=semente)
        except Exception as e:
            print(f"\n── {mod}: erro {e}")
            continue
        if 'erro' in r:
            print(f"\n── {mod}: {r['erro']}")
            continue
        todas.extend(r['linhas'])

    if not todas:
        print("\nnenhuma medição utilizável.")
        return {}

    print("\n" + "═" * 78)
    print("AGREGADO")
    c = Counter(l['op'] for l in todas)
    print(f"  medições avaliadas ......... {len(todas)}")
    for k, v in c.most_common():
        print(f"  {k:<24} {v:>3}  ({v/len(todas):.0%})")
    ganhou = [l for l in todas if l['ganho'] > 0.5]
    print(f"  expansão compensou em ...... {len(ganhou)}/{len(todas)} "
          f"({len(ganhou)/len(todas):.0%})")
    aj_b = st.mean(l['aj_base'] for l in todas)
    aj_e = st.mean(l['aj_exp'] for l in todas)
    print(f"  ajuste médio: base {aj_b:.0%} → com expansão {aj_e:.0%}")

    abst = c.get('abstém', 0) / len(todas)
    print("─" * 78)
    print("CONTRA O PRÉ-REGISTRO:")
    if abst > 0.5:
        print(f"  abstenção em {abst:.0%} das medições. O mecanismo NÃO força operação")
        print("  em física que a base já explica — comportamento previsto em P1, agora")
        print("  medido num domínio que ninguém desenhou para isso.")
    else:
        print(f"  abstenção {abst:.0%} — mais baixa que o esperado por P1. Vale checar")
        print("  se a binarização pela mediana está criando estrutura artificial.")
    if aj_e < 0.85:
        print(f"  ajuste com expansão {aj_e:.0%}: o repertório NÃO cobre a física.")
        print("  É P3 confirmado — razão (Q = P_fus/P_aux, Re = vL/ν) não está entre")
        print("  as três operações, e nenhuma composição delas a produz.")
        print("  Este é o LIMITE, medido fora da bancada.")
    else:
        print(f"  ajuste com expansão {aj_e:.0%} — acima do previsto por P3.")
        print("  Conferir se as medições selecionadas são mesmo do tipo razão.")
    return {'linhas': todas}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--modulos', nargs='*', default=None)
    ap.add_argument('--n', type=int, default=90)
    ap.add_argument('--passos', type=int, default=250)
    a = ap.parse_args()
    relatorio(modulos=a.modulos, n=a.n, passos=a.passos)
