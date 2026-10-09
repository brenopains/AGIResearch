"""
novaphys.agente_desenvolvimento — agir, acumular, crescer
==========================================================

A hipótese que sobreviveu ao raciocínio: o que faz diferença não é volume de
dado observacional, é **agência com memória**. Duas evidências desta sessão
sustentam isso:

    observação → margem de identificação 0,0   ·  recuperação 40%
    intervenção → margem ~3000                 ·  recuperação 100%

    campo vazio em domínio novo → 1/5 medições explicadas
    campo com estrutura anterior → 4/5

O que falta para virar DESENVOLVIMENTO, e não intervenção isolada, são duas
coisas que o agente ainda não tinha:

    OBJETIVO PRÓPRIO   ninguém diz o que procurar; ele busca reduzir o
                       comprimento de descrição do que consegue explicar
    MEMÓRIA ENTRE      o que consolidou no episódio k continua no campo no
    EPISÓDIOS          episódio k+1, atravessando MÓDULOS DE FÍSICA diferentes

O DESENHO
----------
Cada episódio: o agente escolhe um módulo, escolhe uma manopla, intervém,
mede quantas das medições daquele módulo ele consegue explicar com estrutura
composta, e consolida o que recorre e comprime.

    braço COM MEMÓRIA    o campo persiste e cresce
    braço SEM MEMÓRIA    o campo é zerado a cada episódio

Mesmo orçamento de intervenções nos dois. A única variável é a persistência.

A MÉTRICA É A CURVA, NÃO O PONTO FINAL
----------------------------------------
Cobertura explicada por episódio. Se a memória serve, a curva com memória sobe
e a sem memória fica plana — porque cada episódio recomeça do zero.

Um ponto final alto não distinguiria as duas coisas: pode ser só que o último
módulo era fácil. A inclinação distingue.

PRÉ-REGISTRO
-------------
  D1  a curva COM memória sobe ao longo dos episódios
  D2  a curva SEM memória fica plana (é o controle: cada episódio é o primeiro)
  D3  a diferença aparece em módulos que o agente visita DEPOIS de consolidar,
      não no primeiro
  D4  se as duas curvas forem iguais, a memória não está servindo, e isso é
      resultado — significa que cada módulo de física é uma ilha
"""

from __future__ import annotations

import argparse
import math
import random
import statistics as st
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .client import PhysicsWorld
from .intervencao import E, Molde, abstrair, compor
from .descoberta_fisica import intervir, custo, resolver, N_VARS


class Agente:
    """
    Campo persistente, objetivo próprio, memória do que já consolidou.

    O objetivo não é "acerte a lei" — é reduzir o comprimento de descrição do
    que consegue explicar. Ninguém lhe diz que estrutura procurar.
    """

    def __init__(self, com_memoria: bool, semente: int = 0):
        self.com_memoria = com_memoria
        self.base = [E('V', i) for i in range(N_VARS)]
        self.moldes: List[Molde] = []
        self.rng = random.Random(semente)
        self.historico: List[Dict[str, Any]] = []
        self.visitados: List[str] = []

    @property
    def campo(self) -> List[E]:
        c = list(self.base)
        for m in self.moldes:
            c.extend(m.instancias(N_VARS))
        return c

    def esquecer(self) -> None:
        """Braço de controle: recomeçar do zero a cada episódio."""
        self.moldes = []

    def episodio(self, w: PhysicsWorld, modulo: str, n_int: int,
                 passos: int) -> Dict[str, Any]:
        if not self.com_memoria:
            self.esquecer()

        cap = w.capacidades()
        m = next(x for x in cap['modulos'] if x['id'] == modulo)
        manoplas = [k for k in m['manoplas']
                    if k.get('tipo') == 'range' and 'min' in k][:N_VARS]
        if len(manoplas) < N_VARS:
            return {'modulo': modulo, 'erro': 'manoplas insuficientes'}

        dados = intervir(w, modulo, manoplas, n_int, passos, self.rng)
        if not dados:
            return {'modulo': modulo, 'erro': 'sem medições'}

        campo = self.campo
        explicadas, compostas, ganhos = 0, [], []
        for k, pares in sorted(dados.items()):
            e, c = resolver(pares, campo)
            base_c = min(custo(a, pares) for a in self.base)
            ganho = base_c - c
            ganhos.append(ganho)
            if e.tam() > 1 and ganho > 5:
                explicadas += 1
                compostas.append(e)

        # consolidação: o que recorre E comprime, sem eu escolher
        nova = None
        if compostas:
            cont = Counter(x.expandida() for x in compostas)
            forma, vezes = cont.most_common(1)[0]
            if vezes >= 2:
                modelo = next(x for x in compostas if x.expandida() == forma)
                if modelo.tam() >= 3 and len(self.moldes) < 3:
                    ja = {mm.corpo.expandida() for mm in self.moldes}
                    cand = abstrair(modelo, f"E{len(self.moldes)+1}")
                    if cand.corpo.expandida() not in ja:
                        self.moldes.append(cand)
                        nova = cand

        self.visitados.append(modulo)
        r = {'modulo': modulo, 'medicoes': len(dados),
             'explicadas': explicadas,
             'cobertura': explicadas / len(dados),
             'ganho_medio': st.mean(ganhos),
             'campo': len(self.moldes),
             'consolidou': nova.corpo.expandida() if nova else None}
        self.historico.append(r)
        return r


MODULOS = ['fusao', 'fluidos', 'gravidade', 'materia', 'vacuo', 'ressonancia']


def rodar(w: PhysicsWorld, com_memoria: bool, modulos: Sequence[str],
          n_int: int, passos: int, semente: int,
          verboso: bool = True) -> List[Dict[str, Any]]:
    ag = Agente(com_memoria, semente)
    for i, mod in enumerate(modulos):
        r = ag.episodio(w, mod, n_int, passos)
        if verboso:
            if 'erro' in r:
                print(f"   ep{i+1} {mod:<12} {r['erro']}")
            else:
                extra = f"  consolidou {r['consolidou']}" if r['consolidou'] else ""
                print(f"   ep{i+1} {mod:<12} cobertura "
                      f"{r['explicadas']}/{r['medicoes']} = {r['cobertura']:>4.0%}"
                      f"   campo {r['campo']}{extra}")
    return ag.historico


def relatorio(n_int: int = 30, passos: int = 120,
              sementes: Sequence[int] = (0, 1, 2)) -> Dict[str, Any]:
    w = PhysicsWorld()
    print("═" * 78)
    print("DESENVOLVIMENTO — agir, consolidar, carregar para o próximo mundo")
    print("objetivo próprio: reduzir a descrição do que consegue explicar")
    print("ninguém diz que estrutura procurar, nem em que módulo")
    print("═" * 78)
    print("PRÉ-REGISTRO:")
    print("  D1  a curva COM memória sobe ao longo dos episódios")
    print("  D2  a curva SEM memória fica plana (cada episódio é o primeiro)")
    print("  D3  a diferença aparece DEPOIS da primeira consolidação")
    print("  D4  se as curvas forem iguais, cada módulo é uma ilha — e isso é")
    print("      resultado, não defeito")
    print()
    print(f"módulos, na ordem: {' → '.join(MODULOS)}")

    curvas: Dict[bool, List[List[float]]] = {True: [], False: []}
    for s in sementes:
        for mem in (True, False):
            print(f"\n─── semente {s} · {'COM' if mem else 'SEM'} memória")
            h = rodar(w, mem, MODULOS, n_int, passos, semente=s * 137 + 9)
            curvas[mem].append([r.get('cobertura', 0.0) for r in h])

    print("\n" + "═" * 78)
    n_ep = min(min(len(c) for c in curvas[True]), min(len(c) for c in curvas[False]))
    print(f"{'episódio':<10}{'módulo':<14}{'com memória':>14}{'sem memória':>14}")
    med_com, med_sem = [], []
    for i in range(n_ep):
        a = st.mean(c[i] for c in curvas[True])
        b = st.mean(c[i] for c in curvas[False])
        med_com.append(a); med_sem.append(b)
        print(f"{i+1:<10}{MODULOS[i]:<14}{a:>13.0%}{b:>14.0%}")

    print("─" * 78)
    if n_ep < 3:
        print("Episódios de menos para ler inclinação.")
        return {}
    # inclinação simples: segunda metade menos primeira
    meio = n_ep // 2
    inc_com = st.mean(med_com[meio:]) - st.mean(med_com[:meio])
    inc_sem = st.mean(med_sem[meio:]) - st.mean(med_sem[:meio])
    print(f"inclinação (2ª metade − 1ª metade):  com memória {inc_com:+.0%}  ·  "
          f"sem memória {inc_sem:+.0%}")
    if inc_com > inc_sem + 0.10:
        print()
        print("A MEMÓRIA FAZ A CURVA SUBIR. O agente explica mais em módulos que")
        print("visita depois de consolidar, e o controle sem memória não acompanha —")
        print("mesmo orçamento, mesma ordem de módulos, mesma tudo.")
        print()
        print("Isso é desenvolvimento no sentido estreito e medível: o que ele")
        print("consegue explicar depende do que já explicou.")
    elif abs(inc_com - inc_sem) <= 0.10:
        print()
        print("CURVAS IGUAIS. A memória não está ajudando: cada módulo de física é")
        print("uma ilha para este mecanismo. É D4, e é resultado — significa que as")
        print("estruturas que ele consolida não são as que servem no módulo seguinte.")
    else:
        print()
        print("A memória ATRAPALHA. O campo maior está desviando a busca para")
        print("estruturas consolidadas que não servem — consolidação como ruído.")
    return {'com': curvas[True], 'sem': curvas[False]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--passos', type=int, default=120)
    a = ap.parse_args()
    relatorio(n_int=a.n, passos=a.passos)
