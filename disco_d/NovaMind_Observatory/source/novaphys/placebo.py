"""
novaphys.placebo — memória real, memória vazia, memória placebo
================================================================

O experimento anterior mostrou que o agente COM memória mantém cobertura alta
ao longo de seis módulos de física enquanto o SEM memória despenca:

    semente 1 · com memória   69% → 40% → 62% → 14% → 75% → 75%
    semente 1 · sem memória   69% →  0% → 12% →  0% →  0% → 38%

E a cadeia de consolidação atravessou três físicas diferentes:

    fusao (v*v) → fluidos (v*(v*v)) → gravidade ((v*v)/(v*v))

Mas isso NÃO distingue duas coisas:

    (a) carregar o que DESCOBRIU ajuda            ← aprendizagem cumulativa
    (b) ter QUALQUER estrutura no campo ajuda     ← vantagem de vocabulário

Já houve um precedente que obriga a checar: no controle pareado de
transferência, `(v/v)` — que o sistema NÃO descobriu — transferiu melhor que
`(v*v)`, que ele descobriu, por 13,4. Se a consolidação escolhe mal, o ganho
pode não ter nada a ver com a escolha.

TRÊS BRAÇOS
------------
    A  REAL      conserva as estruturas que descobriu
    B  VAZIO     zera o campo a cada episódio
    C  PLACEBO   recebe estruturas ALEATÓRIAS, pareadas com as de A em
                 número de nós, número de slots e cronograma de consolidação

LEITURA, DEFINIDA ANTES DE RODAR
----------------------------------
    A > B  e  A > C   a IDENTIDADE do que foi aprendido importa
    A ≈ C > B         qualquer campo enriquecido ajuda; não é aprendizagem
                      cumulativa específica
    A ≈ B             a memória não produz vantagem
    C > A             a consolidação escolhe mal de forma sistemática

O PLACEBO É PRÉ-REGISTRADO
----------------------------
    SEMENTE_PLACEBO = 20260808

Declarada antes da execução. O conteúdo das estruturas do braço C vem desse
gerador — não de olhar o resultado de A. O que vem de A é apenas o GABARITO de
pareamento: quantos nós, quantos slots, em que episódios consolidar. Parear a
forma sem parear o conteúdo é exatamente o que separa placebo de controle
decorativo.
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

SEMENTE_PLACEBO = 20260808
MODULOS = ['fusao', 'fluidos', 'gravidade', 'materia', 'ressonancia']
OPS = ('+', '-', '*', '/')


def sortear_estrutura(rng: random.Random, n_nos: int, n_slots: int) -> Molde:
    """
    Uma estrutura aleatória com o mesmo tamanho e o mesmo número de slots.

    Sorteia até casar o gabarito. Nada aqui olha para o que o braço A
    descobriu além de quantos nós e quantos slots — o conteúdo é do gerador
    pré-registrado.
    """
    for _ in range(4000):
        def gerar(prof: int) -> E:
            if prof <= 0 or rng.random() < 0.32:
                return E('V', rng.randrange(N_VARS))
            if rng.random() < 0.15:
                return E('abs', gerar(prof - 1))
            return E(rng.choice(OPS), gerar(prof - 1), gerar(prof - 1))
        cand = gerar(3)
        if cand.tam() != n_nos:
            continue
        m = abstrair(cand, 'P')
        if m.n_slots == n_slots:
            return m
    return abstrair(E('*', E('V', 0), E('V', 1)), 'P')


class Agente:
    def __init__(self, braco: str, semente: int,
                 gabarito: Optional[List[Optional[Tuple[int, int]]]] = None):
        self.braco = braco                    # 'real' | 'vazio' | 'placebo'
        self.base = [E('V', i) for i in range(N_VARS)]
        self.moldes: List[Molde] = []
        self.rng = random.Random(semente)
        self.rng_placebo = random.Random(SEMENTE_PLACEBO + semente)
        self.gabarito = gabarito or []
        self.historico: List[Dict[str, Any]] = []

    @property
    def campo(self) -> List[E]:
        c = list(self.base)
        for m in self.moldes:
            c.extend(m.instancias(N_VARS))
        return c

    def episodio(self, w: PhysicsWorld, modulo: str, i_ep: int,
                 n_int: int, passos: int) -> Dict[str, Any]:
        if self.braco == 'vazio':
            self.moldes = []

        cap = w.capacidades()
        m = next(x for x in cap['modulos'] if x['id'] == modulo)
        mans = [k for k in m['manoplas']
                if k.get('tipo') == 'range' and 'min' in k][:N_VARS]
        if len(mans) < N_VARS:
            return {'modulo': modulo, 'erro': 'manoplas insuficientes'}
        dados = intervir(w, modulo, mans, n_int, passos, self.rng)
        if not dados:
            return {'modulo': modulo, 'erro': 'sem medições'}

        campo = self.campo
        explic, compostas = 0, []
        for k, pares in sorted(dados.items()):
            e, c = resolver(pares, campo)
            base_c = min(custo(a, pares) for a in self.base)
            if e.tam() > 1 and base_c - c > 5:
                explic += 1
                compostas.append(e)

        nova = None
        if self.braco == 'real':
            if compostas:
                cont = Counter(x.expandida() for x in compostas)
                forma, vezes = cont.most_common(1)[0]
                if vezes >= 2:
                    modelo = next(x for x in compostas if x.expandida() == forma)
                    ja = {mm.corpo.expandida() for mm in self.moldes}
                    if modelo.tam() >= 3 and len(self.moldes) < 3 \
                            and modelo.expandida() not in ja:
                        nova = abstrair(modelo, f"E{len(self.moldes)+1}")
                        self.moldes.append(nova)
        elif self.braco == 'placebo':
            # consolida no MESMO episódio e com o MESMO gabarito que A
            alvo = self.gabarito[i_ep] if i_ep < len(self.gabarito) else None
            if alvo is not None and len(self.moldes) < 3:
                nova = sortear_estrutura(self.rng_placebo, alvo[0], alvo[1])
                nova = Molde(nova.corpo, nova.n_slots, f"P{len(self.moldes)+1}")
                self.moldes.append(nova)

        r = {'modulo': modulo, 'medicoes': len(dados), 'explicadas': explic,
             'cobertura': explic / len(dados), 'campo': len(self.moldes),
             'consolidou': nova.corpo.expandida() if nova else None,
             'gabarito': (nova.corpo.tam(), nova.n_slots) if nova else None}
        self.historico.append(r)
        return r


def rodar(w: PhysicsWorld, braco: str, semente: int, n_int: int, passos: int,
          gabarito=None, verboso: bool = True) -> List[Dict[str, Any]]:
    ag = Agente(braco, semente, gabarito)
    for i, mod in enumerate(MODULOS):
        r = ag.episodio(w, mod, i, n_int, passos)
        if verboso:
            if 'erro' in r:
                print(f"   ep{i+1} {mod:<12} {r['erro']}")
            else:
                ex = f"  +{r['consolidou']}" if r['consolidou'] else ""
                print(f"   ep{i+1} {mod:<12} "
                      f"{r['explicadas']}/{r['medicoes']} = {r['cobertura']:>4.0%}"
                      f"   campo {r['campo']}{ex}")
    return ag.historico


def relatorio(n_int: int = 15, passos: int = 80,
              sementes: Sequence[int] = (0, 1)) -> Dict[str, Any]:
    w = PhysicsWorld()
    print("═" * 78)
    print("PLACEBO — a identidade do que foi aprendido importa?")
    print(f"semente do placebo PRÉ-REGISTRADA: {SEMENTE_PLACEBO}")
    print("═" * 78)
    print("LEITURA, definida antes de rodar:")
    print("  A > B e A > C   a identidade do aprendido importa")
    print("  A ≈ C > B       qualquer campo enriquecido ajuda")
    print("  A ≈ B           a memória não produz vantagem")
    print("  C > A           a consolidação escolhe mal sistematicamente")
    print(f"\nmódulos: {' → '.join(MODULOS)}")

    curvas: Dict[str, List[List[float]]] = {'real': [], 'vazio': [], 'placebo': []}
    for s in sementes:
        print(f"\n─── semente {s} · A REAL")
        hA = rodar(w, 'real', s * 137 + 9, n_int, passos)
        gab = [r.get('gabarito') for r in hA]
        print(f"─── semente {s} · B VAZIO")
        hB = rodar(w, 'vazio', s * 137 + 9, n_int, passos)
        print(f"─── semente {s} · C PLACEBO  (gabarito de A: "
              f"{[g for g in gab if g]})")
        hC = rodar(w, 'placebo', s * 137 + 9, n_int, passos, gabarito=gab)
        for nome, h in (('real', hA), ('vazio', hB), ('placebo', hC)):
            curvas[nome].append([r.get('cobertura', 0.0) for r in h])

    n = min(min(len(c) for c in v) for v in curvas.values())
    print("\n" + "═" * 78)
    print(f"{'episódio':<10}{'módulo':<14}{'A real':>10}{'B vazio':>10}{'C placebo':>12}")
    med = {k: [] for k in curvas}
    for i in range(n):
        linha = f"{i+1:<10}{MODULOS[i]:<14}"
        for k in ('real', 'vazio', 'placebo'):
            v = st.mean(c[i] for c in curvas[k])
            med[k].append(v)
            linha += f"{v:>10.0%}" if k != 'placebo' else f"{v:>12.0%}"
        print(linha)

    A, B, C = (st.mean(med['real']), st.mean(med['vazio']), st.mean(med['placebo']))
    print("─" * 78)
    print(f"cobertura média — A real {A:.0%}  ·  B vazio {B:.0%}  ·  C placebo {C:.0%}")
    print()
    if A > C + 0.10 and A > B + 0.10:
        print("A IDENTIDADE DO APRENDIDO IMPORTA. O campo com as estruturas que o")
        print("agente descobriu supera tanto o campo vazio quanto o campo com")
        print("estruturas aleatórias de mesmo tamanho e mesmos slots.")
        print("Isso é aprendizagem cumulativa, não vantagem de vocabulário.")
    elif abs(A - C) <= 0.10 and A > B + 0.10:
        print("QUALQUER CAMPO ENRIQUECIDO AJUDA. O placebo empata com a memória real:")
        print("o ganho vem de TER estrutura disponível, não de ter aprendido aquela.")
        print("A vantagem medida no experimento anterior não é aprendizagem")
        print("cumulativa específica.")
    elif abs(A - B) <= 0.10:
        print("A MEMÓRIA NÃO PRODUZ VANTAGEM neste desenho.")
    elif C > A + 0.10:
        print("O PLACEBO VENCE A MEMÓRIA REAL. A consolidação escolhe mal de forma")
        print("sistemática — estruturas aleatórias servem melhor que as descobertas.")
    else:
        print(f"Parcial: A {A:.0%} · B {B:.0%} · C {C:.0%}. Não fecha nenhuma leitura.")
    return {'curvas': curvas}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=15)
    ap.add_argument('--passos', type=int, default=80)
    a = ap.parse_args()
    relatorio(n_int=a.n, passos=a.passos)
