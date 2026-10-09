"""
novaphys.gargalo — o campo carrega informação, ou só amplia a busca?
======================================================================

O placebo leu `A real 50% · B vazio 29% · C placebo 41%` e concluiu
"qualquer campo enriquecido ajuda". Mas há uma hipótese concorrente que
explica esse resultado sem invocar campo, vocabulário ou aprendizagem
nenhuma:

    `resolver()` busca EXAUSTIVAMENTE sobre `compor(campo)`, que é
    quadrático no tamanho do campo.

        campo vazio      3 termos  →    54 candidatos
        1 molde 2 slots  9 termos  →   594 candidatos
        2 moldes        15 termos  →  1710 candidatos
        3 moldes        21 termos  →  3402 candidatos

O braço C foi pareado com A em número de nós e número de slots — portanto em
número de instâncias, portanto em TAMANHO DO ESPAÇO DE BUSCA. `A ≈ C > B` é
exatamente o que se prevê se o campo não carregar informação alguma e o único
efeito for buscar 63× mais expressões.

O placebo controlou o conteúdo e o orçamento JUNTOS. Não dá para separar
os dois com aqueles dados.

AS TRÊS PERGUNTAS
------------------
    G1  IDENTIDADE, tamanho casado
        Quatro campos de UM molde cada — (v*v), (v/v), (v+v), (v-v) — todos
        com 3 nós e 2 slots, todos gerando 6 instâncias, todos com 594
        candidatos. Só o conteúdo difere. Há espalhamento de cobertura?

    G2  TAMANHO, conteúdo aleatório
        Escada de 0, 1, 2, 3 moldes sorteados. A cobertura cresce com o
        número de moldes?

    G3  ORÇAMENTO CASADO
        A mesma escada, mas com a busca truncada no mesmo número de
        candidatos para todos. Se o ganho do campo some quando o orçamento
        é casado, o ganho era busca.

PRÉ-REGISTRO, ANTES DE RODAR
------------------------------
    G1 ≥ 15pp de espalhamento    há o que escolher entre conteúdos; uma
                                 política de consolidação pode importar e o
                                 experimento das 4 políticas se justifica
    G1 < 10pp                    não há o que escolher; nenhuma política
                                 pode ganhar de outra neste desenho

    G3 ganho some com orçamento casado    o "campo cognitivo" é orçamento de
                                          busca; o achado anterior deflaciona
    G3 ganho persiste                     o campo carrega informação além do
                                          tamanho, e aí sim vale construir
                                          memória de hipóteses

DESENHO PAREADO
----------------
As medições de intervenção são coletadas UMA vez por (módulo, semente) e
TODOS os braços são avaliados sobre exatamente os mesmos pares. Nenhuma
diferença entre braços pode vir de ruído do simulador — só do campo. Isso
mata a variância que deixou o placebo com 2 sementes sem poder estatístico.

Nenhum braço aprende nada. Os campos são fixos. A pergunta não é se o agente
consolida bem — é se existe alguma coisa a consolidar.

SEMENTE DOS MOLDES ALEATÓRIOS, PRÉ-REGISTRADA: 20260808
(a mesma de `placebo.py`, para que a escada de tamanho use o mesmo gerador
que gerou o braço placebo)

──────────────────────────────────────────────────────────────────────────────
LIDO DEPOIS DE RODAR — o pré-registro acima tem um defeito, e é meu
──────────────────────────────────────────────────────────────────────────────
G1 e G3 dispararam ao mesmo tempo, apontando em direções opostas:

    G1  espalhamento 21pp a tamanho e orçamento casados  → conteúdo importa
    G3  ganho de aleat3 sobre vazio: +29% irrestrito, +5% casado
                                                        → é orçamento de busca

Não é ambiguidade do mundo, é operacionalização ruim da minha parte. A
comparação de G3 é inválida por dois motivos, ambos visíveis sem olhar o
resultado — eu é que não olhei:

  (a) `aleat3` tem 21 termos e `vazio` tem 3. Truncar os dois em 54 candidatos
      não casa orçamento em nenhum sentido útil: `vazio` fica EXAUSTIVO (54 de
      54) e `aleat3` amostra 1,6% do seu espaço (54 de 3402). A queda do ganho
      mede agulha-no-palheiro, não conteúdo.
  (b) `aleat3` mistura um molde bom com um ruim, então seu desempenho médio não
      fala sobre nenhum conteúdo em particular.

A comparação VÁLIDA já estava declarada — é a de G1: quatro campos com o mesmo
número de termos (9) e o mesmo número de candidatos (594), variando só o
conteúdo. Lê-la em cada coluna de orçamento é a mesma comparação declarada,
não uma nova. E ela sobrevive em todas:

    orçamento      54    594   irrestrito
    (v*v)         40%    49%    49%
    (v/v)         37%    54%    54%
    (v+v)         31%    43%    43%
    (v-v)         14%    33%    33%
    espalhamento  26pp   21pp   21pp

O acaso do sorteio deu ainda um controle melhor do que o planejado: o terceiro
molde saiu DUPLICATA do segundo. Então `aleat2` → `aleat3` é aumento de campo
com conteúdo novo ZERO:

    aleat2   15 termos, 1710 candidatos   55%
    aleat3   21 termos, 3402 candidatos   55%     ← +0pp

Dobrar o espaço de busca sem acrescentar conteúdo não move nada. E a subida de
G2 (`vazio` 26% → `aleat1` 33% → `aleat2` 55%) é inteiramente atribuível a
QUAL molde entrou: `aleat1` = {(v-v)} rende 33%, o mesmo que (v-v) sozinho;
`aleat2` = {(v-v),(v/v)} rende 55%, o mesmo que (v/v) sozinho. Tamanho por si
não explica nada dessa escada.

Conclusão: a hipótese "o campo é só orçamento de busca" está REFUTADA, e o
veredito automático de G3 impresso pelo código está errado. Fica no arquivo
como está, sem remendo, porque reescrever um veredito depois de ver o número
é exatamente o que este experimento existe para não fazer.

CONSEQUÊNCIA DE DESENHO, que muda o experimento seguinte
---------------------------------------------------------
A orçamento irrestrito, guardar um molde ruim é DE GRAÇA — (v-v) rende 33%
contra 26% do vazio, porque um candidato a mais nunca piora um mínimo. A
orçamento apertado, guardar um molde ruim CUSTA: (v-v) cai para 14% contra
26% do vazio, porque dilui a busca.

Logo, qualquer comparação entre políticas de consolidação precisa rodar com
ORÇAMENTO DE BUSCA FIXO. Sem teto, "guardar tudo" vence por construção e
nenhuma política pode perder para outra — não há o que medir.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics as st
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .client import PhysicsWorld
from .intervencao import E, Molde, abstrair, compor
from .descoberta_fisica import intervir, custo, N_VARS, Par

SEMENTE_MOLDES = 20260808
MODULOS = ['fusao', 'fluidos', 'gravidade', 'materia', 'ressonancia']
OPS = ('+', '-', '*', '/')

# orçamentos de busca em que a cobertura é lida, do menor campo ao maior
ORCAMENTOS = (54, 200, 594, 1710, 10 ** 9)


def molde_binario(op: str) -> Molde:
    """(v0 op v1) — 3 nós, 2 slots, 6 instâncias com N_VARS=3."""
    return Molde(E(op, E('V', 0), E('V', 1)), 2, op)


def sortear_molde(rng: random.Random, n_nos: int = 3, n_slots: int = 2) -> Molde:
    """Estrutura aleatória casada em nós e slots. Mesmo gerador do placebo."""
    for _ in range(4000):
        def gerar(prof: int) -> E:
            if prof <= 0 or rng.random() < 0.32:
                return E('V', rng.randrange(N_VARS))
            if rng.random() < 0.15:
                return E('abs', gerar(prof - 1))
            return E(rng.choice(OPS), gerar(prof - 1), gerar(prof - 1))
        c = gerar(3)
        if c.tam() != n_nos:
            continue
        m = abstrair(c, 'R')
        if m.n_slots == n_slots:
            return m
    return molde_binario('*')


def campo_de(moldes: Sequence[Molde]) -> List[E]:
    c: List[E] = [E('V', i) for i in range(N_VARS)]
    for m in moldes:
        c.extend(m.instancias(N_VARS))
    return c


def resolver_curva(pares: Sequence[Par], campo: Sequence[E],
                   ordem: random.Random,
                   orcamentos: Sequence[int] = ORCAMENTOS
                   ) -> Dict[int, Tuple[Optional[E], float]]:
    """
    Uma passada só. Embaralha os candidatos e registra o melhor encontrado
    até cada orçamento — mínimos de prefixo. Truncar a busca sai de graça.

    Embaralhar importa: `compor` devolve os termos base primeiro e as
    compostas depois. Sem embaralhar, um orçamento pequeno veria só os
    termos base e a leitura seria do ordenamento, não do campo.
    """
    cands = compor(campo)
    ordem.shuffle(cands)
    fora: Dict[int, Tuple[Optional[E], float]] = {}
    best, bc = None, math.inf
    ks = sorted(orcamentos)
    i_k = 0
    for i, e in enumerate(cands, 1):
        c = custo(e, pares)
        if c < bc:
            best, bc = e, c
        while i_k < len(ks) and i >= ks[i_k]:
            fora[ks[i_k]] = (best, bc)
            i_k += 1
    while i_k < len(ks):                       # campo menor que o orçamento
        fora[ks[i_k]] = (best, bc)
        i_k += 1
    return fora


def cobertura(dados: Dict[str, List[Par]], campo: Sequence[E],
              base: Sequence[E], ordem: random.Random
              ) -> Dict[int, float]:
    """
    Fração das medições explicadas, por orçamento.

    CRITÉRIO CORRIGIDO. `placebo.py` e `agente_desenvolvimento.py` exigiam
    `e.tam() > 1 and base_c - c > 5`. Mas uma instância de molde é `E('C',…)`
    e `tam()` de um nó 'C' é 1 — é o próprio ponto do MDL, a estrutura
    consolidada custa um símbolo. Logo, quando o molde consolidado explicava
    a medição SOZINHO e perfeitamente, o agente não recebia crédito: só
    contava se a resposta fosse uma COMPOSIÇÃO do molde com outra coisa.

    Isso enviesa contra o braço cujos moldes servem, e torna a cobertura
    não-monótona no orçamento (achar algo melhor podia derrubar a métrica).
    Detectado num smoke test: campo `(v*v)` dava 100% com 54 candidatos e
    0% com 200.

    O critério aqui é só `base_c - c > 5`: achou estrutura que bate qualquer
    variável crua por margem. `base` é idêntica em todos os braços, então
    `base_c` é a mesma referência para todos. É monótono e neutro.

    A variante estrita continua sendo medida, para poder comparar com o que
    os experimentos anteriores mediram.
    """
    ok = {k: 0 for k in ORCAMENTOS}
    estr = {k: 0 for k in ORCAMENTOS}
    for _k, pares in sorted(dados.items()):
        base_c = min(custo(a, pares) for a in base)
        for orc, (e, c) in resolver_curva(pares, campo, ordem).items():
            if e is not None and base_c - c > 5:
                ok[orc] += 1
                if e.tam() > 1:
                    estr[orc] += 1
    n = len(dados)
    out = {k: v / n for k, v in ok.items()}
    out.update({('estrita', k): v / n for k, v in estr.items()})
    return out


def relatorio(n_int: int = 32, passos: int = 80,
              sementes: Sequence[int] = (0, 1, 2, 3, 4, 5)) -> Dict[str, Any]:
    w = PhysicsWorld()
    rm = random.Random(SEMENTE_MOLDES)
    aleatorios = [sortear_molde(rm) for _ in range(3)]

    bracos: List[Tuple[str, List[Molde]]] = [
        ('vazio',   []),
        ('(v*v)',   [molde_binario('*')]),
        ('(v/v)',   [molde_binario('/')]),
        ('(v+v)',   [molde_binario('+')]),
        ('(v-v)',   [molde_binario('-')]),
        ('aleat 1', aleatorios[:1]),
        ('aleat 2', aleatorios[:2]),
        ('aleat 3', aleatorios[:3]),
    ]

    print("═" * 78)
    print("GARGALO — o campo carrega informação, ou só amplia a busca?")
    print(f"moldes aleatórios, semente PRÉ-REGISTRADA: {SEMENTE_MOLDES}")
    print("═" * 78)
    print("PRÉ-REGISTRO:")
    print("  G1  espalhamento entre os 4 conteúdos casados ≥ 15pp → há o que escolher")
    print("  G1  espalhamento < 10pp                             → não há o que escolher")
    print("  G3  ganho some com orçamento casado → o campo é orçamento de busca")
    print("  G3  ganho persiste                  → o campo carrega informação")
    print()
    print(f"pareado: os mesmos pares de intervenção para TODOS os braços")
    print(f"módulos: {' → '.join(MODULOS)}   ·   sementes: {list(sementes)}")
    print(f"moldes sorteados: {[str(m.corpo) for m in aleatorios]}")
    for nome, ms in bracos:
        n_t = len(campo_de(ms))
        print(f"   {nome:<9} campo {n_t:>2} termos → {2*(n_t + 4*n_t*(n_t-1)):>5} candidatos")

    base = [E('V', i) for i in range(N_VARS)]
    # braco → orcamento → lista de coberturas (uma por célula módulo×semente)
    chaves: List[Any] = list(ORCAMENTOS) + [('estrita', k) for k in ORCAMENTOS]
    acc: Dict[str, Dict[Any, List[float]]] = {
        n: {k: [] for k in chaves} for n, _ in bracos}

    for s in sementes:
        for mod in MODULOS:
            rng = random.Random(s * 137 + 9)
            cap = w.capacidades()
            m = next(x for x in cap['modulos'] if x['id'] == mod)
            mans = [k for k in m['manoplas']
                    if k.get('tipo') == 'range' and 'min' in k][:N_VARS]
            if len(mans) < N_VARS:
                print(f"\n  · semente {s} {mod}: manoplas insuficientes")
                continue
            dados = intervir(w, mod, mans, n_int, passos, rng)
            if not dados:
                print(f"\n  · semente {s} {mod}: sem medições")
                continue
            linha = f"  · semente {s} {mod:<12} {len(dados):>2} medições  "
            for nome, ms in bracos:
                # mesma ordem de embaralhamento por (semente, módulo) para
                # todos os braços — o que muda é a lista, não o sorteio
                ordem = random.Random(hash((s, mod, nome)) & 0xFFFF)
                cob = cobertura(dados, campo_de(ms), base, ordem)
                for k, v in cob.items():
                    acc[nome][k].append(v)
                linha += f"{nome} {cob[ORCAMENTOS[-1]]:.0%}  "
            print(linha)

    print("\n" + "═" * 78)
    cab = f"{'braço':<10}" + "".join(
        f"{('irrestrito' if k > 10**8 else str(k)):>12}" for k in ORCAMENTOS)
    print("cobertura média por ORÇAMENTO DE BUSCA (candidatos avaliados)")
    print(cab)
    print("─" * 78)
    med: Dict[str, Dict[Any, float]] = {}
    for nome, _ in bracos:
        med[nome] = {k: (st.mean(v) if v else 0.0) for k, v in acc[nome].items()}
        print(f"{nome:<10}" + "".join(f"{med[nome][k]:>11.0%} " for k in ORCAMENTOS))

    print()
    print("mesma coisa sob o CRITÉRIO ESTRITO dos experimentos anteriores")
    print("(exige tam>1, e portanto não dá crédito quando um molde explica sozinho)")
    print(cab)
    print("─" * 78)
    for nome, _ in bracos:
        print(f"{nome:<10}" + "".join(
            f"{med[nome][('estrita', k)]:>11.0%} " for k in ORCAMENTOS))

    print("\n" + "═" * 78)
    conteudos = ['(v*v)', '(v/v)', '(v+v)', '(v-v)']
    irr = ORCAMENTOS[-1]
    vals = [med[c][irr] for c in conteudos]
    espalh = max(vals) - min(vals)
    print(f"G1 · IDENTIDADE (campo de 9 termos, 594 candidatos, só o conteúdo muda)")
    print("     " + "  ".join(f"{c} {v:.0%}" for c, v in zip(conteudos, vals)))
    print(f"     espalhamento {espalh:.0%}")
    if espalh >= 0.15:
        print("     → HÁ O QUE ESCOLHER. Conteúdos casados em tamanho produzem")
        print("       coberturas diferentes; uma política de consolidação pode importar.")
    elif espalh < 0.10:
        print("     → NÃO HÁ O QUE ESCOLHER. Conteúdos casados em tamanho são")
        print("       intercambiáveis; nenhuma política pode vencer outra aqui.")
    else:
        print("     → zona cinzenta entre os limiares declarados.")

    print()
    print("G2 · TAMANHO (conteúdo aleatório, busca irrestrita)")
    escada = ['vazio', 'aleat 1', 'aleat 2', 'aleat 3']
    print("     " + "  ".join(f"{c} {med[c][irr]:.0%}" for c in escada))
    cresce = all(med[escada[i]][irr] <= med[escada[i + 1]][irr] + 1e-9
                 for i in range(len(escada) - 1))
    print(f"     monótona crescente: {'sim' if cresce else 'não'}")

    print()
    print("G3 · ORÇAMENTO CASADO (todos truncados em 54 candidatos, o do campo vazio)")
    print("     " + "  ".join(f"{c} {med[c][54]:.0%}" for c in escada))
    ganho_irr = med['aleat 3'][irr] - med['vazio'][irr]
    ganho_cap = med['aleat 3'][54] - med['vazio'][54]
    print(f"     ganho de 3 moldes sobre vazio: irrestrito {ganho_irr:+.0%}"
          f"  ·  orçamento casado {ganho_cap:+.0%}")
    print()
    if ganho_irr > 0.10 and ganho_cap <= 0.05:
        print("     → O CAMPO É ORÇAMENTO DE BUSCA. O ganho existe quando o campo")
        print("       maior pode avaliar mais candidatos e desaparece quando o")
        print("       orçamento é casado. O 'campo cognitivo' desta arquitetura não")
        print("       carrega informação sobre o mundo — carrega mais expressões.")
        print("       Nenhuma política de consolidação pode produzir vantagem aqui.")
    elif ganho_cap > 0.10:
        print("     → O CAMPO CARREGA INFORMAÇÃO. Com o mesmo número de candidatos")
        print("       avaliados, o campo enriquecido ainda explica mais. O ganho não")
        print("       é busca. O experimento das políticas de consolidação está")
        print("       fundamentado.")
    else:
        print("     → Parcial. Nem o ganho irrestrito nem o casado fecham leitura.")
    return {'medias': med, 'bruto': acc}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--passos', type=int, default=80)
    ap.add_argument('--sementes', type=int, default=6)
    a = ap.parse_args()
    relatorio(n_int=a.n, passos=a.passos, sementes=tuple(range(a.sementes)))
