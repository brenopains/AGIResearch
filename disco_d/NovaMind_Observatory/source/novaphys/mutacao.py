"""
novaphys.mutacao — pressão de abertura: inventar a ferramenta que falta
========================================================================

A parede, medida em seis bancadas e sempre a mesma frase:

    o sistema reusa a biblioteca que aprendeu; não a AUMENTA quando o mundo
    exige uma transformação que ela não contém.

Esta bancada existe para pressioná-la. Não melhora nada do que já funciona: o
único caminho barato para a solução é criar uma primitiva nova.

DUAS HONESTIDADES ANTES DO DESENHO
-----------------------------------
**1. O regresso desce de novo, e de novo não some.** O mecanismo aqui é
mutação de árvore — trocar uma subárvore, envolver um ramo num teste. São
edições SINTÁTICAS, agnósticas a domínio: não sabem o que é regime, canal ou
limiar. Isso é mais baixo que `SE` e mais baixo que `PARTIR`, e é por isso que
conta como degrau. Mas fui eu que escolhi permitir essas edições, e quem
chamar isso de "inventar do nada" está vendendo o degrau como escada.

**2. "Inventar" aqui tem um significado exato, e é mais estreito do que a
palavra sugere.** Uma primitiva nova só pode ser uma COMPOSIÇÃO dos átomos
existentes. Se o mundo exigir multiplicação e não houver multiplicação entre
os átomos, nenhuma mutação a produz — isso é impossível, não difícil. Então o
que esta bancada mede é:

    achar uma composição PROFUNDA DEMAIS para a busca direta, por escalada
    guiada pelo erro, e então nomeá-la para que fique barata

O que é uma capacidade real e é o que falta hoje — mas não é criação a partir
do nada, e a diferença importa para não repetir o erro de anunciar mais do que
foi medido.

OS TRÊS GRUPOS
---------------
    A  a biblioteca BASTA          resolvido em profundidade 1
    B  não basta, mas há composição CURTA   profundidade 2, ao alcance da
                                            enumeração direta
    C  não basta e não há atalho   profundidade 3 — FORA da enumeração; só
                                   escalando por mutação se chega

B existe para separar duas coisas que se confundem: um sistema que inventa
sem precisar não está sendo criativo, está sendo perdulário. Em B ele deve
COMPOR, não inventar.

O CRITÉRIO DE PROMOÇÃO, DELIBERADAMENTE DURO
---------------------------------------------
Criar alguma coisa não conta. Para a primitiva nova ser aceita, ela tem de:

    · reduzir o comprimento de descrição
    · REAPARECER em tarefas que não participaram da descoberta
    · NÃO ser usada no controle negativo (grupo A), onde não ajuda

Sem a terceira, fabricar abstrações inúteis vence a bancada. Foi essa a
mesma trava que fez o número de `abstracao.py` valer alguma coisa.

O CICLO
--------
    programa atual → onde erra sistematicamente? → mutações pequenas
    → seleção por descrição → recorrência → promoção
"""

from __future__ import annotations

import itertools
import math
import random
import statistics as st
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .abstracao import (No, Primitiva, MAIS, MENOS, _subarvores, minerar,
                        resolver, enumerar)

Obs = Tuple[Dict[str, float], int]
CAMPOS = ('x', 'z', 'w')
CORTES = (1.0, 3.0, 5.0, 7.0, 9.0, 0.25, 0.5, 0.75)


# ══════════════════════════════════════════════ os três grupos
def tarefa(grupo: str, rng: random.Random, n: int = 90) -> List[Obs]:
    """
    A profundidade da solução verdadeira é o que define o grupo — e ela é
    verificável: a enumeração de `abstracao.enumerar` cobre profundidade 2 com
    UM ramo aninhado, e nada além disso.
    """
    cx, cz, cw = rng.uniform(3, 7), rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
    obs: List[Obs] = []
    for _ in range(n):
        ctx = {'x': rng.uniform(0, 10), 'z': rng.random(), 'w': rng.random()}
        if grupo == 'A':                       # profundidade 1
            s = 1 if ctx['x'] < cx else -1
        elif grupo == 'B':                     # profundidade 2 — há atalho
            s = 1 if (ctx['x'] < cx and ctx['z'] < cz) else -1
        elif grupo == 'C':                     # profundidade 3 — sem atalho
            if ctx['x'] < cx:
                s = 1 if ctx['z'] < cz else -1
            else:
                s = 1 if ctx['w'] < cw else -1
        else:
            raise ValueError(grupo)
        obs.append((ctx, s))
    return obs


# ══════════════════════════════════════════════ cortes tirados dos dados
def cortes_dos_dados(obs: Sequence[Obs], campos: Sequence[str],
                     n: int = 9) -> Dict[str, List[float]]:
    """
    Os limiares candidatos saem dos quantis do que foi OBSERVADO, por variável.

    Isto conserta um defeito que arruinou a primeira versão desta bancada, e o
    defeito merece ficar escrito. Eu usava uma grade fixa e comum a todas as
    variáveis — `x` varia em (0,10), `z` e `w` em (0,1), e a grade tinha 5,0.
    O corte verdadeiro em `x` cai em (3,7); com 5,0 como única opção próxima,
    sobravam 12 erros de posicionamento.

    Força bruta sobre profundidade 3 mostrou o estrago: o melhor programa
    profundo custava 36,00, e o programa raso encontrado pela busca direta
    custava 36,00 — EMPATE EXATO. Inventar a estrutura não comprava nada, e
    não havia gradiente nenhum para escalar. A bancada que devia tornar a
    invenção a única saída barata a tornou indiferente.

    É o mesmo defeito já diagnosticado em `substrato.py` e não corrigido lá:
    limiar fixado por mim em vez de derivado do que o sistema percebe.
    """
    out: Dict[str, List[float]] = {}
    for c in campos:
        vals = sorted(ctx[c] for (ctx, _) in obs)
        if not vals:
            out[c] = [0.0]
            continue
        out[c] = [vals[int((i + 1) * len(vals) / (n + 1))] for i in range(n)]
    return out


# ══════════════════════════════════════════════ mutação guiada por erro
def _testes(campos: Sequence[str], cortes: Any) -> List[No]:
    if isinstance(cortes, dict):
        return [No('MENOR', (No('VAR', (c,)), No('NUM', (v,))))
                for c in campos for v in cortes.get(c, [])]
    return [No('MENOR', (No('VAR', (c,)), No('NUM', (v,))))
            for c in campos for v in cortes]


def resolver_raso(obs: Sequence[Obs], campos: Sequence[str],
                  cortes: Any, biblioteca: Sequence[Primitiva],
                  lam: float = 1.5) -> Dict[str, Any]:
    """
    Busca direta até profundidade 2, com testes por variável.

    Escrita aqui em vez de reusar `abstracao.resolver` porque aquela aplica a
    mesma lista de cortes a todas as variáveis — o que é justamente o que
    quebrou a bancada. `abstracao.py` fica intocado: uma variável por vez.
    """
    testes = _testes(campos, cortes)
    progs: List[No] = [MAIS, MENOS]
    for t in testes:
        progs.append(No('SE', (t, MAIS, MENOS)))
        progs.append(No('SE', (t, MENOS, MAIS)))
    for t in testes:
        for t2 in testes:
            if t2 is t:
                continue
            progs.append(No('SE', (t, No('SE', (t2, MAIS, MENOS)), MENOS)))
            progs.append(No('SE', (t, MAIS, No('SE', (t2, MAIS, MENOS)))))
    for p in biblioteca:
        tipos = p.modelo.tipos_dos_buracos()
        opcoes = []
        for tp in tipos:
            opcoes.append(list(campos) if tp == 'campo'
                          else sorted({v for vs in (cortes.values()
                                       if isinstance(cortes, dict) else [cortes])
                                       for v in vs}))
        if len(opcoes) > 4:
            continue
        for combo in itertools.product(*opcoes):
            progs.append(No('LIB', (p,) + combo))

    melhor, melhor_c = None, math.inf
    for p in progs:
        e = sum(1 for (ctx, s) in obs if p.aval(ctx) != s)
        c = e + lam * p.tamanho()
        if c < melhor_c:
            melhor, melhor_c = p, c
    ac = sum(1 for (ctx, s) in obs if melhor.aval(ctx) == s) / max(1, len(obs))
    return {'programa': melhor, 'custo': melhor_c, 'ajuste': ac,
            'tamanho': melhor.tamanho()}


def _caminhos_de_folha(n: No, prefixo: Tuple[int, ...] = ()) -> List[Tuple[int, ...]]:
    """Posição de cada folha de sinal, como caminho de índices de argumento."""
    if n.op == 'SINAL':
        return [prefixo]
    out: List[Tuple[int, ...]] = []
    for i, a in enumerate(n.args):
        if isinstance(a, No):
            out.extend(_caminhos_de_folha(a, prefixo + (i,)))
    return out


def _caminhos_de_teste(n: No, prefixo: Tuple[int, ...] = ()) -> List[Tuple[int, ...]]:
    """
    Posição de cada teste (`MENOR`) no programa.

    A segunda edição permitida existe porque a primeira, sozinha, não bastava:
    com apenas "envolver uma folha num teste", um programa cuja RAIZ testa a
    variável errada fica preso num ótimo local — nenhum envolvimento paga o
    custo, e a escalada não sai do lugar. Foi o que aconteceu: 0% de escalada
    partindo de `se(z<0,5){+}senão{−}` num mundo cuja estrutura começa em `x`.

    Trocar um teste por outro continua sendo edição SINTÁTICA e agnóstica a
    domínio — não diz qual variável olhar, só permite olhar outra.
    """
    if n.op == 'MENOR':
        return [prefixo]
    out: List[Tuple[int, ...]] = []
    for i, a in enumerate(n.args):
        if isinstance(a, No):
            out.extend(_caminhos_de_teste(a, prefixo + (i,)))
    return out


def _substituir(raiz: No, caminho: Tuple[int, ...], novo: No) -> No:
    """
    Troca o nó no CAMINHO dado.

    A primeira versão substituía por identidade (`raiz is alvo`), e isso não
    funcionava por um motivo que não aparece lendo o código de longe: `MAIS` e
    `MENOS` são singletons de módulo, então TODAS as folhas de sinal de um
    programa são literalmente o mesmo objeto. A busca por identidade sempre
    casava com a primeira, e a escalada só conseguia editar um ramo.

    Medido: 0% de escalada no grupo C — o que eu quase reportei como "o
    mecanismo não alcança profundidade 3". Era o mutador não conseguindo mexer
    no ramo que precisava mexer.
    """
    if not caminho:
        return novo
    i, resto = caminho[0], caminho[1:]
    args = list(raiz.args)
    args[i] = _substituir(args[i], resto, novo)
    return No(raiz.op, tuple(args))


def mutar(prog: No, obs: Sequence[Obs], campos: Sequence[str],
          cortes: Sequence[float], lam: float = 1.5,
          rodadas: int = 6, largura: int = 40,
          rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """
    Escalada guiada pelo ERRO, com edições sintáticas genéricas.

    A cada rodada:
      1. separa as observações em que o programa atual erra;
      2. para cada FOLHA de sinal, pergunta se envolvê-la num teste reduziria
         o erro naquela região — é a única edição permitida, e ela não sabe
         nada sobre o domínio;
      3. fica com a melhor por comprimento de descrição.

    Note o que NÃO existe aqui: nenhuma lista de estruturas candidatas, nenhum
    "tente PARTIR", nenhuma dica de qual variável olhar. A direção vem do erro.
    """
    rng = rng or random.Random(0)
    testes = _testes(campos, cortes)

    def custo(p: No) -> float:
        e = sum(1 for (c, s) in obs if p.aval(c) != s)
        return e + lam * p.tamanho()

    # FEIXE, não escalada gulosa.
    #
    # Guloso não atravessa esta bancada, e o motivo é uma barreira medida:
    #     profundidade 1 (partida) ..... custo ~34
    #     profundidade 2 (intermediário) custo ~38,5   ← PIOR
    #     profundidade 3 (destino) ..... custo ~26
    # O passo do meio é pior que o começo, então o guloso para antes de chegar
    # ao que é muito melhor. Ótimo local clássico. Medido: 0% de escalada, e
    # depois 8% com cortes derivados dos dados.
    #
    # O feixe mantém as `largura_feixe` melhores de cada rodada, inclusive as
    # que pioraram, e por isso pode pagar o pedágio. Continua sendo busca
    # genérica: nenhuma noção de domínio entra aqui.
    largura_feixe = 6
    feixe: List[Tuple[float, No]] = [(custo(prog), prog)]
    melhor_global = (custo(prog), prog)
    historia = [str(prog)]
    for _ in range(rodadas):
        vizinhos: List[Tuple[float, No]] = []
        cands = testes if len(testes) <= largura else rng.sample(testes, largura)
        for _, atual in feixe:
            for folha in _caminhos_de_folha(atual):     # edição 1: aprofundar
                for t in cands:
                    for a, b in ((MAIS, MENOS), (MENOS, MAIS)):
                        n_ = _substituir(atual, folha, No('SE', (t, a, b)))
                        vizinhos.append((custo(n_), n_))
            for cam in _caminhos_de_teste(atual):       # edição 2: trocar teste
                for t in cands:
                    n_ = _substituir(atual, cam, t)
                    vizinhos.append((custo(n_), n_))
        if not vizinhos:
            break
        vizinhos.sort(key=lambda kv: kv[0])
        feixe = vizinhos[:largura_feixe]
        if feixe[0][0] < melhor_global[0] - 1e-9:
            melhor_global = feixe[0]
            historia.append(str(melhor_global[1]))

    c_final, atual = melhor_global
    acertos = sum(1 for (c, s) in obs if atual.aval(c) == s)
    return {'programa': atual, 'custo': c_final, 'ajuste': acertos / max(1, len(obs)),
            'passos': len(historia) - 1, 'historia': historia}


def resolver_com_mutacao(obs: Sequence[Obs], campos: Sequence[str],
                         cortes: Sequence[float], biblioteca: Sequence[Primitiva],
                         rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """Busca direta primeiro; a mutação parte do melhor que ela achou."""
    base = resolver_raso(obs, campos, cortes, biblioteca)
    mut = mutar(base['programa'], obs, campos, cortes, rng=rng)
    if mut['custo'] < base['custo'] - 1e-9:
        return {**mut, 'inventou': True, 'custo_base': base['custo'],
                'ajuste_base': base['ajuste']}
    return {**base, 'inventou': False, 'custo_base': base['custo'],
            'ajuste_base': base['ajuste'], 'passos': 0}


# ══════════════════════════════════════════════ o experimento
def experimento(n_desc: int = 8, n_teste: int = 8, semente: int = 0,
                verboso: bool = False) -> Dict[str, Any]:
    rng = random.Random(semente)
    campos = list(CAMPOS)

    # ── fase de descoberta: só o grupo C, onde não há atalho ────────────
    achados: List[No] = []
    aj_dir, aj_mut = [], []
    for _ in range(n_desc):
        obs = tarefa('C', rng)
        cortes = cortes_dos_dados(obs, campos)
        d = resolver_raso(obs, campos, cortes, [])
        m = resolver_com_mutacao(obs, campos, cortes, [], rng=rng)
        aj_dir.append(d['ajuste']); aj_mut.append(m['ajuste'])
        if m['inventou']:
            achados.append(m['programa'])

    # ── promoção: o que reapareceu e comprime ───────────────────────────
    lib = minerar(achados, min_vezes=3) if achados else []
    if verboso:
        print("o que a escalada por erro construiu, e a mineração promoveu:")
        for p in lib:
            print(f"  {p.nome}  esqueleto {p.esqueleto}  vista {p.vezes_vista}×")
            print(f"       exemplar: {p.modelo}")
        if not lib:
            print("  nada foi promovido")

    # ── validação: tarefas NOVAS de cada grupo ──────────────────────────
    res: Dict[str, Any] = {}
    for g in ('A', 'B', 'C'):
        sem, com, usou, aj = [], [], 0, []
        for _ in range(n_teste):
            obs = tarefa(g, rng)
            cortes = cortes_dos_dados(obs, campos)
            s_ = resolver_raso(obs, campos, cortes, [])
            c_ = resolver_raso(obs, campos, cortes, lib)
            sem.append(s_['custo']); com.append(c_['custo']); aj.append(c_['ajuste'])
            usou += int(any(n.op == 'LIB' for n in _subarvores(c_['programa'])))
        res[g] = {'custo_sem': st.mean(sem), 'custo_com': st.mean(com),
                  'usou': usou / n_teste, 'ajuste': st.mean(aj)}

    return {'lib': lib, 'grupos': res,
            'ajuste_direto_C': st.mean(aj_dir), 'ajuste_mutacao_C': st.mean(aj_mut),
            'inventou_em_C': len(achados) / n_desc}


def relatorio(sementes: Tuple[int, ...] = (0, 1, 2)) -> Dict[str, Any]:
    print("═" * 78)
    print("PRESSÃO DE ABERTURA — a única saída barata é criar uma ferramenta nova")
    print("edições permitidas: envolver uma folha num teste. Sintáticas, genéricas,")
    print("sem nenhuma noção de domínio. A direção vem do ERRO.")
    print("═" * 78)

    ls = [experimento(semente=s, verboso=(i == 0)) for i, s in enumerate(sementes)]
    m = lambda k: st.mean([l[k] for l in ls])
    g = lambda gr, k: st.mean([l['grupos'][gr][k] for l in ls])
    print()

    print("DESCOBERTA (grupo C — solução em profundidade 3, fora da enumeração)")
    print(f"  ajuste da busca direta ............ {m('ajuste_direto_C'):.0%}")
    print(f"  ajuste após escalada por erro ..... {m('ajuste_mutacao_C'):.0%}")
    print(f"  escalou em ........................ {m('inventou_em_C'):.0%} das tarefas")
    print(f"  primitivas promovidas ............. "
          f"{st.mean([len(l['lib']) for l in ls]):.1f}")

    print()
    print(f"{'grupo':<8}{'custo sem lib':>15}{'custo com lib':>15}{'usou':>8}"
          f"{'ajuste':>9}   o que testa")
    rot = {'A': 'CONTROLE: a biblioteca basta, não pode usar a nova',
           'B': 'há atalho: deve COMPOR, não inventar',
           'C': 'sem atalho: aqui a nova primitiva tem de aparecer'}
    for gr in ('A', 'B', 'C'):
        print(f"{gr:<8}{g(gr,'custo_sem'):>15.2f}{g(gr,'custo_com'):>15.2f}"
              f"{g(gr,'usou'):>7.0%}{g(gr,'ajuste'):>9.0%}   {rot[gr]}")

    print("─" * 78)
    ganho_C = m('ajuste_mutacao_C') - m('ajuste_direto_C')
    usoC, usoA = g('C', 'usou'), g('A', 'usou')
    comprimeC = g('C', 'custo_sem') - g('C', 'custo_com')

    if ganho_C < 0.05:
        print("A ESCALADA NÃO SAI DO LUGAR. Guiada pelo erro, com as edições")
        print("permitidas, ela não alcança a estrutura de profundidade 3 — então não")
        print("há o que promover, e a parede continua exatamente onde estava.")
    elif not any(l['lib'] for l in ls):
        print(f"A escalada MELHORA o ajuste ({ganho_C:+.0%}), mas nada foi promovido:")
        print("o que ela constrói não se repete o bastante para compensar nomear.")
        print("Construir uma vez não é ganhar uma ferramenta.")
    elif usoA > 0.2:
        print(f"CONTROLE REPROVADO: a primitiva nova é usada em {usoA:.0%} das tarefas do")
        print("grupo A, onde ela não ajuda. Isso é fabricar abstração, e anula a")
        print("leitura dos outros dois grupos.")
    elif usoC > 0.8 and comprimeC > 0:
        print("ATRAVESSOU. A escalada guiada por erro construiu uma estrutura que a")
        print("busca direta não alcança; a mineração a promoveu porque ela se repetiu")
        print("e comprime; ela reaparece em tarefas novas do grupo C e NÃO aparece no")
        print("grupo A, onde não serve.")
        print()
        print("Com a ressalva que estava escrita antes de rodar: 'inventar' aqui é")
        print("achar uma composição funda demais para a busca direta e nomeá-la. Não")
        print("é criar um átomo que não existia — isso continua impossível por")
        print("construção, e continua fora do alcance desta arquitetura.")
    else:
        print(f"Parcial: escalou ({ganho_C:+.0%}), promoveu, mas uso em C {usoC:.0%} e")
        print(f"compressão {comprimeC:+.2f}. Não fecha o critério de promoção.")
    return {'linhas': ls}


if __name__ == "__main__":
    relatorio()
