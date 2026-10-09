"""
novaphys.laco_grade — o laco inteiro num substrato que nao tem nada de continuo
=================================================================================

A PERGUNTA
------------
"Que prova temos de que generaliza?" Ate aqui: uma OPERACAO (ache o que se
conserva) atravessou das orbitas para o ARC (97% x 16%), e o LACO (surpresa,
reanalise por consenso, conceito pela utilidade) generalizou entre dois mundos de
orbitas. Mas o laco so rodou sobre regressao continua com ruido gaussiano. Aqui ele
roda sobre GRADES DISCRETAS, com o NUCLEO DE CODECS DO ARC (`descricao.montar`) no
lugar da regressao — nada de continuo, nada de sigma.

E REATACA UMA FALHA ANTIGA
----------------------------
A lente gravitacional em grade (`lente_limpa2`, `lente_ativa_rep`): com 5 cenas por
tarefa, o nucleo acerta a lei ~50% das vezes — as falhas sao SUBAMOSTRAGEM (os
desvios caem todos para um lado e o outro so aparece no teste). Escolher o
experimento nao resolveu (18 x 15 de 30). O laco tem o remedio certo: juntar as
EXPERIENCIAS de tarefas diferentes do mesmo mundo.

O MUNDO E O LACO
------------------
Grade 10x10, particulas (5) que caem uma linha; ao lado de uma massa (8), desviam
EM DIRECAO a ela (a lente). Uma corrente de 30 tarefas de 4 cenas de treino e 2 de
teste; na tarefa 15 a lei TROCA: passam a desviar PARA LONGE da massa (repulsao).
  SURPRESA (discreta): a lei em uso erra alguma celula de alguma cena nova — no
    mundo deterministico, qualquer erro e anomalia (o analogo do qui-quadrado).
  CONSENSO (versao 2, mudada DEPOIS de um teste de fumaca com semente fora do
    registro e ANTES da medida): duas hipoteses em ordem — (1) mesmo mundo, faltou
    dado: ajusta nas tarefas recentes juntas; se explica as cenas novas, fica; (2)
    o mundo mudou: parte das cenas novas + as anomalias acumuladas desde a ultima
    confirmacao. Em ambos, recruta as tarefas que a lei explica exatamente e
    reajusta (no maximo 24 cenas por ajuste, por custo — declarado). A versao 1
    (partir so das cenas novas) nao juntava nada numa grade sem ruido: no teste de
    fumaca o laco empatou com o isolado e teve crise em TODAS as tarefas.
Bracos, mesmas cenas:
    isolado    cada tarefa do zero com as proprias 4 cenas (o que ja foi medido)
    laco       surpresa + reanalise por consenso
    pool_tudo  surpresa + reanalise juntando TODAS as tarefas guardadas (mistura
               os mundos depois da troca — o controle do consenso)
Nulo: um mundo de desvio SORTEADO (mesma frequencia, direcao ao acaso) — nenhum
braco pode "achar" lei.

O QUE O TESTE DE FUMACA DA VERSAO 2 MOSTROU (semente 1, fora do registro, 12
tarefas, troca na 6): isolado 110110000000, laco 110011000000 com crise em 10 de 12
tarefas (785 s). O laco nunca assenta porque a lei da lente NAO E EXPRIMIVEL
EXATAMENTE na gramatica de codecs: "cai reto A NAO SER que haja massa ao lado" pede
negacao de um teste derivado (a gramatica e de profundidade 1, sem negar testes
derivados), e o nucleo acha leis PARCIAIS que acertam o teste por sorte (~50%) —
nenhuma explica cenas novas exatamente. A repulsao ("diagonal para longe da
massa") pede profundidade 2: 0/6 em todos os bracos. O mundo escolhido estava FORA
da linguagem — o teste mediria o teto da gramatica, nao o laco. Erro de desenho
meu. O teto da linguagem reaparece no substrato de grade, como nas orbitas (n^).

CORRECAO (mesmo dia, teste de viabilidade, semente fora do registro): o
diagnostico acima estava ERRADO. Com a busca ja consertada (completar com copia),
a gramatica de profundidade 1 exprime a lei da lente EXATAMENTE — o "a nao ser que
haja massa ao lado" sai por contagem: a celula abaixo de uma particula isolada tem
`viz8 = 1`. Com 16 cenas: erro de treino 0, 20/20 cenas novas; com 8 cenas: erro
10, 1/20. Faltava DADO, nao linguagem — e os testes de fumaca da lente rodaram
ANTES do conserto da busca. Ligar a negacao de testes derivados (profundidade 2)
PIOROU: uma lei mais barata que fecha o treino e erra 6/20 cenas novas, com 8 e com
16 cenas. Mais linguagem, mais ajuste espurio.

E UM CONFUNDIDOR DE ORCAMENTO (viabilidade, sementes fora do registro): a MESMA lente
com 24 cenas cai para erro 13, 5/20 — e SEM o teto de avaliacoes volta a erro 0,
20/20. O teto do nucleo e TETO_CEL // celulas: com mais dado, MENOS avaliacoes. Isso
conflita com o laco, que junta tarefas: a reanalise de 24 cenas e exatamente onde a
busca e estrangulada. E a REPULSAO nao sai nem sem teto (16 cenas: erro 22, 1/20;
24: erro 34, 1/20) — provavelmente fora da linguagem, NAO provado (feixe e
profundidade tambem limitam). Nao serve de fase 2.

RESULTADO REGISTRADO — RECOLORIR (semente 20261013, 6 correntes, 4 no nulo, 2715 s):
                fase 1   fase 2 (permutacao trocada)
  isolado         20%        16%
  LACO            90%        87%    crises [0,1,2,15,16,...]: aprende e ASSENTA
  pool_tudo       90%        67%    crises seguem ate ~17-18 depois da troca
  nulo (mapa sorteado): 0%
G1 PASSOU (+70). G2 PASSOU (6/6). G3 PASSOU NO LIMITE (+20, limiar +20). N PASSOU.
Previsoes: laco ~90% certo; isolado previ ~45%, deu 20%; G3 previ ~+30, deu +20.
(O relatorio imprimiu "lente"/"repulsao" nos rotulos das fases — texto fixo do
codigo, corrigido depois; o mundo era a recoloracao.)
O LACO INTEIRO — surpresa, reanalise por consenso — funciona num substrato
discreto, com o nucleo de codecs do ARC no lugar da regressao continua: acumular
resolve a subamostragem que a tarefa isolada nao resolve, e a surpresa ve a troca
de mundo em todas as correntes. E foi este experimento que achou o defeito de busca
do nucleo (completar com copia).

CONSERTO DO ORCAMENTO + NOVA FASE 2 (viabilidade, sementes fora do registro):
`ajusta` passa a dar a toda reanalise as avaliacoes de UMA tarefa de N_TR cenas.
Lente com 24 cenas: 20/20 de novo. Fase 2 nova, LENTE_SOBE (a particula sobe e
desvia em direcao a massa): 20/20 com 8, 16 e 24 cenas — viavel.

PRE-REGISTRO DA LENTE NO LACO — `--mundo lente2`, semente 20261016 (nova: o
registro 20261012 previa a repulsao, que se mostrou inviavel), lente -> lente_sobe
na tarefa 15, 4 correntes por braco e 2 no nulo (reduzido ANTES de rodar: o nulo
reanalisa em toda tarefa, ~45 min de CPU por corrente)
  G1  laco - isolado >= +30 na fase 1 (lente)
  G2  crise na troca (tarefa 15 ou 16) em >= 3/4 correntes
  G3  laco - pool_tudo >= +20 na fase 2 (lente_sobe)
  N   nulo: acerto do laco <= 5%
PREVISAO: isolado ~40% na fase 1 (4 cenas), laco ~85% (G1 passa); G2 passa; na
fase 2 o isolado deve ir BEM (a lente_sobe sai com 8 cenas), o laco ~90%, o
pool_tudo mistura os dois mundos ate a janela de 6 tarefas se renovar (~60%) — G3
passa por pouco; N passa.

RESULTADO REGISTRADO — LENTE -> LENTE_SOBE (semente 20261016, 4 correntes, 2 no
nulo, 4462 s):
                fase 1 (lente)   fase 2 (lente_sobe)
  isolado           43%               43%
  LACO              85%               88%    crises [0,1,15,16,...]
  pool_tudo         85%               67%
  nulo: 0%
G1 PASSOU (+42). G2 PASSOU (4/4). G3 PASSOU (+22). N PASSOU. Previsoes certas,
inclusive nas magnitudes (isolado ~40, laco ~85, G3 por pouco) — com uma excecao:
previ o isolado BEM na fase 2 e ele ficou nos mesmos 43%.
(As linhas dos portoes imprimiram "(lente)" e "(repulsao)" — texto fixo antigo,
corrigido depois; a fase 2 era a lente_sobe.)
Segundo mundo de grade em que o laco passa nos quatro portoes, agora com uma lei
FISICA (a lente) e nao so um mapeamento de cores.

VERSAO 3 DO MUNDO — RECOLORIR (lei exprimivel, mas que EXIGE acumular): grade 6x6,
2-3 manchas de cores distintas; a lei e uma PERMUTACAO das 9 cores sem ponto fixo.
Cada tarefa mostra so algumas cores — a tarefa isolada nao sabe o destino das que
nao viu; juntar tarefas resolve. Na tarefa 15 a permutacao TROCA. Nulo: cada cena
com um mapeamento sorteado.

PRE-REGISTRO — mundo RECOLORIR, semente 20261013, 6 correntes por braco, 4 no nulo
(os portoes abaixo valem para este mundo; a semente 20261012 da lente nao foi usada)
PREVISAO (recolorir): isolado ~45%, laco ~90% (G1 passa), G2 passa, pool_tudo mistura
os dois mapeamentos ate a janela de 6 tarefas se renovar (G3 passa, ~+30), N passa.

TESTES DE FUMACA DA RECOLORACAO (semente 1, fora do registro, 12 tarefas, troca
na 6):
  nucleo ANTES do conserto da busca: isolado 000000000000, laco 001100001111
    (crise em todas as 12) — o nucleo escolhia `viz_cor` (decora cor+vizinhanca,
    78% em cenas novas) e nenhuma lei explicava cenas novas exatamente.
  conserto no nucleo (completar com copia, `descricao._montar_base`): a mesma
    busca passa a escolher `[mapa_cor, copia]` (100%).
  nucleo DEPOIS: isolado 000000000000 (a tarefa isolada quase nunca viu todas as
    cores do teste — por construcao), laco 011111001111 com crises so em
    [0, 1, 6, 7, 8]: ASSENTA, detecta a troca na tarefa 6 na hora e recupera em
    2 tarefas; pool_tudo 011111000001, crises de 6 a 11 — so volta quando a
    janela de 24 cenas se renova.
O padrao previsto, numa semente que nao conta. A medida registrada vem agora.

PRE-REGISTRO ORIGINAL (mundo da lente) — mantido como registro, NAO rodado
------------------------------------------------------------------------------------
(Reduzido de 8 antes de rodar: o teste de fumaca mediu ~13 s por ajuste de 4
cenas — 8 correntes levariam ~1h40.)
Acerto = a lei em uso na tarefa reproduz as 2 cenas de teste exatamente.
  G1  laco - isolado >= +30 pontos na fase 1 (lente, tarefas 0-14)
  G2  o laco detecta a troca: crise na tarefa 15 ou 16 em >= 5/6 correntes
  G3  laco - pool_tudo >= +20 na fase 2 (repulsao, tarefas 15-29)
  N   mundo sorteado: acerto do laco <= 5%
PREVISAO: isolado ~40% (4 cenas), laco ~85% (G1 passa), G2 passa, G3 passa (o pool
de tudo carrega as cenas da lente e nao fecha), N passa.
Calibragem: das ultimas 20 previsoes de magnitude, 10 erradas — a maioria para
cima.
"""

from __future__ import annotations

import argparse
import random
import time

from novaphys import descricao as D
from novaphys.arc import dims

SEMENTE = 20261012
L, P, M = 10, 5, 8
N_TAREFAS, TROCA, N_TR, N_TE, MAX_POOL = 30, 15, 4, 2, 24
BRACOS = ("isolado", "laco", "pool_tudo")


def cena(rng, modo):
    g = [[0] * L for _ in range(L)]
    cols = [1, 4, 7]
    rng.shuffle(cols)
    parts = []
    for c in cols[:rng.randint(2, 3)]:
        r = rng.randint(1, 6)
        lado = rng.choice((-1, 0, 1))
        g[r][c] = P
        if lado != 0:
            g[r][c + lado] = M
        parts.append((r, c, lado))
    a = tuple(tuple(l) for l in g)
    b = [[x if x == M else 0 for x in lin] for lin in a]
    for r, c, lado in parts:
        dy = 1
        if modo == "lente":
            dx = lado
        elif modo == "repulsao":
            dx = -lado
        elif modo == "lente_sobe":     # a lente de cabeca para baixo: sobe e
            dx, dy = lado, -1          # desvia EM DIRECAO a massa
        else:
            dx = rng.choice((-1, 0, 1))
        b[r + dy][c + dx] = P
    return a, tuple(tuple(l) for l in b)


LR = 6


def derangement(rng):
    while True:
        p = list(range(1, 10))
        rng.shuffle(p)
        if all(p[i] != i + 1 for i in range(9)):
            return {i + 1: p[i] for i in range(9)}


def cena_cor(rng, perm):
    """2-3 manchas retangulares de cores distintas; saida = cores permutadas."""
    g = [[0] * LR for _ in range(LR)]
    for c in rng.sample(range(1, 10), rng.randint(2, 3)):
        h, w = rng.randint(1, 3), rng.randint(1, 3)
        r0, c0 = rng.randint(0, LR - h), rng.randint(0, LR - w)
        for i in range(r0, r0 + h):
            for j in range(c0, c0 + w):
                g[i][j] = c
    mp = perm if perm != "sorteado" else {c: rng.randint(1, 9) for c in range(1, 10)}
    a = tuple(tuple(l) for l in g)
    b = tuple(tuple(mp[x] if x else 0 for x in l) for l in g)
    return a, b


def _cena(tipo, rng, modo):
    return cena(rng, modo) if tipo == "lente" else cena_cor(rng, modo)


def ajusta(cenas):
    # ORCAMENTO DE UMA TAREFA INTEIRA, sempre. O teto padrao do nucleo e
    # TETO_CEL // celulas: juntar 6 tarefas dava 1/6 das avaliacoes, e a lente com
    # 24 cenas caia de 20/20 para 5/20 so por isso. Aqui toda reanalise tem as
    # avaliacoes de uma tarefa de N_TR cenas — para a tarefa isolada, nada muda.
    # (Mudanca DEPOIS da rodada registrada da recoloracao, antes de qualquer
    # medida da lente no laco.)
    h, w = dims(cenas[0][1]) if cenas else (1, 1)
    teto = max(D.TETO_CEL // (N_TR * h * w), 40) if D.TETO_CEL else None
    try:
        fm, p, e = D.montar(cenas[:MAX_POOL], refoco=True, feixe=4, teto=teto)
        return (fm, p) if p else None
    except Exception:
        return None


def explica(lei, cenas):
    if lei is None:
        return False
    try:
        return all(D.aplicar(lei, a) == b for a, b in cenas)
    except Exception:
        return False


def _recruta(lei, tarefas, sementes):
    """Refaz a lei com as sementes + as tarefas que ela explica exatamente."""
    dentro = sorted(set(sementes) | {i for i, t in enumerate(tarefas) if explica(lei, t)},
                    reverse=True)
    lei2 = ajusta([c for i in dentro for c in tarefas[i]])
    return lei2 if explica(lei2, tarefas[sementes[0]]) else lei


def consenso(guardadas, novas, anomalas):
    """Duas hipoteses, nesta ordem (versao 2, depois do teste de fumaca):
    1. MESMO MUNDO, FALTOU DADO: ajusta nas tarefas mais recentes juntas; se essa lei
       explica as cenas novas, fica com ela (e recruta as que ela explica).
    2. O MUNDO MUDOU: parte das cenas novas + as ANOMALIAS acumuladas desde a ultima
       vez que a teoria foi confirmada (Kuhn: as anomalias formam o paradigma novo).
    A versao 1 partia so das cenas novas e so incorporava tarefas que a lei
    explicasse EXATAMENTE — numa grade sem ruido, lei subamostrada nunca explica a
    tarefa que tem o caso que faltou, e o consenso virava 'cada tarefa do zero'."""
    tarefas = guardadas + [novas]
    novo = len(guardadas)
    recentes = list(range(novo, -1, -1))
    lei = ajusta([c for i in recentes for c in tarefas[i]])
    if explica(lei, novas):
        return _recruta(lei, tarefas, [novo]), True
    sementes = [novo] + sorted(anomalas, reverse=True)
    lei = ajusta([c for i in sementes for c in tarefas[i]])
    return _recruta(lei, tarefas, [novo]), False


def corrente(args):
    braco, s, semente, mundos = args[:4]
    tipo = args[4] if len(args) > 4 else "lente"
    rng = random.Random(semente + 1000 * s)
    guardadas, acertos, crises = [], {}, []
    anomalas = []                # tarefas que surpreenderam desde a ultima confirmacao
    lei = None
    for k in range(N_TAREFAS):
        modo = mundos[0] if k < TROCA else mundos[1]
        tr = [_cena(tipo, rng, modo) for _ in range(N_TR)]
        te = [_cena(tipo, rng, modo) for _ in range(N_TE)]
        if braco == "isolado":
            lei_k = ajusta(tr)
        else:
            if explica(lei, tr):
                anomalas = []            # a teoria foi confirmada por dados novos
            else:
                crises.append(k)
                if braco == "laco":
                    lei, mesmo = consenso(guardadas, tr, anomalas)
                    anomalas = [] if mesmo else anomalas + [k]
                else:
                    todas = [c for t in guardadas for c in t] + tr
                    lei = ajusta(todas[-MAX_POOL:] if len(todas) > MAX_POOL else todas)
            lei_k = lei
        acertos[k] = explica(lei_k, te)
        guardadas.append(tr)
    return braco, s, acertos, crises, mundos[0]


def _ini():
    D.usar_biblioteca(None)


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--correntes", type=int, default=6)
    ap.add_argument("--nulo", type=int, default=4)
    ap.add_argument("--nucleos", type=int, default=3)
    ap.add_argument("--mundo", default="cor")
    a = ap.parse_args()
    n = a.correntes
    if a.mundo == "cor":
        sem = 20261013
        rp = random.Random(sem)
        m1 = derangement(rp)
        m2 = derangement(rp)
        trab = ([(b, s, sem, (m1, m2), "cor") for s in range(n) for b in BRACOS]
                + [("laco", s, sem + 7, ("sorteado", "sorteado"), "cor")
                   for s in range(a.nulo)])
    elif a.mundo == "lente2":
        sem = 20261016
        trab = ([(b, s, sem, ("lente", "lente_sobe"), "lente") for s in range(n)
                 for b in BRACOS]
                + [("laco", s, sem + 7, ("sorteado", "sorteado"), "lente")
                   for s in range(a.nulo)])
    else:
        trab = ([(b, s, SEMENTE, ("lente", "repulsao")) for s in range(n) for b in BRACOS]
                + [("laco", s, SEMENTE + 7, ("sorteado", "sorteado")) for s in range(a.nulo)])
    print("=" * 76)
    print("LACO NA GRADE — surpresa e consenso com o nucleo de codecs do ARC")
    print("=" * 76)
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(corrente, trab, chunksize=1)
    print(f"\n  ({time.time()-t0:.0f}s)\n")
    tab = {}
    print("  acerto (lei em uso reproduz as 2 cenas de teste)")
    for b in BRACOS:
        rr = [r for r in res if r[0] == b and r[4] != "sorteado"]
        f1 = sum(r[2][k] for r in rr for k in range(TROCA)) / (TROCA * n)
        f2 = sum(r[2][k] for r in rr for k in range(TROCA, N_TAREFAS)) / ((N_TAREFAS - TROCA) * n)
        tab[b] = (f1, f2)
        print(f"    {b:9s}  fase 1 {100*f1:3.0f}%   fase 2 (mundo trocado) {100*f2:3.0f}%   "
              f"crises: " + "  ".join(str(r[3][:6]) for r in rr[:4]))
    nul = [r for r in res if r[4] == "sorteado"]
    fn = sum(sum(r[2].values()) for r in nul) / (N_TAREFAS * max(len(nul), 1))
    print(f"    nulo (sorteado), laco: {100*fn:.1f}%")
    lc = [r for r in res if r[0] == "laco" and r[4] != "sorteado"]
    det = sum(1 for r in lc if any(c in (TROCA, TROCA + 1) for c in r[3]))
    print("")
    print("=" * 76)
    g1 = tab["laco"][0] - tab["isolado"][0]
    print(f"G1 {'.' if g1 >= 0.30 else 'FALHOU .'} laco - isolado (fase 1) = {100*g1:+.0f} (limiar +30)")
    print(f"G2 {'.' if det >= n - 1 else 'FALHOU .'} crise na troca (15-16): {det}/{n}")
    g3 = tab["laco"][1] - tab["pool_tudo"][1]
    print(f"G3 {'.' if g3 >= 0.20 else 'FALHOU .'} laco - pool_tudo (fase 2) = {100*g3:+.0f} (limiar +20)")
    print(f"N  {'.' if fn <= 0.05 else 'FALHOU .'} nulo: {100*fn:.1f}% (<= 5%)")


if __name__ == "__main__":
    main()
