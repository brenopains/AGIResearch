"""
novaphys.maestro — especializacoes diferentes juntas rendem mais que uma so?
=============================================================================

A TESE QUE ESTE ARQUIVO TESTA
-------------------------------
"Especializacao diferente junta cria algo novo." Isso e uma afirmacao
verificavel, e ate agora nesta sessao nunca foi verificada — cada solucionador
foi medido sozinho.

Temos tres especialistas de fato diferentes, cada um nascido de um experimento
distinto e com um vies distinto:

    BASE     cadeias das 24 primitivas fixas de `arc.py`
             boa em simetria, recorte, logica entre paineis
    ETER     operadores parametrizados aterrados por tarefa (`eter.py`)
             bom onde o argumento e deduzivel: escala, cor, objeto
    INDUCAO  regras locais tipo automato celular (`arc_inducao.py`)
             boa onde a saida e funcao da vizinhanca, mesma forma

Eles nao sao variacoes de um mesmo metodo: sao gramaticas incompativeis. Se a
tese for verdadeira, a UNIAO deve alcancar tarefas que nenhum alcanca sozinho.

A ARMADILHA QUE TORNA ISSO NAO-TRIVIAL
----------------------------------------
Rodar tres solucionadores e obviamente melhor que rodar um, se cada um receber
o orcamento inteiro. Isso nao prova nada — e so gastar o triplo.

A pergunta honesta e com orcamento TOTAL fixo:

    3 especialistas com K/3 cada  >  1 generalista com K inteiro ?

Cada especialista fica MAIS FRACO no comite do que sozinho. A tese so vence se
a diversidade dos vieses compensar a perda de profundidade de cada um. Esse e
o teste real de "junta cria algo novo", e ele pode perder.

E O MAESTRO?
--------------
"Talvez falte um maestro, uma hierarquia de comando global, como um self."

A versao falsificavel: um alocador que olha a tarefa ANTES de buscar e decide
como repartir o orcamento — em vez de dar K/3 cego para cada um. Ele so tem
direito ao nome se vencer a reparticao uniforme com o MESMO K total.

As caracteristicas que ele pode olhar sao baratas e nao envolvem busca:
mesma forma entrada/saida, razao de dimensoes deduzivel, quantidade de cores,
quantidade de objetos. A politica e aprendida nas 400 tarefas de TREINO e
aplicada nas 400 de AVALIACAO, que ele nunca viu.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  A0  PRE-CONDICAO. Os especialistas tem que ser nao-redundantes: cada um
      resolve ao menos uma tarefa que os outros nao resolvem, com orcamento
      cheio. Se um for subconjunto de outro, nao ha comite — ha copia, e o
      resto da medicao nao significa nada. Verificar ANTES.

  A1  comite (K/3 cada) > melhor especialista sozinho (K inteiro), >= 3 tarefas
      -> a tese vence: diversidade de vies compensa perda de profundidade

  A2  comite ~ melhor especialista
      -> a uniao existe mas o orcamento repartido a consome; juntar so ajuda
         se houver orcamento a mais, o que e trivial e nao e a tese

  A3  comite < melhor especialista
      -> repartir destroi mais do que a diversidade recupera

  A4  maestro > comite uniforme em >= 3 tarefas, mesmo K
      -> alocar por caracteristica da tarefa e real; o "self" ganha o nome

  A5  maestro ~ comite uniforme
      -> nao ha sinal nas caracteristicas baratas; o comite cego ja e o teto

MINHA PREVISAO, escrita antes de rodar: A1 e A5. Espero que a uniao venca
(os vieses sao genuinamente diferentes, e a maioria das solucoes mora em
profundidade baixa, entao cortar o orcamento para K/3 custa pouco), e espero
que o maestro NAO venca o comite cego, porque com ~15 solucoes em 400 tarefas
nao ha amostra para aprender politica de alocacao nenhuma.

Se A4 aparecer eu vou desconfiar de vazamento antes de comemorar.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

from novaphys.arc import Grid, carregar, dims, resolver as resolver_base
from novaphys.eter import (Contexto, aterrar, aplicar_seq,
                           resolver as resolver_eter, PASTA_AVAL)

PASTA_TREINO = (('C:/Users/usuario/Documents/AGI 2.0/backup_novamind_20260627_170507/data/ARC/data/training').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))


# ══════════════════════════════════════════════ especialista 3: inducao local
def _viz(g: Grid, i: int, j: int, r: int = 1) -> Tuple[int, ...]:
    h, w = dims(g)
    return tuple(g[i + a][j + b] if 0 <= i + a < h and 0 <= j + b < w else -1
                 for a in range(-r, r + 1) for b in range(-r, r + 1))


def resolver_inducao(treino: Sequence[Tuple[Grid, Grid]],
                     orcamento: int) -> Tuple[Optional[Dict], int]:
    """
    Aprende cor_saida = f(vizinhanca 3x3 da entrada). So se aplica quando a
    forma nao muda. Consistencia e exigida: se a mesma vizinhanca aparece com
    duas saidas diferentes, a regra nao existe e devolve None.

    CUSTO, e por que ele nao e um parametro. Inducao nao enumera candidatos:
    faz UMA passada pelas celulas dos pares de treino. Isso e menos trabalho
    que avaliar UM candidato de cadeia de profundidade 3, que percorre a
    grade tres vezes por par. Entao o custo dela e 1 candidato, nao um
    orcamento a ser repartido.

    Inventar um orcamento fingido para ela (a primeira versao deste arquivo
    usava `orcamento * 900`, um numero que eu tirei do nada) faria o comite
    parecer mais caro do que e. A contabilidade honesta e: INDUCAO custa 1, e
    BASE e ETER dividem o que sobra.
    """
    for a, b in treino:
        if dims(a) != dims(b):
            return None, 1
    regra: Dict[Tuple[int, ...], int] = {}
    for a, b in treino:
        h, w = dims(a)
        for i in range(h):
            for j in range(w):
                k = _viz(a, i, j)
                if k in regra and regra[k] != b[i][j]:
                    return None, 1
                regra[k] = b[i][j]
    return regra, 1


def aplicar_inducao(regra: Dict, g: Grid) -> Optional[Grid]:
    h, w = dims(g)
    fora = []
    for i in range(h):
        linha = []
        for j in range(w):
            k = _viz(g, i, j)
            if k not in regra:
                return None
            linha.append(regra[k])
        fora.append(tuple(linha))
    return tuple(fora)


# ══════════════════════════════════════════════ interface comum
def rodar(estr: str, treino, teste, orcamento: int) -> bool:
    """True se o especialista resolve o treino E acerta o teste."""
    if estr == "BASE":
        p = resolver_base(treino, prof=3, orcamento=orcamento)
        return p is not None and all(p.aplicar(a) == b for a, b in teste)
    if estr == "ETER":
        seq, mapa, _g, _n = resolver_eter(treino, orcamento, True)
        if seq is None:
            return False
        d = dict(aterrar(Contexto(treino), True))
        return all(aplicar_seq(seq, mapa, a, d) == b for a, b in teste)
    if estr == "INDUCAO":
        r, _g = resolver_inducao(treino, 1)
        if r is None:
            return False
        if not all(aplicar_inducao(r, a) == b for a, b in treino):
            return False
        return all(aplicar_inducao(r, a) == b for a, b in teste)
    raise ValueError(estr)


ESTRATEGIAS = ["BASE", "ETER", "INDUCAO"]


# ══════════════════════════════════════════════ caracteristicas baratas
def caracteristicas(treino) -> Tuple:
    """Olhadas ANTES de qualquer busca. Nenhuma exige rodar solucionador."""
    mesma = all(dims(a) == dims(b) for a, b in treino)
    ctx = Contexto(treino)
    escala = bool(ctx.escalas) or bool(ctx.reducoes)
    ncor = len(ctx.cores_ent)
    menor = all(dims(b)[0] * dims(b)[1] < dims(a)[0] * dims(a)[1]
                for a, b in treino)
    return (mesma, escala, ncor <= 4, menor)


# ══════════════════════════════════════════════ medicao
def perfil(pasta: str, orcamento: int, rotulo: str
           ) -> Dict[str, Set[str]]:
    """Quem resolve o que, com orcamento CHEIO. Base para A0 e para o maestro."""
    tarefas = carregar(pasta)
    fora: Dict[str, Set[str]] = {e: set() for e in ESTRATEGIAS}
    t0 = time.time()
    for nome, tr, te in tarefas:
        for e in ESTRATEGIAS:
            if rodar(e, tr, te, orcamento):
                fora[e].add(nome)
    print(f"  {rotulo} ({len(tarefas)} tarefas, K cheio, "
          f"{time.time()-t0:.0f}s)")
    for e in ESTRATEGIAS:
        print(f"     {e:8s} {len(fora[e]):3d}")
    return fora


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aval", default=PASTA_AVAL)
    ap.add_argument("--treino", default=PASTA_TREINO)
    ap.add_argument("--k", type=int, default=14424, help="orcamento TOTAL")
    a = ap.parse_args()

    print("=" * 78)
    print("MAESTRO — especializacoes diferentes juntas rendem mais que uma so?")
    print("=" * 78)
    print(f"orcamento TOTAL K = {a.k}. Sozinho, cada especialista usa K.")
    print(f"No comite, cada um usa K/3 = {a.k // 3}. Mesmo total.\n")

    # ── A0: nao-redundancia, com orcamento cheio
    print("A0 · PRE-CONDICAO — os especialistas sao nao-redundantes?")
    pa = perfil(a.aval, a.k, "AVALIACAO")
    uniao = set().union(*pa.values())
    exclusivos = {e: pa[e] - set().union(*(pa[x] for x in ESTRATEGIAS
                                           if x != e))
                  for e in ESTRATEGIAS}
    print("\n     exclusivas de cada um:")
    for e in ESTRATEGIAS:
        print(f"       {e:8s} {len(exclusivos[e]):3d}  "
              f"{sorted(exclusivos[e])[:6]}")
    print(f"     uniao (orcamento 3K, nao comparavel): {len(uniao)}")
    if min(len(v) for v in exclusivos.values()) == 0:
        vazios = [e for e in ESTRATEGIAS if not exclusivos[e]]
        print(f"\n     A0 · ATENCAO — {vazios} nao tem tarefa exclusiva.")
        print("     Nao ha comite de verdade ali; leia o resto com isso em mente.")
    else:
        print("\n     A0 · passa — os tres tem territorio proprio")

    # ── A1/A2/A3: comite com orcamento repartido
    k3 = a.k // 3
    print(f"\nA1 · comite cego, K/3 = {k3} para cada, MESMO total")
    tarefas = carregar(a.aval)
    comite: Set[str] = set()
    por_estr: Dict[str, Set[str]] = {e: set() for e in ESTRATEGIAS}
    t0 = time.time()
    for nome, tr, te in tarefas:
        for e in ESTRATEGIAS:
            if rodar(e, tr, te, 1 if e == "INDUCAO" else k3):
                comite.add(nome)
                por_estr[e].add(nome)
    print(f"     ({time.time()-t0:.0f}s)")
    for e in ESTRATEGIAS:
        print(f"       {e:8s} no comite: {len(por_estr[e]):3d}   "
              f"(sozinho com K cheio: {len(pa[e]):3d})")
    melhor = max(ESTRATEGIAS, key=lambda e: len(pa[e]))
    print(f"\n     melhor especialista sozinho com K : {len(pa[melhor])}  "
          f"({melhor})")
    print(f"     comite repartido, mesmo K total   : {len(comite)}")
    d = len(comite) - len(pa[melhor])
    print(f"     delta = {d:+d}")
    if d >= 3:
        print("     A1 · a tese vence — a diversidade de vies compensa")
    elif d <= -1:
        print("     A3 · repartir destroi mais do que a diversidade recupera")
    else:
        print("     A2 · a uniao existe mas o orcamento repartido a consome")

    # ── A4/A5: maestro aprende alocacao no TREINO
    print("\nA4 · maestro — aprende no TREINO qual especialista por perfil")
    pt = perfil(a.treino, k3, "TREINO (K/3, so para aprender politica)")
    tt = carregar(a.treino)
    votos: Dict[Tuple, Dict[str, int]] = {}
    for nome, tr, te in tt:
        c = caracteristicas(tr)
        votos.setdefault(c, {e: 0 for e in ESTRATEGIAS})
        for e in ESTRATEGIAS:
            if nome in pt[e]:
                votos[c][e] += 1
    politica = {c: max(v, key=lambda e: v[e]) for c, v in votos.items()}
    print(f"     {len(politica)} perfis vistos no treino")
    for c, e in sorted(politica.items())[:8]:
        print(f"       mesma={c[0]!s:5s} escala={c[1]!s:5s} "
              f"poucasCores={c[2]!s:5s} menor={c[3]!s:5s} -> {e} "
              f"{votos[c]}")

    maestro: Set[str] = set()
    sem_perfil = 0
    for nome, tr, te in tarefas:
        c = caracteristicas(tr)
        if c in politica:
            e = politica[c]
        else:
            sem_perfil += 1
            e = melhor
        if rodar(e, tr, te, a.k):       # K INTEIRO no escolhido
            maestro.add(nome)
    print(f"\n     perfis nao vistos no treino: {sem_perfil} "
          f"(caem no melhor global)")
    print(f"     maestro (K inteiro no escolhido): {len(maestro)}")
    print(f"     comite cego (K/3 cada)         : {len(comite)}")
    dm = len(maestro) - len(comite)
    print(f"     delta = {dm:+d}")
    if dm >= 3:
        print("     A4 · alocar por caracteristica e real — desconfiar de "
              "vazamento antes de comemorar")
    else:
        print("     A5 · sem sinal nas caracteristicas baratas; o comite cego "
              "ja e o teto")

    print("\n" + "=" * 78)
    print(f"{'melhor especialista sozinho (K)':44s} {len(pa[melhor]):3d}  "
          f"{len(pa[melhor])/len(tarefas):.2%}")
    print(f"{'comite cego (repartido, mesmo K)':44s} {len(comite):3d}  "
          f"{len(comite)/len(tarefas):.2%}")
    print(f"{'maestro (K no escolhido, mesmo K)':44s} {len(maestro):3d}  "
          f"{len(maestro)/len(tarefas):.2%}")

    with open("maestro_resultado.json", "w") as f:
        json.dump({e: sorted(pa[e]) for e in ESTRATEGIAS} |
                  {"comite": sorted(comite), "maestro": sorted(maestro)},
                  f, indent=1)


if __name__ == "__main__":
    main()
