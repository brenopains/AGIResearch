"""novaphys.conjectura_grafo — conjectura automatica com verificador EXATO.

POR QUE ESTE ALVO (criterio destilado de tres fracassos de 16-17/09)
----------------------------------------------------------------------
  farmaco      morreu: o instrumento tem teto (21% de resolucao do dicionario)
  zeta         funcionou, mas o espaco ja foi garimpado por quem tem mais recurso
  construcao   morreu: 10^27 de computacao ate o recorde

O alvo certo precisa de instrumento com resolucao, espaco NAO garimpado, e nao
estar ordens de grandeza do estado da arte. E ha um criterio mais fundo: nossa
vantagem medida NAO e gerar (LLM 0/30; eu 0/7 como palpiteiro) — e NAO NOS
ENGANARMOS (6 defeitos meus caidos por portao em dois dias). Filtro forte com
gerador fraco so vale onde GERAR E DE GRACA. Aqui e: enumeracao exaustiva.

E o modo de falha declarado da area (Graffiti, Fajtlowicz) e exatamente o nosso
ponto forte: produzir milhares de conjecturas verdadeiras e triviais, sem peneira.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
Corpus: `graph_atlas_g` — TODOS os grafos ate 7 vertices. Conexos apenas.
  TREINO   ordem <= 6   (~112 conexos)
  FRONTEIRA ordem == 7  (~853 conexos, escondidos)   <- a regua de 17/09

Candidatas: desigualdades inteiras entre invariantes, formas
  a + b <= c  |  a <= b + c  |  a * b <= c  |  a <= b * c
Aceita so quem vale em TODOS os grafos de treino.

  G1  FRONTEIRA: das aceitas no treino, quantas sobrevivem nos 853 de ordem 7?
      (analogo exato do teste do zeta, onde 3/3 atravessaram)
  G2  NULO PAREADO: embaralhar os VALORES de cada invariante entre grafos
      (destroi a associacao, preserva as marginais). Quantas "relacoes"
      sobrevivem ao mesmo duto? Exigencia: sobreviventes_reais >> nulo.
  G3  APERTO: relacao que nunca e atingida com IGUALDADE e fraca. Exijo que
      >= 1% dos grafos atinjam igualdade — senao e folga, nao teorema.
  G4  NAO-REDUNDANCIA: se A implica B em todo o corpus, B nao conta como nova.
  G5  NOVIDADE: portao final, busca de literatura — e ele e o que matou o
      DepMap, entao entra por ultimo e a mao.

PREVISAO HONESTA, escrita antes:
  - o treino vai aceitar MUITAS (centenas): o corpus de <= 6 e pequeno
  - a fronteira vai matar a maioria — aposto que sobram < 30%
  - o nulo vai produzir MUITA coisa tambem (invariantes sao correlacionados
    por construcao), e essa e a duvida principal do desenho
  - G5: 25% de sair >= 1 relacao apertada, nao-redundante e ausente da
    literatura. A mesma duvida do DepMap: checar novidade e mais dificil que achar
"""
from __future__ import annotations
import itertools, random
import networkx as nx
from networkx.generators.atlas import graph_atlas_g


def invariantes(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    gr = [d for _v, d in G.degree()]
    comp = nx.complement(G)
    try:
        alpha = len(max(nx.find_cliques(comp), key=len))
    except ValueError:
        alpha = n
    try:
        omega = len(max(nx.find_cliques(G), key=len))
    except ValueError:
        omega = 1
    chi = max(nx.coloring.greedy_color(G, strategy="DSATUR").values()) + 1
    dom = len(nx.algorithms.dominating.dominating_set(G))
    mu = len(nx.max_weight_matching(G))
    g = nx.girth(G) if any(nx.cycle_basis(G)) else 2 * n
    return {
        "n": n, "m": m,
        "gmin": min(gr), "gmax": max(gr),
        "alpha": alpha, "omega": omega, "chi": chi,
        "dom": dom, "mu": mu,
        "beta": n - alpha,                 # cobertura por vertices
        "diam": nx.diameter(G), "raio": nx.radius(G),
        "girth": min(g, 2 * n),
        "kv": nx.node_connectivity(G), "ke": nx.edge_connectivity(G),
        "tri": sum(nx.triangles(G).values()) // 3,
        "folhas": sum(1 for d in gr if d == 1),
    }


NOMES = ["n", "m", "gmin", "gmax", "alpha", "omega", "chi", "dom", "mu",
         "beta", "diam", "raio", "girth", "kv", "ke", "tri", "folhas"]

FORMAS = [
    ("{a} + {b} <= {c}", lambda x, y, z: x + y <= z),
    ("{a} <= {b} + {c}", lambda x, y, z: x <= y + z),
    ("{a} * {b} <= {c}", lambda x, y, z: x * y <= z),
    ("{a} <= {b} * {c}", lambda x, y, z: x <= y * z),
]
IGUAL = [
    lambda x, y, z: x + y == z,
    lambda x, y, z: x == y + z,
    lambda x, y, z: x * y == z,
    lambda x, y, z: x == y * z,
]


def candidatas():
    for i, (mol, f) in enumerate(FORMAS):
        for a, b, c in itertools.permutations(NOMES, 3):
            yield (mol.format(a=a, b=b, c=c), i, a, b, c, f, IGUAL[i])


def avalia(cands, dados):
    """Devolve as que valem em TODOS, com a fracao que atinge igualdade."""
    fora = []
    for nome, _i, a, b, c, f, eq in cands:
        vale, ap = True, 0
        for d in dados:
            x, y, z = d[a], d[b], d[c]
            if not f(x, y, z):
                vale = False
                break
            if eq(x, y, z):
                ap += 1
        if vale:
            fora.append((nome, a, b, c, ap / max(len(dados), 1)))
    return fora


def assinatura_igualdade(nome_tupla, dados):
    """G4: duas relacoes que sao atingidas com IGUALDADE pelos MESMOS grafos sao
    a mesma coisa reescrita. Mesmo criterio de particao que corrigiu a contagem
    de operadores inventados (la, assinatura literal contava renomeacao como
    invencao)."""
    _nome, a, b, c, _ap, eq = nome_tupla
    return tuple(i for i, d in enumerate(dados) if eq(d[a], d[b], d[c]))


def main():
    import time, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nulos", type=int, default=5)
    arg = ap.parse_args()
    t0 = time.time()
    A = [g for g in graph_atlas_g() if g.number_of_nodes() >= 2 and nx.is_connected(g)]
    dados = [invariantes(g) for g in A]
    tr = [d for d in dados if d["n"] <= 6]
    te = [d for d in dados if d["n"] == 7]
    print("=" * 76)
    print(f"CONJECTURA EM GRAFOS — treino {len(tr)} (ordem<=6), "
          f"FRONTEIRA {len(te)} (ordem 7)   [{time.time()-t0:.0f}s]")
    print("=" * 76)

    cands = list(candidatas())
    acc = avalia(cands, tr)
    print(f"  aceitas no TREINO: {len(acc)} de {len(cands)}")

    mapa = {nome: (i, a, b, c, f, eq) for nome, i, a, b, c, f, eq in cands}
    sobrev = []
    for nome, a, b, c, apr in acc:
        _i, _a, _b, _c, f, eq = mapa[nome]
        if all(f(d[a], d[b], d[c]) for d in te):
            apt = sum(1 for d in te if eq(d[a], d[b], d[c])) / len(te)
            sobrev.append((nome, a, b, c, apt, eq))
    print(f"  G1 FRONTEIRA: sobrevivem nos 853 de ordem 7: {len(sobrev)} "
          f"({len(sobrev)/max(len(acc),1)*100:.0f}%)")

    # G2 nulo pareado: embaralha cada coluna de invariante entre grafos
    rng = random.Random(20260917)
    nulos = []
    for r in range(arg.nulos):
        emb_tr = [dict(d) for d in tr]
        emb_te = [dict(d) for d in te]
        # NULO ESTRATIFICADO POR ORDEM. Quase todo invariante escala com n;
        # embaralhar entre ordens destroi a ESCALA junto com a estrutura e o
        # nulo fica fraco demais (medido: razao real/nulo so 1,4x). Dentro da
        # mesma ordem, so a associacao grafo<->estrutura morre. Mesma licao do
        # estrato de tecido no farmaco e do substrato de mesma densidade no zeta.
        for bloco in (emb_tr, emb_te):
            por_ordem = {}
            for i, d in enumerate(bloco):
                por_ordem.setdefault(d["n"], []).append(i)
            for _ordem, idxs in por_ordem.items():
                for k in NOMES:
                    if k == "n":
                        continue
                    col = [bloco[i][k] for i in idxs]
                    rng.shuffle(col)
                    for i, v in zip(idxs, col):
                        bloco[i][k] = v
        a2 = avalia(cands, emb_tr)
        s2 = 0
        for nome, a, b, c, _ap in a2:
            _i, _a, _b, _c, f, _eq = mapa[nome]
            if all(f(d[a], d[b], d[c]) for d in emb_te):
                s2 += 1
        nulos.append((len(a2), s2))
    print(f"  G2 NULO pareado ({arg.nulos} replicas): treino "
          f"{[x for x, _ in nulos]}, sobrevivem {[y for _, y in nulos]}")

    # G3 aperto + G4 nao-redundancia por particao de igualdade
    apert = [s for s in sobrev if s[4] >= 0.01]
    print(f"  G3 APERTO (igualdade em >= 1% dos 853): {len(apert)}")
    vistos, unicas = {}, []
    for s in apert:
        chave = assinatura_igualdade(s, te)
        if chave not in vistos:
            vistos[chave] = s
            unicas.append(s)
    print(f"  G4 NAO-REDUNDANTES (particoes de igualdade distintas): {len(unicas)}")
    print()
    print("  as 25 mais APERTADAS (maior fracao com igualdade):")
    for nome, _a, _b, _c, apt, _eq in sorted(unicas, key=lambda x: -x[4])[:25]:
        print(f"     {apt*100:5.1f}%   {nome}")
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
