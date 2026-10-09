"""novaphys.residuo_dirigido — CONSTRUIR o termo olhando o residuo.

Ver PRE_REGISTRO_residuo_dirigido.md. Diferenca de desenho em relacao aos SEIS
nulos anteriores: o codec NAO EXISTE antes da tarefa. Ele e construido a partir
do que a pilha atual deixou por explicar NAQUELA tarefa.

  os 6 nulos    ofereceram vocabulario ao sistema (eu, LLM, enumeracao)
  os 2 sucessos o sistema construiu o termo do dado (h = x*vy - y*vx)

VAZAMENTO: o gerador so ve os pares de TREINO e o residuo sobre eles. A saida de
TESTE nunca entra. Explicar o treino e trabalho legitimo; olhar o teste e trapaca.
"""
from __future__ import annotations
import random
from math import log2

from novaphys import descricao as D
from novaphys.descricao import objetos, fundo, dims, BITS_COR

SEMENTE = 20260917

# ---- as PRIMITIVAS: quantidades da entrada, e as operacoes que as compoem ----
BASE = ("i", "j", "bi", "bj", "cor", "linhas", "colunas", "dobj", "cobj")
UNARIOS = (("", None), ("%2", 2), ("%3", 3), ("%4", 4))
BINARIOS = (("+", lambda a, b: a + b), ("-", lambda a, b: a - b),
            ("min", min), ("max", max), ("==", lambda a, b: int(a == b)))


def observaveis(a, bg, h, w):
    """Para cada celula, as quantidades BASE — so da ENTRADA."""
    nb = [[a[y][x] != bg for x in range(w)] for y in range(h)]
    lin = [sum(r) for r in nb]
    col = [sum(nb[y][x] for y in range(h)) for x in range(w)]
    pos = [(y, x) for y in range(h) for x in range(w) if nb[y][x]]
    out = {}
    for y in range(h):
        for x in range(w):
            d, c = 99, 0
            for (py, px) in pos:
                dd = max(abs(py - y), abs(px - x))
                if dd < d:
                    d, c = dd, a[py][px]
            out[(y, x)] = {"i": y, "j": x, "bi": h - 1 - y, "bj": w - 1 - x,
                           "cor": a[y][x], "linhas": lin[y], "colunas": col[x],
                           "dobj": min(d, 9), "cobj": c}
    return out


def termos():
    """Todos os termos COMPOSTOS, profundidade <= 2. Cada um e um nome e uma
    funcao dos observaveis. E aqui que 'compor operacoes primitivas' acontece."""
    t = []
    for b in BASE:
        for sn, m in UNARIOS:
            t.append((f"{b}{sn}", (lambda b=b, m=m: (lambda o: o[b] % m if m else o[b]))()))
    for x in range(len(BASE)):
        for y in range(x + 1, len(BASE)):
            for on, f in BINARIOS:
                for sn, m in UNARIOS:
                    a_, b_ = BASE[x], BASE[y]
                    t.append((f"({a_}{on}{b_}){sn}",
                              (lambda a_=a_, b_=b_, f=f, m=m:
                               (lambda o: (f(o[a_], o[b_]) % m) if m else f(o[a_], o[b_])))()))
    return t


TERMOS = termos()


def residuo(tr, fm, pilha):
    """Celulas dos pares de TREINO que a pilha atual erra. Nada de teste."""
    fora = []
    for k, (a, b) in enumerate(tr):
        try:
            p = D.aplicar((fm, pilha), a)
        except Exception:
            continue
        if p is None or len(p) != len(b) or len(p[0]) != len(b[0]):
            continue
        if len(a) != len(b) or len(a[0]) != len(b[0]):
            continue
        for y in range(len(b)):
            for x in range(len(b[0])):
                if p[y][x] != b[y][x]:
                    fora.append((k, y, x, b[y][x]))
    return fora


class TermoConstruido:
    """Codec construido DA tarefa. marcador = (indice do termo, mapa valor->cor)."""

    nome = "construido"

    def __init__(self, celulas, tr):
        self.celulas = celulas
        self.tr = tr

    def propor(self, treino, alvo=None):
        if not self.celulas:
            return []
        obs, bgs = {}, {}
        for k, (a, _b) in enumerate(self.tr):
            h, w = dims(a)
            bgs[k] = fundo(a)
            obs[k] = observaveis(a, bgs[k], h, w)
        # NOTA DE DESENHO (17/09): a primeira versao exigia ZERO conflitos.
        # Isso nao estava no pre-registro e e INCONSISTENTE com o nucleo, que
        # precifica todo codec com bits de excecao. Exigir o meu perfeito era
        # uma barra que eu impus sem perceber. Agora ranqueia por conflitos e
        # deixa o MDL da pilha decidir se paga. O resultado ESTRITO fica no
        # registro: 0 termos consistentes em 6 de 6 tarefas testadas.
        cand = []
        for idx, (_nome, f) in enumerate(TERMOS):
            mapa, conf, mau = {}, 0, False
            for (k, y, x, cor) in self.celulas:
                o = obs.get(k, {}).get((y, x))
                if o is None:
                    mau = True
                    break
                v = f(o)
                if v in mapa:
                    if mapa[v] != cor:
                        conf += 1
                else:
                    mapa[v] = cor
            if mau or not mapa or len(mapa) > 10:
                continue
            if len(mapa) >= max(2, len(self.celulas)):
                continue
            cand.append((conf, len(mapa), idx, tuple(sorted(mapa.items()))))
        cand.sort()
        return [(idx, mapa) for _c, _m, idx, mapa in cand[:12]]

    def bits(self, m):
        return log2(max(len(TERMOS), 2)) + len(m[1]) * BITS_COR

    def prever(self, m, a):
        idx, mapa = m[0], dict(m[1])
        _n, f = TERMOS[idx]
        bg = fundo(a)
        h, w = dims(a)
        o = observaveis(a, bg, h, w)
        out = {}
        for y in range(h):
            for x in range(w):
                if a[y][x] != bg:
                    continue
                v = f(o[(y, x)])
                if v in mapa:
                    out[(y, x)] = mapa[v]
        return out


# ------------------------------------------------------------- os tres bracos

def celulas_r2(cel, rng):
    """R2: MESMAS celulas, cores PERMUTADAS. Mesma quantidade de informacao,
    associacao celula<->cor destruida. Placebo pareado do direcionamento."""
    cores = [c for (_k, _y, _x, c) in cel]
    rng.shuffle(cores)
    return [(k, y, x, c) for (k, y, x, _), c in zip(cel, cores)]


def celulas_r3(tr, n, rng):
    """R3: mesmo numero de celulas, sorteadas da grade em vez do residuo."""
    fora = []
    tent = 0
    while len(fora) < n and tent < 50 * max(n, 1):
        tent += 1
        k = rng.randrange(len(tr))
        a, b = tr[k]
        if len(a) != len(b) or len(a[0]) != len(b[0]):
            continue
        y = rng.randrange(len(b))
        x = rng.randrange(len(b[0]))
        fora.append((k, y, x, b[y][x]))
    return fora


def roda(tarefas, residuos, teto, etiqueta, modo, semente=SEMENTE):
    import time
    rng = random.Random(semente)
    t0 = time.time()
    ok, usou = set(), []
    for k, (tarefa, tr, te) in enumerate(tarefas):
        cel = residuos.get(tarefa, [])
        if modo == "base":
            codecs = D.CODECS
        else:
            if modo == "r1":
                c = cel
            elif modo == "r2":
                c = celulas_r2(list(cel), rng)
            else:
                c = celulas_r3(tr, len(cel), rng)
            codecs = D.CODECS + [TermoConstruido(c, tr)]
        try:
            fm, pilha, err = D.montar(tr, codecs=codecs, teto=teto)
            if pilha and err == 0 and all(D.aplicar((fm, pilha), x) == y for x, y in te):
                ok.add(tarefa)
                if modo != "base" and any(not D._marca(it) and it[0].nome == "construido"
                                          for it in pilha):
                    usou.append(tarefa)
        except Exception:
            pass
        if (k + 1) % 25 == 0:
            print(f"    [{etiqueta}] {k+1} tarefas, {len(ok)} resolvidas, "
                  f"{time.time()-t0:.0f}s", flush=True)
    return ok, usou


def main():
    import argparse, time
    from novaphys.arc import carregar
    from novaphys.eter import PASTA_AVAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefas", type=int, default=80)
    ap.add_argument("--teto", type=int, default=400)
    a = ap.parse_args()
    tarefas = carregar(PASTA_AVAL)[: a.tarefas]
    print("=" * 78)
    print(f"RESIDUO DIRIGIDO — {len(tarefas)} tarefas, teto {a.teto} (casado)")
    print("=" * 78)

    # PASSO 1: base, e o RESIDUO de cada tarefa sobre o TREINO
    print("  passo 1: rodando a base e extraindo o residuo do TREINO...", flush=True)
    base, _ = set(), None
    residuos = {}
    t0 = time.time()
    for k, (tarefa, tr, te) in enumerate(tarefas):
        try:
            fm, pilha, err = D.montar(tr, codecs=D.CODECS, teto=a.teto)
            if pilha and err == 0 and all(D.aplicar((fm, pilha), x) == y for x, y in te):
                base.add(tarefa)
            residuos[tarefa] = residuo(tr, fm, pilha) if pilha else []
        except Exception:
            residuos[tarefa] = []
        if (k + 1) % 25 == 0:
            print(f"    [base] {k+1} tarefas, {len(base)} resolvidas, "
                  f"{time.time()-t0:.0f}s", flush=True)
    tam = sorted(len(v) for v in residuos.values())
    print(f"  residuo por tarefa: mediana {tam[len(tam)//2]}, "
          f"maximo {tam[-1]}, zero em {sum(1 for t in tam if t==0)}/{len(tam)}")

    res = {}
    for modo, et in (("r1", "R1 dirigido"), ("r2", "R2 embaralhado"),
                     ("r3", "R3 sem residuo")):
        res[modo] = roda(tarefas, residuos, a.teto, et, modo)

    print("\n" + "=" * 78)
    print(f"  base            {len(base):3d}/{len(tarefas)}")
    for modo, et in (("r1", "R1 dirigido"), ("r2", "R2 embaralhado"),
                     ("r3", "R3 sem residuo")):
        o, u = res[modo]
        print(f"  {et:<16}{len(o):3d}/{len(tarefas)}   NOVAS {len(o-base):2d}, "
              f"perdeu {len(base-o):2d}, entrou na pilha em {len(u)}")
    n1 = len(res['r1'][0] - base); n2 = len(res['r2'][0] - base); n3 = len(res['r3'][0] - base)
    p1 = len(base - res['r1'][0])
    print()
    print(f"D0 {'.' if len(base)==12 else 'FORA .'} sanidade: base {len(base)} (esperado 12)")
    print(f"D1 {'.' if n1>=2 else 'FALHOU .'} R1 traz >= 2 tarefas NOVAS: {n1}")
    print(f"D2 {'.' if (n1-n2>=2 and n1-n3>=2) else 'FALHOU .'} o ganho vem do "
          f"DIRECIONAMENTO: R1-R2={n1-n2}, R1-R3={n1-n3} (>= +2 cada)")
    print(f"D3 {'.' if p1==0 else 'FALHOU .'} nao quebra: R1 perdeu {p1}")


if __name__ == "__main__":
    main()
