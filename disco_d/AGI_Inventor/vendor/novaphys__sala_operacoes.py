"""novaphys.sala_operacoes — sala de espera para OPERACOES.

Ver PRE_REGISTRO_sala_operacoes.md.

A IDEIA: `ENTAO` tem ganho ATRASADO porque as duas metades sao precificadas
SEPARADAS — o primeiro codec custa bits e nao explica celula ate o segundo
chegar, e o feixe ordenado por bits o poda antes. Empacotar (A ENTAO B) como
operacao ATOMICA com custo unico faz o ganho deixar de ser atrasado.

E a candidata NAO entra em D.CODECS ate ser promovida, entao nao gasta teto —
que e o obstaculo 1 (ampliar dilui orcamento) morrendo por construcao.
"""
from __future__ import annotations
import itertools, random, time
from math import log2

from novaphys import descricao as D
from novaphys.descricao import dims

MAX_M = 3          # marcadores por metade (o produto e o custo)
BITS_PAR = 0.0     # preenchido em tempo de execucao: log2(numero de pares)


def _mat(a, pred):
    h, w = dims(a)
    return tuple(tuple(pred.get((i, j), a[i][j]) for j in range(w)) for i in range(h))


class Par:
    """(A ENTAO B) como UMA operacao. B le a grade DEPOIS de A materializar."""

    def __init__(self, A, B, bits_nome=0.0):
        self.A, self.B = A, B
        self.nome = f"par:{A.nome}>{B.nome}"
        self._bn = bits_nome

    def propor(self, treino, alvo=None):
        try:
            ma = list(self.A.propor(treino))[:MAX_M]
            mb = list(self.B.propor(treino))[:MAX_M]
        except Exception:
            return []
        return [(x, y) for x in ma for y in mb]

    def bits(self, m):
        try:
            return self.A.bits(m[0]) + self.B.bits(m[1]) + self._bn
        except Exception:
            return 99.0

    def prever(self, m, a):
        try:
            pa = dict(self.A.prever(m[0], a))
        except Exception:
            pa = {}
        g = _mat(a, pa) if pa else a
        try:
            pb = dict(self.B.prever(m[1], g))
        except Exception:
            pb = {}
        out = dict(pa)
        out.update(pb)              # B sobrescreve: e o que ENTAO faz
        return out


# --------------------------------------------------------- a sala de espera

def _certas(pred, a, b):
    """Celulas que o codec reivindicou E acertou."""
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return set()
    return {(i, j) for (i, j), c in pred.items()
            if 0 <= i < len(b) and 0 <= j < len(b[0]) and c == b[i][j]}


def evidencia(A, B, tr, propostas, res):
    """SINAL BARATO, sem montar pilha e sem gastar teto: o par acerta, no
    RESIDUO, celula que NEM A sozinho NEM B sozinho acertam?
    Se sim, o par carrega informacao que as metades nao carregam — que e a
    definicao operacional de 'a composicao vale mais que as partes'."""
    ma = propostas.get(A.nome, [])[:MAX_M]
    mb = propostas.get(B.nome, [])[:MAX_M]
    if not ma or not mb:
        return 0
    ganho = 0
    for xa in ma:
        for xb in mb:
            for k, (a, b) in enumerate(tr):
                alvo = res.get(k)
                if not alvo:
                    continue
                try:
                    pa = dict(A.prever(xa, a))
                    ca = _certas(pa, a, b)
                    cb = _certas(dict(B.prever(xb, a)), a, b)
                    g = _mat(a, pa) if pa else a
                    pab = dict(pa); pab.update(dict(B.prever(xb, g)))
                    cab = _certas(pab, a, b)
                except Exception:
                    continue
                novas = (cab - ca - cb) & alvo
                ganho += len(novas)
    return ganho


def residuos(tarefas, teto):
    """Residuo do TREINO sob a pilha base. Nao usa saida de teste."""
    base, res = set(), {}
    for tarefa, tr, te in tarefas:
        cel = {}
        try:
            fm, pilha, err = D.montar(tr, codecs=D.CODECS, teto=teto)
            if pilha:
                if err == 0 and all(D.aplicar((fm, pilha), x) == y for x, y in te):
                    base.add(tarefa)
                for k, (a, b) in enumerate(tr):
                    p = D.aplicar((fm, pilha), a)
                    if p is None or len(p) != len(b) or len(p[0]) != len(b[0]):
                        continue
                    cel[k] = {(i, j) for i in range(len(b)) for j in range(len(b[0]))
                              if p[i][j] != b[i][j]}
        except Exception:
            pass
        res[tarefa] = cel
    return base, res


def main():
    import argparse
    from novaphys.arc import carregar
    from novaphys.eter import PASTA_AVAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefas", type=int, default=80)
    ap.add_argument("--teto", type=int, default=400)
    ap.add_argument("--k", type=int, default=3, help="tarefas de evidencia para promover")
    ap.add_argument("--promove", type=int, default=8, help="quantos pares promover")
    a = ap.parse_args()
    tarefas = carregar(PASTA_AVAL)[: a.tarefas]
    t0 = time.time()
    print("=" * 78)
    print(f"SALA DE ESPERA PARA OPERACOES — {len(tarefas)} tarefas, teto {a.teto}")
    print("=" * 78)

    base, res = residuos(tarefas, a.teto)
    comres = [(t, tr, te) for (t, tr, te) in tarefas
              if any(res.get(t, {}).values())]
    print(f"  base {len(base)}/{len(tarefas)} | tarefas com residuo: {len(comres)} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # cache do propor por tarefa (uma vez por codec, nao por par)
    print("  registrando propostas por tarefa...", flush=True)
    cache = {}
    for t, tr, _te in comres:
        d = {}
        for cd in D.CODECS:
            try:
                d[cd.nome] = list(cd.propor(tr))[:MAX_M]
            except Exception:
                d[cd.nome] = []
        cache[t] = d
    print(f"  [{time.time()-t0:.0f}s]", flush=True)

    # SALA: acumula evidencia ENTRE TAREFAS
    print("  medindo evidencia dos 506 pares (sem gastar teto)...", flush=True)
    livro = {}
    for ia, A in enumerate(D.CODECS):
        for B in D.CODECS:
            if A is B:
                continue
            tot, nt = 0, 0
            for t, tr, _te in comres:
                g = evidencia(A, B, tr, cache[t], res[t])
                if g > 0:
                    tot += g; nt += 1
            if nt:
                livro[(A.nome, B.nome)] = (nt, tot)
        if ia % 6 == 5:
            print(f"    {ia+1}/23 codecs, {len(livro)} pares com evidencia "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    ordenado = sorted(livro.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
    print(f"\n  pares com QUALQUER evidencia: {len(livro)} de 506")
    print(f"  {'tarefas':>8}{'celulas':>9}   par")
    for (na, nb), (nt, tot) in ordenado[:14]:
        print(f"  {nt:8d}{tot:9d}   {na} > {nb}")

    byn = {c.nome: c for c in D.CODECS}
    bits_nome = log2(506)
    prom = [Par(byn[na], byn[nb], bits_nome)
            for (na, nb), (nt, _g) in ordenado if nt >= a.k][: a.promove]
    print(f"\n  PROMOVIDOS (evidencia em >= {a.k} tarefas): {len(prom)}")
    for p in prom:
        print(f"     {p.nome}")

    def roda(extra, etq):
        ok, usou = set(), []
        for t, tr, te in tarefas:
            try:
                fm, pilha, err = D.montar(tr, codecs=D.CODECS + extra, teto=a.teto)
                if pilha and err == 0 and all(D.aplicar((fm, pilha), x) == y for x, y in te):
                    ok.add(t)
                    if any(not D._marca(it) and it[0].nome.startswith("par:") for it in pilha):
                        usou.append(t)
            except Exception:
                pass
        print(f"    [{etq}] {len(ok)}/{len(tarefas)} [{time.time()-t0:.0f}s]", flush=True)
        return ok, usou

    com, usou = roda(prom, "promovidos")
    rng = random.Random(20260917)
    todos = [(A, B) for A in D.CODECS for B in D.CODECS if A is not B]
    plac = [Par(A, B, bits_nome) for A, B in rng.sample(todos, len(prom))]
    pl, _ = roda(plac, "placebo sorteado")
    k1 = [Par(byn[na], byn[nb], bits_nome)
          for (na, nb), (nt, _g) in ordenado if nt >= 1][: a.promove]
    p1, _ = roda(k1, "K=1 (sem sala)")

    print("\n" + "=" * 78)
    print(f"  base              {len(base):3d}/{len(tarefas)}")
    print(f"  promovidos K={a.k}    {len(com):3d}   NOVAS {len(com-base)}, perdeu {len(base-com)}, usou em {len(usou)}")
    print(f"  placebo sorteado  {len(pl):3d}   NOVAS {len(pl-base)}, perdeu {len(base-pl)}")
    print(f"  K=1 (sem sala)    {len(p1):3d}   NOVAS {len(p1-base)}, perdeu {len(base-p1)}")
    print()
    n1, n2, n4 = len(com - base), len(pl - base), len(p1 - base)
    print(f"W0 {'.' if len(base)==12 else 'FORA .'} sanidade: base {len(base)}")
    print(f"W1 {'.' if n1>=2 else 'FALHOU .'} paga: {n1} novas (>= 2)")
    print(f"W2 {'.' if n1-n2>=2 else 'FALHOU .'} bate o placebo: {n1}-{n2} = {n1-n2} (>= +2)")
    print(f"W3 {'.' if usou else 'FALHOU .'} os promovidos APARECEM em pilha vencedora: {len(usou)}")
    print(f"W4 {'.' if n1-n4>=1 else 'FALHOU .'} a SALA importa (K={a.k} x K=1): {n1}-{n4} = {n1-n4}")


if __name__ == "__main__":
    main()
