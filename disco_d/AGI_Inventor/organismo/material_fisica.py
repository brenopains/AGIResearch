"""
organismo.material_fisica — as capacidades que ja existiam, como MATERIAL
==========================================================================
Reimplementacao limpa (o original fica intocado em AGI 2.0/novaphys) das quatro
capacidades da linha de fisica que passaram portao com placebo:

  lei        MDL guloso sobre termos  c * X1^e1 * X2^e2 * direcao   (orbita.py)
             AGORA CONJUNTO: varios conjuntos de dados (encontros diferentes do
             mesmo mundo, com escala diferente) partilham a FORMA e tem
             coeficientes proprios. Estrutura paga uma vez; coeficientes por dado.
  grandeza   invariante por autovetor generalizado sobre monomios de grau <= 2
             (inventa.py), em lote; o alfabeto CRESCE com os conceitos que o
             organismo incorporou (conceito vira atomo de conceito novo).
  direcao    girar 90 graus / somar / subtrair direcoes que ja existem
             (gramatica.py), julgada pela lei que encurta.
  pergunta   limiar sobre um atomo ou separador sobre dois, escolhido por MDL
             (a operacao "atributo exato/qualquer" do construtor e do xadrez).

Constantes do SUBSTRATO (a "fisica do cerebro", declaradas, nao sao tetos de
conteudo cognitivo): EXPS, MAX_FAT=2 (fatores por termo), MAX_K=3 (monomios por
grandeza), ANGULOS=24 (separador de pergunta), SEGURANCA_TERMOS=8 (protecao de
laco, nunca atingida nos testes de unidade), SEGURANCA_DISTINTAS=64 (idem).

Todo metodo devolve tambem o CUSTO em unidades de computo (avaliacoes de
candidato x pontos de dado / 1e4), a moeda unica do orcamento de todos os bracos.
"""
from __future__ import annotations

import itertools
from math import comb, log2

import numpy as np

from mundo_fisico import SIG_A

EXPS = np.array([-4, -3, -2, -1, 1, 2, 3, 4], dtype=float)
MAX_FAT = 2
MAX_K = 3
ANGULOS = 24
SEGURANCA_TERMOS = 8
SEGURANCA_DISTINTAS = 64
PISO = 1e-3


def custo(n):
    return float(n) / 1e4


# ─── atomos e direcoes ───────────────────────────────────────────────────────

class Conceito:
    """Grandeza inventada: suporte (monomios sobre atomos) + pesos + escala.
    Atomos podem ser outros conceitos (profundidade cresce)."""

    def __init__(self, suporte, w, esc, atomos_pais, nome):
        self.suporte = tuple(suporte)       # tuplas de nomes de atomos
        self.w = np.asarray(w, float)
        self.esc = float(esc)
        self.pais = dict(atomos_pais)       # nome -> Conceito (so os usados)
        self.nome = nome

    @property
    def profundidade(self):
        return 1 + max([c.profundidade for c in self.pais.values()], default=0)

    def chave_suporte(self):
        return tuple(sorted(self.suporte))

    def __call__(self, S):
        A = atomos_inv(S, self.pais)
        cols = [_mono(A, m) for m in self.suporte]
        return (np.stack(cols, -1) @ self.w) / self.esc


def _mono(A, m):
    out = A[m[0]]
    for a in m[1:]:
        out = out * A[a]
    return out


def atomos_inv(S, conceitos):
    x, y, vx, vy = S[..., 0], S[..., 1], S[..., 2], S[..., 3]
    r = np.hypot(x, y)
    A = {"x": x, "y": y, "vx": vx, "vy": vy, "r": r, "1/r": 1 / r}
    for nome, c in conceitos.items():
        A[nome] = c(S)
    return A


def escalares(S, conceitos):
    """Variaveis da lei: invariantes a rotacao de base + conceitos incorporados."""
    x, y, vx, vy = S.T
    r = np.hypot(x, y)
    V = {"r": r, "v": np.hypot(vx, vy), "vr": (x * vx + y * vy) / r}
    for nome, c in conceitos.items():
        V[nome] = c(S)
    nomes = list(V)
    return nomes, np.stack([V[n] for n in nomes])


DIR_BASE = ("r^", "t^", "v^")


def _base_dirs(S):
    x, y, vx, vy = S.T
    r = np.hypot(x, y)
    v = np.maximum(np.hypot(vx, vy), 1e-9)
    return {"r^": np.stack([x / r, y / r], 1),
            "t^": np.stack([-y / r, x / r], 1),
            "v^": np.stack([vx / v, vy / v], 1)}


def avalia_dir(expr, base):
    """expr: nome base | ('rot', e) | ('soma', a, b) | ('dif', a, b)."""
    if isinstance(expr, str):
        return base[expr]
    op = expr[0]
    if op == "rot":
        u = avalia_dir(expr[1], base)
        return np.stack([-u[:, 1], u[:, 0]], 1)
    a, b = avalia_dir(expr[1], base), avalia_dir(expr[2], base)
    w = a + b if op == "soma" else a - b
    n = np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-9)
    return w / n


def nome_dir(expr):
    if isinstance(expr, str):
        return expr
    if expr[0] == "rot":
        return f"rot({nome_dir(expr[1])})"
    s = "+" if expr[0] == "soma" else "-"
    return f"({nome_dir(expr[1])}{s}{nome_dir(expr[2])})"


def direcoes(S, extras):
    base = _base_dirs(S)
    out = dict(base)
    for e in extras:
        out[nome_dir(e)] = avalia_dir(e, base)
    return out


# ─── a lei ───────────────────────────────────────────────────────────────────

_CANDS = {}


def candidatos(P, definido=None):
    """Expoente negativo so para variavel de sinal definido nos dados (uma
    variavel que cruza zero tem potencia negativa singular — regra do substrato)."""
    definido = tuple(bool(d) for d in (definido if definido is not None else [True] * P))
    chave = (P, definido)
    if chave not in _CANDS:
        linhas, meta = [], []
        for k in range(1, MAX_FAT + 1):
            for vs in itertools.combinations(range(P), k):
                for es in itertools.product(EXPS, repeat=k):
                    if any(x < 0 and not definido[v] for v, x in zip(vs, es)):
                        continue
                    e = np.zeros(P)
                    e[list(vs)] = es
                    linhas.append(e)
                    meta.append((k, vs, es))
        _CANDS[chave] = (np.array(linhas), meta)
    return _CANDS[chave]


def sinal_definido(Xs):
    """Xs: lista de (P, N). Variavel tem sinal definido se nao cruza zero em nenhum."""
    P = Xs[0].shape[0]
    out = []
    for i in range(P):
        pos = all((X[i] > 0).all() for X in Xs)
        neg = all((X[i] < 0).all() for X in Xs)
        out.append(pos or neg)
    return out


def _F(E, X):
    la = np.log(np.maximum(np.abs(X), PISO))
    neg = (X < 0).astype(float)
    par = (np.abs(E) % 2) @ neg
    with np.errstate(over="ignore", invalid="ignore"):
        out = (1 - 2 * (par % 2)) * np.exp(np.clip(E @ la, -60, 60))
    return out


def bits_estrutura(P, k, ndirs):
    return 1 + log2(ndirs) + log2(MAX_FAT) + log2(comb(P, k) * len(EXPS) ** k)


class Lei:
    """Forma (termos) + coeficientes de UM conjunto de dados. Chamavel."""

    def __init__(self, termos, coefs, conceitos, extras):
        self.termos = list(termos)       # (vars nomes, exps, dir nome)
        self.coefs = np.asarray(coefs, float)
        self.conceitos = dict(conceitos)
        self.extras = list(extras)

    def forma(self):
        return tuple(sorted((tuple(v), tuple(float(x) for x in e), d)
                            for v, e, d in self.termos))

    def base(self, S):
        nomes, X = escalares(S, self.conceitos)
        idx = {n: j for j, n in enumerate(nomes)}
        dirs = direcoes(S, self.extras)
        cols = []
        for vs, es, d in self.termos:
            e = np.zeros(len(nomes))
            for v, x in zip(vs, es):
                e[idx[v]] = x
            cols.append(_F(e[None, :], X)[0][:, None] * dirs[d])
        return cols

    def __call__(self, S):
        cols = self.base(S)
        out = np.zeros((len(S), 2))
        for c, col in zip(self.coefs, cols):
            out = out + c * col
        return out

    def __repr__(self):
        ts = []
        for (vs, es, d), c in zip(self.termos, self.coefs):
            f = "*".join(f"{v}^{int(e)}" for v, e in zip(vs, es))
            ts.append(f"{c:+.3g} {f} {d}")
        return " ".join(ts) if ts else "0"


def _refit(cols_por_dado, A_por_dado):
    coefs = []
    for cols, A in zip(cols_por_dado, A_por_dado):
        if not cols:
            coefs.append(np.zeros(0))
            continue
        M = np.stack([c.ravel() for c in cols], 1)
        c, *_ = np.linalg.lstsq(M, A.ravel(), rcond=None)
        coefs.append(c)
    return coefs


def ajusta_lei(dados, conceitos, extras, forma=None, limite=float("inf")):
    """dados: lista de (S, A). Devolve (formas+coefs por dado como Lei, bits, custo).
    Se `forma` e dada, so reajusta os coeficientes (barato)."""
    custo_total = 0.0
    if forma is not None:
        termos = [(tuple(v), tuple(e), d) for v, e, d in forma]
        leis = []
        cols_d = []
        for S, A in dados:
            L = Lei(termos, np.zeros(len(termos)), conceitos, extras)
            cols_d.append(L.base(S))
        coefs = _refit(cols_d, [A for _S, A in dados])
        for c in coefs:
            leis.append(Lei(termos, c, conceitos, extras))
        custo_total += custo(sum(len(S) for S, _ in dados) * max(len(termos), 1))
        return leis, bits_lei(leis, dados), custo_total

    nomes, _ = escalares(dados[0][0], conceitos)
    P = len(nomes)
    Xs, dirs_d = [], []
    for S, _A in dados:
        _n, X = escalares(S, conceitos)
        Xs.append(X)
        dirs_d.append(direcoes(S, extras))
    E, meta = candidatos(P, sinal_definido(Xs))
    dnames = list(dirs_d[0])
    nd = len(dnames)
    Fs = [_F(E, X) for X in Xs]          # (ncand, N_d)
    ok_all = np.logical_and.reduce([np.isfinite(F).all(1) for F in Fs])
    dens = [np.where(ok_all[:, None], F, 0.0) for F in Fs]
    den2 = [(F ** 2).sum(1) for F in dens]
    est = np.array([bits_estrutura(P, k, nd) for k in range(1, MAX_FAT + 1)])
    kv = np.array([m[0] for m in meta])
    neffs = [A.size for _S, A in dados]
    coef_bits = sum(0.5 * log2(n) for n in neffs)
    As = [A for _S, A in dados]
    estado = {"custo": 0.0}

    def termo_de(i, di):
        k, vs, es = meta[i]
        return (tuple(nomes[j] for j in vs), tuple(float(x) for x in es), dnames[di])

    def colunas(termos):
        return [Lei(termos, np.zeros(len(termos)), conceitos, extras).base(S)
                for S, _A in dados]

    def residuo_e_base(termos):
        """Residuo do ajuste conjunto e base ortonormal das colunas (por dado)."""
        if not termos:
            return [A.copy() for A in As], [None] * len(dados)
        cols_d = colunas(termos)
        coefs = _refit(cols_d, As)
        R, Qs = [], []
        for cols, c, A in zip(cols_d, coefs, As):
            R.append(A - sum(cc * col for cc, col in zip(c, cols)))
            M = np.stack([col.ravel() for col in cols], 1)
            Q, _ = np.linalg.qr(M)
            Qs.append(Q.reshape(len(A), 2, -1))
        return R, Qs

    def melhor_candidato(R, Qs, proibidos):
        """Busca ortogonal: ganho de cada candidato DEPOIS de descontar o que os
        termos ja aceitos explicam (OMP)."""
        melhor_g = np.full(len(E), -np.inf)
        melhor_d = np.zeros(len(E), int)
        for di, d in enumerate(dnames):
            g_d = np.zeros(len(E))
            for j in range(len(dados)):
                U = dirs_d[j][d]
                sse = float((R[j] ** 2).sum())
                num = dens[j] @ (R[j] * U).sum(1)
                den = den2[j].copy()
                if Qs[j] is not None:
                    G = np.einsum("nck,nc->nk", Qs[j], U)
                    Pj = dens[j] @ G
                    den = den - (Pj ** 2).sum(1)
                bom = den > 1e-9 * np.maximum(den2[j], 1e-300)
                red = np.where(bom, num ** 2 / np.where(bom, den, 1.0), 0.0)
                red = np.minimum(red, sse * (1 - 1e-12))
                g_d += 0.5 * neffs[j] * np.log2(sse / (sse - red))
            m = g_d > melhor_g
            melhor_g = np.where(m, g_d, melhor_g)
            melhor_d = np.where(m, di, melhor_d)
        estado["custo"] += custo(len(E) * nd * sum(neffs))
        tot = melhor_g - est[kv - 1] - coef_bits
        tot = np.where(ok_all, tot, -np.inf)
        for t in proibidos:
            pass
        ordem = np.argsort(-tot)
        for i in ordem[:8]:
            t = termo_de(int(i), int(melhor_d[i]))
            if t not in proibidos:
                return float(tot[i]), t
        return -np.inf, None

    def bits_de(termos):
        cols_d = colunas(termos)
        coefs = _refit(cols_d, As)
        leis = [Lei(termos, c, conceitos, extras) for c in coefs]
        return bits_lei(leis, dados) if termos else _bits_vazio(dados, P, nd)

    # 1) guloso ortogonal
    termos = []
    for _ in range(SEGURANCA_TERMOS):
        if estado["custo"] >= limite:
            break
        R, Qs = residuo_e_base(termos)
        g, t = melhor_candidato(R, Qs, termos)
        if t is None or g <= 0:
            break
        termos.append(t)
    # 2) trocas e remocoes enquanto o comprimento de descricao cair
    b_atual = bits_de(termos)
    for _passada in range(3):
        mudou = False
        for pos in range(len(termos)):
            if pos >= len(termos) or estado["custo"] >= limite:
                break
            outros = termos[:pos] + termos[pos + 1:]
            R, Qs = residuo_e_base(outros)
            _g, t = melhor_candidato(R, Qs, termos)
            opcoes = [outros] + ([outros[:pos] + [t] + outros[pos:]] if t is not None else [])
            for op in opcoes:
                b = bits_de(op)
                if b < b_atual - 1e-9:
                    termos, b_atual, mudou = list(op), b, True
                    break
        if not mudou:
            break
    custo_total += estado["custo"]
    coefs = _refit(colunas(termos), As) if termos else [np.zeros(0) for _ in dados]
    leis = [Lei(termos, c, conceitos, extras) for c in coefs]
    return leis, b_atual, custo_total


def _bits_vazio(dados, P, nd):
    b = 0.0
    for _S, A in dados:
        sse = float((A ** 2).sum())
        b += 0.5 * A.size * log2(max(sse / A.size, 1e-300))
    return b


def bits_lei(leis, dados):
    """Comprimento de descricao: erro (gaussiano, sigma livre) + estrutura + coefs."""
    if not leis:
        return float("inf")
    L = leis[0]
    if not L.termos:
        return _bits_vazio(dados, 0, 0)
    nomes, _ = escalares(dados[0][0], L.conceitos)
    P = len(nomes)
    nd = len(direcoes(dados[0][0], L.extras))
    b = 0.0
    for lei, (S, A) in zip(leis, dados):
        sse = float(((A - lei(S)) ** 2).sum())
        b += 0.5 * A.size * log2(max(sse / A.size, 1e-300))
        b += len(L.termos) * 0.5 * log2(A.size)
    for vs, _es, _d in L.termos:
        b += bits_estrutura(P, len(vs), nd)
    return b


def qui2(lei, S, A):
    r = A - lei(S)
    return float((r ** 2).mean() / SIG_A ** 2)


def limiar_qui2(n):
    return 1 + 3 * np.sqrt(2 / max(n, 1))


def valida_lei(dados, conceitos, extras, forma, rng):
    """Validacao interna (o organismo nao ve o verificador): ajusta a forma em
    2/3 de cada dado e mede qui2 no 1/3 restante. Devolve (passa, qui2, custo)."""
    tr, ho = [], []
    for S, A in dados:
        idx = rng.permutation(len(S))
        c = max(2, len(S) // 3)
        ho.append((S[idx[:c]], A[idx[:c]]))
        tr.append((S[idx[c:]], A[idx[c:]]))
    leis, _b, cst = ajusta_lei(tr, conceitos, extras, forma=forma)
    q, n = 0.0, 0
    for lei, (S, A) in zip(leis, ho):
        q += qui2(lei, S, A) * A.size
        n += A.size
    q /= max(n, 1)
    return q <= limiar_qui2(n), q, cst


# ─── a grandeza conservada ───────────────────────────────────────────────────

def alfabeto_inv(T, conceitos):
    A = atomos_inv(T, conceitos)
    at = list(A)
    nomes, cols = [], []
    for a in at:
        nomes.append((a,))
        cols.append(A[a])
    for i in range(len(at)):
        for j in range(i, len(at)):
            nomes.append((at[i], at[j]))
            cols.append(A[at[i]] * A[at[j]])
    return nomes, np.stack(cols, -1)


def _covs(F):
    n, p, m = F.shape
    G = F.reshape(-1, m)
    G = G - G.mean(0)
    Tc = G.T @ G / len(G)
    Fc = F - F.mean(1, keepdims=True)
    W = np.einsum("npi,npj->ij", Fc, Fc) / (n * p)
    return W, Tc


def _lam(Q):
    dentro = ((Q - Q.mean(1, keepdims=True)) ** 2).mean()
    total = ((Q - Q.mean()) ** 2).mean()
    return dentro / total if total > 0 else 1.0


def poupanca(lam, N, ntraj):
    return 0.5 * N * log2(1 / max(lam, 1e-300)) - 0.5 * ntraj * log2(max(N / ntraj, 1))


def estrutura_inv(m, k, N):
    return log2(MAX_K) + log2(comb(m, k)) + 0.5 * (k - 1) * log2(N)


def _gen_eig_lote(W, Tc, idx):
    """Menor autovalor generalizado de (W_s, T_s) para cada subconjunto (linhas de idx)."""
    k = idx.shape[1]
    Ts = Tc[idx[:, :, None], idx[:, None, :]]
    Ws = W[idx[:, :, None], idx[:, None, :]]
    ev = np.linalg.eigvalsh(Ts)
    ok = ev[:, 0] > 1e-9 * np.maximum(ev[:, -1], 1e-300)
    lam = np.full(len(idx), np.inf)
    vec = np.zeros((len(idx), k))
    if ok.any():
        L = np.linalg.cholesky(Ts[ok])
        Li = np.linalg.inv(L)
        M = Li @ Ws[ok] @ np.transpose(Li, (0, 2, 1))
        M = 0.5 * (M + np.transpose(M, (0, 2, 1)))
        vals, vecs = np.linalg.eigh(M)
        y = vecs[:, :, 0]
        w = np.einsum("nji,nj->ni", Li, y)
        lam[ok] = vals[:, 0]
        vec[ok] = w
    return lam, vec


def inventa(Ttr, Tho, conceitos, atomos_permitidos=None):
    """Todas as invariantes distintas com poupanca positiva no treino E fora da
    amostra (trajetorias separadas). Sem limite de quantidade: a selecao e da
    economia do organismo. Devolve (lista de (suporte, w, ganho, esc), custo)."""
    cs = conceitos if atomos_permitidos is None else \
        {k: v for k, v in conceitos.items() if k in atomos_permitidos}
    nomes, Ftr = alfabeto_inv(Ttr, cs)
    _, Fho = alfabeto_inv(Tho, cs)
    m = len(nomes)
    W, Tc = _covs(Ftr)
    Ntr = Ftr.shape[0] * Ftr.shape[1]
    Nho = Fho.shape[0] * Fho.shape[1]
    cands = []
    n_sub = 0
    for k in range(1, MAX_K + 1):
        idx = np.array(list(itertools.combinations(range(m), k)))
        n_sub += len(idx)
        lam, vec = _gen_eig_lote(W, Tc, idx)
        g = np.array([poupanca(max(l, 1e-300), Ntr, Ftr.shape[0]) if np.isfinite(l)
                      else -np.inf for l in lam]) - estrutura_inv(m, k, Ntr)
        for i in np.where(g > 0)[0]:
            cands.append((float(g[i]), idx[i].tolist(), vec[i]))
    cst = custo(n_sub * Ntr)
    cands.sort(key=lambda c: -c[0])
    ref = Ttr.reshape(-1, 4)[::3]
    Aref = alfabeto_inv(ref, cs)[1]
    Aref = (Aref - Aref.mean(0)) / (Aref.std(0) + 1e-12)
    kept_q = np.zeros((0, len(ref)))
    escolhidas = []
    vistos = 0
    for g, s, w in cands:
        vistos += 1
        q_ho = Fho[..., s] @ w
        if poupanca(_lam(q_ho), Nho, Fho.shape[0]) <= 0:
            continue
        q = Aref[:, s] @ w
        sd = q.std()
        if sd < 1e-12:
            continue
        q = (q - q.mean()) / (sd * np.sqrt(len(q)))
        if len(kept_q) and np.abs(kept_q @ q).max() > 0.95:
            continue
        kept_q = np.vstack([kept_q, q])
        esc = float((Ftr[..., s] @ w).std())
        escolhidas.append((tuple(nomes[i] for i in s), w, g, esc))
        if len(escolhidas) >= SEGURANCA_DISTINTAS:
            break
    cst += custo(vistos * len(ref))
    # corte na maior quebra relativa de ganho (comparacao, nao limiar absoluto)
    if len(escolhidas) > 1:
        gs = np.array([e[2] for e in escolhidas])
        razoes = np.log(gs[:-1] / np.maximum(gs[1:], 1e-9))
        escolhidas = escolhidas[:int(np.argmax(razoes)) + 1]
    pais_usados = lambda sup: {a: cs[a] for mono in sup for a in mono if a in cs}
    out = [(sup, w, g, esc, pais_usados(sup)) for sup, w, g, esc in escolhidas]
    return out, cst


def reajusta_suporte(conc, T):
    """Mesmo suporte, pesos refeitos nas trajetorias atuais (variante de contexto)."""
    A = atomos_inv(T, conc.pais)
    F = np.stack([_mono(A, m) for m in conc.suporte], -1)
    W, Tc = _covs(F)
    lam, vec = _gen_eig_lote(W, Tc, np.arange(len(conc.suporte))[None, :])
    cst = custo(F.shape[0] * F.shape[1] * len(conc.suporte))
    if not np.isfinite(lam[0]):
        return None, lam[0], cst
    w = vec[0]
    esc = float((F @ w).std())
    if esc < 1e-12:
        return None, lam[0], cst
    return Conceito(conc.suporte, w, esc, conc.pais, conc.nome), float(lam[0]), cst


def lam_conceito(conc, T):
    try:
        return float(_lam(conc(T))), custo(T.shape[0] * T.shape[1])
    except Exception:
        return 1.0, 0.0


# ─── direcoes derivadas ──────────────────────────────────────────────────────

def deriva_direcoes(extras, S):
    """Candidatas: girar cada direcao; somar/subtrair pares. Sem duplicatas."""
    base = _base_dirs(S)
    exist = list(DIR_BASE) + list(extras)
    vals = [avalia_dir(e, base) for e in exist]
    cands = []
    for e in exist:
        cands.append(("rot", e))
    for a, b in itertools.combinations(exist, 2):
        cands.append(("soma", a, b))
        cands.append(("dif", a, b))
    novos = []
    for c in cands:
        u = avalia_dir(c, base)
        dup = False
        for w in vals + [avalia_dir(n, base) for n in novos]:
            if abs(float((u * w).sum(1).mean())) >= 0.999:
                dup = True
                break
        if not dup:
            novos.append(c)
    return novos, custo(len(cands) * len(S))


# ─── a pergunta ──────────────────────────────────────────────────────────────

def atomos_q(S, conceitos):
    x, y, vx, vy = S.T
    r = np.hypot(x, y)
    V = {"r": r, "v": np.hypot(vx, vy), "vr": (x * vx + y * vy) / r,
         "x": x, "y": y, "vx": vx, "vy": vy}
    for nome, c in conceitos.items():
        V[nome] = c(S)
    return V


def _melhor_limiar(z, y):
    """Melhor limiar e sinal para separar y por z. Devolve (erros, limiar, sinal)."""
    o = np.argsort(z)
    zs, ys = z[o], y[o]
    n = len(y)
    # rotulo 1 acima do limiar: erros = uns abaixo + zeros acima
    uns_ab = np.concatenate([[0], np.cumsum(ys)])
    zeros_ac = (n - np.arange(n + 1)) - (ys.sum() - uns_ab)
    e1 = uns_ab + zeros_ac
    e0 = n - e1
    i1, i0 = int(np.argmin(e1)), int(np.argmin(e0))
    def lim(i):
        if i == 0:
            return zs[0] - 1e-9
        if i == n:
            return zs[-1] + 1e-9
        return 0.5 * (zs[i - 1] + zs[i])
    if e1[i1] <= e0[i0]:
        return int(e1[i1]), lim(i1), 1
    return int(e0[i0]), lim(i0), -1


def prever_q(Sex, yex, Sq, conceitos, atomos=None):
    """Escolhe por MDL entre limiar de 1 atomo e separador linear de 2 atomos.
    Devolve (rotulos das consultas, acuracia leave-one-out estimada, descricao, custo)."""
    V = atomos_q(Sex, conceitos)
    Vq = atomos_q(Sq, conceitos)
    nomes = [a for a in V if atomos is None or a in atomos]
    n = len(yex)
    Z = {a: (V[a] - V[a].mean()) / (V[a].std() + 1e-12) for a in nomes}
    Zq = {a: (Vq[a] - V[a].mean()) / (V[a].std() + 1e-12) for a in nomes}
    na = len(nomes)
    melhor = None
    for a in nomes:
        err, lim, sg = _melhor_limiar(Z[a], yex)
        bits = err * log2(max(n, 2)) + log2(max(na, 1)) + log2(n + 1)
        if melhor is None or bits < melhor[0]:
            melhor = (bits, ("1", a, lim, sg))
    angs = np.linspace(0, np.pi, ANGULOS, endpoint=False)
    for a, b in itertools.combinations(nomes, 2):
        for th in angs:
            z = np.cos(th) * Z[a] + np.sin(th) * Z[b]
            err, lim, sg = _melhor_limiar(z, yex)
            bits = (err * log2(max(n, 2)) + log2(max(comb(na, 2), 1)) + log2(ANGULOS)
                    + log2(n + 1) + 1)
            if bits < melhor[0]:
                melhor = (bits, ("2", a, b, th, lim, sg))
    cst = custo((na + comb(na, 2) * ANGULOS) * n * 2)

    def aplica(desc, Zd):
        if desc[0] == "1":
            _, a, lim, sg = desc
            z = Zd[a]
        else:
            _, a, b, th, lim, sg = desc
            z = np.cos(th) * Zd[a] + np.sin(th) * Zd[b]
        return ((z > lim).astype(int) if sg == 1 else (z <= lim).astype(int))

    desc = melhor[1]
    # leave-one-out barato: refaz so o limiar da descricao escolhida
    acertos = 0
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        if desc[0] == "1":
            z = Z[desc[1]]
        else:
            z = np.cos(desc[3]) * Z[desc[1]] + np.sin(desc[3]) * Z[desc[2]]
        _e, lim, sg = _melhor_limiar(z[m], yex[m])
        p = int(z[i] > lim) if sg == 1 else int(z[i] <= lim)
        acertos += int(p == yex[i])
    cst += custo(n * n)
    return aplica(desc, Zq), acertos / n, desc, cst
