"""novaphys.residuo_farmaco — o residuo do farmaco nomeia o alvo que ninguem anotou.

Pilha (e a pilha do nucleo, nao um metodo novo):
  camada 0  first-writer: eixo de estresse/ciclo — o que TODO farmaco em dose
            alta faz. NAO se residualiza por regressao num confundidor colinear
            (foi o que matou o PRISM e a hipermutacao); escreve-se primeiro.
  camada 1  first-writer: assinatura do ALVO ANOTADO, tirada do dicionario de
            knockdown. Tambem escrita primeiro.
  residuo   o que sobrou.
  camada 2  qual gene do dicionario comprime o residuo, em BITS.

Ver PRE_REGISTRO_residuo_farmaco.md.
"""
import json
import numpy as np

# ---------------------------------------------------------------- utilidades

def z(M, eixo=1):
    """Centra e escala cada linha; NaN vira 0 depois de centrar."""
    M = np.asarray(M, np.float32)
    mu = np.nanmean(M, axis=eixo, keepdims=True)
    sd = np.nanstd(M, axis=eixo, keepdims=True)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    out = (M - np.where(np.isfinite(mu), mu, 0.0)) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _proj_out(R, B):
    """Remove de cada linha de R a projecao no subespaco gerado por B (k x G)."""
    if B is None or len(B) == 0:
        return R
    Q, _ = np.linalg.qr(np.asarray(B, np.float32).T)      # G x k ortonormal
    return (R - (R @ Q) @ Q.T).astype(np.float32)


def camada0(D, k=8):
    """Eixo de estresse: as k direcoes que TODO farmaco compartilha.
    Sao as componentes principais da propria matriz de farmaco — isto e, o que
    e comum a tudo. Devolve a base (k x G)."""
    A = D - D.mean(0, keepdims=True)
    # SVD economica via Gram no espaco de genes (G << n em geral nao vale; usa randomizada)
    rng = np.random.default_rng(0)
    G = A.shape[1]
    Om = rng.standard_normal((G, k + 10), dtype=np.float32)
    Y = A @ Om
    Q, _ = np.linalg.qr(Y)
    B = (Q.T @ A)                                          # (k+10) x G
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    return Vt[:k].astype(np.float32)


def camada0_cv(D, k=8, folds=5, semente=0):
    """Camada 0 com AJUSTE CRUZADO. Estimar a base de estresse no MESMO dado de
    onde ela e removida vaza: o PCA aprende parte do dicionario, e o residuo sai
    torto na direcao dele (medido: razao 1,29 contra 0,99 com base independente).
    Aqui a base de cada dobra vem das OUTRAS dobras."""
    n = D.shape[0]
    rng = np.random.default_rng(semente)
    dobra = rng.integers(0, folds, n)
    R = np.empty_like(D)
    for f in range(folds):
        sel = dobra == f
        if not sel.any():
            continue
        B = camada0(D[~sel], k=k)
        R[sel] = _proj_out(D[sel], B)
    return R


def nulo_do_duto(D, S, alvos, gidx, n_rep=10, **kw):
    """O nulo tem que passar pelo DUTO INTEIRO, nao so pelo fim.
    Permuta as coordenadas de gene do dado CRU e re-roda tudo — inclusive a
    ESTIMATIVA da camada de estresse. Assim o artefato da propria derivacao
    (estimar um subespaco no mesmo dado de onde ele e removido) entra no nulo em
    vez de virar achado. Mesmo controle que matou 15 das 18 anunciadas em zeta.
    """
    rng = np.random.default_rng(12345)
    out = []
    for _ in range(n_rep):
        pi = rng.permutation(D.shape[1])
        B, _ = pipeline(D[:, pi], S, alvos, gidx, n_nulos=0, **kw)
        out.append(float(np.nanmax(B[np.isfinite(B)])))
    return np.array(out)


def pipeline(D, S, alvos, gidx, k=8, folds=5, semente=0, n_nulos=10):
    """Pilha completa com as duas correcoes que o sintetico exigiu:
      (1) AJUSTE CRUZADO — a base de estresse de cada dobra vem das outras;
      (2) MESMO ESPACO  — o dicionario passa pela MESMA projecao do residuo.
    Sem (2), <r,s> vive no subespaco do residuo mas <s,s> vive no espaco inteiro,
    e assinaturas com muita massa no subespaco removido saem mal calibradas.
    """
    n = D.shape[0]
    rng = np.random.default_rng(semente)
    dobra = rng.integers(0, folds, n)
    B = np.full((n, S.shape[0]), -np.inf, np.float32)
    nulos = []
    for f in range(folds):
        sel = np.where(dobra == f)[0]
        if len(sel) == 0:
            continue
        base = camada0(D[dobra != f], k=k)
        Rf = _proj_out(D[sel], base)
        Sf = z(_proj_out(S, base))                 # << o dicionario no MESMO espaco
        Rf, _ = camada1(Rf, [alvos[i] for i in sel], Sf, gidx)
        Rf = z(Rf)
        ban = {i: [gidx[g] for g in alvos[j] if g in gidx]
               for i, j in enumerate(sel)}
        B[sel] = varre(Rf, Sf, testaveis=S.shape[0], banido=ban)
        if n_nulos:
            nulos.append(nulo_max(Rf, Sf, n_rep=n_nulos, testaveis=S.shape[0], semente=f))
    return B, (np.concatenate(nulos) if nulos else np.array([]))


def camada1(D, alvos, S, gidx):
    """Remove de cada condicao a assinatura dos seus alvos ANOTADOS.
    alvos: lista (por linha de D) de listas de simbolos. S: dicionario (nG x G).
    Feito linha a linha porque cada farmaco tem alvo diferente."""
    R = D.copy()
    tocadas = 0
    for i, lista in enumerate(alvos):
        idx = [gidx[g] for g in lista if g in gidx]
        if not idx:
            continue
        R[i:i + 1] = _proj_out(R[i:i + 1], S[idx])
        tocadas += 1
    return R, tocadas


# ------------------------------------------------------------------ varredura

def varre(R, S, testaveis=None, banido=None):
    """Bits que cada assinatura de knockdown economiza em cada residuo.

    R (n x G) residuo ja z-scored. S (m x G) dicionario ja z-scored.
    Ajuste de um coeficiente: r' = r - alpha*s, alpha = <r,s>/<s,s>.
    Reducao de variancia -> bits; menos o custo de ESTRUTURA (nomear o candidato
    e escrever alpha), contado so sobre candidatos TESTAVEIS — o defeito de
    contabilidade que custou 3,7 bits por par no DepMap.
    """
    R = np.ascontiguousarray(R, np.float32)
    S = np.ascontiguousarray(S, np.float32)
    n, G = R.shape
    m = S.shape[0]
    ss = np.einsum("ij,ij->i", S, S)                       # <s,s>
    ss = np.where(ss > 1e-6, ss, np.inf).astype(np.float32)
    rr = np.einsum("ij,ij->i", R, R)                       # <r,r>
    rr = np.maximum(rr, 1e-6).astype(np.float32)

    C = R @ S.T                                            # n x m   <r,s>
    expl = (C * C) / ss[None, :]                           # variancia explicada
    frac = np.clip(1.0 - expl / rr[:, None], 1e-9, 1.0)
    bits = -0.5 * G * np.log2(frac)                        # ganho bruto

    nte = testaveis if testaveis else m
    custo = np.log2(max(nte, 2)) + 0.5 * np.log2(max(G, 2))
    bits -= custo
    if banido is not None:
        for i, js in banido.items():
            if js:
                bits[i, list(js)] = -np.inf
    return bits.astype(np.float32)


# ------------------------------------------------------------------- nulo

def nulo_max(R, S, n_rep=10, testaveis=None, semente=0):
    """Nulo que quebra o alinhamento farmaco<->gene preservando as duas margens:
    permuta as COORDENADAS DE GENE do residuo (uma permutacao por replica, a
    mesma para todas as condicoes, para nao destruir a correlacao entre elas).
    Devolve o maximo de bits observado em cada replica."""
    rng = np.random.default_rng(semente)
    G = R.shape[1]
    out = []
    for _ in range(n_rep):
        pi = rng.permutation(G)
        B = varre(R[:, pi], S, testaveis=testaveis)
        out.append(float(np.nanmax(B)))
    return np.array(out)


def limiar_fd(nulos_max, n_cond, alvo_fd=1.0):
    """Limiar por falsas-descobertas-esperadas <= alvo_fd.
    Cada replica do nulo da o maximo sobre n_cond condicoes; o limiar que deixa
    passar alvo_fd por rodada e o quantil correspondente dos maximos."""
    if len(nulos_max) == 0:
        return 0.0
    return float(np.nanmax(nulos_max))


# ------------------------------------------------------- verdade plantada

def plantado(n=400, G=1500, m=300, semente=7, efeitos=(1.0, 0.5, 0.25, 0.12),
             k_estresse=6, ruido=1.0):
    """Gera farmaco/knockdown falsos com SEGUNDO ALVO plantado em forca conhecida.
    Mede: recupera? e quantos espurios? — antes de ver o dado real."""
    rng = np.random.default_rng(semente)
    S = rng.standard_normal((m, G), np.float32)            # dicionario de knockdown
    E = rng.standard_normal((k_estresse, G), np.float32)   # eixo de estresse comum

    D = np.zeros((n, G), np.float32)
    alvo_anot = rng.integers(0, m, n)                      # alvo anotado
    alvo_2o = np.full(n, -1)
    forca = np.zeros(n, np.float32)
    grupos = np.array_split(np.arange(n), len(efeitos) + 1)
    for gi, idx in enumerate(grupos):
        for i in idx:
            v = ruido * rng.standard_normal(G).astype(np.float32)
            v += 3.0 * (rng.standard_normal(k_estresse).astype(np.float32) @ E)  # estresse forte
            v += 2.0 * S[alvo_anot[i]]                                           # alvo anotado
            if gi < len(efeitos):
                j = int(rng.integers(0, m))
                while j == alvo_anot[i]:
                    j = int(rng.integers(0, m))
                v += efeitos[gi] * S[j]
                alvo_2o[i] = j
                forca[i] = efeitos[gi]
            D[i] = v
    return D, S, alvo_anot, alvo_2o, forca, efeitos


def roda_plantado(**kw):
    D, S, anot, segundo, forca, efeitos = plantado(**kw)
    gidx = {i: i for i in range(S.shape[0])}
    Dz, Sz = z(D), z(S)

    alvos = [[a] for a in anot]
    B, _ = pipeline(Dz, Sz, alvos, gidx, k=8, folds=5, n_nulos=0)
    nm = nulo_do_duto(Dz, Sz, alvos, gidx, n_rep=10, k=8, folds=5)
    lim = limiar_fd(nm, Dz.shape[0])
    print(f"  nulo: max por replica {np.round(nm,1)}")
    print(f"  LIMIAR = {lim:.1f} bits (max do nulo em 10 replicas)")
    melhor = B.argmax(1)
    bits = B.max(1)
    print(f"  {'forca':>8} {'n':>5} {'acertou':>9} {'espurio':>9}")
    for f in list(efeitos) + [0.0]:
        sel = np.isclose(forca, f)
        if not sel.any():
            continue
        aceito = sel & (bits > lim)
        if f > 0:
            ok = aceito & (melhor == segundo)
            print(f"  {f:8.2f} {sel.sum():5d} {ok.sum():9d} {int((aceito & ~ok).sum()):9d}")
        else:
            print(f"  {'nenhum':>8} {sel.sum():5d} {'-':>9} {int(aceito.sum()):9d}   <- espurios puros")
    return B, segundo, forca


if __name__ == "__main__":
    print("VERDADE PLANTADA — poder do duto antes de ver o dado real")
    print("(estresse comum 3x mais forte que o alvo anotado, que e 2x o 2o alvo)")
    roda_plantado()


# ------------------------------------- verdade plantada, com REPLICACAO

def plantado2(n_drogas=120, n_lin=20, G=1200, m=250, semente=11,
              efeitos=(0.8, 0.4, 0.2, 0.1), k_estresse=6, ruido=1.0):
    """Cada droga aparece em n_lin linhagens. Alvo anotado e 2o alvo sao
    propriedade da MOLECULA (iguais em toda linhagem); estresse e ruido variam
    por condicao. E assim que o dado real e — e por isso o criterio de aceite
    e repetir entre linhagens, nao passar do limiar uma vez."""
    rng = np.random.default_rng(semente)
    S = rng.standard_normal((m, G), np.float32)
    E = rng.standard_normal((k_estresse, G), np.float32)
    anot = rng.integers(0, m, n_drogas)
    seg = np.full(n_drogas, -1); forca = np.zeros(n_drogas, np.float32)
    gr = np.array_split(np.arange(n_drogas), len(efeitos) + 1)
    for gi, idx in enumerate(gr):
        if gi >= len(efeitos):
            continue
        for i in idx:
            j = int(rng.integers(0, m))
            while j == anot[i]:
                j = int(rng.integers(0, m))
            seg[i] = j; forca[i] = efeitos[gi]
    D = np.zeros((n_drogas * n_lin, G), np.float32)
    droga = np.repeat(np.arange(n_drogas), n_lin)
    for r in range(len(D)):
        i = droga[r]
        v = ruido * rng.standard_normal(G).astype(np.float32)
        v += 3.0 * (rng.standard_normal(k_estresse).astype(np.float32) @ E)
        v += 2.0 * S[anot[i]]
        if seg[i] >= 0:
            v += forca[i] * S[seg[i]]
        D[r] = v
    return D, S, droga, anot, seg, forca, efeitos


def roda_plantado2(T=None, **kw):
    D, S, droga, anot, seg, forca, ef = plantado2(**kw)
    gidx = {i: i for i in range(S.shape[0])}
    Dz, Sz = z(D), z(S)
    alvos = [[anot[i]] for i in droga]
    B, _ = pipeline(Dz, Sz, alvos, gidx, k=8, folds=5, n_nulos=0)
    nm = nulo_do_duto(Dz, Sz, alvos, gidx, n_rep=10, k=8, folds=5)
    lim = float(np.nanmax(nm))
    n_lin = (droga == 0).sum()
    T = T or max(3, int(0.4 * n_lin))
    print(f"  limiar do nulo do duto = {lim:.1f} bits; exige repetir em >= {T}/{n_lin} linhagens")

    mel = B.argmax(1); bits = B.max(1)
    passou = bits > lim
    nd = len(anot)
    voto = np.zeros((nd, S.shape[0]), np.int32)
    for r in range(len(B)):
        if passou[r]:
            voto[droga[r], mel[r]] += 1
    top = voto.argmax(1); nvot = voto.max(1)
    aceito = nvot >= T
    print(f"  {'forca':>8} {'n':>4} {'aceito':>7} {'acertou':>8} {'espurio':>8}")
    for f in list(ef) + [0.0]:
        sel = np.isclose(forca, f)
        a = sel & aceito
        ok = a & (top == seg)
        if f > 0:
            print(f"  {f:8.2f} {sel.sum():4d} {a.sum():7d} {ok.sum():8d} {int((a & ~ok).sum()):8d}")
        else:
            print(f"  {'nenhum':>8} {sel.sum():4d} {a.sum():7d} {'-':>8} {int(a.sum()):8d}  <- espurios puros")
