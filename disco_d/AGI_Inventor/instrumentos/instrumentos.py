"""
instrumentos — experiencias, sensores e variaveis que nascem dos contraexemplos
================================================================================

A hipotese (PRE_REGISTRO_instrumentos.md): quando situacoes que a representacao
acha iguais respondem de forma diferente a mesma experiencia, falta uma variavel;
a experiencia que expoe a diferenca e o material para construi-la.

  SONDA        salva a situacao, roda duas experiencias dela (restaurando entre
               elas) e registra os desfechos — todo passo conta no orcamento
  SENSOR       NMF sobre |desfecho(e_i) - desfecho(e_j)| da MESMA situacao: padroes
               espaciais onde agir de outro jeito faz diferenca; phi_k = mudanca
               dentro do padrao k
  CONFUSAO     entre as situacoes VIZINHAS na representacao atual, as experiencias
               cujo desfecho o modelo erra mais — "achei igual, deu diferente"
  CONSTRUCAO   variantes das experiencias que contradizem o modelo: prefixo, com
               espera antes, e a costura de duas contraditorias
  VARIAVEL     z_{e,k}(r0) = antecipacao de phi_k apos executar e, admitida se R2 >= 0,1
               em sondas retidas; a definicao executavel (programa + sensor +
               estimador) fica guardada
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import NMF
from sklearn.ensemble import ExtraTreesRegressor

H = 8                                            # horizonte de toda experiencia


# ============================================================ experiencias
def biblioteca_inicial(nA, rng, n_pares=16):
    progs = [((a, H),) for a in range(nA)]
    pares = [(a, b) for a in range(nA) for b in range(nA) if a != b]
    for k in rng.permutation(len(pares))[:n_pares]:
        a, b = pares[k]
        progs.append(((a, H // 2), (b, H // 2)))
    return progs


def passos_de(prog):
    return [a for a, k in prog for _ in range(k)]


def de_passos(seq):
    out = []
    for a in seq:
        if out and out[-1][0] == a:
            out[-1] = (a, out[-1][1] + 1)
        else:
            out.append((a, 1))
    return tuple(out)


def executa(mundo, prog, o0, o_ant, trans, base_fn):
    """Roda o programa; guarda as transicoes reais; devolve o desfecho."""
    o, oa, R, fim = o0, o_ant, 0.0, False
    for a in passos_de(prog):
        s = base_fn(o, oa)
        o2, r, d = mundo.passo(a)
        trans.append((s, a, r, base_fn(o2, o), d))
        oa, o = o, o2
        R += r
        if d:
            fim = True
            break
    return o, R, fim


def neutro(o0, oH, R):
    """Medida NEUTRA comum a todos os bracos (so para a avaliacao de antecipacao)."""
    d = (oH - o0).reshape(21, 2, 16, 2).mean((1, 3)).ravel()
    return np.concatenate([d, [R]]).astype(np.float32)


# ============================================================ o laboratorio
class Laboratorio:
    """Guarda as sondas e faz crescer sensores, experiencias e variaveis (braco I)."""

    def __init__(self, nA, rng, braco):
        self.rng = rng
        self.braco = braco                        # "U", "C", "I", "P" (I-placebo)
        self.progs = biblioteca_inicial(nA, rng)
        self.n_L0 = len(self.progs)
        self.reg = []                             # (idx_situacao, r0, prog_id, neutro, dif)
        self.pares = []                           # (|dif_i - dif_j|) da mesma situacao
        self.W = None                             # sensores (NMF)
        self.vars = []                            # (prog_id, k, modelo)
        self.eventos = {"sensores": 0, "experiencias": 0, "variaveis": 0, "rejeitadas": 0}
        self._arv = None
        self._pred = {}

    # ---------------------------------------------------- representacao
    def z(self, R0, n=None):
        R0 = np.atleast_2d(R0)
        vs = self.vars if n is None else self.vars[:n]
        if not vs:
            return np.zeros((len(R0), 0), np.float32)
        return np.stack([m.predict(R0) for _p, _k, m in vs], 1).astype(np.float32)

    def rep(self, R0, n=None):
        """r0 (+ as n primeiras variaveis; elas so se acumulam, nunca saem)."""
        if self.braco in ("I", "P"):
            return np.concatenate([np.atleast_2d(R0), self.z(R0, n)], 1)
        return np.atleast_2d(R0)

    # ---------------------------------------------------- qual experiencia
    def escolhe(self, r0):
        n = len(self.progs)
        if self.braco == "U" or len(self.reg) < 60:
            return list(self.rng.choice(self.n_L0 if self.braco == "U" else n, 2, replace=False))
        if self._arv is None:
            return list(self.rng.choice(n, 2, replace=False))
        tentadas = {}
        _d, idx = self._arv.query(self.rep(r0)[0], k=min(12, len(self._arv_ids)))
        for j in np.atleast_1d(idx):
            p, res = self._arv_ids[j]
            tentadas.setdefault(p, []).append(res)
        nota = np.array([np.mean(tentadas[p]) if p in tentadas else np.inf
                         for p in range(n)])                    # nunca tentada: prioridade
        nota += self.rng.random(n) * 1e-6
        return list(np.argsort(-nota)[:2])

    # ---------------------------------------------------- registro e revisao
    def registra(self, idx_sit, r0, pid, neut, dif):
        self.reg.append((idx_sit, r0, pid, neut, dif))

    def registra_par(self, dif_i, dif_j):
        self.pares.append(np.abs(dif_i - dif_j))

    def _residuos(self):
        """Erro do modelo atual (regressao na representacao) por registro."""
        X = self.rep(np.stack([r[1] for r in self.reg]))
        Y = np.stack([r[3] for r in self.reg])
        P = np.array([r[2] for r in self.reg])
        res = np.zeros(len(self.reg))
        self._pred = {}
        for p in np.unique(P):
            m = P == p
            if m.sum() < 8:
                res[m] = np.linalg.norm(Y[m] - Y[m].mean(0), axis=1)
                continue
            Xm = np.concatenate([X[m], np.ones((m.sum(), 1))], 1)
            A = Xm.T @ Xm + 1.0 * np.eye(Xm.shape[1])
            B = np.linalg.solve(A, Xm.T @ Y[m])
            self._pred[p] = B
            res[m] = np.linalg.norm(Y[m] - Xm @ B, axis=1)
        return X, res, P

    def revisa(self):
        """Periodico: modelo de confusao; e, no braco I/P, sensores, experiencias, variaveis."""
        if len(self.reg) < 60:
            return
        X, res, P = self._residuos()
        self._arv = cKDTree(X)
        self._arv_ids = list(zip(P, res))
        if self.braco not in ("I", "P"):
            return
        self._sensores()
        self._constroi(res, P)
        n_antes = len(self.vars)
        self._variaveis()
        if len(self.vars) != n_antes:            # a representacao mudou: refaz a vizinhanca
            X, res, P = self._residuos()
            self._arv = cKDTree(X)
            self._arv_ids = list(zip(P, res))

    def _sensores(self):
        if len(self.pares) < 40:
            return
        M = np.stack(self.pares[-1500:])
        k = min(12, len(M) // 4)
        nmf = NMF(n_components=k, init="nndsvda", max_iter=300, random_state=0)
        nmf.fit(M + 1e-6)
        W = nmf.components_
        W = W / (W.sum(1, keepdims=True) + 1e-9)
        if self.W is None or W.shape != self.W.shape:
            self.eventos["sensores"] += k
        self.W = W

    def phi(self, dif):
        """Os sensores aplicados a uma diferenca de quadro (21x16)."""
        return self.W @ dif

    def _constroi(self, res, P):
        """Experiencias novas a partir das que CONTRADIZEM o modelo."""
        if len(self.progs) >= 48:
            return
        medias = {p: res[P == p].mean() for p in np.unique(P) if (P == p).sum() >= 8}
        if len(medias) < 2:
            return
        top = sorted(medias, key=lambda p: -medias[p])[:3]
        novos = []
        for p in top:
            s = passos_de(self.progs[p])
            for c in (2, 6):                                          # cortes que L0 nao tem
                novos.append(de_passos(s[:c] + [0] * (H - c)))        # so o comeco
                novos.append(de_passos([0] * (H - c) + s[:c]))        # esperar antes
        a, b = passos_de(self.progs[top[0]]), passos_de(self.progs[top[1]])
        for c in (2, 6):
            novos.append(de_passos(a[:c] + b[c:]))                    # costura de duas
        for n in novos:
            if n not in self.progs and len(self.progs) < 48:
                self.progs.append(n)
                self.eventos["experiencias"] += 1

    def _variaveis(self):
        """z_{e,k}: antecipar o sensor k depois da experiencia e, so com o presente."""
        if self.W is None:
            return
        R0 = np.stack([r[1] for r in self.reg])
        P = np.array([r[2] for r in self.reg])
        D = np.stack([r[4] for r in self.reg])
        if self.braco == "P":                        # PLACEBO: acao <-> consequencia
            P = P[self.rng.permutation(len(P))]      # destruida antes de aprender
        F = D @ self.W.T                             # phi de cada registro
        ja = {(p, k) for p, k, _m in self.vars}
        cands = []
        for p in np.unique(P):
            m = np.nonzero(P == p)[0]
            if len(m) < 40:
                continue
            var = F[m].var(0)
            for k in np.argsort(-var)[:3]:
                if (p, k) not in ja:
                    cands.append((var[k], p, k, m))
        cands.sort(key=lambda c: -c[0])
        for _v, p, k, m in cands[:6]:
            if len(self.vars) >= 16:
                break
            corte = int(0.8 * len(m))
            tr, te = m[:corte], m[corte:]
            if len(te) < 8:
                continue
            mod = ExtraTreesRegressor(n_estimators=20, min_samples_leaf=5, max_features=0.3,
                                      random_state=0, n_jobs=1).fit(R0[tr], F[tr, k])
            y, yp = F[te, k], mod.predict(R0[te])
            r2 = 1 - ((y - yp) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
            if r2 >= 0.1:
                if self.vars:
                    zs = self.z(R0[te])
                    if np.nanmax(np.abs([np.corrcoef(yp, zs[:, j])[0, 1]
                                         for j in range(zs.shape[1])])) > 0.9:
                        self.eventos["rejeitadas"] += 1
                        continue
                self.vars.append((int(p), int(k), mod))
                self.eventos["variaveis"] += 1
            else:
                self.eventos["rejeitadas"] += 1

    def descreve(self):
        """O procedimento completo de cada variavel (programa + sensor)."""
        out = []
        for p, k, _m in self.vars:
            w = self.W[k].reshape(21, 16) if self.W is not None else None
            cen = None
            if w is not None:
                yy, xx = np.mgrid[0:21, 0:16]
                cen = (float((w * yy).sum()), float((w * xx).sum()))
            out.append({"programa": [list(x) for x in self.progs[p]], "sensor": int(k),
                        "centro_do_sensor_21x16": cen})
        return out
