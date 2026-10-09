"""
novaphys.percepcao — a representacao aprendida pelo proprio sistema
=====================================================================

A LACUNA, AGORA MEDIDA E NAO SO SUSPEITA
------------------------------------------
[[escala-nao-e-o-gargalo]] fechou a pergunta do usuario ("falta capacidade ou e
escala de dataset?"): com 24x mais treino no HAR nosso metodo foi -0,007 e a
distancia para sete linhas de estatistica CRESCEU (0,158 -> 0,195). O buraco e
de REPRESENTACAO. Toda entrada deste projeto e codificada por mim — quantizacao,
atomos HDC, histograma de simbolos — e [[fluxo-nao-tinha-tempo]] ja tinha
mostrado que essa codificacao era invariante a permutacao do tempo. Informacao
jogada fora na entrada nao volta com mais exemplos.

A PECA
--------
Um codificador que aprende a propria representacao do sinal CRU, SEM ROTULO. A
etica de descoberta do projeto fica: nenhum rotulo entra no treino do
codificador. Rotulo so aparece depois, do mesmo jeito que no pipeline antigo —
para NOMEAR grupos pela maioria no treino e para medir.

Objetivo auto-supervisionado: contrastivo. Duas visoes aumentadas da MESMA
janela (ruido, escala por canal, deslocamento no tempo) devem cair perto; visoes
de janelas diferentes, longe. E o que a literatura de HAR auto-supervisionado usa
(TS-TCC, SimCLR para series), e por isso serve de primeiro teste: se isso nao
funcionar aqui, o defeito e meu e nao do metodo.

DOIS PROTOCOLOS DE AVALIACAO, E O PRIMEIRO E O QUE DECIDE
-----------------------------------------------------------
  DESCOBERTA  12 grupos por k-medias sobre a representacao congelada, cada grupo
              nomeado pela maioria no treino — EXATAMENTE o protocolo do
              `PipelineCongelado` de `har_pessoas`. Comparavel ao nosso 0,675.
  SONDA       regressao logistica sobre a representacao congelada. Mede a
              qualidade da representacao com rotulo; comparavel ao rival de 7
              linhas (0,870), que tambem usa rotulo.

O PLACEBO QUE DECIDE SE FOI APRENDIZADO
-----------------------------------------
Rede convolucional com pesos ALEATORIOS e sabidamente forte em serie temporal
(e o principio do ROCKET). Entao o braco ALEATORIO tem a mesma arquitetura, sem
treino nenhum. Se o aprendido nao passar dele, quem trabalha e a arquitetura
que eu escolhi, nao a aprendizagem — e isso seria so trocar uma codificacao a
mao por outra.

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  V0  PRE-CONDICAO. A perda contrastiva CAI durante o treino (ultima epoca
      < 0,8 x primeira). Se nao cai, o codificador nao aprendeu e nada depois
      significa coisa alguma.
  V1  A PERCEPCAO APRENDIDA FECHA A LACUNA: no protocolo DESCOBERTA o aprendido
      chega a >= 0,772 (fecha pelo menos METADE da distancia de 0,195 entre nos
      e o rival) E supera o ALEATORIO por >= 0,05
      -> a capacidade que faltava e esta; representacao aprendida sem rotulo
         substitui a codificacao a mao.
  V2  aprendido > nosso 0,675 + 0,03 mas < 0,772 -> ajuda, fecha menos da
      metade
  V3  aprendido ~ ALEATORIO (diferenca < 0,05) -> quem trabalha e a
      arquitetura, nao a aprendizagem
  V4  aprendido <= nosso 0,675 + 0,03 -> nao melhora

PREVISAO: V1 a 0,40, V2 a 0,30, V3 a 0,20, V4 a 0,10. Mecanismo para V1:
contrastivo em HAR chega a ~0,85-0,90 em sonda linear na literatura; o protocolo
de descoberta perde algo para k-medias, mas partindo de 0,85 sobra folga sobre
0,772. Risco maior e V3: convolucao aleatoria ja separa atividades de movimento
de atividades paradas, que e a maior parte da acuracia.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from novaphys.har import carrega

K_GRUPOS = 12            # o mesmo de har_pessoas
EPOCAS = 60
LOTE = 256
TEMP = 0.2
DIM = 128
SEMENTES = (0, 1, 2)
NOSSO_MAO = 0.675        # [[escala-nao-e-o-gargalo]], 21 pessoas
RIVAL_7L = 0.870
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Codificador(nn.Module):
    """CNN 1-D pequena: 9 canais x 128 amostras -> vetor de DIM."""

    def __init__(self, canais: int = 9, dim: int = DIM) -> None:
        super().__init__()
        self.rede = nn.Sequential(
            nn.Conv1d(canais, 64, 8, padding=4), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, dim, 3, padding=1), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.projecao = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(),
                                      nn.Linear(dim, 64))

    def forward(self, x):
        return self.rede(x).squeeze(-1)


def _aumenta(x: torch.Tensor) -> torch.Tensor:
    """Visao aumentada: ruido, escala por canal, deslocamento circular no tempo."""
    n, c, t = x.shape
    x = x * (1 + 0.1 * torch.randn(n, c, 1, device=x.device))
    x = x + 0.05 * torch.randn_like(x)
    desl = torch.randint(-16, 17, (n,), device=x.device)
    idx = (torch.arange(t, device=x.device)[None, :] - desl[:, None]) % t
    return torch.gather(x, 2, idx[:, None, :].expand(n, c, t))


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z1.shape[0]
    s = z @ z.T / TEMP
    s.fill_diagonal_(-1e9)
    alvo = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(s, alvo)


def treina(X: np.ndarray, semente: int, epocas: int = EPOCAS
           ) -> Tuple[Codificador, list]:
    """Treino auto-supervisionado. NENHUM rotulo entra aqui."""
    torch.manual_seed(semente)
    enc = Codificador().to(DEV)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV)
    perdas = []
    for _ep in range(epocas):
        perm = torch.randperm(len(Xt), device=DEV)
        tot, nb = 0.0, 0
        enc.train()
        for i in range(0, len(Xt) - LOTE + 1, LOTE):
            xb = Xt[perm[i:i + LOTE]]
            z1 = enc.projecao(enc(_aumenta(xb)))
            z2 = enc.projecao(enc(_aumenta(xb)))
            L = _nt_xent(z1, z2)
            opt.zero_grad()
            L.backward()
            opt.step()
            tot += float(L)
            nb += 1
        perdas.append(tot / max(nb, 1))
    return enc, perdas


@torch.no_grad()
def representa(enc: Codificador, X: np.ndarray) -> np.ndarray:
    enc.eval()
    out = []
    for i in range(0, len(X), 1024):
        xb = torch.tensor(X[i:i + 1024], dtype=torch.float32, device=DEV)
        out.append(enc(xb).cpu().numpy())
    return np.concatenate(out)


def descoberta(Rtr, ytr, Rte, yte, semente: int) -> float:
    """O protocolo do PipelineCongelado: k-medias, nome pela maioria no treino."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    s = StandardScaler().fit(Rtr)
    km = KMeans(K_GRUPOS, n_init=10, random_state=semente).fit(s.transform(Rtr))
    nome = {}
    for c in range(K_GRUPOS):
        r = ytr[km.labels_ == c]
        nome[c] = Counter(r.tolist()).most_common(1)[0][0] if len(r) else 1
    g = km.predict(s.transform(Rte))
    return float(np.mean(np.array([nome[int(c)] for c in g]) == yte))


def sonda(Rtr, ytr, Rte, yte) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    s = StandardScaler().fit(Rtr)
    m = LogisticRegression(max_iter=3000).fit(s.transform(Rtr), ytr)
    return float(np.mean(m.predict(s.transform(Rte)) == yte))


def experimento() -> Dict:
    Xtr, ytr = carrega("train", None)
    Xte, yte = carrega("test", None)
    # (n, 128, 9) -> (n, 9, 128), normalizado por canal com estatistica do TREINO
    Xtr = np.transpose(Xtr, (0, 2, 1)).astype(np.float32)
    Xte = np.transpose(Xte, (0, 2, 1)).astype(np.float32)
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    print("=" * 78)
    print("PERCEPCAO — representacao aprendida sem rotulo fecha a lacuna?")
    print("=" * 78)
    print("  treino %d janelas (21 pessoas), teste %d (9 pessoas NOVAS), %s"
          % (len(Xtr), len(Xte), DEV))
    print("  o codificador NUNCA ve rotulo; rotulo so nomeia grupos e mede\n",
          flush=True)

    res = {"APRENDIDO": {"desc": [], "sonda": []},
           "ALEATORIO": {"desc": [], "sonda": []}}
    quedas = []
    for s in SEMENTES:
        t0 = time.time()
        enc, perdas = treina(Xtr, s)
        quedas.append(perdas[-1] / perdas[0])
        torch.manual_seed(s)
        alea = Codificador().to(DEV)          # mesma arquitetura, SEM treino
        for nome, e in (("APRENDIDO", enc), ("ALEATORIO", alea)):
            Rtr, Rte = representa(e, Xtr), representa(e, Xte)
            res[nome]["desc"].append(descoberta(Rtr, ytr, Rte, yte, s))
            res[nome]["sonda"].append(sonda(Rtr, ytr, Rte, yte))
        print("  semente %d: perda %.3f -> %.3f | aprendido desc %.3f sonda %.3f"
              " | aleatorio desc %.3f sonda %.3f | %.0fs"
              % (s, perdas[0], perdas[-1], res["APRENDIDO"]["desc"][-1],
                 res["APRENDIDO"]["sonda"][-1], res["ALEATORIO"]["desc"][-1],
                 res["ALEATORIO"]["sonda"][-1], time.time() - t0), flush=True)

    def m(v):
        return statistics.mean(v), statistics.pstdev(v)
    print("\n  %-26s %16s %16s" % ("", "DESCOBERTA", "SONDA (c/ rotulo)"))
    print("  %-26s %16.3f %16s" % ("codificacao A MAO (nossa)", NOSSO_MAO, "-"))
    for nome in ("ALEATORIO", "APRENDIDO"):
        d, sd_ = m(res[nome]["desc"])
        p, sp = m(res[nome]["sonda"])
        print("  %-26s %9.3f +-%.3f %9.3f +-%.3f" % (nome, d, sd_, p, sp))
    print("  %-26s %16s %16.3f" % ("rival 7 linhas", "-", RIVAL_7L))

    ad = m(res["APRENDIDO"]["desc"])[0]
    rd = m(res["ALEATORIO"]["desc"])[0]
    q = statistics.mean(quedas)
    meta = NOSSO_MAO + (RIVAL_7L - NOSSO_MAO) / 2
    print("\n" + "=" * 78)
    if q >= 0.8:
        ver = "V0 falhou"
        print("V0 FALHOU: a perda so caiu a %.2f da inicial. Nao aprendeu." % q)
    elif ad - rd < 0.05:
        ver = "V3"
        print("V3 . quem trabalha e a ARQUITETURA: aprendido %.3f x aleatorio"
              % ad)
        print("     %.3f. Troquei uma codificacao a mao por outra." % rd)
    elif ad >= meta:
        ver = "V1"
        print("V1 . A PERCEPCAO APRENDIDA FECHA A LACUNA: descoberta %.3f contra"
              % ad)
        print("     a mao %.3f (meta %.3f), e %+.3f sobre o aleatorio. Sem rotulo."
              % (NOSSO_MAO, meta, ad - rd))
    elif ad > NOSSO_MAO + 0.03:
        ver = "V2"
        print("V2 . ajuda (%.3f contra %.3f) mas fecha menos da metade." %
              (ad, NOSSO_MAO))
    else:
        ver = "V4"
        print("V4 . nao melhora a codificacao a mao (%.3f x %.3f)." %
              (ad, NOSSO_MAO))
    r = {"veredito": ver, "queda_perda": q, "meta": meta,
         "aprendido": res["APRENDIDO"], "aleatorio": res["ALEATORIO"]}
    json.dump(r, open("resultados/percepcao.json", "w"), indent=1)
    return r


if __name__ == "__main__":
    experimento()
