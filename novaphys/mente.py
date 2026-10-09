"""
novaphys.mente — uma mente contínua, não 250 scripts
======================================================

A LACUNA
----------
Nao existe "o agente". Cada experimento deste projeto roda isolado, aprende do
zero e morre. O usuario descreveu o que falta: "a mente da AGI estar totalmente
conectada, input e output passando por tudo e acumulando". E [[memoria-
entrincheira]] + [[laco-autonomo]] ja mostraram as pecas de uma memoria que se
revisa — surpresa contra o ruido detecta a crise (10/10), reanalise com olhos
frescos conserta o entrincheiramento (90% x 30%), e o laco fechado bate o meu
calendario (83% x 68%). So que num mundo so, de orbitas, com o ruido conhecido.

Isto e a primeira mente que VIVE um fluxo real, do comeco ao fim, sem reiniciar.

O MUNDO: GAMEPLAY REAL COM UMA LEI ESCONDIDA
---------------------------------------------
357 episodios de Minecraft (VPT), cada linha com a acao do jogador (mouse,
teclas) e o estado seguinte (yaw, pitch). A lei obvia: mexer o mouse gira a
camera. Medido: correlacao 0,719.

Mas em 18,9% dos passos o inventario esta aberto, e ai o mouse move o CURSOR e a
camera nao gira: correlacao 0,028. O mundo tem DUAS leis, e o sinal que separa
as duas (`isGuiOpen`) **a mente nunca ve**. Ela so ve acao e consequencia. Esse
flag escondido vira o gabarito.

O QUE A MENTE FAZ, PASSO A PASSO — cada peca veio de uma medida
------------------------------------------------------------------
  PREVE     antes de ver o passo, preve o giro com a lei que melhor explicou os
            ultimos passos (preve-e-depois-aprende: toda previsao e fora da
            amostra por construcao)
  SURPRESA  erro da lei atual contra o piso de ruido DELA
            ([[laco-autonomo]]: surpresa contra o ruido, nao duvida)
  CRISE     surpresa sustentada em todas as leis que ela tem -> a linguagem nao
            da conta
  REVISA    cria lei nova e a ajusta SO com as experiencias que surpreenderam,
            com olhos frescos, sem as leis velhas ([[memoria-entrincheira]],
            rodada 3: tirar as leis velhas vale +36)
  ACUMULA   as leis e as experiencias atravessam os episodios; o que ela
            descobriu no episodio 3 serve no 50
  SEPARA    uma lei por regime, sem fundir ([[integrar-exige-dose-e-afinidade]]:
            fundir envenena)

QUATRO BRACOS
---------------
  FUNDIDO   uma lei so, acumulando tudo. E o que envenena: 19% dos passos puxam
            a inclinacao para zero.
  ISOLADO   a MENTE, mas com a memoria apagada a cada episodio — e o estado
            atual do projeto: capacidade sem continuidade.
  MENTE     a mente continua.
  ORACULO   duas leis escolhidas pelo flag escondido. Teto, nao concorrente.

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  M0  PRE-CONDICAO. ORACULO erra menos que FUNDIDO por margem clara (>= 10% do
      erro do FUNDIDO). Se nao, o regime nao importa para prever e a pergunta
      nao se faz.
  M1  A MENTE DESCOBRE A LEI ESCONDIDA: fecha >= metade da distancia entre
      FUNDIDO e ORACULO, E a atribuicao de lei dela bate com o flag que ela nunca
      viu com pureza >= 0,90 (a base, chutando sempre "fechado", e 0,81)
  M2  CONTINUIDADE PAGA: nos primeiros 200 passos de cada episodio novo a MENTE
      erra menos que o ISOLADO — a experiencia de antes chega ao episodio novo
  M3  a mente nao descobre (fecha < 1/4 da distancia) -> surpresa + revisao nao
      bastam em dado real e ruidoso

PREVISAO: M1 a 0,50, M2 a 0,65, M3 a 0,25. Mecanismo de M1: o regime GUI dura
muitos passos seguidos, entao uma janela curta de erro basta para trocar de lei.
Risco nomeado: quando o mouse nao se mexe as duas leis preveem giro zero e o
passo nao informa nada; se a maioria dos passos for assim, a troca de lei fica
lenta e a pureza cai. Por isso a pureza e medida so nos passos em que o mouse
se mexeu, e isso esta dito aqui antes.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

RAIZ_VPT = r"D:\agi_dados\vpt\data"
JANELA = 6            # passos recentes para escolher a lei vigente
CRISE_K = 4.0         # surpresa: erro > K x piso de ruido da lei
CRISE_W = 6           # ... sustentada por W passos informativos seguidos
MAX_LEIS = 6
GUARDA = 400          # experiencias guardadas por lei (para reanalisar)
CLIP = 30.0           # graus: teleporte/respawn nao e lei de camera
INICIO = 200          # "primeiros passos" de um episodio, para M2


def episodios(n: int) -> List[List[Tuple[np.ndarray, np.ndarray, bool]]]:
    """
    Cada passo vira (x, y, gui): x = [dx, dy, 1] do mouse, y = [dyaw, dpitch]
    que DE FATO aconteceu no passo seguinte, gui = o flag escondido. A mente so
    recebe x e y; `gui` fica guardado para o gabarito.
    """
    fs = sorted(glob.glob(os.path.join(RAIZ_VPT, "*", "*.jsonl")))[:n]
    out = []
    for f in fs:
        ep, prev = [], None
        for linha in open(f, encoding="utf-8"):
            try:
                d = json.loads(linha)
            except Exception:
                continue
            if prev is not None:
                dyaw = d["yaw"] - prev["yaw"]
                dyaw = (dyaw + 180.0) % 360.0 - 180.0
                dpit = d["pitch"] - prev["pitch"]
                if abs(dyaw) <= CLIP and abs(dpit) <= CLIP:
                    x = np.array([d["mouse"]["dx"], d["mouse"]["dy"], 1.0])
                    y = np.array([dyaw, dpit])
                    ep.append((x, y, bool(d["isGuiOpen"])))
            prev = d
        if len(ep) > 50:
            out.append(ep)
    return out


class Lei:
    """Uma lei linear y = W x, por minimos quadrados recursivos."""

    def __init__(self) -> None:
        self.A = np.eye(3) * 1e-2
        self.B = np.zeros((3, 2))
        self.W = np.zeros((3, 2))
        self.res = deque(maxlen=200)         # residuos recentes: piso de ruido
        self.exp: deque = deque(maxlen=GUARDA)
        self.n = 0

    def preve(self, x):
        return x @ self.W

    def aprende(self, x, y):
        self.A += np.outer(x, x)
        self.B += np.outer(x, y)
        self.n += 1
        if self.n % 5 == 0 or self.n < 50:
            self.W = np.linalg.solve(self.A, self.B)
        self.res.append(float(np.abs(y - self.preve(x)).mean()))
        self.exp.append((x, y))

    def piso(self) -> float:
        return max(statistics.median(self.res), 0.05) if len(self.res) > 20 else 1.0

    @staticmethod
    def de(experiencias) -> "Lei":
        """REANALISE com olhos frescos: so estas experiencias, sem lei velha."""
        L = Lei()
        for x, y in experiencias:
            L.aprende(x, y)
        return L


def _informativo(x) -> bool:
    return abs(x[0]) + abs(x[1]) > 0


class Mente:
    """Leis separadas, lei vigente por erro recente, crise -> revisao."""

    def __init__(self) -> None:
        self.leis: List[Lei] = [Lei()]
        self.recente: List[deque] = [deque(maxlen=JANELA)]
        self.vigente = 0
        self.surpresos: deque = deque(maxlen=CRISE_W * 4)
        self.seq_crise = 0
        self.revisoes = 0

    def preve(self, x):
        return self.leis[self.vigente].preve(x)

    def vive(self, x, y) -> int:
        """Ve a consequencia, aprende, e devolve a lei a que o passo pertenceu."""
        if not _informativo(x):
            self.leis[self.vigente].aprende(x, y)
            return self.vigente
        erros = [float(np.abs(y - L.preve(x)).mean()) for L in self.leis]
        for i, e in enumerate(erros):
            self.recente[i].append(e)
        dono = int(np.argmin(erros))
        # CRISE: nenhuma lei que eu tenho explica, contra o proprio ruido
        if all(e > CRISE_K * L.piso() for e, L in zip(erros, self.leis)):
            self.seq_crise += 1
            self.surpresos.append((x, y))
        else:
            self.seq_crise = 0
        if self.seq_crise >= CRISE_W and len(self.leis) < MAX_LEIS:
            # REVISAO: lei nova ajustada so com o que surpreendeu
            self.leis.append(Lei.de(list(self.surpresos)))
            self.recente.append(deque(maxlen=JANELA))
            self.revisoes += 1
            self.seq_crise = 0
            self.surpresos.clear()
            dono = len(self.leis) - 1
        self.leis[dono].aprende(x, y)
        # lei vigente para o PROXIMO passo: menor erro recente
        med = [statistics.mean(r) if r else 1e9 for r in self.recente]
        self.vigente = int(np.argmin(med))
        return dono


def experimento(n_ep: int = 80) -> Dict:
    eps = episodios(n_ep)
    tot = sum(len(e) for e in eps)
    print("=" * 78)
    print("MENTE — uma mente continua descobre a lei que ninguem mostrou?")
    print("=" * 78)
    print("  %d episodios de gameplay real, %d passos, em sequencia, sem reiniciar"
          % (len(eps), tot))
    gui_frac = sum(g for e in eps for _x, _y, g in e) / max(tot, 1)
    print("  inventario aberto em %.1f%% dos passos — a mente NUNCA ve isso\n"
          % (100 * gui_frac))

    t0 = time.time()
    # --- FUNDIDO: uma lei, acumulando tudo
    F = Lei()
    # --- ORACULO: duas leis, escolhidas pelo flag escondido
    O = {False: Lei(), True: Lei()}
    # --- MENTE continua
    M = Mente()
    erros = {k: [] for k in ("FUNDIDO", "ISOLADO", "MENTE", "ORACULO")}
    inicio = {k: [] for k in ("ISOLADO", "MENTE")}
    atrib: List[Tuple[int, bool]] = []

    for k, ep in enumerate(eps):
        I = Mente()                              # ISOLADO: memoria apagada
        for t, (x, y, gui) in enumerate(ep):
            conta = k >= 2 and _informativo(x)   # 2 episodios de aquecimento
            for nome, prev in (("FUNDIDO", F.preve(x)),
                               ("ISOLADO", I.preve(x)),
                               ("MENTE", M.preve(x)),
                               ("ORACULO", O[gui].preve(x))):
                e = float(np.abs(y - prev).mean())
                if conta:
                    erros[nome].append(e)
                    if t < INICIO and nome in inicio:
                        inicio[nome].append(e)
            F.aprende(x, y)
            O[gui].aprende(x, y)
            I.vive(x, y)
            dono = M.vive(x, y)
            if conta:
                atrib.append((dono, gui))
        if (k + 1) % 20 == 0:
            print("    %3d/%d episodios | leis na mente: %d | revisoes: %d | %.0fs"
                  % (k + 1, len(eps), len(M.leis), M.revisoes, time.time() - t0),
                  flush=True)

    m = {k: statistics.mean(v) for k, v in erros.items()}
    print("\n  erro medio de previsao do giro (graus), passos em que o mouse mexeu,")
    print("  sempre PREVE-ANTES-DE-VER:")
    for k in ("FUNDIDO", "ISOLADO", "MENTE", "ORACULO"):
        print("    %-9s %.3f" % (k, m[k]))

    # pureza: cada lei da mente mapeada ao flag da maioria dos passos dela
    from collections import Counter
    por_lei: Dict[int, Counter] = {}
    for dono, gui in atrib:
        por_lei.setdefault(dono, Counter())[gui] += 1
    acerto = sum(c.most_common(1)[0][1] for c in por_lei.values())
    pureza = acerto / max(len(atrib), 1)
    base = max(Counter(g for _d, g in atrib).values()) / max(len(atrib), 1)
    print("\n  a mente criou %d leis (%d revisoes por crise)" % (len(M.leis), M.revisoes))
    for d_, c in sorted(por_lei.items()):
        tot_ = sum(c.values())
        print("    lei %d: %6d passos | inventario aberto em %5.1f%%"
              % (d_, tot_, 100 * c[True] / tot_))
    print("  PUREZA contra o flag que ela nunca viu: %.3f (base %.3f)" % (pureza, base))

    mi, ii = statistics.mean(inicio["MENTE"]), statistics.mean(inicio["ISOLADO"])
    print("\n  primeiros %d passos de cada episodio novo: MENTE %.3f x ISOLADO %.3f"
          % (INICIO, mi, ii))

    dist = m["FUNDIDO"] - m["ORACULO"]
    fecha = (m["FUNDIDO"] - m["MENTE"]) / dist if dist > 0 else 0.0
    print("\n" + "=" * 78)
    ver = []
    if dist < 0.10 * m["FUNDIDO"]:
        ver.append("M0 falhou")
        print("M0 FALHOU: o regime escondido nao muda a previsao (%.3f x %.3f)."
              % (m["FUNDIDO"], m["ORACULO"]))
    else:
        if fecha >= 0.5 and pureza >= 0.90:
            ver.append("M1")
            print("M1 . A MENTE DESCOBRIU A LEI ESCONDIDA: fecha %.0f%% da distancia"
                  % (100 * fecha))
            print("     ate o oraculo e separa os regimes com pureza %.3f contra um"
                  % pureza)
            print("     flag que ela nunca viu (base %.3f)." % base)
        elif fecha < 0.25:
            ver.append("M3")
            print("M3 . nao descobre: fecha so %.0f%% da distancia." % (100 * fecha))
        else:
            ver.append("parcial")
            print("PARCIAL: fecha %.0f%% da distancia, pureza %.3f (barras: 50%%"
                  % (100 * fecha, pureza))
            print("     e 0,90).")
        if mi < ii:
            ver.append("M2")
            print("M2 . CONTINUIDADE PAGA: no comeco de episodio novo a mente erra")
            print("     %.3f contra %.3f do isolado — o que ela viveu antes chega." % (mi, ii))
        else:
            print("M2 falhou: continuidade nao ajuda no comeco (%.3f x %.3f)." % (mi, ii))
    r = {"veredito": ver, "erro": m, "fecha": fecha, "pureza": pureza,
         "base": base, "leis": len(M.leis), "revisoes": M.revisoes,
         "inicio": {"MENTE": mi, "ISOLADO": ii}, "episodios": len(eps),
         "passos": tot}
    json.dump(r, open("resultados/mente.json", "w"), indent=1)
    return r


if __name__ == "__main__":
    experimento()
