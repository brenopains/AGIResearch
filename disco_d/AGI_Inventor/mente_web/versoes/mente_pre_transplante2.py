"""
mente_web.mente (v1) — o organismo vivendo na internet de verdade
==================================================================
Condicao inicial (igual a v0): nenhuma URL de conteudo, nenhuma lingua, nenhum
dicionario, nenhuma tarefa. Sabe que tem portas aleatorias, barras de busca,
links, midia, "voltar". O resto e decisao dela.

O QUE MUDOU DA v0 (a vida de 5 min que ficou presa pesquisando "port", "ver",
"que" esta guardada intacta em vidas/v0_teste_5min e versoes/v0):

1. PERGUNTA = ESTADO COGNITIVO INCOMPLETO, nao palavra. Quatro tipos de lacuna,
   todos percebidos pela propria mente:
     depois      "nao sei o que vem depois de X"        (previsao que falhou)
     inesperado  "X aparece sem que eu o antecipe"      (reaparece sem explicacao)
     divergente  "X e usado de jeitos diferentes em lugares diferentes" (a unica
                 contradicao que esta mente consegue perceber: ainda nao tem
                 proposicoes)
     ligacao     "X vai junto com aquela imagem?"       (relacao incompleta)
   Cada lacuna guarda as JANELAS onde surgiu (memoria de trabalho do quebra-
   cabeca: ate 3 trechos de ~56 bytes) e sua incompletude u e REMEDIDA nelas.
2. MEMORIA DE INTENCAO: problema aberto -> acao informacional -> evidencia ->
   atualizacao -> proxima acao. Toda acao feita POR uma lacuna remede u dela e
   de ate 8 lacunas do mesmo tipo NAO perseguidas, no mesmo intervalo (controle
   pareado); pagou = caiu mais que a mediana dos controles. Sem cooldown: a acao
   que nao mudou nada perde valor NAQUELE contexto (pagina + tracos da lacuna +
   a propria historia daquela lacuna) e a repeticao some como consequencia.
   Reformular = escolher outra formulacao (alvo, alvo + vizinho, a estrutura de
   cima) ou outra fonte — tudo Thompson.
3. PRESSOES, NAO UMA FORMULA: progresso de compressao (curiosidade diversiva,
   Berlyne 1960), fechar lacuna (curiosidade especifica, Loewenstein 1994) e
   confirmar entre modalidades disputam o mesmo Thompson valor/custo. Sob acaso
   todas pagam ~1/2; o que paga mais que o acaso ganha espaco. Continuam sendo
   formulas nossas — plurais e medidas, nao uma so.
4. SUBIR NA HIERARQUIA: pedacos (picos de surpresa, Harris 1955) -> estruturas
   (voltam em >= 2 dominios) -> estruturas de estruturas (pares vizinhos coesos
   pela razao de verossimilhanca de Dunning 1993, em >= 2 dominios), recursivo.
   Ninguem diz que "palavra" e a unidade certa.
5. MULTIMODAL QUE COMECA FRACO: imagem, som e texto no mesmo episodio sao so
   candidatos; a ligacao so e CONFIRMADA quando PREVE episodios independentes
   (ganho em bits medido antes de aprender daquele episodio), com um nulo
   (som deslocado no tempo; imagem trocada) contado ao lado.
6. BARRAS DESCOBERTAS: alem das 5 dadas, todo formulario GET de um campo que
   ela encontra vira uma barra possivel; ela aprende quais servem.

ESQUECER: o dado bruto passa. Ficam modelo de bytes, estruturas, fusoes,
prototipos, associacoes, lacunas (com suas janelas) e resumos.
"""
from __future__ import annotations

import collections
import json
import math
import os
import pickle
import time
import zlib

import numpy as np

import digestao as G
from corpo import BARRAS, PORTAS, Corpo, dominio

LARGURA = 3.0            # nucleo de contexto (organismo, vida 2)
N0 = 2.0                 # prior Beta
KAPPA0 = N0              # peso inicial da populacao como prior; depois APRENDIDO
                         # (Bayes empirico sobre a variacao real entre lacunas)
TAU_ATIV = 200.0         # esquecimento da ativacao (passos)
SONO_CADA = 25
MAX_LEXICO = 250_000
MAX_ESTRUTURAS = 40_000
MAX_PARES = 250_000
MAX_LACUNAS = 300
MAX_BARRAS_NOVAS = 25
FUSOES_POR_SONO = 100
G2_CRIT = 10.83          # Dunning 1993, p < 0,001 (convencao estatistica, declarada)
JA, JD = 24, 16          # janela: bytes antes e depois da estrutura
TIPOS = ("depois", "inesperado", "divergente", "ligacao")
FORMULACOES = ("alvo", "vizinho", "acima")
EXPLORA = (["seguir_novo", "seguir_familiar", "seguir_acaso", "assistir", "ver", "voltar"]
           + [f"porta:{p}" for p in PORTAS])
DIM_X = 2 * G.D_CTX
DOM_DADOS = {"wiby.me", "pt.wikipedia.org", "en.wikipedia.org", "commons.wikimedia.org",
             "archive.org"}


def txt(b: bytes):
    return b.decode("utf-8", "replace")


def limpa(b: bytes):
    return txt(b).strip(" \t\n.,;:()[]{}\"'|-")


def entropia_beta(pos, neg):
    p = (pos + 1.0) / (pos + neg + 2.0)
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def g2(k11, k12, k21, k22):
    def xlx(x):
        return x * math.log(x) if x > 0 else 0.0
    n = k11 + k12 + k21 + k22
    return 2.0 * (xlx(k11) + xlx(k12) + xlx(k21) + xlx(k22) - xlx(k11 + k12) - xlx(k21 + k22)
                  - xlx(k11 + k21) - xlx(k12 + k22) + xlx(n))


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


# ═════════════════════════════════════════════════════════════════════════════
class Reg:
    """Registros (contexto, pagou?) de um braco; posterior por nucleo de contexto."""

    def __init__(self, cap=500):
        self.X = np.zeros((cap, DIM_X), np.float16)
        self.ok = np.zeros(cap, np.float32)
        self.n, self.cap = 0, cap

    def add(self, x, ok):
        i = self.n % self.cap
        self.X[i], self.ok[i] = x, float(ok)
        self.n += 1

    def pop(self, x):
        """(media, massa efetiva) da populacao de contextos parecidos."""
        m = min(self.n, self.cap)
        if m == 0:
            return 0.5, 0.0
        s = self.X[:m].astype(np.float32) @ x
        w = np.exp(LARGURA * (s - 1.0))
        a = float(w @ self.ok[:m])
        tot = float(w.sum())
        return (a + 1.0) / (tot + 2.0), tot


class Assoc:
    """Ligacao entre itens e prototipos que comeca FRACA e so se confirma quando
    preve episodios independentes (ganho medido com as contagens de ANTES)."""

    def __init__(self):
        self.c = {}                       # (s, p) -> [n, pos, neg, eps(set), soma_ganho]
        self.cs = collections.Counter()
        self.cp = collections.Counter()
        self.tot = 0
        self.conf = set()

    def observa(self, pares, ep, nP):
        ganhos, novas, vistos = [], [], set()
        for s, p in pares:
            cs = self.cs[s]
            if cs == 0:
                continue
            pm = (self.cp[p] + 1.0) / (self.tot + max(nP, 1))   # 1 desfecho possivel: ganho 0
            e = self.c.get((s, p))
            pc = ((e[0] if e else 0) + 2.0 * pm) / (cs + 2.0)
            gn = math.log2(pc / pm)
            ganhos.append(gn)
            if e is not None and (s, p) not in vistos and ep not in e[3]:
                vistos.add((s, p))
                e[4] += gn
                e[1 if gn > 0 else 2] += 1
                if e[1] >= 2 and e[4] > 0 and (s, p) not in self.conf:
                    self.conf.add((s, p))
                    novas.append((s, p))
        for s, p in pares:
            e = self.c.get((s, p))
            if e is None:
                e = self.c[(s, p)] = [0, 0, 0, set(), 0.0]
            e[0] += 1
            if len(e[3]) < 20:
                e[3].add(ep)
            self.cs[s] += 1
            self.cp[p] += 1
            self.tot += 1
        return (float(np.mean(ganhos)) if ganhos else 0.0), novas


# ═════════════════════════════════════════════════════════════════════════════
class Mente:
    VERSAO = "v1"
    FUSAO_DUNNING = True              # v2: fusoes nascem por autointervencao no sono
    FORMULACOES = FORMULACOES

    def __init__(self, semente, pasta, registrar=True):
        self.rng = np.random.default_rng(semente)
        self.pasta = pasta
        os.makedirs(pasta, exist_ok=True)
        self.corpo = Corpo()
        self.texto = G.Texto()
        self.lex = {}                     # pedaco -> [n, dominios, t_ult]
        self.E = {}                       # estrutura (qualquer nivel) -> dict
        self.fusoes = {}                  # (q, p) -> q+p
        self.pais = {}                    # estrutura -> estruturas de cima que a contem
        self.pares = {}                   # (q, p) vizinhos -> [n, dominios]
        self.n_adj = 0
        self.glob = G.Prototipos(88, 1024, "g")
        self.aud = G.Prototipos(24, 512, "a")
        self.av, self.av_nulo = Assoc(), Assoc()        # imagem -> som
        self.gr, self.gr_nulo = Assoc(), Assoc()        # texto -> imagem/som
        self.ult_protos = collections.deque(maxlen=200)
        self.convergencias = set()
        self.lac = {}                     # (tipo, chave) -> lacuna
        self.u0_tipo = {t: [] for t in TIPOS}
        self.hq = {"depois": collections.deque(maxlen=3000),
                   "inesperado": collections.deque(maxlen=3000)}
        self.q90 = {"depois": 1e9, "inesperado": 1e9}
        self.ref = collections.deque(maxlen=300)   # janelas comuns (nao-lacuna)
        self.ref_med = {"depois": None, "inesperado": None}
        self.fechadas = []
        self.pendentes = []
        self._congeladas = set()
        self.regs = {}                    # braco -> Reg
        self.regs_lac = Reg()             # perseguir lacunas deste tipo pagou?
        self.regs_form = {}               # formulacao -> Reg
        self.custo = {}
        self.funciona = {}               # braco -> [executou, falhou]: a ferramenta FUNCIONA?
        self.rende = {}                  # braco -> [pagou, nao pagou] em TODAS as perguntas
        self.vis = G.Prototipos(64, 1024, "v")   # retalhos 8x8: a visao aprende por si
        self.kappa = KAPPA0
        self.hist_prog = {"texto": [], "midia": [], "lacuna": []}
        self.hist_surp_bloco = []
        self.dom = {}
        self.barras_novas = {}            # dominio -> formulario
        self.buscas = []
        self.lidas = set()                # hash das URLs ja lidas
        self.t = 0
        self.vida_s = 0.0
        self.pagina = None
        self.n_pag = 0
        self.tau = TAU_ATIV
        self.n_nasc = 0
        self.log = open(os.path.join(pasta, "ontogenia.jsonl") if registrar else os.devnull, "a",
                        encoding="utf-8")

    # ── registro ─────────────────────────────────────────────────────────────
    def ev(self, _evento, **kw):
        self.log.write(json.dumps({"t": self.t, "vida_s": round(self.vida_s, 1), "ev": _evento, **kw},
                                  ensure_ascii=False, default=str) + "\n")
        self.log.flush()

    def _dom(self, d):
        e = self.dom.get(d)
        if e is None:
            e = self.dom[d] = {"paginas": 0, "cred": 0.5}
        return e

    # ── ler: digestao do texto ───────────────────────────────────────────────
    def ler(self, pg):
        self._descobre_barras(pg)
        dados = pg.texto
        if len(dados) < 200:
            return None
        pid = self.n_pag
        self.n_pag += 1
        reserva = dados[-512:] if len(dados) >= 1500 else None
        corpo_txt = dados[:-512] if reserva is not None else dados
        L0 = float(self.texto.surpresa(reserva, False).mean()) if reserva is not None else None
        blocos, lidos, primeiro = [], 0, None
        hist = self.hist_surp_bloco
        q25 = np.quantile(hist, 0.25) if len(hist) > 30 else -1
        q90 = np.quantile(hist, 0.90) if len(hist) > 30 else 1e9
        motivo = "fim"
        for a in range(0, min(len(corpo_txt), 65536), 2048):
            bloco = corpo_txt[a:a + 2048]
            if len(bloco) < 64:
                break
            sur = self.texto.surpresa(bloco, True)
            m = float(sur.mean())
            blocos.append(m)
            if primeiro is None:
                primeiro = m
            lidos += len(bloco)
            self._lexico(bloco, sur, pg, pid)
            if len(blocos) >= 2:
                if np.mean(blocos[-2:]) < q25:
                    motivo = "previsivel"
                    break
                if len(blocos) >= 4 and np.mean(blocos[-2:]) > q90 and blocos[-1] >= blocos[-3]:
                    motivo = "incompreensivel"
                    break
        hist.extend(blocos)
        if len(hist) > 5000:
            del hist[:1000]
        prog = None
        if reserva is not None:
            L1 = float(self.texto.surpresa(reserva, False).mean())
            prog = L0 - L1
            self.hist_prog["texto"].append(prog)
            if len(self.hist_prog["texto"]) > 3000:
                del self.hist_prog["texto"][:1000]
        self._dom(pg.dominio)["paginas"] += 1
        self._apos_ler(pg, corpo_txt, blocos)
        self.ev("leu", url=pg.url, dominio=pg.dominio, lidos=lidos, total=len(dados),
                bits_primeiro_bloco=primeiro, bits_final=blocos[-1] if blocos else None,
                progresso=prog, motivo=motivo)
        return prog

    def _descobre_barras(self, pg):
        for f in pg.formularios:
            d = dominio(f[0])
            if d in DOM_DADOS or d in self.barras_novas or len(self.barras_novas) >= MAX_BARRAS_NOVAS:
                continue
            self.barras_novas[d] = f
            self.ev("barra_descoberta", dominio=d, acao=f[0], campo=f[1], em=pg.url)

    def _lexico(self, bloco, sur, pg, pid):
        seq = []
        for (a, b) in G.segmenta(bloco, sur):
            p = bloco[a:b]
            if len(p) >= 3:
                e = self.lex.get(p)
                if e is None:
                    if len(self.lex) < MAX_LEXICO * 1.2:
                        self.lex[p] = [1, {pg.dominio}, self.t]
                else:
                    e[0] += 1
                    e[2] = self.t
                    if len(e[1]) < 12:
                        e[1].add(pg.dominio)
                    # regularidade = volta em >= 2 dominios INDEPENDENTES
                    if p not in self.E and len(e[1]) >= 2:
                        self._nasce(p, pg, 1, None)
            seq.append((a, b, p))
        # SUBIR: aplicar as fusoes aprendidas, de baixo para cima
        if self.fusoes:
            for _ in range(5):
                out, i, mudou = [], 0, False
                while i < len(seq):
                    if i + 1 < len(seq):
                        k = self.fusoes.get((seq[i][2], seq[i + 1][2]))
                        if k is not None:
                            self._apos_fusao((seq[i][2], seq[i + 1][2]), sur, seq[i][0], seq[i + 1][1])
                            out.append((seq[i][0], seq[i + 1][1], k))
                            i += 2
                            mudou = True
                            continue
                    out.append(seq[i])
                    i += 1
                seq = out
                if not mudou:
                    break
        ant = None
        for (a, b, p) in seq:
            s = self.E.get(p)
            if s is None:
                ant = None
                continue
            self._ocorre(s, p, pg, bloco, sur, a, b)
            if ant is not None:
                self.n_adj += 1
                par = (ant, p)
                e = self.pares.get(par)
                if e is None:
                    if len(self.pares) < MAX_PARES * 1.2:
                        self.pares[par] = [1, {pg.dominio}]
                else:
                    e[0] += 1
                    if len(e[1]) < 12:
                        e[1].add(pg.dominio)
                self._apos_par(par, float(sur[a:b].sum()))
                # contexto distribucional, separado em "casa" e "fora"
                sa = self.E[ant]
                self._ctx_add(s, pg.dominio, b"<" + ant)
                self._ctx_add(sa, pg.dominio, b">" + p)
            ant = p

    def _ctx_add(self, s, dom, chave):
        nome = "ctx_c" if dom == s["dom_nasc"] else "ctx_f"
        v = s[nome]
        if v is None:
            v = s[nome] = np.zeros(G.D_CTX, np.float16)
        v += G._idx(chave)

    def _nasce(self, p, pg, nivel, partes):
        self.E[p] = dict(nasc=self.t, dom_nasc=pg.dominio if pg else "fusao", doms=set(), n=0,
                         nivel=nivel, partes=partes, u_dep=None, u_prop=None, ult=self.t,
                         reusos=0, util_fora=0, ctx_c=None, ctx_f=None, div=False)
        if pg is not None:
            self.E[p]["doms"].add(pg.dominio)
        self.n_nasc += 1
        if nivel >= 2 or self.n_nasc % 25 == 1:
            self.ev("nascimento", pedaco=txt(p), nivel=nivel,
                    dominio=pg.dominio if pg else None,
                    partes=[txt(x) for x in partes] if partes else None)

    def _ocorre(self, s, p, pg, bloco, sur, a, b):
        s["n"] += 1
        s["ult"] = self.t
        dep = float(sur[b:b + JD].mean()) if b + 2 < len(sur) else None
        prop = float(sur[a:b].mean())
        al = 0.1
        if dep is not None:
            s["u_dep"] = dep if s["u_dep"] is None else (1 - al) * s["u_dep"] + al * dep
        s["u_prop"] = prop if s["u_prop"] is None else (1 - al) * s["u_prop"] + al * prop
        if pg.dominio not in s["doms"]:
            if len(s["doms"]) < 12:
                s["doms"].add(pg.dominio)
            s["reusos"] += 1
            med = np.median(self.hist_surp_bloco[-500:]) if self.hist_surp_bloco else 8.0
            util = prop < med
            if util:
                s["util_fora"] += 1
            if s["reusos"] <= 2 and len(p) >= 5:
                self.ev("reuso", pedaco=txt(p), nivel=s["nivel"], de=s["dom_nasc"],
                        em=pg.dominio, util=bool(util))
        if len(s["doms"]) < 2 or len(p) < 3:
            return
        # o que nao entendo AQUI acende uma lacuna (comparado com a minha historia)
        for tipo, v in (("depois", dep), ("inesperado", prop)):
            if v is None:
                continue
            h = self.hq[tipo]
            h.append(v)
            if len(h) % 250 == 0 and len(h) >= 250:
                self.q90[tipo] = float(np.quantile(h, 0.9))
                # regua de "entendido": um encontro TIPICO, medido do mesmo jeito
                # (surpresa antes de aprender) que a lacuna e medida
                self.ref_med[tipo] = float(np.quantile(h, 0.5))
            if v > self.q90[tipo]:
                self._acende(tipo, p, p, pg, bloco, a, b)
            elif tipo == "depois" and self.rng.random() < 0.05:
                i0 = max(0, a - JA)
                self.ref.append((bloco[i0:b + JD], a - i0, b - i0))
        if s["div"] and dep is not None:
            self._acende("divergente", p, p, pg, bloco, a, b)

    # ── lacunas: o mesoestado ────────────────────────────────────────────────
    def _acende(self, tipo, chave, alvo, pg, bloco=None, a=0, b=0, proto=None):
        key = (tipo, chave)
        g = self.lac.get(key)
        nova = g is None
        if nova:
            if len(self.lac) >= MAX_LACUNAS * 1.5:
                return None
            g = self.lac[key] = dict(tipo=tipo, alvo=alvo, aberta=self.t, ativ=[], jan=[],
                                     doms_jan=set(), u=None, u0=None, tent=[], estado="aberta",
                                     proto=proto, pos=0, neg=0, fechou=None)
        if g["estado"] != "aberta":
            return g
        if key in self._congeladas:                # o que eu mesmo fui buscar nao conta
            return g                               # como "isto voltou a me intrigar"
        g["ativ"].append(self.t)
        if len(g["ativ"]) > 30:
            g["ativ"] = g["ativ"][-30:]
        if bloco is not None and len(g["jan"]) < 3 and key not in self._congeladas \
                and (pg.dominio not in g["doms_jan"] or len(g["jan"]) < 2):
            i0 = max(0, a - JA)
            g["jan"].append((pg.dominio, pg.url, bloco[i0:b + JD], a - i0, b - i0))
            g["doms_jan"].add(pg.dominio)
        g["u"] = self._u(key)
        if nova:
            g["u0"] = g["u"]
            if g["u0"] is not None:
                self.u0_tipo[tipo].append(g["u0"])
            self.ev("lacuna", tipo=tipo, alvo=txt(alvo), u0=g["u0"],
                    proto=list(proto) if proto else None)
        return g

    def _u(self, key):
        g = self.lac.get(key)
        if g is None:
            return None
        if g["tipo"] == "ligacao":
            return entropia_beta(g["pos"], g["neg"])
        # entender = PREVER os proximos encontros com X (surpresa antes de
        # aprender, media movel), nao lembrar a frase antiga: lembrar sofre
        # interferencia de tudo o que se aprende depois (fumaca d: net < 0)
        s = self.E.get(g["alvo"])
        if s is None:
            return None
        return s["u_prop"] if g["tipo"] == "inesperado" else s["u_dep"]

    def _fechada(self, g):
        if g["u"] is None:
            return False
        if g["tipo"] == "ligacao":
            return g["pos"] >= 2 or (g["neg"] >= 3 and g["pos"] == 0)
        ref = self.ref_med["inesperado" if g["tipo"] == "inesperado" else "depois"]
        return ref is not None and g["u"] <= ref

    def _fecha(self, key, como):
        g = self.lac[key]
        g["estado"], g["fechou"] = "fechada", self.t
        resumo = dict(tipo=g["tipo"], alvo=txt(g["alvo"]), aberta=g["aberta"], fechou=self.t,
                      tentativas=len(g["tent"]), u0=g["u0"], u=g["u"], como=como,
                      bracos_que_pagaram=[x["braco"] for x in g["tent"] if x["net"] > 0])
        self.fechadas.append(resumo)
        self.ev("fechou", **resumo)

    def ativacao(self, g):
        if not g["ativ"]:
            return 0.0
        u = np.asarray(g["ativ"], float)
        return float(np.exp(-(self.t - u) / self.tau).sum())

    def candidatas(self, k=5):
        med = {t: (np.median(v[-500:]) if v else 1.0) for t, v in self.u0_tipo.items()}
        out = []
        for key, g in self.lac.items():
            if g["estado"] != "aberta" or g["u"] is None:
                continue
            if g["tipo"] != "ligacao" and not g["jan"]:
                continue
            a = self.ativacao(g)
            if a < 0.05:
                continue
            out.append((a * g["u"] / max(med[g["tipo"]], 1e-6), key))
        out.sort(key=lambda z: -z[0])
        return [k for _s, k in out[:k]]

    def ctx_lacuna(self, g):
        s = self.E.get(g["alvo"])
        n = s["n"] if s else 0
        nd = len(s["doms"]) if s else 0
        niv = min(s["nivel"], 3) if s else 1
        u0s = self.u0_tipo[g["tipo"]][-500:]
        ur = int(np.searchsorted(np.quantile(u0s, [0.25, 0.5, 0.75]), g["u"])) if len(u0s) > 8 else 0
        feats = [b"tipo:" + g["tipo"].encode(), b"nivel:%d" % niv,
                 b"freq:%d" % int(np.searchsorted([4, 16, 64, 256, 1024], n)),
                 b"doms:%d" % int(np.searchsorted([2, 3, 5, 8], nd)),
                 b"len:%d" % int(np.searchsorted([4, 6, 10, 16, 32], len(g["alvo"]))),
                 b"u:%d" % ur, b"tent:%d" % int(np.searchsorted([1, 3, 6], len(g["tent"])))]
        v = np.zeros(G.D_CTX, np.float32)
        for f in feats:
            v += G._idx(f)
        v += 0.7 * G._idx(b"alvo:" + g["alvo"])
        return _unit(v)

    # ── perceber o contexto da pagina pelas PROPRIAS estruturas ──────────────
    def ctx_pagina(self, pg):
        v = np.zeros(G.D_CTX, np.float32)
        if pg is None:
            return _unit(v + G._idx(b"sem_pagina"))
        if pg.texto:
            amostra = pg.texto[:4096]
            sur = self.texto.surpresa(amostra, False)
            for (a, b) in G.segmenta(amostra, sur):
                if amostra[a:b] in self.E:
                    v += G._idx(amostra[a:b])
        v = _unit(v) + 0.5 * G._idx(b"tipo:" + pg.tipo.encode()) / math.sqrt(G.D_CTX) \
            + 0.5 * G._idx(b"midia:" + str(bool(pg.midias)).encode()) / math.sqrt(G.D_CTX)
        return _unit(v)

    # ── escolha ──────────────────────────────────────────────────────────────
    def _custo(self, e):
        """Custo de um braco nunca tentado = mediana dos custos que ela ja conhece."""
        c = self.custo.get(e)
        if c is None:
            c = float(np.median(list(self.custo.values()))) if self.custo else 3.0
        return max(c, 0.5)

    def _p_funciona(self, e):
        """Separado de 'pagou': uma ferramenta que nao funciona (robots, erro) nao
        funciona em contexto nenhum — aprende-se de uma vez, sem nucleo de contexto."""
        ok, fa = self.funciona.get(e, (0, 0))
        return float(self.rng.beta(ok + 1, fa + 1))

    def _reg(self, tabela, nome):
        r = tabela.get(nome)
        if r is None:
            r = tabela[nome] = Reg()
        return r

    def _theta(self, reg, x, proprio=None):
        """Sem lacuna: toda a evidencia de contextos parecidos (como na v0).
        Com lacuna: a populacao vira PRIOR (peso <= kappa aprendido) e a historia PROPRIA
        desta lacuna com este braco decide."""
        m, massa = reg.pop(x)
        if proprio is None:
            return float(self.rng.beta(massa * m + N0 * 0.5, massa * (1 - m) + N0 * 0.5))
        k = min(massa, getattr(self, "kappa", KAPPA0))
        a = k * m + N0 * 0.5 + proprio[0]
        b = k * (1 - m) + N0 * 0.5 + proprio[1]
        return float(self.rng.beta(a, b))

    def _theta_ferramenta(self, e, g):
        """Ferramenta inutil se aprende DE UMA VEZ, nao pergunta por pergunta: o que
        ela sabe da ferramenta em TODAS as perguntas (sem nucleo de contexto, que
        diluia a evidencia: Google UK 185 tentativas, 0% de retorno) + a historia
        desta pergunta com ela."""
        ok, no = self.rende.get(e, (0, 0))
        po, pn = self._historia(g, e)
        return float(self.rng.beta(1 + ok + po, 1 + no + pn))

    def _historia(self, g, braco=None):
        ts = [x for x in g["tent"] if braco is None or x["braco"] == braco]
        pos = sum(1 for x in ts if x.get("ok", x["net"] > 0))
        return pos, len(ts) - pos

    def bracos_lacuna(self, g, pg):
        out = [f"pesquisar:{b}" for b in BARRAS] + [f"pesquisar:@{d}" for d in self.barras_novas]
        if pg is not None and pg.links and self._link_lacuna(g, pg) is not None:
            out.append("seguir_lacuna")
        return out

    def disponiveis(self, pg):
        out = [f"porta:{p}" for p in PORTAS]
        if self.corpo.pode_voltar():                 # botao "voltar" so acende com historico
            out.append("voltar")
        if pg is not None and pg.links:
            out += ["seguir_novo", "seguir_familiar", "seguir_acaso"]
        if pg is not None and any(m[0] in ("video", "audio") for m in pg.midias):
            out.append("assistir")
        if pg is not None and any(m[0] == "imagem" for m in pg.midias):
            out.append("ver")
        return out

    def escolhe(self, pg):
        xp = self.ctx_pagina(pg)
        x0 = np.concatenate([xp, G._idx(b"sem_lacuna") / math.sqrt(G.D_CTX)]) / math.sqrt(2)
        melhor = None
        for e in self.disponiveis(pg):
            v = self._theta(self._reg(self.regs, e), x0) * self._p_funciona(e) / self._custo(e)
            if melhor is None or v > melhor[0]:
                melhor = (v, e, None, x0)
        cands = self.candidatas(5)
        if cands:
            # 1) qual problema aberto perseguir (populacao + historia propria)
            best = None
            for key in cands:
                g = self.lac[key]
                xg = np.concatenate([xp, self.ctx_lacuna(g)]) / math.sqrt(2)
                th = self._theta(self.regs_lac, xg, self._historia(g))
                if best is None or th > best[0]:
                    best = (th, key, xg)
            _th, key, xg = best
            g = self.lac[key]
            # 2) que acao informacional, para ESTE problema
            for e in self.bracos_lacuna(g, pg):
                v = self._theta_ferramenta(e, g) * self._p_funciona(e) / self._custo(e)
                if v > melhor[0]:
                    melhor = (v, e, key, xg)
        return melhor[1], melhor[2], melhor[3]

    def _formulacao(self, g, xg):
        ops = []
        for f in self.FORMULACOES:
            q = self._formula(g, f)
            if q:
                ops.append((self._theta(self._reg(self.regs_form, f), xg), f, q))
        if not ops:
            return None, None
        ops.sort(key=lambda z: -z[0])
        return ops[0][1], ops[0][2]

    def _formula(self, g, f):
        alvo = limpa(g["alvo"])
        if len(alvo) < 2:
            return None
        if f == "alvo":
            return alvo
        if f == "vizinho":
            v = self._vizinho(g)
            return f"{alvo} {limpa(v)}" if v and len(limpa(v)) >= 2 else None
        if f == "acima":
            ps = [p for p in self.pais.get(g["alvo"], ()) if p in self.E]
            if not ps:
                return None
            p = max(ps, key=lambda z: self.E[z]["n"])
            q = limpa(p)
            return q if len(q) >= 2 and q != alvo else None
        return None

    def _vizinho(self, g):
        c = collections.Counter()
        for (_d, _u, jan, ia, ib) in g["jan"]:
            sur = self.texto.surpresa(jan, False)
            segs = G.segmenta(jan, sur)
            for i, (a, b) in enumerate(segs):
                if a <= ia < b or (a >= ia and b <= ib):
                    for j in (i - 1, i + 1):
                        if 0 <= j < len(segs):
                            p = jan[segs[j][0]:segs[j][1]]
                            if len(p) >= 4 and p != g["alvo"] and p in self.E:
                                c[p] += 1
                    break
        return c.most_common(1)[0][0] if c else None

    def _link_lacuna(self, g, pg):
        alvo = g["alvo"].strip()
        if len(alvo) < 3:
            return None
        for anc, u in pg.links[:400]:
            if alvo in anc and u != pg.url:
                return u
        return None

    def _controles(self, key):
        tipo = key[0]
        ks = [k for k, g in self.lac.items() if k != key and g["tipo"] == tipo
              and g["estado"] == "aberta" and g["u"] is not None]
        if len(ks) > 8:
            ks = [ks[i] for i in self.rng.choice(len(ks), 8, replace=False)]
        return ks

    # ── um passo de vida ─────────────────────────────────────────────────────
    def passo(self):
        t0 = time.time()
        pg = self.pagina
        estr, key, x = self.escolhe(pg)
        nova, prog, alvo, modal, consulta, form = None, None, None, "texto", None, None
        g = self.lac.get(key) if key is not None else None
        ctrl, u_a = [], {}
        if g is not None:
            ctrl = self._controles(key)
            self._congeladas = set([key] + ctrl)
            u_a = {k: self._u(k) for k in [key] + ctrl}
        if estr.startswith("porta:"):
            nova = self.corpo.aleatorio(estr.split(":")[1])
        elif estr == "voltar":
            nova = self.corpo.voltar()
        elif estr in ("seguir_novo", "seguir_familiar", "seguir_acaso"):
            u = self._escolhe_link(pg, estr)
            if u:
                nova = self.corpo.abrir(u)
        elif estr == "seguir_lacuna":
            u = self._link_lacuna(g, pg)
            consulta = txt(g["alvo"])
            if u:
                nova = self.corpo.abrir(u)
        elif estr.startswith("pesquisar:"):
            form, consulta = self._formulacao(g, x)
            if consulta:
                barra = estr.split(":", 1)[1]
                self.buscas.append((self.t, barra, consulta))
                if len(self.buscas) > 5000:
                    del self.buscas[:1000]
                self.ev("pesquisa", barra=barra, consulta=consulta, formulacao=form,
                        lacuna=g["tipo"], alvo=txt(g["alvo"]), u=g["u"])
                if barra.startswith("@"):
                    nova = self.corpo.usar_formulario(self.barras_novas[barra[1:]], consulta)
                else:
                    nova = self.corpo.pesquisar(barra, consulta)
        elif estr == "assistir":
            prog, alvo = self._assistir(pg)
            modal = "midia"
        elif estr == "ver":
            prog, alvo = self._ver(pg)
            modal = "midia"
        repetida = False
        if nova is not None and nova.tipo != "erro":
            self.pagina = nova
            alvo = nova.url
            hu = zlib.crc32(nova.url.encode("utf-8", "replace"))   # estavel entre processos
            repetida = hu in self.lidas
            if len(self.lidas) < 200_000:
                self.lidas.add(hu)
            prog = self.ler(nova)
            if estr == "pesquisar:midia" and nova.midias:
                self._ver(nova)                       # pesquisar imagem = olhar o que veio
            if estr == "porta:imagem" and nova.midias:
                prog, _u = self._ver(nova)            # a porta das imagens: abrir = olhar
                modal = "midia"
        if nova is not None and nova.tipo == "erro" and nova.origem == "sem_rede":
            # o MUNDO sumiu (nao foi a ferramenta que falhou): nada de aprender que
            # tudo "nao funciona"; ela espera e consolida, e volta quando a rede volta
            self._congeladas = set()
            self.offline = getattr(self, "offline", 0) + 1
            if self.offline == 1 or self.offline % 20 == 0:
                self.ev("sem_rede", estrategia=estr, vezes=self.offline)
            if self.offline % 10 == 1:
                self.dorme()
            self.telemetria(forcar=True)
            time.sleep(30)
            return
        if getattr(self, "offline", 0):
            self.ev("rede_voltou", depois_de=self.offline)
            self.offline = 0
        dt = time.time() - t0
        self.vida_s += dt
        falhou = (nova is not None and nova.tipo == "erro") or \
            (prog is None and estr != "voltar" and nova is None and alvo is None)
        self.custo[estr] = (0.8 * self.custo[estr] + 0.2 * max(dt, 1.0)) if estr in self.custo             else max(dt, 1.0)
        fu = self.funciona.setdefault(estr, [0, 0])
        fu[1 if falhou else 0] += 1
        if g is not None:
            ok = self._mede_intencao(key, ctrl, u_a, estr, form, consulta, nova, falhou, x, repetida)
            self._congeladas = set()
        else:
            hist = self.hist_prog[modal]
            ok = prog is not None and len(hist) > 5 and prog > np.median(hist[-200:])
            if prog is not None or falhou or nova is None:
                self._reg(self.regs, estr).add(x, ok and not falhou)
        self.ev("acao", estrategia=estr, lacuna=(key[0] if key else None),
                alvo_lacuna=(txt(g["alvo"]) if g is not None else None), consulta=consulta,
                destino=alvo, progresso=prog, pagou=bool(ok), segundos=round(dt, 2),
                erro=(nova.origem if nova is not None and nova.tipo == "erro" else None))
        self._confere_pendentes()
        self.t += 1
        if self.t % SONO_CADA == 0:
            self.dorme()
        self._apos_passo()
        self.telemetria()

    def _mede_intencao(self, key, ctrl, u_a, estr, form, consulta, nova, falhou, x, repetida=False):
        """problema -> acao -> evidencia -> atualizacao: a lacuna mudou mais que
        as lacunas parecidas que NAO foram perseguidas no mesmo intervalo?"""
        u_d = {k: self._u(k) for k in [key] + ctrl}
        g = self.lac[key]
        da = (u_a[key] - u_d[key]) if (u_a.get(key) is not None and u_d[key] is not None) else 0.0
        dc = [u_a[k] - u_d[k] for k in ctrl if u_a.get(k) is not None and u_d.get(k) is not None]
        net = da - (float(np.median(dc)) if dc else 0.0)
        if falhou or repetida:
            net = min(net, 0.0)                    # reler o que ja li nao e evidencia nova
        # pagou = mudou MAIS que o meu tipico E mais que a flutuacao das lacunas
        # que nao persegui no mesmo intervalo (desvio robusto dos controles)
        h = self.hist_prog["lacuna"]
        ruido = 1.4826 * float(np.median(np.abs(np.asarray(dc) - np.median(dc)))) if len(dc) >= 3 else 0.0
        ok = net > 0 and net > ruido and net > (float(np.median(h[-200:])) if len(h) > 5 else 0.0)
        h.append(net)
        if len(h) > 3000:
            del h[:1000]
        g["u"] = u_d[key]
        for k in ctrl:
            if k in self.lac and u_d.get(k) is not None:
                self.lac[k]["u"] = u_d[k]
        destino = nova.url if nova is not None else None
        mesma = destino is not None and any(destino == j[1] for j in g["jan"])
        g["tent"].append(dict(t=self.t, braco=estr, form=form, consulta=consulta,
                              u_antes=u_a.get(key), u_depois=u_d[key], net=net, ok=bool(ok)))
        self._reg(self.regs, estr).add(x, ok)
        rd = self.rende.setdefault(estr, [0, 0])
        rd[0 if ok else 1] += 1
        self.regs_lac.add(x, ok)
        if form:
            self._reg(self.regs_form, form).add(x, ok)
        self.ev("intencao", tipo=g["tipo"], alvo=txt(g["alvo"]), braco=estr, formulacao=form,
                consulta=consulta, u_antes=u_a.get(key), u_depois=u_d[key], delta=da,
                controles_mediana=(float(np.median(dc)) if dc else None), n_controles=len(dc),
                net=net, pagou=bool(ok), ruido=ruido, destino=destino, mesma_pagina=mesma,
                repetida=bool(repetida), falhou=bool(falhou),
                tentativa=len(g["tent"]))
        self.pendentes.append((self.t + 50, key, list(ctrl), dict(u_a)))
        if self._fechada(g):
            self._fecha(key, "perseguida")
        return ok

    def _confere_pendentes(self):
        resta = []
        for due, key, ctrl, u_a in self.pendentes:
            if due != self.t:
                if due > self.t:
                    resta.append((due, key, ctrl, u_a))
                continue
            if key not in self.lac or u_a.get(key) is None:
                continue
            ua = self._u(key)
            if ua is None:
                continue
            dc = [u_a[k] - self._u(k) for k in ctrl
                  if k in self.lac and u_a.get(k) is not None and self._u(k) is not None]
            da = u_a[key] - ua
            self.ev("intencao_50", tipo=key[0], alvo=txt(self.lac[key]["alvo"]), delta=da,
                    controles_mediana=(float(np.median(dc)) if dc else None),
                    net=da - (float(np.median(dc)) if dc else 0.0))
        self.pendentes = resta

    def _escolhe_link(self, pg, modo):
        L = [(a, u) for a, u in pg.links if u != pg.url]
        if not L:
            return None
        if modo == "seguir_novo":
            novos = [u for a, u in L if dominio(u) not in self.dom]
            return novos[int(self.rng.integers(len(novos)))] if novos else None
        if modo == "seguir_familiar":
            sc = [(sum(1 for p in self._pedacos(a) if p in self.E), u) for a, u in L[:150]]
            sc.sort(key=lambda z: -z[0])
            return sc[0][1] if sc and sc[0][0] > 0 else None
        return L[int(self.rng.integers(len(L)))][1]

    # ── midia ────────────────────────────────────────────────────────────────
    def _np(self):
        return len(self.glob.P) + len(self.aud.P)

    def _aterra(self, itens, protos, ep, abre=()):
        """texto x prototipos do mesmo episodio; nulo: prototipo de OUTRO episodio."""
        pares = [(s, pr) for s in itens for pr in protos]
        ganho, novas = self.gr.observa(pares, ep, self._np())
        if self.ult_protos:
            falso = [self.ult_protos[int(self.rng.integers(len(self.ult_protos)))] for _ in protos]
            self.gr_nulo.observa([(s, pr) for s in itens for pr in falso], ep, self._np())
        for (s, pr) in novas:
            self.ev("aterramento", pedaco=txt(s), modalidade=pr[0], prototipo=int(pr[1]),
                    confirmacoes=self.gr.c[(s, pr)][1], ganho_bits=round(self.gr.c[(s, pr)][4], 3),
                    nulo_confirmadas=len(self.gr_nulo.conf))
        # evidencia para as lacunas de ligacao abertas sobre estes itens
        presentes = set(protos)
        for key, g in self.lac.items():
            if g["tipo"] == "ligacao" and g["estado"] == "aberta" and g["alvo"] in itens:
                if g["proto"] in presentes:
                    g["pos"] += 1
                else:
                    g["neg"] += 1
                g["u"] = self._u(key)
        for s in abre:
            for pr in protos:
                e = self.gr.c.get((s, pr))
                if e is not None and e[0] == 1:
                    self._acende("ligacao", s + b"#" + f"{pr[0]}{pr[1]}".encode(), s, None,
                                 proto=pr)
        for pr in protos:
            self.ult_protos.append(pr)
        return ganho

    def _assistir(self, pg):
        ms = [m for m in pg.midias if m[0] in ("video", "audio")]
        if not ms:
            return None, None
        tipo, u, leg = ms[int(self.rng.integers(len(ms)))]
        seq, visuais, nq = [], [], 0
        for _tm, quadro, audio in self.corpo.assistir(u, 30.0):
            if quadro is not None:
                nq += 1
                if nq % 5 == 1:
                    visuais.append(self._curiosidade_visual(quadro))
            gq = self.glob.codifica(G.descritor_global(quadro)[None], self.t)[0] \
                if quadro is not None else None
            band = G.coclea(audio) if audio is not None else None
            aa = self.aud.codifica(band[None], self.t)[0] if band is not None else None
            seq.append((gq, aa))
        pares = [(a, b) for a, b in seq if a is not None and b is not None]
        L = len(seq)
        desl = [(seq[i][0], seq[(i + L // 2) % L][1]) for i in range(L)] if L else []
        desl = [(a, b) for a, b in desl if a is not None and b is not None]
        prog = None
        if len(pares) >= 6:
            prog, novas = self.av.observa(pares, u, len(self.aud.P))
            self.av_nulo.observa(desl, u, len(self.aud.P))
            for (gv, aa) in novas:
                self.ev("ligacao_av", visual=int(gv), som=int(aa), confirmacoes=self.av.c[(gv, aa)][1],
                        nulo_confirmadas=len(self.av_nulo.conf))
        if visuais:
            prog = (prog or 0.0) + float(np.mean(visuais))
        if prog is not None:
            self.hist_prog["midia"].append(prog)
        # texto da pagina e da legenda x o que se viu e ouviu
        itens = list(dict.fromkeys(self._estruturas_texto(pg)[:8] +
                                   [p for p in self._pedacos(leg) if p in self.E and len(p) >= 3][:4]))
        cv = collections.Counter(a for a, _b in seq if a is not None).most_common(1)
        ca = collections.Counter(b for _a, b in seq if b is not None).most_common(1)
        protos = ([("v", int(cv[0][0]))] if cv else []) + ([("a", int(ca[0][0]))] if ca else [])
        if itens and protos:
            self._aterra(itens, protos, u, abre=[p for p in self._pedacos(leg) if p in self.E
                                                 and len(p) >= 4][:2])
        self.ev("assistiu", url=u, tipo=tipo, passos=L, legenda=txt(leg)[:80],
                prototipos_v=len(self.glob.P), prototipos_a=len(self.aud.P), progresso=prog)
        return prog, u

    def _curiosidade_visual(self, q):
        """Progresso de compressao da VISAO, como o do texto (Schmidhuber): aprende
        12 retalhos da imagem e mede quanto melhorou a previsao dos outros 4, que
        ela ainda nao tinha aprendido. Uma crianca olha porque olhar ensina."""
        R = G.retalhos(q)
        ordem = self.rng.permutation(len(R))
        tr, te = R[ordem[:12]], R[ordem[12:]]

        def erro(X):
            if len(self.vis.P) == 0:
                return float(np.sqrt((X ** 2).sum(1)).mean())
            d = np.sqrt(((X[:, None, :] - self.vis.P[None]) ** 2).sum(-1)).min(1)
            return float(d.mean())
        e0 = erro(te)
        self.vis.codifica(tr, self.t)
        e1 = erro(te)
        self.vis.codifica(te, self.t)
        return e0 - e1

    def _ver(self, pg):
        ims = [m for m in pg.midias if m[0] == "imagem"][:3]
        estr_pag = self._estruturas_texto(pg)[:6]
        ganhos, vistos, visuais = [], 0, []
        for _t, u, alt in ims:
            q = self.corpo.ver(u)
            if q is None:
                continue
            vistos += 1
            visuais.append(self._curiosidade_visual(q))
            gq = self.glob.codifica(G.descritor_global(q)[None], self.t)[0]
            leg = [p for p in self._pedacos(alt) if p in self.E and len(p) >= 3][:5]
            itens = list(dict.fromkeys(leg + estr_pag))[:10]
            if itens:
                ganhos.append(self._aterra(itens, [("v", int(gq))], u,
                                           abre=[p for p in leg if len(p) >= 4][:2]))
        if not vistos:
            return None, None
        pv = float(np.mean(visuais)) if visuais else 0.0
        prog = pv + (float(np.mean(ganhos)) if ganhos else 0.0)
        self.hist_prog["midia"].append(prog)
        self.ev("viu", n=vistos, prototipos_g=len(self.glob.P), prototipos_retalho=len(self.vis.P),
                progresso_visual=round(pv, 4), progresso=prog)
        return prog, ims[0][1]

    def _pedacos(self, dados: bytes):
        if len(dados) < 3:
            return []
        sur = self.texto.surpresa(dados, False)
        return [dados[a:b] for a, b in G.segmenta(dados, sur)]

    def _estruturas_texto(self, pg):
        if not pg.texto:
            return []
        amostra = pg.texto[:4096]
        sur = self.texto.surpresa(amostra, False)
        c = collections.Counter(amostra[a:b] for a, b in G.segmenta(amostra, sur))
        return [p for p, _n in c.most_common() if p in self.E and len(p) >= 3]

    # ── sono: subir, consolidar, podar, medir ────────────────────────────────
    def dorme(self):
        antes_lex, antes_E, antes_P = len(self.lex), len(self.E), len(self.pares)
        if len(self.lex) > MAX_LEXICO:
            velhos = [p for p, e in self.lex.items() if e[0] == 1 and self.t - e[2] > 50 and p not in self.E]
            for p in velhos[: len(self.lex) - MAX_LEXICO // 2]:
                del self.lex[p]
        if len(self.E) > MAX_ESTRUTURAS:
            presas = {g["alvo"] for g in self.lac.values()}
            fr = sorted((p for p in self.E if p not in presas),
                        key=lambda p: self.E[p]["n"] + 5 * self.E[p]["util_fora"])
            for p in fr[: len(self.E) - MAX_ESTRUTURAS]:
                del self.E[p]
        if len(self.pares) > MAX_PARES:
            for k in [k for k, e in self.pares.items() if e[0] == 1][: len(self.pares) - MAX_PARES // 2]:
                del self.pares[k]
        # SUBIR: pares vizinhos coesos (Dunning) em >= 2 dominios viram unidade
        N = max(self.n_adj, 1)
        cand = []
        por_dom = collections.defaultdict(lambda: [0, 0])
        for par, (n, ds) in self.pares.items():
            for d in ds:
                por_dom[d][1] += 1
                if len(ds) >= 2:
                    por_dom[d][0] += 1
            if n < 3 or len(ds) < 2 or par in self.fusoes:
                continue
            q, p = par
            if q not in self.E or p not in self.E or len(q) + len(p) > 80:
                continue
            nq, npp = self.E[q]["n"], self.E[p]["n"]
            if n * N <= nq * npp:
                continue
            k12, k21 = max(nq - n, 0), max(npp - n, 0)
            k22 = max(N - n - k12 - k21, 0)
            sc = g2(n, k12, k21, k22)
            if sc > G2_CRIT:
                cand.append((sc, par, n, ds))
        cand.sort(key=lambda z: -z[0])
        subiram = collections.Counter()
        exemplos = []
        for sc, (q, p), n, ds in (cand[:FUSOES_POR_SONO] if self.FUSAO_DUNNING else []):
            k = q + p
            self.fusoes[(q, p)] = k
            niv = max(self.E[q]["nivel"], self.E[p]["nivel"]) + 1
            if k not in self.E:
                self._nasce(k, None, niv, (q, p))
                self.E[k]["doms"] = set(ds)
            else:
                self.E[k]["partes"] = self.E[k]["partes"] or (q, p)
            self.pais.setdefault(q, set()).add(k)
            self.pais.setdefault(p, set()).add(k)
            subiram[niv] += 1
            if len(exemplos) < 8:
                exemplos.append((txt(k), niv, round(sc, 1)))
        for d, (c, n) in por_dom.items():
            self._dom(d)["cred"] = (c + 1) / (n + 2)
        # divergencia de uso: o mesmo X com vizinhanca diferente em casa e fora
        cands = [(p, s) for p, s in self.E.items() if s["ctx_c"] is not None and s["ctx_f"] is not None
                 and len(s["doms"]) >= 3 and s["n"] >= 10 and len(p) >= 3]
        if len(cands) >= 30:
            cands = sorted(cands, key=lambda z: -z[1]["n"])[:3000]
            A = np.stack([s["ctx_c"] for _p, s in cands]).astype(np.float32)
            B = np.stack([s["ctx_f"] for _p, s in cands]).astype(np.float32)
            cos = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)
            corte = np.quantile(cos, 0.1)
            for (p, s), c in zip(cands, cos):
                s["div"] = bool(c <= corte)
        for s in self.E.values():                     # vetores float16 nao estouram
            for nome in ("ctx_c", "ctx_f"):
                v = s[nome]
                if v is not None and np.abs(v).max() > 20000:
                    v *= 0.5
        # lacunas: fecham sozinhas? esquecer as frias; soltar as ja fechadas
        abertas = 0
        for key, g in list(self.lac.items()):
            if g["estado"] == "aberta":
                g["u"] = self._u(key) if g["tipo"] != "ligacao" or g["u"] is None else g["u"]
                if self._fechada(g):
                    self._fecha(key, "sozinha" if not g["tent"] else "perseguida")
                else:
                    abertas += 1
            elif g["estado"] == "fechada" and self.t - g["fechou"] > 60:
                del self.lac[key]
        esquecidas = abandonadas = 0
        if abertas > MAX_LACUNAS:
            ab = sorted((k for k, g in self.lac.items() if g["estado"] == "aberta"),
                        key=lambda k: self.ativacao(self.lac[k]) * (self.lac[k]["u"] or 0))
            for k in ab[: abertas - MAX_LACUNAS]:
                g = self.lac.pop(k)
                if g["tent"]:
                    abandonadas += 1
                    self.ev("abandonou", tipo=g["tipo"], alvo=txt(g["alvo"]), tentativas=len(g["tent"]),
                            u0=g["u0"], u=g["u"])
                else:
                    esquecidas += 1
        # quanto a populacao de perguntas preve UMA pergunta: se todas as lacunas
        # pagam parecido, a populacao manda; se variam muito, a historia propria
        # manda (Bayes empirico, metodo dos momentos; tira o numero fixo)
        taxas, nn = [], []
        for g in self.lac.values():
            n = len(g["tent"])
            if n >= 2:
                taxas.append(sum(1 for x in g["tent"] if x.get("ok")) / n)
                nn.append(n)
        if len(taxas) >= 5:
            m = float(np.mean(taxas))
            v = float(np.var(taxas)) - float(np.mean([m * (1 - m) / n for n in nn]))
            self.kappa = 200.0 if v <= 1e-6 else float(np.clip(m * (1 - m) / v - 1, 1.0, 200.0))
        # convergencia: texto-imagem, texto-som e imagem-som confirmados juntos
        for (s, pr) in list(self.gr.conf):
            if pr[0] != "v":
                continue
            for (s2, pa) in self.gr.conf:
                if s2 == s and pa[0] == "a" and (pr[1], pa[1]) in self.av.conf:
                    tri = (s, pr[1], pa[1])
                    if tri not in self.convergencias:
                        self.convergencias.add(tri)
                        self.ev("convergencia", pedaco=txt(s), visual=pr[1], som=pa[1])
        niveis = collections.Counter(min(s["nivel"], 5) for s in self.E.values())
        tipos_ab = collections.Counter(g["tipo"] for g in self.lac.values() if g["estado"] == "aberta")
        top = [(self.lac[k]["tipo"], txt(self.lac[k]["alvo"])) for k in self.candidatas(8)]
        altos = sorted((p for p, s in self.E.items() if s["nivel"] >= 2), key=lambda p: -self.E[p]["n"])[:12]
        self.ev("sono", lexico=len(self.lex), lexico_podado=antes_lex - len(self.lex),
                estruturas=len(self.E), estruturas_podadas=antes_E - len(self.E),
                pares=len(self.pares), pares_podados=antes_P - len(self.pares),
                fusoes=len(self.fusoes), subiram=dict(subiram), exemplos_subiram=exemplos,
                niveis=dict(niveis), lacunas_abertas=dict(tipos_ab), esquecidas=esquecidas,
                abandonadas=abandonadas, fechadas_total=len(self.fechadas), perguntas=top,
                mais_frequentes_altos=[txt(p) for p in altos],
                prototipos_g=len(self.glob.P), prototipos_a=len(self.aud.P),
                regua_entendido=self.ref_med, kappa=round(self.kappa, 2),
                ligacoes_av=len(self.av.conf), ligacoes_av_nulo=len(self.av_nulo.conf),
                aterrados=len(self.gr.conf), aterrados_nulo=len(self.gr_nulo.conf),
                convergencias=len(self.convergencias), barras_novas=len(self.barras_novas),
                dominios=len(self.dom), bytes_lidos=self.texto.bytes_lidos,
                pedidos=self.corpo.n_pedidos)
        if (self.t // SONO_CADA) % 10 == 0:
            self._vizinhancas()

    def _vizinhancas(self):
        """Observacao (nao entra na decisao): quem e vizinho de quem pelo uso —
        candidatos a 'conceito' que ninguem nomeou."""
        cands = [(p, s) for p, s in self.E.items() if len(limpa(p)) >= 4
                 and (s["ctx_c"] is not None or s["ctx_f"] is not None)]
        if len(cands) < 50:
            return
        cands = sorted(cands, key=lambda z: -z[1]["n"])[:1500]
        V = np.stack([(s["ctx_c"].astype(np.float32) if s["ctx_c"] is not None else 0)
                      + (s["ctx_f"].astype(np.float32) if s["ctx_f"] is not None else 0)
                      for _p, s in cands])
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        S = V @ V.T
        np.fill_diagonal(S, -1)
        out = []
        for i in range(0, min(len(cands), 600), 20):
            j = np.argsort(-S[i])[:5]
            out.append([txt(cands[i][0])] + [txt(cands[k][0]) for k in j])
        self.ev("vizinhancas", amostra=out)

    # ── telemetria: o organismo por dentro, para o painel (so leitura) ────────
    def telemetria(self, forcar=False):
        agora = time.time()
        if not forcar and agora - getattr(self, "_tel_t", 0.0) < 3:
            return
        self._tel_t = agora
        try:
            ab = [g for g in self.lac.values() if g["estado"] == "aberta"]
            ab.sort(key=lambda g: -self.ativacao(g))
            lac = [{"tipo": g["tipo"], "alvo": txt(g["alvo"])[:40], "ativ": round(self.ativacao(g), 3),
                    "u": None if g["u"] is None else round(g["u"], 3), "tent": len(g["tent"]),
                    "pagou": sum(1 for x in g["tent"] if x.get("ok")), "idade": self.t - g["aberta"]}
                   for g in ab[:48]]
            bracos = []
            for e in set(self.custo) | set(self.funciona):
                r = self.regs.get(e)
                m = min(r.n, r.cap) if r is not None else 0
                fu = self.funciona.get(e, [0, 0])
                bracos.append({"braco": e, "custo": round(self.custo.get(e, 0.0), 2),
                               "funciona": list(fu), "usos": max(int(fu[0] + fu[1]), r.n if r is not None else 0),
                               "pagou": None if not m else round(float(r.ok[:m].mean()), 3)})
            bracos.sort(key=lambda b: -b["usos"])
            niveis = collections.Counter(min(s["nivel"], 6) for s in self.E.values())
            tel = {"parede": agora, "t": self.t, "vida_s": round(self.vida_s, 1), "versao": self.VERSAO,
                   "pagina": {"url": self.pagina.url if self.pagina else None,
                              "dominio": self.pagina.dominio if self.pagina else None,
                              "links": len(self.pagina.links) if self.pagina else 0,
                              "midias": len(self.pagina.midias) if self.pagina else 0},
                   "memoria": {"lexico": len(self.lex), "estruturas": len(self.E), "pares": len(self.pares),
                               "fusoes": len(self.fusoes), "lacunas": len(self.lac), "lidas": len(self.lidas),
                               "prototipos_visuais": len(self.glob.P), "prototipos_sonoros": len(self.aud.P),
                               "bytes_lidos": self.texto.bytes_lidos, "niveis": dict(niveis)},
                   "mesoestado": {"abertas": len(ab), "por_tipo": dict(collections.Counter(g["tipo"] for g in ab)),
                                  "fechadas": len(self.fechadas), "kappa": round(self.kappa, 1),
                                  "lacunas": lac},
                   "bracos": bracos[:24], "barras_novas": list(self.barras_novas)[:25],
                   "multimodal": {"texto_imagem": len(self.gr.conf), "texto_imagem_nulo": len(self.gr_nulo.conf),
                                  "imagem_som": len(self.av.conf), "imagem_som_nulo": len(self.av_nulo.conf),
                                  "convergencias": len(self.convergencias)},
                   "offline": getattr(self, "offline", 0), "pedidos": self.corpo.n_pedidos}
            self._telemetria_extra(tel)
            arq = os.path.join(self.pasta, "telemetria.json")
            with open(arq + ".tmp", "w", encoding="utf-8") as f:
                json.dump(tel, f, ensure_ascii=False)
            os.replace(arq + ".tmp", arq)
        except Exception as e:                       # telemetria nunca pode matar a mente
            self.ev("telemetria_falhou", msg=str(e)[:160])

    def _telemetria_extra(self, tel):
        pass

    # ── ganchos (vazios na v1; a v2 os usa) ──────────────────────────────────
    def _apos_par(self, par, bits):
        pass

    def _apos_fusao(self, par, sur, a, b):
        pass

    def _apos_ler(self, pg, dados, blocos):
        pass

    def _apos_passo(self):
        pass

    # ── memoria persistente ──────────────────────────────────────────────────
    def salva(self, sub=None):
        """Salvamento ATOMICO (escreve ao lado e troca de uma vez; guarda a versao
        anterior): desligar o PC no meio nunca corrompe a memoria dela.
        Fotografias saem comprimidas (o modelo de bytes e quase todo zero)."""
        pasta = os.path.join(self.pasta, sub) if sub else self.pasta
        os.makedirs(pasta, exist_ok=True)
        estado = {k: v for k, v in self.__dict__.items() if k not in ("corpo", "texto", "log")}
        estado["bytes_lidos"] = self.texto.bytes_lidos
        if sub:
            np.savez_compressed(os.path.join(pasta, "texto.npz"), cont=self.texto.cont, tot=self.texto.tot)
            with open(os.path.join(pasta, "estado.pkl"), "wb") as f:
                pickle.dump(estado, f)
        else:
            novos = []
            for nome, arr in (("texto_cont.npy", self.texto.cont), ("texto_tot.npy", self.texto.tot)):
                tmp = os.path.join(pasta, nome + ".tmp")
                with open(tmp, "wb") as f:
                    np.save(f, arr)
                novos.append((tmp, os.path.join(pasta, nome)))
            tmp = os.path.join(pasta, "estado.pkl.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(estado, f)
            novos.append((tmp, os.path.join(pasta, "estado.pkl")))
            for tmp, final in novos:
                if os.path.exists(final):
                    os.replace(final, final + ".anterior")
                os.replace(tmp, final)
        self.ev("salvou", onde=sub or ".")

    @classmethod
    def carrega(cls, pasta, semente, log=True):
        m = cls(semente, pasta, registrar=log)
        arq = os.path.join(pasta, "estado.pkl")
        if not os.path.exists(arq):
            return m
        suf = ""
        try:
            with open(arq, "rb") as f:
                estado = pickle.load(f)
        except Exception:                          # salvamento interrompido: usa a anterior
            suf = ".anterior"
            with open(arq + suf, "rb") as f:
                estado = pickle.load(f)
        bl = estado.pop("bytes_lidos")
        estado.pop("pasta", None)
        novos = [k for k in m.__dict__ if k not in estado and not k.startswith("_")
                 and k not in ("corpo", "texto", "log", "pasta")]
        m.__dict__.update(estado)
        if novos and log:
            m.ev("transplante", orgaos=novos)
        npz = os.path.join(pasta, "texto.npz")
        if os.path.exists(npz):
            z = np.load(npz)
            m.texto.cont, m.texto.tot = z["cont"], z["tot"]
        else:
            m.texto.cont = np.load(os.path.join(pasta, "texto_cont.npy" + suf))
            m.texto.tot = np.load(os.path.join(pasta, "texto_tot.npy" + suf))
        m.texto.bytes_lidos = bl
        if log:
            m.ev("acordou", passos=m.t)
        return m
