"""
mente_web.mente — o organismo vivendo na internet de verdade
==============================================================
Condicao inicial: nenhuma URL de conteudo, nenhuma lingua, nenhum dicionario,
nenhuma tarefa. Sabe que tem portas aleatorias, barras de busca, links, midia,
"voltar". O resto e decisao dela.

O QUE MOVE: progresso de compreensao (compressao-progresso, Schmidhuber 1991/
2009): quanto a experiencia tornou previsivel o que ela ainda nao tinha visto da
mesma pagina (amostra reservada, medida antes e depois). Toda escolha e Thompson
do valor por custo (custo = segundos de vida), por contexto — e o contexto e
percebido pelas PROPRIAS estruturas que ela criou (nao por tracos que eu escolhi).

MESOESTADO: perguntas abertas = estruturas que voltam em varios lugares mas cujo
entorno ela ainda nao preve (incompreensao alta, comparada com as outras
estruturas). Ativacao decai com o tempo e reacende quando a estrutura reaparece.
A pergunta mais ativa vira consulta nas barras de busca e criterio para escolher
link — busca sem ninguem pedir.

PROCEDENCIA: credibilidade do dominio = fracao dos seus padroes (pares de
estruturas vizinhas) que aparecem tambem em dominios independentes. Potencial
gerativo de uma estrutura = em quantos outros dominios ela apareceu e foi util
(reduziu surpresa). Crenca e imaginacao sao eixos separados.

ESQUECER: o dado bruto passa e vai embora. Ficam modelo de bytes, estruturas,
prototipos, ligacoes, perguntas, resumos de episodio. Memoria e finita (podas no
sono, registradas).
"""
from __future__ import annotations

import collections
import json
import math
import os
import pickle
import time

import numpy as np

import digestao as G
from corpo import BARRAS, PORTAS, Corpo

LARGURA = 3.0            # nucleo de contexto (vida 2: largura fixa 3 = aprendida)
N0 = 2.0
TAU_ATIV = 200.0         # esquecimento da ativacao (em passos; metaplasticidade ajusta)
SONO_CADA = 25           # passos entre consolidacoes
MAX_LEXICO = 400_000     # memoria externa do lexico bruto
MAX_ESTRUTURAS = 60_000  # memoria externa de estruturas (poda por forca quando passa)
ESTRATEGIAS = (["seguir_pergunta", "seguir_novo", "seguir_familiar", "seguir_acaso",
                "assistir", "ver", "voltar"] + [f"pesquisar:{b}" for b in BARRAS]
               + [f"porta:{p}" for p in PORTAS])


def txt(b: bytes):
    return b.decode("utf-8", "replace")


class Mente:
    def __init__(self, semente, pasta):
        self.rng = np.random.default_rng(semente)
        self.pasta = pasta
        os.makedirs(pasta, exist_ok=True)
        self.corpo = Corpo()
        self.texto = G.Texto()
        self.lex = {}           # pedaco -> [n, paginas(set), t_ult]
        self.E = {}             # estrutura: pedaco -> dict
        self.vis = G.Prototipos(64, 1024, "v")
        self.aud = G.Prototipos(24, 512, "a")
        self.av = {}            # (v, a) -> [n, set(videos)]
        self.ground = {}        # (pedaco, 'v'|'a', p) -> [n, set(paginas)]
        self.ligacoes_av = set()
        self.av_v = collections.Counter()
        self.aterrados = set()
        self.regs = {s: [] for s in ESTRATEGIAS}   # [ctx, ok]
        self.custo = {s: 8.0 for s in ESTRATEGIAS}
        self.hist_prog = {"texto": [], "midia": []}
        self.hist_surp_bloco = []
        self.padroes = {}       # (a, b) -> set(dominios)
        self.dom = collections.defaultdict(lambda: {"paginas": 0, "cred": 0.5, "padroes": set()})
        self.episodios = []
        self.buscas = []
        self.t = 0
        self.vida_s = 0.0
        self.pagina = None
        self.n_pag = 0
        self.tau = TAU_ATIV
        self.log = open(os.path.join(pasta, "ontogenia.jsonl"), "a", encoding="utf-8")

    # ── registro ─────────────────────────────────────────────────────────────
    def ev(self, tipo, **kw):
        self.log.write(json.dumps({"t": self.t, "vida_s": round(self.vida_s, 1), "ev": tipo, **kw},
                                  ensure_ascii=False, default=str) + "\n")
        self.log.flush()

    # ── ler: digestao do texto ───────────────────────────────────────────────
    def ler(self, pg):
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
            # forrageio: sai quando a pagina deixa de render, comparado com a
            # propria historia (previsivel demais = nada novo; ruido que nao cai)
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
        self.dom[pg.dominio]["paginas"] += 1
        estr_pag = self._estruturas_texto(pg)
        resumo = dict(url=pg.url, dominio=pg.dominio, lidos=lidos, total=len(dados),
                      bits_primeiro_bloco=primeiro, bits_final=blocos[-1] if blocos else None,
                      progresso=prog, motivo=motivo, n_estruturas=len(estr_pag))
        self.episodios.append({k: resumo[k] for k in ("url", "dominio", "progresso")})
        self.ev("leu", **resumo)
        return prog

    def _lexico(self, bloco, sur, pg, pid):
        segs = G.segmenta(bloco, sur)
        anteriores = []
        for (a, b) in segs:
            p = bloco[a:b]
            if len(p) < 3:
                continue
            # quanto custa ANTECIPAR este pedaco (surpresa media dos proprios bytes)
            depois = float(sur[a:b].mean())
            e = self.lex.get(p)
            if e is None:
                self.lex[p] = [1, {pg.dominio}, self.t]
            else:
                e[0] += 1
                e[2] = self.t
                if len(e[1]) < 12:
                    e[1].add(pg.dominio)
                # regularidade = volta em >= 2 dominios INDEPENDENTES
                if p not in self.E and len(e[1]) >= 2:
                    self._nasce(p, pg)
            s = self.E.get(p)
            if s is not None:
                self._ocorre(s, p, pg, depois, anteriores)
            anteriores.append(p)
            if len(anteriores) > 2:
                anteriores.pop(0)

    def _nasce(self, p, pg):
        self.E[p] = dict(nasc=self.t, dom_nasc=pg.dominio, doms={pg.dominio}, n=0,
                         incompr=None, ctx=np.zeros(G.D_CTX, np.float32),
                         usos=[], reusos=0, util_fora=0, pesquisada=None,
                         incompr_na_busca=None)
        self.ev("nascimento", pedaco=txt(p), dominio=pg.dominio)

    def _ocorre(self, s, p, pg, depois, anteriores):
        s["n"] += 1
        s["usos"].append(self.t)
        if len(s["usos"]) > 50:
            s["usos"] = s["usos"][-50:]
        a = 0.1
        s["incompr"] = depois if s["incompr"] is None else (1 - a) * s["incompr"] + a * depois
        for q in anteriores:
            s["ctx"] += G._idx(q)
            sq = self.E.get(q)
            if sq is not None:
                sq["ctx"] += G._idx(p)
                par = (q, p)
                ds = self.padroes.setdefault(par, set())
                if len(ds) < 12:
                    ds.add(pg.dominio)
        if pg.dominio not in s["doms"]:
            s["doms"].add(pg.dominio)
            s["reusos"] += 1
            med = np.median(self.hist_surp_bloco[-500:]) if self.hist_surp_bloco else 8.0
            if depois < med:
                s["util_fora"] += 1
            if s["reusos"] <= 3:
                self.ev("reuso", pedaco=txt(p), de=s["dom_nasc"], em=pg.dominio,
                        util=bool(depois < med))

    # ── perceber o contexto pelas PROPRIAS estruturas ────────────────────────
    def contexto(self, pg):
        v = np.zeros(G.D_CTX, np.float32)
        if pg is None or not pg.texto:
            return v
        amostra = pg.texto[:4096]
        sur = self.texto.surpresa(amostra, False)
        for (a, b) in G.segmenta(amostra, sur):
            p = amostra[a:b]
            if p in self.E:
                v += G._idx(p)
        v += 0.5 * G._idx(b"tipo:" + pg.tipo.encode()) + 0.5 * G._idx(b"midia:" + str(bool(pg.midias)).encode())
        return v

    # ── mesoestado: perguntas abertas ────────────────────────────────────────
    def ativacao(self, s):
        if not s["usos"]:
            return 0.0
        u = np.asarray(s["usos"], float)
        return float(np.exp(-(self.t - u) / self.tau).sum())

    def perguntas(self, k=5):
        cands = [(p, s) for p, s in self.E.items() if s["incompr"] is not None and len(p) >= 4
                 and len(s["doms"]) >= 2]
        if not cands:
            return []
        inc = np.array([s["incompr"] for _p, s in cands])
        corte = np.quantile(inc, 0.75)
        out = []
        for p, s in cands:
            if s["incompr"] >= corte:
                a = self.ativacao(s)
                if a > 0.05:
                    out.append((a * s["incompr"], p, s))
        out.sort(key=lambda x: -x[0])
        return out[:k]

    # ── escolha ──────────────────────────────────────────────────────────────
    def _posterior(self, estr, x):
        R = self.regs[estr]
        a = b = 0.0
        if R:
            nx = np.linalg.norm(x) + 1e-9
            C = np.stack([r[0] for r in R[-400:]])
            ok = np.array([r[1] for r in R[-400:]], float)
            s = (C @ x) / (np.linalg.norm(C, axis=1) * nx + 1e-9)
            w = np.exp(LARGURA * (s - 1.0))
            a, b = float(w @ ok), float(w.sum() - w @ ok)
        return a + 1 + N0 * 0.5, b + 1 + N0 * 0.5

    def disponiveis(self, pg):
        out = ["voltar"] + [f"porta:{p}" for p in PORTAS]
        if pg is not None and pg.links:
            out += ["seguir_novo", "seguir_familiar", "seguir_acaso"]
            if self._link_pergunta(pg) is not None:
                out.append("seguir_pergunta")
        if pg is not None and any(m[0] in ("video", "audio") for m in pg.midias):
            out.append("assistir")
        if pg is not None and any(m[0] == "imagem" for m in pg.midias):
            out.append("ver")
        if self.perguntas(1):
            out += [f"pesquisar:{b}" for b in BARRAS]
        return out

    def escolhe(self, pg):
        x = self.contexto(pg)
        melhor, vm = None, -1
        for e in self.disponiveis(pg):
            a, b = self._posterior(e, x)
            th = self.rng.beta(a, b)
            v = th / max(self.custo[e], 0.5)
            if v > vm:
                melhor, vm = e, v
        return melhor, x

    def _link_pergunta(self, pg):
        qs = self.perguntas(5)
        if not qs or not pg.links:
            return None
        for _sc, p, _s in qs:
            for anc, u in pg.links:
                if p in anc:
                    return (p, anc, u)
        return None

    # ── um passo de vida ─────────────────────────────────────────────────────
    def passo(self):
        t0 = time.time()
        pg = self.pagina
        estr, x = self.escolhe(pg)
        motivo, alvo, prog, modal = None, None, None, "texto"
        nova = None
        if estr.startswith("porta:"):
            nova = self.corpo.aleatorio(estr.split(":")[1])
        elif estr == "voltar":
            nova = self.corpo.voltar()
        elif estr == "seguir_pergunta":
            r = self._link_pergunta(pg)
            if r:
                p, anc, u = r
                motivo = txt(p)
                self._marca_reativacao(p)
                nova = self.corpo.abrir(u)
        elif estr in ("seguir_novo", "seguir_familiar", "seguir_acaso"):
            u = self._escolhe_link(pg, estr)
            if u:
                nova = self.corpo.abrir(u)
        elif estr.startswith("pesquisar:"):
            qs = self.perguntas(5)
            if qs:
                _sc, p, s = qs[int(self.rng.integers(len(qs)))] if self.rng.random() < 0.3 else qs[0]
                consulta = txt(p).strip(" \t\n.,;:()[]\"'")
                if len(consulta) >= 2:
                    barra = estr.split(":")[1]
                    motivo = consulta
                    s["pesquisada"] = self.t
                    s["incompr_na_busca"] = s["incompr"]
                    self._marca_reativacao(p)
                    self.buscas.append((self.t, barra, consulta))
                    self.ev("pesquisa", barra=barra, consulta=consulta,
                            origem="pergunta do mesoestado", incompr=s["incompr"])
                    nova = self.corpo.pesquisar(barra, consulta)
        elif estr == "assistir":
            prog, alvo = self._assistir(pg)
            modal = "midia"
        elif estr == "ver":
            prog, alvo = self._ver(pg)
            modal = "midia"
        if nova is not None and nova.tipo != "erro":
            self.pagina = nova
            alvo = nova.url
            prog = self.ler(nova)
        dt = time.time() - t0
        self.vida_s += dt
        falhou = (nova is not None and nova.tipo == "erro") or             (prog is None and estr not in ("voltar",) and nova is None and alvo is None)
        # uma tentativa nunca custa zero: abrir a mao e olhar ja gasta vida
        self.custo[estr] = 0.8 * self.custo[estr] + 0.2 * max(dt, 1.0)
        hist = self.hist_prog[modal]
        ok = prog is not None and len(hist) > 5 and prog > np.median(hist[-200:])
        if prog is not None or falhou or nova is None:
            self.regs[estr].append([x, float(ok and not falhou)])
            if len(self.regs[estr]) > 2000:
                self.regs[estr] = self.regs[estr][-1500:]
        self.ev("acao", estrategia=estr, motivo=motivo, alvo=alvo, progresso=prog,
                pagou=bool(ok), segundos=round(dt, 2),
                erro=(nova.origem if nova is not None and nova.tipo == "erro" else None))
        self._confere_perguntas()
        self.t += 1
        if self.t % SONO_CADA == 0:
            self.dorme()

    def _escolhe_link(self, pg, modo):
        L = [(a, u) for a, u in pg.links if u != pg.url]
        if not L:
            return None
        if modo == "seguir_novo":
            from corpo import dominio
            novos = [u for a, u in L if dominio(u) not in self.dom]
            if novos:
                return novos[int(self.rng.integers(len(novos)))]
            return None
        if modo == "seguir_familiar":
            sc = [(sum(1 for p in self._pedacos(a) if p in self.E), u) for a, u in L[:150]]
            sc.sort(key=lambda z: -z[0])
            return sc[0][1] if sc and sc[0][0] > 0 else None
        return L[int(self.rng.integers(len(L)))][1]

    def _marca_reativacao(self, p):
        s = self.E.get(p)
        if s is None:
            return
        ult = s["usos"][-2] if len(s["usos"]) >= 2 else s["nasc"]
        gap = self.t - ult
        if gap >= 20:
            self.ev("reativacao", pedaco=txt(p), intervalo=gap)

    def _confere_perguntas(self):
        """Pergunta pesquisada cuja incompreensao caiu = pergunta (parcialmente)
        resolvida pela propria busca."""
        for p, s in self.E.items():
            if s["pesquisada"] is not None and s["incompr_na_busca"] is not None \
                    and self.t - s["pesquisada"] in (10, 50):
                antes, agora = s["incompr_na_busca"], s["incompr"]
                self.ev("pergunta_depois", pedaco=txt(p), passos=self.t - s["pesquisada"],
                        incompr_antes=antes, incompr_agora=agora, caiu=bool(agora < antes))

    # ── midia ────────────────────────────────────────────────────────────────
    def _assistir(self, pg):
        ms = [m for m in pg.midias if m[0] in ("video", "audio")]
        if not ms:
            return None, None
        tipo, u, leg = ms[int(self.rng.integers(len(ms)))]
        surp, passos, vids = [], 0, u
        estr_pag = self._estruturas_texto(pg)
        for tm, quadro, audio in self.corpo.assistir(u, 30.0):
            passos += 1
            vs = []
            if quadro is not None:
                vs = self.vis.codifica(G.retalhos(quadro), self.t)
            band = G.coclea(audio) if audio is not None else None
            aa = self.aud.codifica(band[None], self.t)[0] if band is not None else None
            if vs and aa is not None:
                cnt = collections.Counter(vs).most_common(2)
                for v, _c in cnt:
                    k = (v, aa)
                    e = self.av.setdefault(k, [0, set()])
                    nv = self.av_v[v]
                    surp.append(-math.log2((e[0] + 1) / (nv + 2)))
                    e[0] += 1
                    self.av_v[v] += 1
                    if len(e[1]) < 20:
                        e[1].add(vids)
                    if len(e[1]) >= 2 and k not in self.ligacoes_av:
                        self.ligacoes_av.add(k)
                        self.ev("ligacao_av", visual=int(v), som=int(aa), videos=len(e[1]))
            for p in estr_pag[:20]:
                for mod, proto in ([("v", v) for v in set(vs)] + ([("a", aa)] if aa is not None else [])):
                    kk = (p, mod, int(proto))
                    g = self.ground.setdefault(kk, [0, set()])
                    g[0] += 1
                    g[1].add(pg.url)
                    if len(g[1]) >= 2 and kk not in self.aterrados:
                        self.aterrados.add(kk)
                        self.ev("aterramento", pedaco=txt(p), modalidade=mod, prototipo=int(proto))
        prog = None
        if len(surp) >= 6:
            h = len(surp) // 2
            prog = float(np.mean(surp[:h]) - np.mean(surp[h:]))
            self.hist_prog["midia"].append(prog)
        self.ev("assistiu", url=u, tipo=tipo, passos=passos, legenda=txt(leg)[:80],
                prototipos_v=len(self.vis.P), prototipos_a=len(self.aud.P), progresso=prog)
        return prog, u

    def _ver(self, pg):
        ims = [m for m in pg.midias if m[0] == "imagem"][:3]
        estr_pag = self._estruturas_texto(pg)
        novos0 = self.vis.nascidos
        for _t, u, alt in ims:
            q = self.corpo.ver(u)
            if q is None:
                continue
            vs = self.vis.codifica(G.retalhos(q), self.t)
            alvo = [p for p in self._pedacos(alt) if p in self.E and len(p) >= 3][:5] + estr_pag[:5]
            for p in alvo:
                for v in set(vs):
                    kk = (p, "v", int(v))
                    g = self.ground.setdefault(kk, [0, set()])
                    g[0] += 1
                    g[1].add(u)
                    if len(g[1]) >= 2 and kk not in self.aterrados:
                        self.aterrados.add(kk)
                        self.ev("aterramento", pedaco=txt(p), modalidade="v", prototipo=int(v))
        novos = self.vis.nascidos - novos0
        prog = -float(novos) / max(len(ims), 1)
        self.hist_prog["midia"].append(prog)
        return prog, (ims[0][1] if ims else None)

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

    # ── sono: consolidar, podar, medir ───────────────────────────────────────
    def dorme(self):
        antes_lex, antes_E = len(self.lex), len(self.E)
        if len(self.lex) > MAX_LEXICO:
            velhos = [p for p, e in self.lex.items() if e[0] == 1 and self.t - e[2] > 50 and p not in self.E]
            for p in velhos[: len(self.lex) - MAX_LEXICO // 2]:
                del self.lex[p]
        if len(self.E) > MAX_ESTRUTURAS:
            fr = sorted(self.E, key=lambda p: (self.E[p]["n"] + 5 * self.E[p]["util_fora"]))
            for p in fr[: len(self.E) - MAX_ESTRUTURAS]:
                del self.E[p]
        # credibilidade: padroes do dominio que aparecem em dominios independentes
        por_dom = collections.defaultdict(lambda: [0, 0])
        for par, ds in self.padroes.items():
            for d in ds:
                por_dom[d][1] += 1
                if len(ds) >= 2:
                    por_dom[d][0] += 1
        for d, (c, n) in por_dom.items():
            self.dom[d]["cred"] = (c + 1) / (n + 2)
        qs = [txt(p) for _sc, p, _s in self.perguntas(8)]
        pref = collections.Counter()
        for e, R in self.regs.items():
            pref[e] = len(R)
        top_e = sorted(self.E.items(), key=lambda kv: -kv[1]["n"])[:15]
        self.ev("sono", lexico=len(self.lex), lexico_podado=antes_lex - len(self.lex),
                estruturas=len(self.E), estruturas_podadas=antes_E - len(self.E),
                prototipos_v=len(self.vis.P), prototipos_a=len(self.aud.P),
                ligacoes_av=len(self.ligacoes_av), aterrados=len(self.aterrados),
                perguntas=qs, mais_frequentes=[txt(p) for p, _s in top_e],
                dominios=len(self.dom), bytes_lidos=self.texto.bytes_lidos,
                pedidos=self.corpo.n_pedidos)

    # ── memoria persistente ──────────────────────────────────────────────────
    def salva(self):
        np.save(os.path.join(self.pasta, "texto_cont.npy"), self.texto.cont)
        np.save(os.path.join(self.pasta, "texto_tot.npy"), self.texto.tot)
        estado = {k: v for k, v in self.__dict__.items()
                  if k not in ("corpo", "texto", "log", "dom")}
        estado["dom"] = dict(self.dom)
        estado["bytes_lidos"] = self.texto.bytes_lidos
        with open(os.path.join(self.pasta, "estado.pkl"), "wb") as f:
            pickle.dump(estado, f)
        self.ev("salvou")

    @classmethod
    def carrega(cls, pasta, semente):
        m = cls(semente, pasta)
        arq = os.path.join(pasta, "estado.pkl")
        if not os.path.exists(arq):
            return m
        with open(arq, "rb") as f:
            estado = pickle.load(f)
        dom = estado.pop("dom")
        bl = estado.pop("bytes_lidos")
        m.__dict__.update(estado)
        m.dom = collections.defaultdict(lambda: {"paginas": 0, "cred": 0.5, "padroes": set()}, dom)
        m.texto.cont = np.load(os.path.join(pasta, "texto_cont.npy"))
        m.texto.tot = np.load(os.path.join(pasta, "texto_tot.npy"))
        m.texto.bytes_lidos = bl
        m.ev("acordou", passos=m.t)
        return m
