"""
organismo.organismo — a mente como ORGANISMO: estruturas com existencia graduada
==================================================================================
Implementa as ideias da pasta D:/AGI_Inventor/ideias. Cada ideia tem uma chave
(FLAGS) para que o braco de ablacao a desligue SOZINHA. O mapa ideia -> mecanismo
-> portao esta em DESENHO.md. Resumo:

  SUBSTRATO (invariavel): material do dominio (lei, grandeza, direcao, pergunta),
  HDC de contexto, medida de custo, procedencia, verificacao interna.

  ORGANISMO (plastico), tudo a mesma moeda: ESTRUTURA. Operacoes do material,
  conceitos, direcoes, leis, problemas abertos, episodios, coalizoes. Nenhuma tem
  "existe/nao existe": tem REGISTROS (contexto, pagou?) que dao uma crenca de
  eficacia POR CONTEXTO, forca, estabilidade, ativacao recente, potencial
  gerativo, fonte. Escolha de acao = amostra de Thompson do valor por custo.

    mesoestado    problemas abertos guardam o que o organismo PENSOU (dados
                  acumulados de varios encontros, candidatos tentados, melhor lei
                  parcial) e sao reacendidos pelo contexto (FLAGS['meso'])
    graduado      nada de MAX_*: limite e o orcamento externo (FLAGS['tetos'] liga
                  os numeros magicos do projeto, para o braco de comparacao)
    metacontrole  a ordem das operacoes e aprendida por contexto (FLAGS['meta'])
    coalizoes     sequencias que resolvem em >= 2 problemas viram uma estrutura
                  executavel de uma vez — habito consolidado (FLAGS['coal'])
    ecologia      credito atrasado: a estrutura que habilitou outra recebe credito
                  quando a descendente paga, mesmo depois (FLAGS['eco'])
    regime        temperatura sobe quando falha apesar do investimento: alarga o
                  contexto, aplaina a crenca, deixa entrar o improvavel (FLAGS['regime'])
    procedencia   dois eixos: fonte controla CRENCA; potencial gerativo controla
                  IMAGINACAO; nada e apagado por ser ficcao (FLAGS['proc2'])
    sonho         sobra de orcamento vira replay: problemas abertos x estruturas
                  novas, no dado guardado (FLAGS['sonho'])
    metaplast.    a constante de esquecimento por tipo se ajusta pelo que a
                  reativacao rende (sempre ligada; so registrada)

Todo evento vai para o registro de ontogenia (JSONL): nascimento, uso, credito,
reativacao com intervalo, coalizao, regime, fonte, compactacao.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field

import numpy as np

import hdc
import material_fisica as F
from mundo_fisico import SIG_A

# ─── chaves das ideias ───────────────────────────────────────────────────────

FLAGS_ORGANISMO = dict(meso=True, tetos=False, meta=True, coal=True, eco=True,
                       regime=True, proc2=True, sonho=True, rag=False,
                       placebo_meso=False, pipeline=False)

BRACOS = {
    "ORGANISMO": {},
    "SEM_MESO": dict(meso=False),
    "PLACEBO_MESO": dict(placebo_meso=True),
    "TETOS": dict(tetos=True),
    "SEM_META": dict(meta=False),
    "SEM_COAL": dict(coal=False),
    "SEM_ECO": dict(eco=False),
    "SEM_REGIME": dict(regime=False),
    "SEM_PROC2": dict(proc2=False),
    "SEM_SONHO": dict(sonho=False),
    "PIPELINE": dict(meso=False, tetos=True, meta=False, coal=False, eco=False,
                     regime=False, proc2=False, sonho=False, rag=True, pipeline=True),
}

# Numeros magicos do projeto (so ligados nos bracos TETOS/PIPELINE)
TETOS = dict(conceito=3, direcao=3, lei=6, aberto=6, episodio=12, coalizao=4,
             vocab=3, rag=3, admitidos=3)

# Constantes do SUBSTRATO do organismo (declaradas; nenhuma e teto de conteudo)
BETA0 = 8.0          # largura do nucleo de contexto quando frio
N0 = 2.0             # forca da crenca a priori (pseudo-contagens)
EMA_T = 0.1          # taxa da temperatura de vida
TAU0 = 20.0          # constante inicial de esquecimento (problemas)
ETA_META = 0.1       # taxa da metaplasticidade
REGULAR = 2          # "regularidade" = repetir em 2 problemas distintos (projeto)
T_FIXA = 0.25        # temperatura do braco SEM_REGIME (declarada antes)
MEM_PONTOS = 40000   # memoria externa (pontos de dado guardados), como o orcamento
SEGURANCA_ACOES = 80 # protecao de laco por problema (nunca deve limitar)


# ─── estrutura ───────────────────────────────────────────────────────────────

@dataclass
class Estrutura:
    id: str
    tipo: str
    payload: dict
    fonte: str
    nasc: int
    pais: list = field(default_factory=list)
    regs: list = field(default_factory=list)      # [t, ctx, ok]
    forca: float = 0.0
    estab: float = 0.0
    desc_cred: float = 0.0
    desc_n: float = 0.0
    usos: list = field(default_factory=list)      # instantes de uso com sucesso
    origem: tuple | None = None                   # (id da operacao, indice do registro)
    ctx_nasc: np.ndarray | None = None

    def potencial_gerativo(self):
        return (self.desc_cred + 1.0) / (self.desc_n + 2.0)


class Organismo:
    def __init__(self, flags: dict, semente: int, log=None):
        self.f = dict(FLAGS_ORGANISMO)
        self.f.update(flags)
        self.rng = np.random.default_rng(semente)
        self.E: dict[str, Estrutura] = {}
        self.t = 0
        self.T_vida = T_FIXA if not self.f["regime"] else 0.0
        self.tau = {}
        self.fonte_cred = {"percebido": [9.0, 1.0]}
        self.padroes: dict[tuple, set] = {}
        self.log = log
        self.n_ids = itertools.count()
        self.admitidos = []           # pipeline: ids admitidos (booleano)
        self._cpt = {}                # custo aprendido por unidade de trabalho
        self._cache = {}
        for op in ("forma", "ajustar_base", "ajustar_ctx", "ajustar_novos", "ajustar_com",
                   "ajustar_amplo", "juntar", "inventar", "reajustar", "derivar",
                   "prever_base", "prever_ctx", "prever_novos", "prever_com", "macro"):
            self._nova("op", {"nome": op}, "substrato", eid=f"op:{op}")
        # metacognicao: "quando eu acho que acertei, aqui, eu acerto?"
        self._nova("op", {"nome": "confiar"}, "substrato", eid="op:confiar")

    # ── registro de ontogenia ────────────────────────────────────────────────
    def evento(self, _ev, **kw):
        if self.log is not None:
            kw = {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in kw.items()}
            self.log.write(json.dumps({"t": self.t, "ev": _ev, **kw}, ensure_ascii=False,
                                      default=str) + "\n")

    def _nova(self, tipo, payload, fonte, pais=(), eid=None, ctx=None):
        eid = eid or f"{tipo}:{next(self.n_ids)}"
        e = Estrutura(eid, tipo, payload, fonte, self.t, list(pais), ctx_nasc=ctx)
        self.E[eid] = e
        if tipo != "op":
            self.evento("nascimento", id=eid, tipo=tipo, fonte=fonte, pais=list(pais),
                        desc=_descreve(e))
        if self.f["tetos"] and tipo in TETOS:
            self._aplica_teto(tipo)
        return e

    def _aplica_teto(self, tipo):
        mesmos = [e for e in self.E.values() if e.tipo == tipo]
        lim = TETOS[tipo]
        if len(mesmos) > lim:
            mesmos.sort(key=lambda e: (e.forca, e.nasc))
            for e in mesmos[:len(mesmos) - lim]:
                self._dissolve(e, "teto")

    def _dissolve(self, e, motivo):
        self.evento("dissolucao", id=e.id, tipo=e.tipo, motivo=motivo, forca=e.forca)
        self.E.pop(e.id, None)
        if e.id in self.admitidos:
            self.admitidos.remove(e.id)

    # ── crenca de eficacia por contexto ──────────────────────────────────────
    def _temperatura(self, ws=None):
        if not self.f["regime"]:
            return T_FIXA
        T = self.T_vida
        if ws is not None and ws.orcamento > 0:
            T = T + (1 - T) * min(1.0, ws.gasto_sem_progresso / ws.orcamento)
        return float(min(max(T, 0.0), 1.0))

    def _cred_fonte(self, fonte):
        a, b = self.fonte_cred.get(fonte, [1.0, 1.0])
        return a / (a + b), a + b - 2

    def ativacao(self, e):
        if not e.usos:
            return 0.0
        tau = self.tau.get(e.tipo, TAU0)
        return float(np.exp(-(self.t - np.asarray(e.usos, float)) / tau).sum())

    def _sims(self, e, x):
        """Semelhanca do contexto x com cada registro de e (uma matmul, em cache)."""
        n = len(e.regs)
        chave = (e.id, id(x), n)
        s = self._cache.get(chave)
        if s is None:
            if getattr(e, "_C", None) is None or e._C.shape[0] != n:
                C = np.stack([c for _t, c, _o in e.regs]) if n else np.zeros((0, hdc.D))
                nr = np.linalg.norm(C, axis=1, keepdims=True)
                e._C = C / np.maximum(nr, 1e-12)
                e._ok = np.array([o for _t, _c, o in e.regs], float)
            nx = np.linalg.norm(x)
            s = e._C @ (x / max(nx, 1e-12)) if n else np.zeros(0)
            self._cache[chave] = s
        return s, e._ok

    def posterior(self, e, x, T):
        beta = BETA0 * (1 - T) + 1.0
        a = b = 0.0
        if e.regs:
            s, oks = self._sims(e, x)
            w = np.exp(beta * (s - 1.0))
            a = float(w @ oks)
            b = float(w.sum() - a)
        esc = 1.0 / (1.0 + 3.0 * T)
        a, b = a * esc, b * esc
        m = self._prior(e, T)
        a0, b0 = 1 + N0 * m, 1 + N0 * (1 - m)
        act = min(self.ativacao(e), 3.0) if self.f["meso"] else 0.0
        return a + a0 + act, b + b0

    def _prior(self, e, T):
        """Dois eixos. Lei = AFIRMACAO sobre um mundo: a fonte controla a crenca
        (quente, a crenca pesa menos). Conceito/direcao = IDEIA: frio, neutra;
        quente, vale o potencial gerativo. Sem proc2: a fonte controla tudo."""
        if e.tipo == "lei":
            cf, _n = self._cred_fonte(e.fonte)
            return cf * (1 - T) + 0.5 * T if self.f["proc2"] else cf
        if e.tipo in ("conceito", "direcao"):
            if self.f["proc2"]:
                return 0.5 * (1 - T) + e.potencial_gerativo() * T
            cf, _n = self._cred_fonte(e.fonte)
            return cf
        return 0.5

    def amostra(self, e, x, T):
        a, b = self.posterior(e, x, T)
        return float(self.rng.beta(a, b))

    def media(self, e, x, T):
        a, b = self.posterior(e, x, T)
        return a / (a + b)

    # ── percepcao do problema (substrato, igual em todos os bracos) ──────────
    def percebe(self, prob, lei_base):
        feats = {"tipo": prob.tipo, "fonte": prob.fonte}
        if prob.tipo == "L":
            ts = lei_base.termos
            feats["imp1"] = _chave_termo(ts[0]) if len(ts) > 0 else "nada"
            feats["imp2"] = _chave_termo(ts[1]) if len(ts) > 1 else "nada"
            feats["nterm"] = min(len(ts), 4)
        else:
            feats["pergunta"] = prob.molde
        if prob.T is not None:
            T = prob.T
            x, y, vx, vy = T[..., 0], T[..., 1], T[..., 2], T[..., 3]
            r = np.hypot(x, y)
            for nome, q in (("lam_r", r), ("lam_v2", vx ** 2 + vy ** 2),
                            ("lam_rv", x * vx + y * vy)):
                lam = max(F._lam(q), 1e-6)
                feats[nome] = int(np.round(np.log10(lam) * 2))
        return hdc.codifica(feats), feats

    # ── um problema ──────────────────────────────────────────────────────────
    def resolve(self, prob, orcamento):
        self._cache = {}
        ws = Espaco(prob, None, orcamento)
        if prob.tipo == "L":
            ws.x = hdc.codifica({"tipo": "L", "fonte": prob.fonte})
            self._executa(ws, "ajustar_base", None)
            x, feats = self.percebe(prob, ws.melhor["resposta"])
        else:
            x, feats = self.percebe(prob, None)
        ws.x = x
        if self.f["placebo_meso"]:
            ws.x_reat = self._contexto_sorteado(x)
        else:
            ws.x_reat = x
        self._reativacoes(ws)
        if self.f["pipeline"] or not self.f["meta"]:
            self._agenda_fixa(ws)
        else:
            self._agenda_aprendida(ws)
        resposta = ws.resposta()
        if prob.verificavel:
            ok, nota = prob.verifica(resposta)
        else:
            ok, nota = bool(ws.valido), None
        if prob.verificavel and ws.valido:
            self._registra(self.E["op:confiar"], ws.x, ok)
        self._registra_bum(ws, ok)
        self._credito(ws, ok)
        self._atualiza_abertos(ws, ok)
        self._atualiza_regime(ok, ws)
        sobra = max(0.0, orcamento - ws.gasto)
        gasto_sonho = 0.0
        if self.f["sonho"] and sobra > 0:
            gasto_sonho = self._sonha(sobra)
        self._compacta()
        self.evento("problema", idx=prob.idx, fam=prob.fam, tipo=prob.tipo,
                    fonte=prob.fonte, molde=prob.molde, ok=bool(ok), nota=nota,
                    gasto=ws.gasto, sonho=gasto_sonho, T=self._temperatura(ws),
                    acoes=[(a["op"], a.get("arg")) for a in ws.acoes],
                    valido=bool(ws.valido), n_estr=len(self.E))
        self.t += 1
        return ok, ws

    def _contexto_sorteado(self, x):
        outros = [e.ctx_nasc for e in self.E.values()
                  if e.tipo == "aberto" and e.ctx_nasc is not None]
        if not outros:
            return hdc.codifica({"placebo": int(self.rng.integers(1e9))})
        return outros[int(self.rng.integers(len(outros)))]

    def _reativacoes(self, ws):
        """Guarda quem estava LATENTE no inicio do problema (ativacao ~ 0)."""
        ws.latentes = {e.id: (self.t - (max(e.usos) if e.usos else e.nasc))
                       for e in self.E.values()
                       if e.tipo != "op" and self.ativacao(e) < 0.05}

    def _registra_bum(self, ws, ok):
        """O 'BUM': algo latente voltou e entrou na resposta que deu certo."""
        if not ok or ws.melhor is None:
            return
        usados = set(ws.melhor.get("usados", []))
        usados |= {a["arg"] for a in ws.acoes if a["op"] in ("juntar", "forma", "macro")
                   and a["id"] in self._ancestrais(ws)}
        for sid in usados:
            if sid in getattr(ws, "latentes", {}) and sid in self.E:
                self.evento("reativacao", id=sid, tipo=self.E[sid].tipo,
                            intervalo=ws.latentes[sid])

    # ── a agenda aprendida (metacontrole) ────────────────────────────────────
    def _candidatas(self, ws):
        T = self._temperatura(ws)
        prob = ws.prob
        out = []
        if prob.tipo == "L":
            for e in self._de_tipo("lei"):
                if e.id not in ws.testadas:
                    out.append(("forma", e))
            out.append(("ajustar_base", None))
            out.append(("ajustar_ctx", None))
            if ws.novos_atomos():
                out.append(("ajustar_novos", None))
            for cid in sorted(ws.em_jogo):
                ce = self.E.get(cid)
                if ce is not None and ce.tipo in ("conceito", "direcao") and cid not in ws.testados_com:
                    out.append(("ajustar_com", ce))
            if T > 0.5 or not self.f["regime"]:
                out.append(("ajustar_amplo", None))
            out.append(("derivar", None))
        else:
            out.append(("prever_base", None))
            out.append(("prever_ctx", None))
            if ws.novos_atomos():
                out.append(("prever_novos", None))
            for cid in sorted(ws.em_jogo):
                ce = self.E.get(cid)
                if ce is not None and ce.tipo == "conceito" and cid not in ws.testados_com:
                    out.append(("prever_com", ce))
        fontes = self._de_tipo("aberto") + (self._de_tipo("episodio")
                                            if not self.f["meso"] else [])
        for e in fontes:
            if e.id not in ws.juntados:
                out.append(("juntar", e))
        if ws.trajs_atuais() is not None or ws.trajs:
            out.append(("inventar", None))
        if ws.trajs_atuais() is not None:
            for e in self._de_tipo("conceito"):
                if e.id not in ws.reajustados:
                    out.append(("reajustar", e))
        if self.f["coal"]:
            for e in self._de_tipo("coalizao"):
                if e.id not in ws.macros and e.payload["tipo_prob"] == prob.tipo:
                    out.append(("macro", e))
        return out

    def _custo_previsto(self, op, e, ws):
        """Custo APRENDIDO pelo uso (media movel do custo real da operacao,
        escalada pelo tamanho do dado); a formula so vale antes do primeiro uso."""
        n = sum(len(S) for S, _A in ws.dados) if ws.dados else 16
        oe = self.E[f"op:{op}"]
        cm = oe.payload.get("custo_med")
        if op == "juntar" and e is not None:
            add = sum(len(S) for S, _A in e.payload.get("dados", []))
            ab = self.E["op:ajustar_base"].payload
            por_ponto = (ab.get("custo_med") or 10.0) / max(ab.get("n_med") or 24, 1)
            return 0.2 + add * por_ponto
        if op.startswith("ajustar") or op.startswith("prever") or op == "inventar":
            w = self._trabalho(op, e, ws)
            cpt = self._cpt.get(op[:6])
            if cpt is not None:
                return cpt * w
            return F.custo(w * 8)
        if cm is not None and op != "macro":
            return cm
        return self._custo_formula(op, e, ws, n)

    def _trabalho(self, op, e, ws):
        if op == "inventar":
            from math import comb
            a = 6 + len(ws.atomos_inv(self))
            m = a + a * (a + 1) // 2
            Tc = ws.trajs_atuais() if ws.trajs_atuais() is not None else (ws.trajs[-1] if ws.trajs else None)
            npts = (Tc.shape[0] * Tc.shape[1] // 2) if Tc is not None else 100
            return (m + comb(m, 2) + comb(m, 3)) * npts
        if op.startswith("ajustar") and op == "ajustar_com":
            ws.alvo_com = e.id if e is not None else ws.alvo_com
        conc, ext, _u = ws.vocab_para(op, self, dry=True)
        if op.startswith("ajustar"):
            P = 3 + len(conc)
            nc = P * 8 + P * (P - 1) // 2 * 64
            n = sum(len(S) for S, _A in ws.dados)
            return nc * (3 + len(ext)) * 2 * n
        na = 7 + len(conc)
        return (na + na * (na - 1) // 2 * 24) * 32

    def _custo_formula(self, op, e, ws, n):
        if op == "forma":
            return 0.05 + F.custo(n * 8)
        if op.startswith("ajustar"):
            nv = 3 + len(ws.vocab_para(op, self, dry=True)[0])
            nd = 3 + len(ws.vocab_para(op, self, dry=True)[1])
            return F.custo(nv * nv * 64 * nd * n * 2 * 6)
        if op.startswith("prever"):
            na = 7 + len(ws.vocab_para(op, self, dry=True)[0])
            return F.custo((na + na * na * 12) * 32)
        if op == "juntar":
            return 0.2
        if op == "inventar":
            m = 6 + len(ws.atomos_inv(self))
            nm = m + m * (m + 1) // 2
            return F.custo(nm ** 3 / 6 * 100)
        if op == "reajustar":
            return 0.2
        if op == "derivar":
            ne = 3 + len(ws.dirs_em_jogo(self))
            return F.custo((ne + ne * ne) * n * 2 * 400)
        if op == "macro":
            return 0.5 * len(e.payload["passos"])
        return 1.0

    def _confia(self, ws):
        """Parar com resposta validada? Amostra da crenca de que, NESTE contexto,
        validar internamente basta (aprendida pelo verificador)."""
        if not ws.valido:
            return False
        th = self.amostra(self.E["op:confiar"], ws.x, self._temperatura(ws))
        ws.desconfiou = ws.desconfiou or th < 0.5
        return th >= 0.5

    def _agenda_aprendida(self, ws):
        for _seg in range(SEGURANCA_ACOES):
            if ws.gasto >= ws.orcamento or self._confia(ws):
                break
            T = self._temperatura(ws)
            cands = [c for c in self._candidatas(ws) if not ws.ja_feito(c[0], self, c[1])]
            if not cands:
                break
            melhor, vmelhor = None, -1.0
            for op, e in cands:
                th = self.amostra(self.E[f"op:{op}"], ws.x, T)
                if e is not None:
                    xe = ws.x_reat if e.tipo == "aberto" else ws.x
                    th *= self.amostra(e, xe, T)
                c = self._custo_previsto(op, e, ws)
                if c > ws.orcamento - ws.gasto + 1e-9 and op not in ("forma", "reajustar"):
                    continue
                v = th / max(c, 1e-3)
                if v > vmelhor:
                    melhor, vmelhor = (op, e), v
            if melhor is None:
                break
            self._executa(ws, *melhor)

    # ── a agenda fixa (pipeline e SEM_META) ──────────────────────────────────
    def _agenda_fixa(self, ws):
        prob = ws.prob
        ordem = []
        if prob.tipo == "L":
            ordem += [("forma", e) for e in self._de_tipo("lei")]
            ordem.append(("ajustar_ctx", None))
            fontes = self._fontes_fixas(ws)
            ordem += [("juntar", e) for e in fontes]
            if fontes:
                ordem.append(("ajustar_ctx", None))
            if prob.T is not None:
                ordem += [("reajustar", e) for e in self._de_tipo("conceito")]
                if self._de_tipo("conceito"):
                    ordem.append(("ajustar_ctx", None))
                ordem.append(("inventar", None))
                ordem.append(("ajustar_novos", None))
            ordem.append(("derivar", None))
            ordem.append(("ajustar_novos", None))
            ordem.append(("ajustar_amplo", None))
        else:
            ordem.append(("prever_ctx", None))
            if prob.T is not None:
                ordem += [("reajustar", e) for e in self._de_tipo("conceito")]
                ordem.append(("prever_ctx", None))
                ordem.append(("inventar", None))
                ordem.append(("prever_novos", None))
        for op, e in ordem:
            if ws.valido or ws.gasto >= ws.orcamento:
                break
            if e is not None and e.id not in self.E:
                continue
            if ws.ja_feito(op, self):
                continue
            if op == "ajustar_novos" and not ws.novos_atomos():
                continue
            c = self._custo_previsto(op, e, ws)
            if c > ws.orcamento - ws.gasto + 1e-9 and op not in ("forma", "reajustar"):
                continue
            self._executa(ws, op, e)

    def _fontes_fixas(self, ws):
        """Pipeline: RAG — os k episodios mais parecidos (k magico). SEM_META:
        abertos por crenca de contexto, sem k."""
        if self.f["pipeline"]:
            eps = self._de_tipo("episodio")
            eps.sort(key=lambda e: -hdc.sim(ws.x, e.ctx_nasc))
            return eps[:TETOS["rag"]]
        T = self._temperatura(ws)
        fontes = self._de_tipo("aberto") if self.f["meso"] else self._de_tipo("episodio")
        melhor, mm = None, 0.5
        for e in fontes:
            xe = ws.x_reat if e.tipo == "aberto" else ws.x
            m = self.media(e, xe, T)
            if m > mm:
                melhor, mm = e, m
        return [melhor] if melhor is not None else []

    def _de_tipo(self, tipo):
        return [e for e in self.E.values() if e.tipo == tipo]

    # ── execucao de uma operacao ─────────────────────────────────────────────
    def _executa(self, ws, op, e):
        acao = {"op": op, "arg": e.id if e is not None else None, "id": len(ws.acoes),
                "produz": [], "usa": []}
        prob = ws.prob
        antes = ws.chave_melhor()
        gasto0 = ws.gasto
        if op == "forma":
            ws.testadas.add(e.id)
            forma = e.payload["forma"]
            conc, ext, usa = ws.resolve_forma(e, self)
            if conc is None:
                ws.gasta(0.05, False)
            else:
                ok, q, c = F.valida_lei([(prob.S, prob.A)], conc, ext, forma, self.rng)
                ok = ok and not self._refutada(forma, ws)
                leis, bits, c2 = F.ajusta_lei([(prob.S, prob.A)], conc, ext, forma=forma)
                ws.gasta(c + c2, False)
                acao["usa"] = [e.id] + usa
                ws.propoe_lei(leis[0], bits, ok, acao, conc, ext)
        elif op.startswith("ajustar"):
            acao["trabalho"] = self._trabalho(op, e, ws)
            if op == "ajustar_com":
                ws.alvo_com = e.id
                ws.testados_com.add(e.id)
                acao["arg"] = e.id
            conc, ext, usa = ws.vocab_para(op, self)
            ws.assinaturas.add(ws.assinatura(op, usa))
            leis, bits, c = F.ajusta_lei(ws.dados, conc, ext,
                                         limite=max(ws.orcamento - ws.gasto, 1.0))
            lei = leis[0]
            forma = [(v, e_, d) for v, e_, d in lei.termos]
            ok, q, c2 = F.valida_lei([(prob.S, prob.A)], conc, ext, forma, self.rng)
            ok = ok and not self._refutada(forma, ws)
            ws.gasta(c + c2, False)
            acao["usa"] = usa
            acao["vocab"] = sorted(usa)
            ws.propoe_lei(lei, bits, ok, acao, conc, ext)
        elif op.startswith("prever"):
            acao["trabalho"] = self._trabalho(op, e, ws)
            if op == "prever_com":
                ws.alvo_com = e.id
                ws.testados_com.add(e.id)
                acao["arg"] = e.id
            conc, _ext, usa = ws.vocab_para(op, self)
            ws.assinaturas.add(ws.assinatura(op, usa))
            rot, loo, desc, c = F.prever_q(prob.S, prob.y, prob.consultas(), conc)
            ws.gasta(c, False)
            acao["usa"] = usa
            ws.propoe_rotulos(rot, loo, desc, acao, conc)
        elif op == "juntar":
            ws.juntados.add(e.id)
            ws.dados += [d for d in e.payload["dados"] if prob.tipo == "L"]
            ws.trajs += e.payload["trajs"]
            for cid in e.payload.get("candidatos", []):
                if cid in self.E:
                    ws.em_jogo.add(cid)
            ws.gasta(0.2, False)
            acao["usa"] = [e.id]
            ws.juntou.append(e.id)
        elif op == "inventar":
            acao["trabalho"] = self._trabalho(op, e, ws)
            ws.assinaturas.add(("inventar", len(ws.trajs), tuple(sorted(ws.atomos_inv(self)))))
            Tc = ws.trajs_atuais()
            if Tc is None:
                Tc = ws.trajs[-1]
            n = len(Tc)
            if n < 4:
                ws.gasta(0.1, False)
            else:
                atomos = ws.atomos_inv(self)
                conc = {k: v for k, v in atomos.items()}
                out, c = F.inventa(Tc[: n // 2], Tc[n // 2:], conc)
                ws.gasta(c, False)
                acao["usa"] = [ws.id_de_atomo(k) for k in conc]
                for sup, w, g, esc, pais in out:
                    if self._duplicata_conceito(sup, w, esc, pais, Tc):
                        continue
                    nome = f"c{next(self.n_ids)}"
                    C = F.Conceito(sup, w, esc, pais, nome)
                    pid = [ws.id_de_atomo(p) for p in pais]
                    novo = self._nova("conceito", {"c": C, "ganho": g}, prob.fonte,
                                      pais=[p for p in pid if p], ctx=ws.x)
                    novo.origem = ("op:inventar", None)
                    ws.novos.add(novo.id)
                    ws.em_jogo.add(novo.id)
                    acao["produz"].append(novo.id)
        elif op == "reajustar":
            ws.reajustados.add(e.id)
            Tc = ws.trajs_atuais()
            C2, lam, c = F.reajusta_suporte(e.payload["c"], Tc)
            ws.gasta(c, False)
            if C2 is not None:
                ws.variantes[e.id] = C2
                ws.em_jogo.add(e.id)
                acao["usa"] = [e.id]
                acao["produz"].append(e.id + "~")
        elif op == "derivar":
            ws.assinaturas.add(("derivar", tuple(sorted(ws.dirs_em_jogo(self)))))
            exist = [self.E[i].payload["expr"] for i in ws.dirs_em_jogo(self)]
            novos, c = F.deriva_direcoes(exist, prob.S)
            ws.gasta(c, False)
            for expr in novos:
                if any(F.nome_dir(d.payload["expr"]) == F.nome_dir(expr)
                       for d in self._de_tipo("direcao")):
                    did = next(d.id for d in self._de_tipo("direcao")
                               if F.nome_dir(d.payload["expr"]) == F.nome_dir(expr))
                    ws.em_jogo.add(did)
                    continue
                novo = self._nova("direcao", {"expr": expr}, prob.fonte,
                                  pais=[], ctx=ws.x)
                novo.origem = ("op:derivar", None)
                ws.novos.add(novo.id)
                ws.em_jogo.add(novo.id)
                acao["produz"].append(novo.id)
        elif op == "macro":
            ws.macros.add(e.id)
            acao["usa"] = [e.id]
            for passo, sid in e.payload["passos"]:
                if ws.valido or ws.gasto >= ws.orcamento:
                    break
                se = self.E.get(sid) if sid else None
                if sid and se is None:
                    continue
                if passo == "reajustar" and ws.trajs_atuais() is None:
                    continue
                if passo == "juntar" and se is not None and se.id in ws.juntados:
                    continue
                sub = self._executa(ws, passo, se)
                sub["dentro_de"] = acao["id"]
        oe = self.E[f"op:{op}"]
        real = ws.gasto - gasto0
        if (op.startswith("ajustar") or op.startswith("prever") or op == "inventar") and acao.get("trabalho"):
            k = op[:6]
            cpt = real / acao["trabalho"]
            self._cpt[k] = cpt if k not in self._cpt else 0.8 * self._cpt[k] + 0.2 * cpt
        nd = sum(len(S) for S, _A in ws.dados) if ws.dados else 16
        if oe.payload.get("custo_med") is None:
            oe.payload["custo_med"], oe.payload["n_med"] = real, nd
        else:
            oe.payload["custo_med"] = 0.8 * oe.payload["custo_med"] + 0.2 * real
            oe.payload["n_med"] = 0.8 * oe.payload["n_med"] + 0.2 * nd
        acao["custo"] = real
        acao["progresso"] = ws.chave_melhor() != antes
        if acao["progresso"]:
            ws.gasto_sem_progresso = 0.0
        ws.acoes.append(acao)
        return acao

    def _refutada(self, forma, ws):
        if self.f["pipeline"]:
            return False
        f2 = tuple((tuple(v), tuple(e_), d) for v, e_, d in forma)
        T = self._temperatura(ws)
        for e in self._de_tipo("lei"):
            if e.payload["forma"] == f2 and e.regs:
                if self.media(e, ws.x, T) < 0.5:
                    return True
        return False

    def _duplicata_conceito(self, sup, w, esc, pais, Tc):
        S = Tc.reshape(-1, 4)[::5]
        q = F.Conceito(sup, w, esc, pais, "tmp")(S)
        for e in self._de_tipo("conceito"):
            try:
                q2 = e.payload["c"](S)
            except Exception:
                continue
            if np.std(q2) > 0 and abs(np.corrcoef(q, q2)[0, 1]) > 0.999:
                return True
        return False

    # ── credito ──────────────────────────────────────────────────────────────
    def _ancestrais(self, ws):
        """Acoes na cadeia de dependencia da resposta final."""
        if ws.melhor is None:
            return set()
        final = ws.melhor["acao"]
        anc = {final["id"]}
        usados = set(ws.melhor.get("usados", []))
        for a in ws.acoes:
            if set(a["produz"]) & usados or (a["op"] == "juntar" and a["arg"] in usados):
                anc.add(a["id"])
            if a["op"] == "reajustar" and (a["arg"] in usados):
                anc.add(a["id"])
        for a in ws.acoes:
            if a.get("dentro_de") in anc:
                anc.add(a["id"])
            if a["id"] in anc and a.get("dentro_de") is not None:
                anc.add(a["dentro_de"])
        return anc

    def _registra(self, e, x, ok):
        e.regs.append([self.t, x, float(ok)])
        e._C = None
        if ok:
            e.usos.append(self.t)
            e.forca += 1.0
            e.estab = 1 - 1 / (1 + len(set(int(t) for t, _c, o in e.regs if o > 0)))
        else:
            e.forca *= 0.9
            if e.ctx_nasc is not None and hdc.sim(x, e.ctx_nasc) > 0.5:
                e.estab *= 0.7

    def _credito(self, ws, ok):
        x = ws.x
        anc = self._ancestrais(ws) if ok else set()
        if not self.f["eco"] and ok and ws.melhor is not None:
            anc = {ws.melhor["acao"]["id"]}
        usados = set(ws.melhor.get("usados", [])) if (ok and ws.melhor) else set()
        for a in ws.acoes:
            a_ok = a["id"] in anc
            opk = f"op:{a['op']}"
            self._registra(self.E[opk], x, a_ok)
            if a["arg"] and a["arg"] in self.E and a["op"] != "macro":
                self._registra(self.E[a["arg"]], ws.x_reat if self.E[a["arg"]].tipo == "aberto" else x, a_ok)
            if a["op"] == "macro" and a["arg"] in self.E:
                self._registra(self.E[a["arg"]], x, a_ok)
        # atomos de vocabulario: pagou se a resposta final os usa
        tentados = set()
        for a in ws.acoes:
            tentados |= set(a.get("usa", []))
        for sid in tentados:
            if sid in self.E and self.E[sid].tipo in ("conceito", "direcao", "lei"):
                if not any(a["arg"] == sid for a in ws.acoes):
                    self._registra(self.E[sid], x, sid in usados)
        # candidatos novos que nao foram tentados: so existem (potencial)
        if ok:
            for sid in usados:
                self._propaga(sid, x)
            self._fonte(ws, True, usados)
        else:
            self._fonte(ws, False, tentados)
        # leis: a resposta validada vira estrutura (forma + conceitos usados)
        if ws.melhor is not None and ws.prob.tipo == "L" and (ok or ws.valido):
            self._guarda_lei(ws, ok)
        # SEM_MESO: o que nao pagou neste problema e descartado (booleano)
        if not self.f["meso"]:
            for sid in list(ws.novos):
                if sid in self.E and sid not in usados:
                    self._dissolve(self.E[sid], "booleano")
        if self.f["pipeline"]:
            for sid in usados:
                if sid in self.E and self.E[sid].tipo in ("conceito", "direcao") and sid not in self.admitidos:
                    self.admitidos.append(sid)
            while len(self.admitidos) > TETOS["admitidos"] * 2:
                self.admitidos.pop(0)
            for sid in list(ws.novos):
                if sid in self.E and sid not in self.admitidos:
                    self._dissolve(self.E[sid], "nao_admitido")
        if not self.f["proc2"]:
            self._apaga_ficcao()
        if self.f["coal"] and ok:
            self._coalizao(ws, anc)

    def _propaga(self, sid, x):
        """Ecologia: a estrutura que GEROU esta (pais, operacao de origem) recebe
        credito atrasado; potencial gerativo dos pais sobe."""
        if not self.f["eco"]:
            return
        e = self.E.get(sid)
        if e is None:
            return
        for p in e.pais:
            pe = self.E.get(p)
            if pe is not None:
                pe.desc_cred += 1
                pe.desc_n += 1
                self.evento("credito_atrasado", de=sid, para=p)
        if e.origem is not None and e.nasc < self.t:
            op = self.E.get(e.origem[0])
            if op is not None:
                op.regs.append([e.nasc, e.ctx_nasc if e.ctx_nasc is not None else x, 1.0])
                op._C = None
                self.evento("credito_atrasado", de=sid, para=e.origem[0], nasc=e.nasc)

    def _fonte(self, ws, ok, ids):
        """Crenca da fonte so muda por AFIRMACOES (leis) testadas em problema
        percebido. Ideias (conceitos, direcoes) nao mudam a crenca da fonte."""
        if ws.prob.fonte != "percebido":
            return
        for sid in ids:
            e = self.E.get(sid)
            if e is None or e.tipo != "lei" or e.fonte in ("percebido", "sonhado"):
                continue
            # so conta como teste da AFIRMACAO onde o mundo parece o mesmo em que
            # ela foi feita (peso pelo mesmo nucleo de contexto da crenca)
            w = float(np.exp(BETA0 * (hdc.sim(ws.x, e.ctx_nasc) - 1.0))) if e.ctx_nasc is not None else 0.0
            if w < 1e-3:
                continue
            ab = self.fonte_cred.setdefault(e.fonte, [1.0, 1.0])
            ab[0 if ok else 1] += w
            self.evento("fonte", fonte=e.fonte, ok=bool(ok), peso=w, cred=ab[0] / sum(ab))

    def _apaga_ficcao(self):
        for e in list(self.E.values()):
            if e.tipo == "op":
                continue
            cf, n = self._cred_fonte(e.fonte)
            if n >= 3 and cf < 0.5:
                self._dissolve(e, "fonte_nao_crivel")

    def _guarda_lei(self, ws, ok):
        m = ws.melhor
        lei = m["resposta"]
        forma = tuple((tuple(v), tuple(e_), d) for v, e_, d in lei.termos)
        for e in self._de_tipo("lei"):
            if e.payload["forma"] == forma:
                if not any(a["arg"] == e.id for a in ws.acoes):
                    self._registra(e, ws.x, ok)
                ws.leis_guardadas.append(e.id)
                return
        usados = [u for u in m.get("usados", []) if u in self.E and self.E[u].tipo in ("conceito", "direcao")]
        nova = self._nova("lei", {"forma": forma, "repr": repr(lei), "atomos": usados},
                          ws.prob.fonte, pais=usados, ctx=ws.x)
        nova.origem = (f"op:{m['acao']['op']}", None)
        ws.leis_guardadas.append(nova.id)
        self._registra(nova, ws.x, ok)

    def _coalizao(self, ws, anc):
        passos = []
        for a in ws.acoes:
            if a["id"] in anc and a["op"] not in ("macro",) and a.get("dentro_de") is None:
                arg = a["arg"]
                if (a["op"].startswith("ajustar") or a["op"].startswith("prever")) \
                        and not a["op"].endswith("_com"):
                    arg = None
                passos.append((a["op"], arg))
        if len(passos) < 2:
            return
        chave = (ws.prob.tipo, tuple(passos))
        vistos = self.padroes.setdefault(chave, set())
        vistos.add(ws.prob.idx)
        if len(vistos) >= REGULAR:
            for e in self._de_tipo("coalizao"):
                if e.payload["chave"] == chave:
                    return
            c = self._nova("coalizao", {"passos": list(passos), "chave": chave,
                                        "tipo_prob": ws.prob.tipo},
                           "percebido", pais=[p for _o, p in passos if p], ctx=ws.x)
            self._registra(c, ws.x, True)
            self.evento("coalizao", id=c.id, passos=[list(p) for p in passos],
                        problemas=sorted(vistos))

    # ── mesoestado: problemas abertos e episodios ────────────────────────────
    def _atualiza_abertos(self, ws, ok):
        prob = ws.prob
        dados = [(prob.S, prob.A)] if prob.tipo == "L" else []
        trajs = [prob.T] if prob.T is not None else []
        if self.f["meso"]:
            juntou = [i for i in ws.juntou if i in self.E and self.E[i].tipo == "aberto"]
            cands = sorted(ws.em_jogo & {e.id for e in self.E.values()
                                          if e.tipo in ("conceito", "direcao")})
            if not juntou and ws.melhor is not None and ws.melhor["acao"]["op"] == "forma":
                lid = ws.melhor["acao"]["arg"]
                for e in self._de_tipo("aberto"):
                    if lid in e.payload.get("leis", []):
                        juntou = [e.id]
                        break
            if juntou:
                ab = self.E[juntou[0]]
                ab.payload["dados"] += dados
                ab.payload["trajs"] += trajs
                ab.payload["candidatos"] = sorted(set(ab.payload.get("candidatos", [])) | set(cands))
                ab.payload["encontros"] += 1
                ab.payload["resolvido"] = ab.payload.get("resolvido", False) or ok
                if ws.melhor is not None and prob.tipo == "L":
                    ab.payload["melhor"] = repr(ws.melhor["resposta"])
                ab.payload.setdefault("leis", [])
                ab.payload["leis"] += [i for i in ws.leis_guardadas if i not in ab.payload["leis"]]
            elif dados or trajs:
                ctx = ws.x
                ab = self._nova("aberto", {"dados": dados, "trajs": trajs,
                                           "candidatos": cands, "encontros": 1,
                                           "resolvido": ok, "leis": list(ws.leis_guardadas),
                                           "melhor": repr(ws.melhor["resposta"]) if (ws.melhor and prob.tipo == "L") else None},
                                prob.fonte, ctx=ctx)
                if ok:
                    self._registra(ab, ctx, True)
        else:
            if dados or trajs:
                self._nova("episodio", {"dados": dados, "trajs": trajs, "candidatos": []},
                           prob.fonte, ctx=ws.x)

    def _compacta(self):
        """Memoria externa finita: quando passa, os dados mais velhos das fontes de
        menor forca sao descartados (a estrutura fica, o dado bruto vai)."""
        tot = sum(sum(len(S) for S, _A in e.payload.get("dados", [])) +
                  sum(T.shape[0] * T.shape[1] for T in e.payload.get("trajs", []))
                  for e in self.E.values() if e.tipo in ("aberto", "episodio"))
        if tot <= MEM_PONTOS:
            return
        fontes = sorted([e for e in self.E.values() if e.tipo in ("aberto", "episodio")],
                        key=lambda e: (e.forca, e.nasc))
        for e in fontes:
            if tot <= MEM_PONTOS:
                break
            while (e.payload["dados"] or e.payload["trajs"]) and tot > MEM_PONTOS:
                if e.payload["trajs"]:
                    T = e.payload["trajs"].pop(0)
                    tot -= T.shape[0] * T.shape[1]
                else:
                    S, _A = e.payload["dados"].pop(0)
                    tot -= len(S)
            self.evento("compactacao", id=e.id)

    # ── regime ───────────────────────────────────────────────────────────────
    def _atualiza_regime(self, ok, ws):
        if not self.f["regime"]:
            return
        antes = self.T_vida
        self.T_vida = (1 - EMA_T) * self.T_vida + EMA_T * (0.0 if ok else 1.0)
        if abs(self.T_vida - antes) > 0.05 or (antes < 0.5 <= self.T_vida) or (antes >= 0.5 > self.T_vida):
            self.evento("regime", T=self.T_vida, ok=bool(ok))
        # metaplasticidade: o que estava latente e pagou alonga a memoria do tipo
        for sid in ws.melhor.get("usados", []) if ws.melhor else []:
            e = self.E.get(sid)
            if e is None or len(e.usos) < 2:
                continue
            gap = e.usos[-1] - e.usos[-2]
            tau = self.tau.get(e.tipo, TAU0)
            if ok and gap > tau:
                self.tau[e.tipo] = tau + ETA_META * (gap - tau)
                self.evento("metaplasticidade", tipo=e.tipo, tau=self.tau[e.tipo])

    # ── sonho ────────────────────────────────────────────────────────────────
    def _sonha(self, sobra):
        """Replay: problemas abertos nao resolvidos x estruturas nascidas depois do
        ultimo encontro deles, ajustados no dado GUARDADO. O que valida vira lei
        com fonte 'sonhado' (crenca baixa ate um problema percebido confirmar)."""
        gasto = 0.0
        abertos = [e for e in self._de_tipo("aberto")
                   if not e.payload.get("resolvido") and e.payload["dados"]]
        if not abertos:
            return 0.0
        T = self._temperatura()
        abertos.sort(key=lambda e: -self.amostra(e, e.ctx_nasc, T))
        for ab in abertos:
            if gasto >= sobra:
                break
            ult = ab.payload.get("ult_sonho", ab.nasc)
            novos = [e for e in self.E.values() if e.tipo in ("conceito", "direcao")
                     and e.nasc > ult]
            if not novos:
                continue
            novos += [e for e in self.E.values() if e.tipo in ("conceito", "direcao")
                      and e.nasc <= ult and any(o > 0 for _t, _c, o in e.regs)]
            conc = {e.payload["c"].nome: e.payload["c"] for e in novos if e.tipo == "conceito"}
            ext = [e.payload["expr"] for e in novos if e.tipo == "direcao"]
            for i in ab.payload.get("candidatos", []):
                ce = self.E.get(i)
                if ce is not None and ce.tipo == "direcao":
                    ext.append(ce.payload["expr"])
            dados = ab.payload["dados"]
            leis, bits, c = F.ajusta_lei(dados, conc, ext)
            gasto += c
            ab.payload["ult_sonho"] = self.t
            lei = leis[0]
            forma = [(v, e_, d) for v, e_, d in lei.termos]
            okv, q, c2 = F.valida_lei(dados, conc, ext, forma, self.rng)
            gasto += c2
            self.evento("sonho", aberto=ab.id, valida=bool(okv), lei=repr(lei))
            if okv:
                f2 = tuple((tuple(v), tuple(e_), d) for v, e_, d in lei.termos)
                if not any(e.payload["forma"] == f2 for e in self._de_tipo("lei")):
                    usados = [e.id for e in novos if (e.tipo == "conceito" and
                              any(e.payload["c"].nome in v for v, _e, _d in lei.termos))
                              or (e.tipo == "direcao" and any(F.nome_dir(e.payload["expr"]) == d
                                                              for _v, _e, d in lei.termos))]
                    nova = self._nova("lei", {"forma": f2, "repr": repr(lei), "atomos": usados},
                                      "sonhado", pais=usados, ctx=ab.ctx_nasc)
                    nova.origem = ("op:ajustar_amplo", None)
                    ab.payload.setdefault("candidatos", []).append(nova.id)
        return gasto


# ─── o espaco de trabalho de um problema ─────────────────────────────────────

class Espaco:
    def __init__(self, prob, x, orcamento):
        self.prob, self.x, self.orcamento = prob, x, orcamento
        self.x_reat = x
        self.gasto = 0.0
        self.gasto_sem_progresso = 0.0
        self.dados = [(prob.S, prob.A)] if prob.tipo == "L" else []
        self.trajs = []
        self.acoes = []
        self.melhor = None
        self.valido = False
        self.testadas, self.juntados, self.reajustados, self.macros = set(), set(), set(), set()
        self.novos, self.em_jogo = set(), set()
        self.variantes = {}
        self.juntou = []
        self._mapa = {}          # nome do conceito -> id da estrutura
        self._mapa_dir = {}      # id da direcao -> nome da direcao
        self.assinaturas = set()
        self.desconfiou = False
        self.testados_com = set()
        self.confirmados = set()
        self.alvo_com = None
        self.leis_guardadas = []

    def assinatura(self, op, usa):
        tipo = "prever" if op.startswith("prever") else "ajustar"
        if op.endswith("_com"):
            return (tipo, "com", self.alvo_com, tuple(sorted(usa)), len(self.dados))
        return (tipo, tuple(sorted(usa)), len(self.dados),
                tuple(sorted(self.variantes)))

    def ja_feito(self, op, org, e=None):
        if op.endswith("_com"):
            return e is not None and e.id in self.testados_com
        if op == "inventar":
            return ("inventar", len(self.trajs), tuple(sorted(self.atomos_inv(org)))) in self.assinaturas
        if op == "derivar":
            return ("derivar", tuple(sorted(self.dirs_em_jogo(org)))) in self.assinaturas
        if not (op.startswith("ajustar") or op.startswith("prever")):
            return False
        _c, _e, usa = self.vocab_para(op, org, dry=True)
        return self.assinatura(op, usa) in self.assinaturas

    def gasta(self, c, progresso):
        self.gasto += c
        if not progresso:
            self.gasto_sem_progresso += c

    def trajs_atuais(self):
        return self.prob.T

    def novos_atomos(self):
        return bool(self.novos)

    def chave_melhor(self):
        if self.melhor is None:
            return None
        return (round(self.melhor["bits"], 6), self.valido)

    # vocabulario por estrategia (o metacontrole escolhe a estrategia; a economia
    # escolhe os atomos dentro dela pelo orcamento que sobra)
    def vocab_para(self, op, org, dry=False):
        T = org._temperatura(self)
        x = self.x
        cands = []
        for e in org.E.values():
            if e.tipo == "conceito" or (e.tipo == "direcao" and not op.startswith("prever")):
                cands.append(e)
        if org.f["pipeline"]:
            ids = [i for i in org.admitidos if i in org.E]
            ids += [i for i in self.novos if op.endswith("novos")]
            if op.endswith("amplo"):
                ids = [e.id for e in cands]
            sel = [org.E[i] for i in dict.fromkeys(ids) if i in org.E and (org.E[i].tipo != "direcao" or not op.startswith("prever"))]
        elif op.endswith("base"):
            sel = []
        elif op.endswith("_com"):
            sel = self._vocab_ctx(org, cands, x, T)
            alvo = org.E.get(self.alvo_com)
            if alvo is not None and alvo not in sel:
                sel = sel + [alvo]
        elif op.endswith("novos"):
            sel = [e for e in cands if e.id in self.novos]
            sel += [e for e in cands if e.id in self.em_jogo and e not in sel]
        elif op.endswith("amplo"):
            sel = cands
        else:   # ctx: crenca de contexto, sem numero de atomos fixo
            sel = self._vocab_ctx(org, cands, x, T)
        conc, ext, usa = {}, [], []
        for e in sel:
            if e.tipo == "conceito":
                C = self.variantes.get(e.id, e.payload["c"])
                conc[C.nome] = C
                self._mapa[C.nome] = e.id
            else:
                if op.startswith("prever"):
                    continue
                ext.append(e.payload["expr"])
                self._mapa_dir[e.id] = F.nome_dir(e.payload["expr"])
            usa.append(e.id)
        return conc, ext, usa

    def _vocab_ctx(self, org, cands, x, T):
        """Atomos cuja crenca de pagar NESTE contexto passa de 1/2 (decisao de
        Bayes com perda simetrica), mais os ja confirmados neste problema."""
        medias = [(org.media(e, x, T), e) for e in cands if e.regs]
        medias.sort(key=lambda m: -m[0])
        sel = [e for m, e in medias if m > 0.5]
        sel += [org.E[i] for i in self.confirmados if i in org.E and org.E[i] not in sel]
        if org.f["tetos"]:
            sel = sel[:TETOS["vocab"]]
        return sel

    def atomos_inv(self, org):
        """Atomos para a invencao: conceitos que ja pagaram em algum lugar."""
        out = {}
        for e in org.E.values():
            if e.tipo == "conceito" and any(o > 0 for _t, _c, o in e.regs):
                C = self.variantes.get(e.id, e.payload["c"])
                out[C.nome] = C
                self._mapa[C.nome] = e.id
        if org.f["tetos"]:
            out = dict(list(out.items())[:TETOS["vocab"]])
        return out

    def id_de_atomo(self, nome):
        return self._mapa.get(nome)

    def dirs_em_jogo(self, org):
        """Direcoes que ja pagaram em algum lugar: as unicas de onde se deriva."""
        return [e.id for e in org.E.values() if e.tipo == "direcao"
                and any(o > 0 for _t, _c, o in e.regs)]

    def resolve_forma(self, lei_e, org):
        conc, ext, usa = {}, [], []
        for aid in lei_e.payload.get("atomos", []):
            a = org.E.get(aid)
            if a is None:
                return None, None, []
            if a.tipo == "conceito":
                C = self.variantes.get(aid, a.payload["c"])
                conc[C.nome] = C
                self._mapa[C.nome] = aid
            else:
                ext.append(a.payload["expr"])
                self._mapa_dir[aid] = F.nome_dir(a.payload["expr"])
            usa.append(aid)
        # a forma pode citar conceitos pelo nome: conferir que todos existem
        nomes = {n for v, _e, _d in lei_e.payload["forma"] for n in v}
        base = {"r", "v", "vr"}
        if not nomes <= (base | set(conc)):
            return None, None, []
        dnames = {d for _v, _e, d in lei_e.payload["forma"]}
        if not dnames <= (set(F.DIR_BASE) | {F.nome_dir(e) for e in ext}):
            return None, None, []
        return conc, ext, usa

    def propoe_lei(self, lei, bits, ok, acao, conc, ext):
        nomes = {n for v, _e, _d in lei.termos for n in v}
        dn = {d for _v, _e, d in lei.termos}
        realmente = [self._mapa[n] for n in nomes if n in self._mapa]
        realmente += [sid for sid in acao.get("usa", []) if sid.startswith("direcao")
                      and self._mapa_dir.get(sid) in dn]
        if acao["op"] == "forma":
            realmente.append(acao["arg"])
        juntados = [a["arg"] for a in self.acoes if a["op"] == "juntar"]
        if len(self.dados) > 1:
            realmente += juntados
        melhor = self.melhor
        if (ok and not self.valido) or melhor is None or (ok == self.valido and bits < melhor["bits"]):
            self.melhor = {"resposta": lei, "bits": bits, "acao": acao,
                           "usados": sorted(set(realmente))}
            self.valido = self.valido or ok
            if ok:
                self.confirmados |= {s for s in realmente if s.startswith(("conceito", "direcao"))}
            if ok:
                self.gasto_sem_progresso = 0.0

    def propoe_rotulos(self, rot, loo, desc, acao, conc):
        ok = loo >= 0.9
        nomes = {desc[1]} if desc[0] == "1" else {desc[1], desc[2]}
        usados = [self._mapa[n] for n in nomes if n in self._mapa]
        if (ok and not self.valido) or self.melhor is None or (ok == self.valido and loo > self.melhor["bits"] * -1):
            self.melhor = {"resposta": rot, "bits": -loo, "acao": acao, "usados": usados,
                           "desc": str(desc[:2] if desc[0] == "1" else desc[:3])}
            self.valido = self.valido or ok

    def resposta(self):
        return None if self.melhor is None else self.melhor["resposta"]


def _chave_termo(t):
    vs, es, d = t
    return "*".join(f"{v}{int(e)}" for v, e in zip(vs, es)) + "|" + d


def _descreve(e):
    if e.tipo == "conceito":
        C = e.payload["c"]
        return {"suporte": [list(m) for m in C.suporte], "prof": C.profundidade}
    if e.tipo == "direcao":
        return {"expr": F.nome_dir(e.payload["expr"])}
    if e.tipo == "lei":
        return {"lei": e.payload.get("repr")}
    if e.tipo == "coalizao":
        return {"passos": [list(p) for p in e.payload["passos"]]}
    return {}
