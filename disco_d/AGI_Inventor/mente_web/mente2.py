"""
mente_web.mente2 (v2) — a segunda crianca: plasticidade LOCAL, autointervencao
================================================================================
Herda tudo da v1 (corpo, sentidos, lacunas, intencao, multimodal) e troca o
MOTOR de consolidacao. Ideias do usuario de 08/10 (Kavli 2026: traducao local de
proteinas; etiqueta e captura; Nobel 2026: optogenetica; ripples antes do
imprevisivel e ripples que recompoem sequencias).

1. TRACO LOCAL (etiqueta). Toda combinacao de duas estruturas vizinhas que ela
   percebe ganha um traco proporcional ao ERRO DE PREVISAO local (bits que a
   segunda peca custou). O traco decai sozinho. Nao ha portao de nascimento:
   nem ">= 2 dominios", nem Dunning. Existir e barato; durar e caro.
2. SONO DISPARADO POR INCERTEZA (ripples). O sono nao vem a cada 25 passos: vem
   quando a surpresa acumulada desde o ultimo sono passa da que tipicamente se
   acumula. Mais incerteza -> mais RECURSO de consolidacao (mais testes).
3. OPTOGENETICA INTERNA (captura). No sono, os elementos mais marcados disputam o
   recurso. Para cada um, ela re-percebe a experiencia recente (hipocampo: uma
   janela de cada pagina recente) COM e SEM aquele elemento e mede em bits quanto
   a descricao dela muda (MDL: dados + lexico). Ligar o que paga = consolidar;
   desligar o que nao paga = deprimir. Estados intermediarios de verdade:
   traco -> candidata -> consolidada -> (re-testada) -> deprimida.
4. IMAGINACAO COMPOSICIONAL (ripples que recompoem). No sono ela troca uma peca
   de uma fusao consolidada pela vizinha de USO (contexto distribucional) e
   preve uma combinacao que NUNCA viu ("de Lisboa" + Porto~Lisboa -> "de Porto").
   Junto, um PLACEBO de imaginacao: troca por uma peca de frequencia parecida mas
   sem parentesco de uso. Se a realidade confirmar a imaginada mais que o
   placebo, ela esta gerando linguagem que nunca leu. Duas confirmacoes no mesmo
   molde = CONSTRUCAO com lacuna ("de ___").
5. PENSAMENTO -> LINGUAGEM. Ao perseguir uma pergunta, alem de "alvo", "alvo +
   vizinho" e "a estrutura de cima", ela pode: dizer o que ESPERA que venha
   depois ("frase"), juntar o IRMAO de uso ("irmao"), ou encaixar o alvo numa
   CONSTRUCAO que ela aprendeu ("construcao"). Qual forma paga para que duvida e
   aprendido (Thompson), nao dado.

O que ja existe (dito antes): etiqueta e captura (Frey e Morris 1997), regras de
tres fatores, tracos de elegibilidade; ablacao como medida de contribuicao (LeCun
1990); lexico hierarquico por MDL (de Marcken 1996); construcoes com lacuna de
texto cru (ADIOS, Solan et al. 2005); replay em RL (Dyna, 1990).
Novo seria: UM mecanismo local (traco + recurso + autointervencao) governando o
nascimento e a morte das estruturas de uma mente que vive na internet real a
partir do byte, e a imaginacao composicional medida contra placebo na realidade.
"""
from __future__ import annotations

import collections
import math

import numpy as np

import digestao as G
from mente import FORMULACOES, SONO_CADA, Mente, limpa, txt

TAU_TRACO = 200.0         # decaimento do traco (passos) — o mesmo da ativacao da v1
JANELAS_HIPOCAMPO = 32    # experiencia recente re-percebida no sono (1 janela/pagina)
TAM_JANELA = 512
TESTES_BASE = 40          # recurso de sono por ciclo tipico (orcamento do PC, declarado)
IMAGINA_POR_SONO = 20     # fusoes consolidadas que servem de molde a cada sono
MAX_IMAGINADAS = 60_000


def aplica(seq, fusoes):
    for _ in range(6):
        out, i, mudou = [], 0, False
        n = len(seq)
        while i < n:
            if i + 1 < n:
                k = fusoes.get((seq[i], seq[i + 1]))
                if k is not None:
                    out.append(k)
                    i += 2
                    mudou = True
                    continue
            out.append(seq[i])
            i += 1
        seq = out
        if not mudou:
            break
    return seq


def _xlogx(c):
    return c * math.log2(c) if c > 0 else 0.0


class Mente2(Mente):
    VERSAO = "v2"
    FUSAO_DUNNING = False
    FORMULACOES = FORMULACOES + ("frase", "irmao", "construcao")

    def __init__(self, semente, pasta, registrar=True):
        super().__init__(semente, pasta, registrar)
        self.traco = {}                  # (q,p) -> [traco, t]   (pares ainda nao consolidados)
        self.w = {}                      # (q,p) -> forca consolidada (bits) das fusoes ativas
        self.testadas = collections.Counter()   # (q,p) -> quantas vezes foi testada
        self.deprimidas = 0
        self.hipocampo = collections.deque(maxlen=JANELAS_HIPOCAMPO)
        self.pressao = 0.0
        self.pressao_passo = collections.deque(maxlen=500)
        self._pressao_agora = 0.0
        self.ult_sono2 = 0
        self.n_sonos2 = 0
        self.imaginadas = {}             # (q,p) -> dict   (imaginacao real)
        self.imaginadas_nulo = {}        # (q,p) -> dict   (placebo de imaginacao)
        self.confirm = {"real": 0, "nulo": 0}
        self.molde_conf = collections.defaultdict(set)   # molde -> pecas confirmadas
        self.construcoes = {}            # molde -> set(pecas)
        self._viz_cache = None
        self._sucessor = {}              # estrutura -> o vizinho que mais a segue
        self._jan_sono = []

    # ── 1. traco local ───────────────────────────────────────────────────────
    def _traco_de(self, par):
        e = self.traco.get(par)
        if e is None:
            return 0.0
        return e[0] * math.exp(-(self.t - e[1]) / TAU_TRACO)

    def _marca(self, par, bits):
        e = self.traco.get(par)
        if e is None:
            if len(self.traco) > 300_000:
                return
            self.traco[par] = [bits, self.t]
        else:
            e[0] = e[0] * math.exp(-(self.t - e[1]) / TAU_TRACO) + bits
            e[1] = self.t

    def _apos_par(self, par, bits):
        if par not in self.fusoes:
            self._marca(par, bits)
        # a realidade confirmou algo que eu so tinha IMAGINADO?
        for tab, nome in ((self.imaginadas, "real"), (self.imaginadas_nulo, "nulo")):
            im = tab.get(par)
            if im is not None and im["confirmada"] is None:
                im["confirmada"] = self.t
                self.confirm[nome] += 1
                if nome == "real":
                    self._marca(par, bits)          # previsao cumprida tambem marca
                    molde = im["molde"]
                    self.molde_conf[molde].add(im["peca"])
                    self.ev("imaginou_e_viu", par=[txt(par[0]), txt(par[1])],
                            molde=[txt(x) if x else "___" for x in molde], espera=self.t - im["t"],
                            confirmadas=self.confirm["real"], placebo=self.confirm["nulo"])
                    if len(self.molde_conf[molde]) >= 2 and molde not in self.construcoes:
                        self.construcoes[molde] = set(self.molde_conf[molde])
                        self.ev("construcao", molde=[txt(x) if x else "___" for x in molde],
                                pecas=[txt(x) for x in list(self.molde_conf[molde])[:8]])
                else:
                    self.ev("placebo_viu", par=[txt(par[0]), txt(par[1])],
                            confirmadas=self.confirm["real"], placebo=self.confirm["nulo"])

    def _apos_fusao(self, par, sur, a, b):
        # fusao consolidada em uso tambem carrega traco (para ser re-testada)
        self._marca(par, float(sur[a:b].sum()) * 0.1)

    # ── 2. sono disparado por incerteza ──────────────────────────────────────
    def _apos_ler(self, pg, dados, blocos):
        if blocos:
            h = self.hist_surp_bloco
            med = float(np.median(h[-2000:])) if len(h) > 30 else float(np.median(blocos))
            self._pressao_agora += sum(max(0.0, b - med) for b in blocos)
        if len(dados) >= 600:
            i = int(self.rng.integers(0, max(1, len(dados) - TAM_JANELA)))
            self.hipocampo.append(dados[i:i + TAM_JANELA])

    def _apos_passo(self):
        self.pressao += self._pressao_agora
        self.pressao_passo.append(self._pressao_agora)
        self._pressao_agora = 0.0
        if len(self.hipocampo) < 8:
            return
        ref = SONO_CADA * float(np.mean(self.pressao_passo)) if self.pressao_passo else 0.0
        desde = self.t - self.ult_sono2
        if (ref > 0 and self.pressao >= ref and desde >= 5) or desde >= 4 * SONO_CADA:
            recurso = int(np.clip(round(TESTES_BASE * (self.pressao / ref if ref > 0 else 1.0)),
                                  10, 3 * TESTES_BASE))
            self.sonha(recurso)
            self.pressao = 0.0
            self.ult_sono2 = self.t

    # ── 3. optogenetica interna ──────────────────────────────────────────────
    def _base_hipocampo(self):
        seqs1 = []
        bytes_cont = np.zeros(256, np.float64)
        for jan in self.hipocampo:
            sur = self.texto.surpresa(jan, False)
            seqs1.append([jan[a:b] for a, b in G.segmenta(jan, sur)])
            bytes_cont += np.bincount(np.frombuffer(jan, np.uint8), minlength=256)
        h0 = -np.log2((bytes_cont + 1) / (bytes_cont.sum() + 256))
        return seqs1, h0

    def _lex_custo(self, u, fundidas, h0, V):
        if u in fundidas:
            return 2.0 * math.log2(V + 1)         # soletrar = apontar as duas partes
        return float(sum(h0[x] for x in u)) + math.log2(V + 1)

    def _teste(self, par, liga, seqs1, perc, C, N, fundidas, h0, V):
        """Bits que a descricao da experiencia recente economiza COM o par fundido
        (comparado a SEM). Re-percebe so as janelas onde as duas pecas aparecem."""
        q, p = par
        alvo = q + p
        fus = dict(self.fusoes)
        if liga:
            fus[par] = alvo
        else:
            fus.pop(par, None)
        idx = [i for i, jan in enumerate(self._jan_sono) if alvo in jan]
        if not idx:
            return None
        dC = collections.Counter()
        for i in idx:
            novo = aplica(seqs1[i], fus)
            for u in perc[i]:
                dC[u] -= 1
            for u in novo:
                dC[u] += 1
        dN = sum(dC.values())
        fund2 = set(fundidas)
        if liga:
            fund2.add(alvo)
        else:
            fund2.discard(alvo)
        Ld_antes = _xlogx(N)
        Ld_depois = _xlogx(N + dN)
        Ll = 0.0
        for u, d in dC.items():
            if d == 0:
                continue
            c0 = C.get(u, 0)
            c1 = c0 + d
            Ld_antes -= _xlogx(c0)
            Ld_depois -= _xlogx(c1)
            if c0 == 0 and c1 > 0:
                Ll += self._lex_custo(u, fund2, h0, V)
            elif c0 > 0 and c1 == 0:
                Ll -= self._lex_custo(u, fundidas, h0, V)
        L_mudanca = (Ld_depois - Ld_antes) + Ll       # depois - antes
        # devolve bits ECONOMIZADOS pelo par fundido
        return -L_mudanca if liga else L_mudanca

    def sonha(self, recurso):
        self.n_sonos2 += 1
        seqs1, h0 = self._base_hipocampo()
        self._jan_sono = list(self.hipocampo)
        perc = [aplica(s, self.fusoes) for s in seqs1]
        C = collections.Counter(u for s in perc for u in s)
        N = sum(C.values())
        V = len(C)
        fundidas = set(self.fusoes.values())
        # os mais marcados disputam o recurso (candidatas e ja consolidadas juntas)
        texto_hip = b"".join(self.hipocampo)
        disputa = []
        for par, e in self.traco.items():
            tr = e[0] * math.exp(-(self.t - e[1]) / TAU_TRACO)
            if tr > 0 and (par[0] + par[1]) in texto_hip:
                disputa.append((tr, par))
        disputa.sort(key=lambda z: -z[0])
        # um quarto do recurso re-testa consolidadas presentes na experiencia
        # recente, as menos testadas primeiro (o que foi ligado pode ser desligado)
        retestar = sorted((par for par in self.fusoes if (par[0] + par[1]) in texto_hip),
                          key=lambda par: self.testadas[par])
        n_re = min(len(retestar), recurso // 4)
        fila = list(retestar[:n_re])
        vistos = set(fila)
        fila += [par for _tr, par in disputa if par not in vistos][: recurso - n_re]
        ligou = deprimiu = rejeitou = 0
        exemplos = []
        self._sono_detalhe = {"t": self.t, "recurso": recurso, "testes": []}
        for par in fila:
            ativa = par in self.fusoes
            if not ativa and (par[0] not in self.E or par[1] not in self.E):
                continue
            d = self._teste(par, not ativa, seqs1, perc, C, N, fundidas, h0, V)
            self.testadas[par] += 1
            if d is None:
                continue
            nome_par = txt(par[0] + par[1])
            if ativa:
                self.w[par] = 0.5 * self.w.get(par, 0.0) + 0.5 * d
                self._sono_detalhe["testes"].append({"u": nome_par, "bits": round(d, 1),
                                                     "acao": "desfez" if self.w[par] < 0 else "manteve"})
                if self.w[par] < 0:                       # deprimir: desligar
                    k = self.fusoes.pop(par)
                    self.w.pop(par, None)
                    self.deprimidas += 1
                    deprimiu += 1
                    if deprimiu <= 3:
                        self.ev("deprimiu", unidade=txt(k), bits=round(d, 2))
            elif d > 0:                                   # capturar: ligar
                self._sono_detalhe["testes"].append({"u": nome_par, "bits": round(d, 1), "acao": "ligou"})
                k = par[0] + par[1]
                self.fusoes[par] = k
                self.w[par] = d
                niv = max(self.E[par[0]]["nivel"], self.E[par[1]]["nivel"]) + 1
                if k not in self.E:
                    self._nasce(k, None, niv, par)
                    self.E[k]["doms"] = set(self.pares.get(par, [0, set()])[1]) or {"fusao"}
                self.pais.setdefault(par[0], set()).add(k)
                self.pais.setdefault(par[1], set()).add(k)
                ligou += 1
                if len(exemplos) < 8:
                    exemplos.append((txt(k), round(d, 1)))
            else:
                rejeitou += 1
                self._sono_detalhe["testes"].append({"u": nome_par, "bits": round(d, 1), "acao": "rejeitou"})
            self.traco.pop(par, None)                     # o recurso foi gasto nele
        imag = self._imagina()
        self.ev("sonho", recurso=recurso, disputa=len(disputa), ligou=ligou, rejeitou=rejeitou,
                deprimiu=deprimiu, exemplos=exemplos, fusoes=len(self.fusoes),
                tracos=len(self.traco), imaginadas=imag, confirmadas=dict(self.confirm),
                construcoes=len(self.construcoes))

    # ── 4. imaginacao composicional (com placebo) ────────────────────────────
    def _vizinhos(self):
        cands = [(p, s) for p, s in self.E.items() if len(limpa(p)) >= 2
                 and (s["ctx_c"] is not None or s["ctx_f"] is not None)]
        if len(cands) < 50:
            return None
        cands = sorted(cands, key=lambda z: -z[1]["n"])[:2000]
        V = np.stack([(s["ctx_c"].astype(np.float32) if s["ctx_c"] is not None else 0)
                      + (s["ctx_f"].astype(np.float32) if s["ctx_f"] is not None else 0)
                      for _p, s in cands])
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        nomes = [p for p, _s in cands]
        freq = np.array([s["n"] for _p, s in cands], float)
        return nomes, {p: i for i, p in enumerate(nomes)}, V, freq

    def _imagina(self):
        viz = self._vizinhos()
        if viz is None or not self.w:
            return 0
        nomes, ix, V, freq = viz
        moldes = sorted(self.w, key=lambda k: -self.w[k])[:IMAGINA_POR_SONO]
        n = 0
        for (q, p) in moldes:
            for lado, fixo, troca in ((0, p, q), (1, q, p)):
                i = ix.get(troca)
                if i is None:
                    continue
                sims = V @ V[i]
                sims[i] = -1
                ordem = np.argsort(-sims)
                proibidos = set(ordem[:20].tolist()) | {i}
                for j in ordem[:3]:
                    nova = nomes[j]
                    par = (nova, fixo) if lado == 0 else (fixo, nova)
                    if par in self.pares or par in self.imaginadas or nova == troca:
                        continue
                    molde = (None, fixo) if lado == 0 else (fixo, None)
                    # placebo: mesma faixa de frequencia, SEM parentesco de uso
                    banda = np.nonzero((freq >= freq[j] / 2) & (freq <= freq[j] * 2))[0]
                    banda = [k for k in banda.tolist() if k not in proibidos]
                    if not banda:
                        continue
                    k = banda[int(self.rng.integers(len(banda)))]
                    falso = (nomes[k], fixo) if lado == 0 else (fixo, nomes[k])
                    if falso in self.pares or falso in self.imaginadas_nulo:
                        continue
                    self.imaginadas[par] = dict(t=self.t, molde=molde, peca=nova, confirmada=None)
                    self.imaginadas_nulo[falso] = dict(t=self.t, molde=molde, peca=nomes[k],
                                                       confirmada=None)
                    n += 1
        for tab in (self.imaginadas, self.imaginadas_nulo):
            if len(tab) > MAX_IMAGINADAS:
                for k in list(tab)[: len(tab) - MAX_IMAGINADAS]:
                    del tab[k]
        return n

    # ── 5. pensamento -> linguagem ───────────────────────────────────────────
    def _formula(self, g, f):
        q = super()._formula(g, f)
        if q is not None or f in ("alvo", "vizinho", "acima"):
            return q
        alvo_b = g["alvo"]
        alvo = limpa(alvo_b)
        if len(alvo) < 2:
            return None
        if f == "frase":                      # dizer o que espero que venha depois
            seq, atual = [alvo], alvo_b
            for _ in range(2):
                atual = self._sucessor.get(atual)
                if atual is None:
                    break
                seq.append(limpa(atual))
            return " ".join(x for x in seq if x) if len(seq) > 1 else None
        if f == "irmao":                      # juntar o irmao de uso
            viz = self._viz_cache
            if viz is None:
                return None
            nomes, ix, V, _fr = viz
            i = ix.get(alvo_b)
            if i is None:
                return None
            sims = V @ V[i]
            sims[i] = -1
            irmao = limpa(nomes[int(np.argmax(sims))])
            return f"{alvo} {irmao}" if irmao and irmao != alvo else None
        if f == "construcao":                 # encaixar o alvo numa construcao aprendida
            for molde, pecas in self.construcoes.items():
                if alvo_b in pecas or any(limpa(x) == alvo for x in pecas):
                    a, b = molde
                    return " ".join(limpa(x) for x in ((a, alvo_b) if a else (alvo_b, b)) if x)
            return None
        return None

    def _telemetria_extra(self, tel):
        ref = SONO_CADA * float(np.mean(self.pressao_passo)) if self.pressao_passo else 0.0
        w = list(self.w.values())
        tel["plasticidade"] = {
            "pressao": round(self.pressao, 3), "limiar_sono": round(ref, 3),
            "desde_ultimo_sono": self.t - self.ult_sono2, "sonos": self.n_sonos2,
            "tracos": len(self.traco), "consolidadas": len(self.fusoes), "deprimidas": self.deprimidas,
            "testadas": len(self.testadas), "hipocampo": len(self.hipocampo),
            "forca_mediana": round(float(np.median(w)), 1) if w else None,
            "forca_max": round(float(np.max(w)), 1) if w else None}
        tel["imaginacao"] = {"imaginadas": len(self.imaginadas), "placebo_imaginadas": len(self.imaginadas_nulo),
                             "confirmadas": dict(self.confirm), "construcoes": len(self.construcoes),
                             "moldes": [" ".join(txt(x) if x else "___" for x in k)
                                        for k in list(self.construcoes)[-8:]]}

    def _detalhe_extra(self, d):
        rec = list(self.imaginadas.items())[-14:]
        nul = list(self.imaginadas_nulo.items())[-14:]
        d["imaginacao"] = {
            "recentes": [{"par": txt(p[0] + p[1]), "molde": " ".join(txt(x) if x else "___" for x in im["molde"]),
                          "peca": txt(im["peca"]), "confirmada": im["confirmada"] is not None} for p, im in rec][::-1],
            "placebo": [{"par": txt(p[0] + p[1]), "confirmada": im["confirmada"] is not None} for p, im in nul][::-1],
            "confirmadas": dict(self.confirm),
            "construcoes": [{"molde": " ".join(txt(x) if x else "___" for x in k), "pecas": [txt(x) for x in list(v)[:8]]}
                            for k, v in list(self.construcoes.items())[-8:]]}
        sd = getattr(self, "_sono_detalhe", None)
        if sd:
            d["sono"] = {**sd, "testes": sd["testes"][:40]}

    def dorme(self):
        super().dorme()
        if (self.t // SONO_CADA) % 4 == 0:
            self._viz_cache = self._vizinhos()
            melhor = {}
            for (q, p), e in self.pares.items():
                if e[0] >= 2 and (q not in melhor or e[0] > melhor[q][0]):
                    melhor[q] = (e[0], p)
            self._sucessor = {q: p for q, (_n, p) in melhor.items()}
        estados = collections.Counter()
        for par, e in list(self.traco.items()):
            tr = e[0] * math.exp(-(self.t - e[1]) / TAU_TRACO)
            if tr < 1e-3:
                del self.traco[par]                 # traco que nao foi capturado decai
            else:
                estados["traco"] += 1
        self.ev("plasticidade", tracos=estados["traco"], consolidadas=len(self.fusoes),
                deprimidas_total=self.deprimidas, testadas=len(self.testadas),
                sonos=self.n_sonos2, imaginadas=len(self.imaginadas),
                confirmadas=dict(self.confirm), construcoes=len(self.construcoes),
                w_mediana=round(float(np.median(list(self.w.values()))), 2) if self.w else None)
