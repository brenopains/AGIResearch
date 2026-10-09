"""
mente_web.marco — o que a mente das 6 h faz que a das 30 min nao fazia
=======================================================================
Uso: python marco.py vidas/v1

Le as fotografias (foto_1800 ... foto_21600), a mente final e uma mente VAZIA
(t = 0), e mede todas no MESMO material: a bateria fixa sondas/bateria.pkl, que a
mente nunca leu. A regua do avaliador (o que e palavra, o que e palavra funcional)
NUNCA entra na mente: so serve para julgar de fora.

Tambem le a ontogenia (ontogenia.jsonl) por janelas de vida: 0-30 min, 30-60 min,
1-2 h, 2-4 h, 4-6 h. Grava <vida>/marcos.json.
"""
import collections
import glob
import json
import math
import os
import pickle
import re
import sys
import tempfile

import numpy as np

import digestao as G
from mente import Mente, limpa
from mente2 import Mente2

ESP = set(b" \t\n\r")
JANELAS = ((0, 1800), (1800, 3600), (3600, 7200), (7200, 14400), (14400, 10 ** 9))
NOME_J = ("0-30min", "30-60min", "1-2h", "2-4h", "4-6h")
PAL = re.compile(r"\w+", re.U)


def analisa(m, dados, com_fusoes=True):
    sur = m.texto.surpresa(dados, False)
    seq = [(a, b, dados[a:b]) for a, b in G.segmenta(dados, sur)]
    if com_fusoes and m.fusoes:
        for _ in range(5):
            out, i, mudou = [], 0, False
            while i < len(seq):
                if i + 1 < len(seq):
                    k = m.fusoes.get((seq[i][2], seq[i + 1][2]))
                    if k is not None:
                        out.append((seq[i][0], seq[i + 1][1], k))
                        i += 2
                        mudou = True
                        continue
                out.append(seq[i])
                i += 1
            seq = out
            if not mudou:
                break
    return seq, sur


def f1(pred, ouro):
    if not pred or not ouro:
        return 0.0
    c = len(pred & ouro)
    p, r = c / len(pred), c / len(ouro)
    return 0.0 if c == 0 else 2 * p * r / (p + r)


def regua(bateria):
    """Do avaliador: frequencia de palavras inteiras e de pedacos dentro de palavras."""
    inteira = collections.Counter()
    for t in bateria["textos"]:
        for w in PAL.findall(t["texto"].decode("utf-8", "replace").lower()):
            inteira[w] += 1
    funcionais = {w for w, _n in inteira.most_common(100)}
    corpus = " ".join(t["texto"].decode("utf-8", "replace").lower() for t in bateria["textos"])
    return inteira, funcionais, corpus


def classe_consulta(q, inteira, funcionais, corpus):
    ws = PAL.findall(q.lower())
    if not ws:
        return "fragmento"
    w = max(ws, key=len)
    if w in funcionais:
        return "funcional"
    sub = len(re.findall(r"\w" + re.escape(w) + r"|" + re.escape(w) + r"\w", corpus))
    if inteira[w] == 0 and sub == 0:
        return "desconhecido"
    return "fragmento" if sub > inteira[w] else "palavra"


def mdl_bits_por_byte(m, bateria):
    """Comprimento de descricao da bateria com o lexico da mente (dados: unigrama
    das unidades percebidas; lexico: soletrar pecas, apontar partes das fusoes)."""
    C = collections.Counter()
    tot_b = 0
    bc = np.zeros(256)
    for t in bateria["textos"]:
        d = t["texto"]
        tot_b += len(d)
        bc += np.bincount(np.frombuffer(d, np.uint8), minlength=256)
        seq, _ = analisa(m, d)
        C.update(p for _a, _b, p in seq)
    h0 = -np.log2((bc + 1) / (bc.sum() + 256))
    N = sum(C.values())
    Ld = N * math.log2(N) - sum(c * math.log2(c) for c in C.values())
    V = len(C)
    fund = set(m.fusoes.values())
    Ll = sum((2 * math.log2(V + 1)) if u in fund else (float(h0[np.frombuffer(u, np.uint8)].sum())
                                                        + math.log2(V + 1)) for u in C)
    return round((Ld + Ll) / max(tot_b, 1), 4)


def mede_foto(m, bateria, regua_):
    inteira, funcionais, corpus = regua_
    res = {}
    por = collections.defaultdict(list)
    F1, F1_n1, palavra, frase, n2, cob, nunid = [], [], 0, 0, 0, [], 0
    for t in bateria["textos"]:
        d = t["texto"]
        seq, sur = analisa(m, d)
        por[t["porta"]].append(float(sur.mean()))
        ouro = {i for i in range(len(d)) if d[i] not in ESP and (i == 0 or d[i - 1] in ESP)}
        F1.append(f1({a for a, _b, _p in seq}, ouro))
        seq1, _ = analisa(m, d, com_fusoes=False)
        F1_n1.append(f1({a for a, _b, _p in seq1}, ouro))
        toks = set(PAL.findall(d.decode("utf-8", "replace").lower()))
        dentro = 0
        for a, b, p in seq:
            s = m.E.get(p)
            if s is None:
                continue
            dentro += b - a
            nunid += 1
            u = limpa(p).lower()
            if len(u) >= 3 and " " not in u and u in toks:
                palavra += 1
            if re.search(r"\w\s+\w", u):
                frase += 1
            if s["nivel"] >= 2:
                n2 += 1
        cob.append(dentro / max(len(d), 1))
    res["bits_por_byte"] = {k: round(float(np.mean(v)), 3) for k, v in por.items()}
    res["f1_fronteira"] = round(float(np.mean(F1)), 3)
    res["f1_fronteira_so_nivel1"] = round(float(np.mean(F1_n1)), 3)
    res["unidades_conhecidas"] = nunid
    res["frac_palavra_inteira"] = round(palavra / max(nunid, 1), 3)
    res["frac_frase"] = round(frase / max(nunid, 1), 3)
    res["frac_nivel2"] = round(n2 / max(nunid, 1), 3)
    res["cobertura"] = round(float(np.mean(cob)), 3)
    res["estruturas"] = len(m.E)
    res["por_nivel"] = dict(collections.Counter(min(s["nivel"], 5) for s in m.E.values()))
    res["fusoes"] = len(m.fusoes)
    ab = [g for g in m.lac.values() if g["estado"] == "aberta"]
    res["lacunas_abertas"] = dict(collections.Counter(g["tipo"] for g in ab))
    cls = collections.Counter(classe_consulta(limpa(g["alvo"]), inteira, funcionais, corpus) for g in ab)
    res["alvos_das_lacunas"] = dict(cls)
    # imagens nunca vistas: o titulo preve o prototipo da cena melhor que o acaso?
    if len(m.glob.P) and m.gr.tot:
        real, nulo = [], []
        ims = bateria["imagens"]
        perm = np.random.default_rng(0).permutation(len(ims))
        protos = [int(m.glob.codifica(G.descritor_global(im["quadro"])[None], 0, aprende=False)[0])
                  for im in ims]
        for i, im in enumerate(ims):
            for alvo_i, lista in ((i, real), (int(perm[i]), nulo)):
                itens = [p for _a, _b, p in analisa(m, ims[alvo_i]["titulo"])[0] if p in m.E]
                pr = ("v", protos[i])
                nP = len(m.glob.P) + len(m.aud.P) + 1
                pm = (m.gr.cp[pr] + 1.0) / (m.gr.tot + nP + 1.0)
                gs = []
                for s in itens:
                    cs = m.gr.cs[s]
                    if cs:
                        e = m.gr.c.get((s, pr))
                        gs.append(math.log2((((e[0] if e else 0) + 2 * pm) / (cs + 2)) / pm))
                if gs:
                    lista.append(float(np.mean(gs)))
        res["imagens_ganho_bits"] = {"real": round(float(np.mean(real)), 4) if real else None,
                                     "nulo_titulo_trocado": round(float(np.mean(nulo)), 4) if nulo else None,
                                     "n": len(real)}
    res["confirmadas"] = {"texto_imagem_som": len(m.gr.conf), "nulo": len(m.gr_nulo.conf),
                          "imagem_som": len(m.av.conf), "imagem_som_nulo": len(m.av_nulo.conf),
                          "convergencias": len(m.convergencias)}
    res["barras_descobertas"] = len(m.barras_novas)
    res["mdl_bits_por_byte"] = mdl_bits_por_byte(m, bateria)
    res["versao"] = getattr(m, "VERSAO", "v1")
    if hasattr(m, "confirm"):
        res["v2"] = {"imaginadas": len(m.imaginadas), "confirmadas": dict(m.confirm),
                     "construcoes": len(m.construcoes), "deprimidas": m.deprimidas,
                     "sonos": m.n_sonos2, "fusoes_testadas": len(m.testadas)}
    res["passos"], res["vida_min"] = m.t, round(m.vida_s / 60, 1)
    return res


def sinal(k, n):
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * min(p, 1 - p + math.comb(n, k) / 2 ** n))


def janelas(evs, regua_, jan=JANELAS, nomes=NOME_J):
    inteira, funcionais, corpus = regua_
    out = []
    for (a, b), nome in zip(jan, nomes):
        E = [e for e in evs if a <= e.get("vida_s", 0) < b]
        if not E:
            continue
        acs = collections.Counter(e["estrategia"] for e in E if e["ev"] == "acao")
        it = [e for e in E if e["ev"] == "intencao" and not e["mesma_pagina"] and not e["falhou"]]
        pos = sum(1 for e in it if e["net"] > 0)
        cons = [e["consulta"] for e in E if e["ev"] == "pesquisa"]
        cc = collections.Counter(classe_consulta(q, inteira, funcionais, corpus) for q in cons)
        # insistir: mesma lacuna, mesmo braco e mesma consulta logo depois de nao mudar nada
        ult, ins, base = {}, 0, 0
        for e in [e for e in E if e["ev"] == "intencao"]:
            k = e["alvo"]
            if k in ult and not ult[k].get("pagou", ult[k]["net"] > 0):
                base += 1
                if ult[k]["braco"] == e["braco"] and ult[k]["consulta"] == e["consulta"]:
                    ins += 1
            ult[k] = e
        out.append(dict(janela=nome, acoes=sum(acs.values()), top_acoes=acs.most_common(6),
                        intencoes=len(it), frac_net_pos=round(pos / max(len(it), 1), 3),
                        p_sinal=round(sinal(pos, len(it)), 4),
                        net_mediano=round(float(np.median([e["net"] for e in it])), 4) if it else None,
                        net_medio=round(float(np.mean([e["net"] for e in it])), 4) if it else None,
                        frac_pagou=round(sum(1 for e in it if e.get("pagou")) / max(len(it), 1), 3),
                        consultas=len(cons), consultas_classe=dict(cc),
                        insistir_sem_mudanca=f"{ins}/{base}",
                        fechou_perseguida=sum(1 for e in E if e["ev"] == "fechou" and e["como"] == "perseguida"),
                        fechou_sozinha=sum(1 for e in E if e["ev"] == "fechou" and e["como"] == "sozinha"),
                        abandonou=sum(1 for e in E if e["ev"] == "abandonou"),
                        barras_descobertas=sum(1 for e in E if e["ev"] == "barra_descoberta"),
                        subiu_nivel2=sum(1 for e in E if e["ev"] == "nascimento" and e.get("nivel", 1) >= 2),
                        aterramentos=sum(1 for e in E if e["ev"] == "aterramento"),
                        ligacoes_av=sum(1 for e in E if e["ev"] == "ligacao_av"),
                        convergencias=sum(1 for e in E if e["ev"] == "convergencia")))
    return out


def latentes(fotos):
    """Nasceu na 1a hora, ficou parada de 1 h a 4 h, voltou depois das 4 h."""
    if not all(k in fotos for k in (3600, 7200, 14400, "final")):
        return None
    m1, m2, m4, mf = fotos[3600], fotos[7200], fotos[14400], fotos["final"]
    vol = []
    for p, s in m1.E.items():
        s2, s4, sf = m2.E.get(p), m4.E.get(p), mf.E.get(p)
        if s2 and s4 and sf and s["n"] == s2["n"] == s4["n"] and sf["n"] > s4["n"]:
            vol.append(p)
    alvos = {g["alvo"] for g in mf.lac.values()} | {f["alvo"].encode() for f in mf.fechadas}
    return dict(nasceu_1h_parada_1a4h_voltou=len(vol),
                viraram_lacuna=sum(1 for p in vol if p in alvos),
                exemplos=[limpa(p) for p in sorted(vol, key=lambda p: -len(p))[:15]])


if __name__ == "__main__":
    vida = sys.argv[1]
    evs0 = [json.loads(l) for l in open(os.path.join(vida, "ontogenia.jsonl"), encoding="utf-8")]
    v2 = any(e["ev"] == "versao" and e.get("versao") == "v2" for e in evs0)
    Classe = Mente2 if v2 else Mente
    bateria = pickle.load(open("sondas/bateria.pkl", "rb"))
    rg = regua(bateria)
    fotos = {}
    vazia = Mente(0, tempfile.mkdtemp(), registrar=False)
    tabela = {"t=0 (vazia)": mede_foto(vazia, bateria, rg)}
    for d in sorted(glob.glob(os.path.join(vida, "foto_*")), key=lambda x: int(x.split("_")[-1])):
        mc = int(d.split("_")[-1])
        fotos[mc] = Classe.carrega(d, 0, log=False)
        tabela[f"{mc // 60} min"] = mede_foto(fotos[mc], bateria, rg)
        print("medida", d, flush=True)
    if os.path.exists(os.path.join(vida, "estado.pkl")):
        fotos["final"] = Classe.carrega(vida, 0, log=False)
        tabela["final"] = mede_foto(fotos["final"], bateria, rg)
    evs = [json.loads(l) for l in open(os.path.join(vida, "ontogenia.jsonl"), encoding="utf-8")]
    out = dict(fotos=tabela, janelas=janelas(evs, rg), latentes=latentes(fotos),
               vida_toda=janelas(evs, rg, ((0, 10 ** 12),), ("vida toda",)))
    json.dump(out, open(os.path.join(vida, "marcos.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    for k, v in tabela.items():
        print(f"\n== {k}")
        for kk, vv in v.items():
            print(f"   {kk}: {vv}")
    print("\n== janelas de vida")
    for j in out["janelas"]:
        print("  ", j)
    print("\n== vida toda:", out["vida_toda"])
    print("\n== latentes:", out["latentes"])
