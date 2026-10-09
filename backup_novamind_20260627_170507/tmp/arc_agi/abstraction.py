"""
abstraction.py — O PASSO DE ABSTRACAO ("aha"): a maquina constroi um MODELO DE MUNDO da tarefa.
Em vez de testar primitivas fixas, ela:
  1) SEGMENTA objetos na entrada e na saida (conhecimento-nucleo: objetidade);
  2) CORRELACIONA cada objeto de entrada com o seu correspondente na saida (por sobreposicao de celulas);
  3) INDUZ a regra de SEGUNDA ORDEM: o que mudou (cor/posicao/existencia) e FUNCAO de QUAL propriedade?
     - busca a CHAVE (cor, tamanho, rank de tamanho, nº de buracos, rank de posicao) e APRENDE o mapa chave->efeito;
     - se o mesmo mapa explica TODOS os objetos de TODOS os demos -> achou a LOGICA (o "modelo de mundo" da tarefa).
  4) APLICA o mapa aprendido na entrada de teste.
Isso e correlacao/logica emergente dos dados (nao primitiva minha): o alfabeto e "objeto+propriedade+efeito";
a FRASE (qual propriedade governa qual efeito, e o mapa) e descoberta pelos dados. Honesto: ainda e um fecho
(o conjunto de chaves/efeitos), porem expressa regras condicionais-por-propriedade que os templates fixos nao pegam.
"""
import tem_arc2 as T
from collections import Counter

def _objs(g):
    bg = T.bg_color(g); os = T.components(g, bg); H, W = T.dims(g)
    for o in os:
        o["holes"] = T.holes(g, o, bg)
        o["cellset"] = set((y, x) for y, x in o["cells"])
    return bg, os, H, W

def _ranks(objs, keyfn, reverse=True):
    """rank (1=maior) de cada objeto por uma propriedade, dentro da grade."""
    vals = sorted(set(keyfn(o) for o in objs), reverse=reverse)
    return {v: i + 1 for i, v in enumerate(vals)}

# chaves candidatas: a propriedade do objeto de ENTRADA que pode governar o efeito
def _key_funcs(objs):
    szrank = _ranks(objs, lambda o: o["size"], True)
    horank = _ranks(objs, lambda o: o["holes"], True)
    toprank = _ranks(objs, lambda o: o["top"], False)
    leftrank = _ranks(objs, lambda o: o["left"], False)
    cc = Counter(o["color"] for o in objs)
    return {
        "cor":        lambda o: o["color"],
        "tamanho":    lambda o: o["size"],
        "rank_tam":   lambda o, r=szrank: r[o["size"]],
        "n_buracos":  lambda o: o["holes"],
        "rank_buraco":lambda o, r=horank: r[o["holes"]],
        "rank_topo":  lambda o, r=toprank: r[o["top"]],
        "rank_esq":   lambda o, r=leftrank: r[o["left"]],
        "freq_cor":   lambda o, cc=cc: cc[o["color"]],
    }

def _out_color_of(o, go):
    """cor (unica) que as celulas do objeto assumem na saida; None se nao for unica."""
    cols = set(go[y][x] for (y, x) in o["cells"])
    return next(iter(cols)) if len(cols) == 1 else None

# ---------- INDUTOR 1: RECOLORIR por propriedade (mapa chave->cor) ----------
def induce_recolor(task):
    for p in task["train"]:
        if T.dims(p["input"]) != T.dims(p["output"]): return None
    keynames = None
    learned = {}     # keyname -> {keyval: out_color}
    for p in task["train"]:
        gi, go = p["input"], p["output"]
        bg, objs, H, W = _objs(gi)
        if not objs: return None
        kf = _key_funcs(objs)
        if keynames is None: keynames = list(kf.keys()); learned = {k: {} for k in keynames}
        for o in objs:
            oc = _out_color_of(o, go)
            if oc is None: return None
            for kn in keynames:
                kv = kf[kn](o); m = learned[kn]
                if kv in m and m[kv] != oc: m["__bad__"] = True
                else: m[kv] = oc
    # escolhe a CHAVE mais simples que e consistente e realmente muda algo
    for kn in keynames:
        m = learned[kn]
        if m.get("__bad__"): continue
        if all(c is not None for c in m.values()) and len(m) >= 1:
            mp = {k: v for k, v in m.items() if k != "__bad__"}
            def fn(g, kn=kn, mp=mp):
                bg, objs, H, W = _objs(g); out = T.gcopy(g); kf = _key_funcs(objs)
                for o in objs:
                    kv = kf[kn](o)
                    if kv in mp:
                        for y, x in o["cells"]: out[y][x] = mp[kv]
                return out
            return ("recolor_por[" + kn + "]", fn)
    return None

# ---------- INDUTOR 2: MANTER/DELETAR por propriedade (predicado aprendido) ----------
def induce_select(task):
    for p in task["train"]:
        if T.dims(p["input"]) != T.dims(p["output"]): return None
    keynames = None; survive = {}    # keyname -> {keyval: bool(sobrevive?)}
    for p in task["train"]:
        gi, go = p["input"], p["output"]
        bg, objs, H, W = _objs(gi)
        if not objs: return None
        kf = _key_funcs(objs)
        if keynames is None: keynames = list(kf.keys()); survive = {k: {} for k in keynames}
        for o in objs:
            # sobreviveu se as celulas continuam com a propria cor na saida; sumiu se viraram fundo
            outcols = set(go[y][x] for (y, x) in o["cells"])
            kept = (outcols == {o["color"]}); gone = (outcols == {bg})
            if not (kept or gone): return None
            for kn in keynames:
                kv = kf[kn](o); m = survive[kn]
                if kv in m and m[kv] != kept: m["__bad__"] = True
                else: m[kv] = kept
    for kn in keynames:
        m = survive[kn]
        if m.get("__bad__"): continue
        vals = {k: v for k, v in m.items() if k != "__bad__"}
        if len(set(vals.values())) < 2: continue   # precisa discriminar (uns ficam, outros somem)
        def fn(g, kn=kn, vals=vals):
            bg, objs, H, W = _objs(g); out = [[bg] * W for _ in range(H)]; kf = _key_funcs(objs)
            for o in objs:
                kv = kf[kn](o)
                if vals.get(kv, False):
                    for y, x in o["cells"]: out[y][x] = o["color"]
            return out
        return ("manter_por[" + kn + "]", fn)
    return None

# ---------- INDUTOR 3: MOVER por propriedade (vetor aprendido por chave) ----------
def induce_move(task):
    for p in task["train"]:
        if T.dims(p["input"]) != T.dims(p["output"]): return None
    keynames = None; vecs = {}
    for p in task["train"]:
        gi, go = p["input"], p["output"]
        bg, oi, H, W = _objs(gi); bg2, oo, _, _ = _objs(go)
        if not oi: return None
        kf = _key_funcs(oi)
        if keynames is None: keynames = list(kf.keys()); vecs = {k: {} for k in keynames}
        for o in oi:
            cand = [b for b in oo if T.shape_sig(b) == T.shape_sig(o) and b["color"] == o["color"]]
            if not cand: return None
            v = (cand[0]["top"] - o["top"], cand[0]["left"] - o["left"])
            for kn in keynames:
                kv = kf[kn](o); m = vecs[kn]
                if kv in m and m[kv] != v: m["__bad__"] = True
                else: m[kv] = v
    for kn in keynames:
        m = vecs[kn]
        if m.get("__bad__"): continue
        mp = {k: v for k, v in m.items() if k != "__bad__"}
        if not mp or all(v == (0, 0) for v in mp.values()): continue
        def fn(g, kn=kn, mp=mp):
            bg, objs, H, W = _objs(g); out = T.gcopy(g); kf = _key_funcs(objs)
            for o in objs:
                for y, x in o["cells"]: out[y][x] = bg
            for o in objs:
                kv = kf[kn](o); v = mp.get(kv, (0, 0))
                for y, x in o["cells"]:
                    ny, nx = y + v[0], x + v[1]
                    if 0 <= ny < H and 0 <= nx < W: out[ny][nx] = o["color"]
            return out
        return ("mover_por[" + kn + "]", fn)
    return None

INDUCERS = [induce_recolor, induce_select, induce_move]

def abstraction_candidates(task):
    """retorna [(nome, fn)] das regras de 2a ordem que a abstracao induziu (cada uma = um modelo de mundo da tarefa)."""
    out = []
    for ind in INDUCERS:
        try:
            r = ind(task)
        except Exception:
            r = None
        if r: out.append(r)
    return out

if __name__ == "__main__":
    import json, glob, pathlib, time
    from solver import ARC_DIR
    base = set()
    bp = pathlib.Path(__file__).parent / "unified_solved.json"
    if bp.exists(): base = set(json.load(open(bp)))
    ids = sorted(p.name[:-5] for p in ARC_DIR.glob("*.json"))
    sc = 0; got = []; novas = []; t0 = time.time()
    for i, tid in enumerate(ids):
        t = json.load(open(ARC_DIR / (tid + ".json"))); t.setdefault("_id", tid)
        cands = abstraction_candidates(t)
        ok = False
        for nm, fn in cands:
            if T.vtrain(fn, t) and T.vtest(fn, t): ok = True; break
        if ok:
            sc += 1; got.append(tid)
            if tid not in base: novas.append(tid)
        if (i + 1) % 100 == 0: print(f"  {i+1}/400  abstrai {sc}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nABSTRACAO: {sc}/400 resolvidas | NOVAS (que o simbolico nao pegava): {len(novas)} {novas}", flush=True)
