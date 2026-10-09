"""
forge.py — CRIACAO ABERTA DE PRIMITIVAS (a maquina sintetiza, nao eu injeto).
Fundamento (neurociencia/cognicao): conhecimento-nucleo (Spelke: objetidade) + ATENCAO SELETIVA (escolher por
uma propriedade) + ACAO composicional. Uma primitiva = SELETOR(predicado sobre objetos, inferido) x ACAO
(recolorir/mover/deletar/manter/preencher-bbox, com PARAMETRO ajustado dos pares de treino). O forge BUSCA o
espaco (seletor x acao) e AJUSTA os parametros aos dados; cada combinacao que casa TODO o treino e uma PRIMITIVA
NOVA que a propria maquina descobriu (eu so dei o alfabeto: os predicados-nucleo e as acoes; ela escreve a palavra).
Honesto: ainda e um fecho (a gramatica seletor x acao), mas o espaco e enorme e a primitiva especifica e
sintetizada PELOS DADOS, nao escrita por mim. Mede quantas tarefas a sintese aberta resolve alem dos indutores.
"""
import tem_arc2 as T

def _feats(g):
    bg = T.bg_color(g); objs = T.components(g, bg); H, W = T.dims(g)
    for o in objs:
        o["holes"] = T.holes(g, o, bg)
        o["border"] = (o["top"] == 0 or o["left"] == 0 or o["bot"] == H - 1 or o["right"] == W - 1)
    return bg, objs

# ===== SELETORES-NUCLEO (predicados parametrizados; alguns gerados pelos dados) =====
def selectors(objs):
    sels = {}
    if not objs: return sels
    mx = max(o["size"] for o in objs); mn = min(o["size"] for o in objs)
    mh = max(o["holes"] for o in objs)
    sels["all"] = lambda o: True
    sels["largest"] = lambda o, mx=mx: o["size"] == mx
    sels["smallest"] = lambda o, mn=mn: o["size"] == mn
    sels["most_holes"] = lambda o, mh=mh: o["holes"] == mh and mh > 0
    sels["border"] = lambda o: o["border"]
    sels["interior"] = lambda o: not o["border"]
    # unicidade: o objeto cuja COR e unica (so um objeto com aquela cor)
    from collections import Counter
    cc = Counter(o["color"] for o in objs)
    sels["unique_color"] = lambda o, cc=cc: cc[o["color"]] == 1
    sels["common_color"] = lambda o, cc=cc: cc[o["color"]] == max(cc.values())
    for c in set(o["color"] for o in objs):     # seletor por cor (gerado pelos dados)
        sels[f"color={c}"] = (lambda o, c=c: o["color"] == c)
    return sels

# ===== ACOES (sintetizam o parametro a partir do treino; retornam fn(g) ou None) =====
def _fit_recolor(task, selname):
    """objetos do seletor viram UMA cor consistente em todo o treino."""
    target = None
    for p in task["train"]:
        gi, go = p["input"], p["output"]
        if T.dims(gi) != T.dims(go): return None
        bg, objs = _feats(gi); sels = selectors(objs); sel = sels.get(selname)
        if sel is None: return None
        sel_any = False
        for o in objs:
            if sel(o):
                sel_any = True
                cols = {go[y][x] for y, x in o["cells"]}
                if len(cols) != 1: return None
                oc = next(iter(cols))
                if target is None: target = oc
                elif target != oc: return None
        if not sel_any: return None
    if target is None: return None
    def fn(g):
        bg, objs = _feats(g); sels = selectors(objs); sel = sels.get(selname); out = T.gcopy(g)
        if sel is None: return out
        for o in objs:
            if sel(o):
                for y, x in o["cells"]: out[y][x] = target
        return out
    return fn

def _fit_delete(task, selname):
    """objetos do seletor viram fundo."""
    for p in task["train"]:
        if T.dims(p["input"]) != T.dims(p["output"]): return None
    def fn(g):
        bg, objs = _feats(g); sels = selectors(objs); sel = sels.get(selname); out = T.gcopy(g)
        if sel is None: return out
        for o in objs:
            if sel(o):
                for y, x in o["cells"]: out[y][x] = bg
        return out
    return fn

def _fit_keep(task, selname):
    """mantem so os objetos do seletor; resto vira fundo."""
    for p in task["train"]:
        if T.dims(p["input"]) != T.dims(p["output"]): return None
    def fn(g):
        bg, objs = _feats(g); sels = selectors(objs); sel = sels.get(selname); H, W = T.dims(g)
        out = [[bg] * W for _ in range(H)]
        if sel is None: return out
        for o in objs:
            if sel(o):
                for y, x in o["cells"]: out[y][x] = o["color"]
        return out
    return fn

def _fit_move(task, selname):
    """objetos do seletor transladam por um vetor consistente (fit do treino)."""
    vec = None
    for p in task["train"]:
        gi, go = p["input"], p["output"]
        if T.dims(gi) != T.dims(go): return None
        bg, oi = _feats(gi); bg2, oo = _feats(go); sels = selectors(oi); sel = sels.get(selname)
        if sel is None: return None
        sub_i = [o for o in oi if sel(o)]
        if not sub_i: return None
        for o in sub_i:                          # casa por (shape,color) com o objeto correspondente no output
            cand = [b for b in oo if T.shape_sig(b) == T.shape_sig(o) and b["color"] == o["color"]]
            if not cand: return None
            v = (cand[0]["top"] - o["top"], cand[0]["left"] - o["left"])
            if vec is None: vec = v
            elif vec != v: return None
    if not vec or vec == (0, 0): return None
    def fn(g):
        bg, objs = _feats(g); sels = selectors(objs); sel = sels.get(selname); H, W = T.dims(g); out = T.gcopy(g)
        if sel is None: return out
        for o in objs:
            if sel(o):
                for y, x in o["cells"]: out[y][x] = bg
        for o in objs:
            if sel(o):
                for y, x in o["cells"]:
                    ny, nx = y + vec[0], x + vec[1]
                    if 0 <= ny < H and 0 <= nx < W: out[ny][nx] = o["color"]
        return out
    return fn

def _fit_outline(task, selname):
    """objetos do seletor viram so o seu CONTORNO (borda do objeto); interior vira fundo."""
    for p in task["train"]:
        if T.dims(p["input"]) != T.dims(p["output"]): return None
    def fn(g):
        bg, objs = _feats(g); sels = selectors(objs); sel = sels.get(selname); out = T.gcopy(g)
        if sel is None: return out
        for o in objs:
            if not sel(o): continue
            cs = set((y, x) for y, x in o["cells"])
            for y, x in o["cells"]:
                edge = any((y+dy, x+dx) not in cs for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)))
                if not edge: out[y][x] = bg
        return out
    return fn

def _fit_fillbbox(task, selname):
    """preenche a CAIXA (bounding-box) dos objetos do seletor com a cor do objeto."""
    for p in task["train"]:
        if T.dims(p["input"]) != T.dims(p["output"]): return None
    def fn(g):
        bg, objs = _feats(g); sels = selectors(objs); sel = sels.get(selname); out = T.gcopy(g)
        if sel is None: return out
        for o in objs:
            if not sel(o): continue
            for y in range(o["top"], o["bot"] + 1):
                for x in range(o["left"], o["right"] + 1): out[y][x] = o["color"]
        return out
    return fn

def _fit_gravity(task, selname):
    """gravidade: objetos do seletor caem na direcao (down/up/left/right) inferida do treino ate a borda."""
    H0, W0 = T.dims(task["train"][0]["input"])
    for p in task["train"]:
        if T.dims(p["input"]) != T.dims(p["output"]): return None
    def apply_dir(g, sel, dy, dx):
        bg, objs = _feats(g); H, W = T.dims(g); out = [[bg] * W for _ in range(H)]
        keep = [o for o in objs if not sel(o)]
        for o in keep:
            for y, x in o["cells"]: out[y][x] = o["color"]
        movers = [o for o in objs if sel(o)]
        order = sorted(movers, key=lambda o: (-(o["bot"]) if dy > 0 else o["top"]) if dy else (-(o["right"]) if dx > 0 else o["left"]))
        for o in order:
            cells = list(o["cells"]); shift = 0
            while True:
                ok = True
                for y, x in cells:
                    ny, nx = y + dy * (shift + 1), x + dx * (shift + 1)
                    if not (0 <= ny < H and 0 <= nx < W) or out[ny][nx] != bg: ok = False; break
                if ok: shift += 1
                else: break
            for y, x in cells: out[y + dy * shift][x + dx * shift] = o["color"]
        return out
    dirs = {"down": (1, 0), "up": (-1, 0), "left": (0, -1), "right": (0, 1)}
    good = None
    for dn, (dy, dx) in dirs.items():
        ok = True
        for p in task["train"]:
            bg, objs = _feats(p["input"]); sels = selectors(objs); sel = sels.get(selname)
            if sel is None: ok = False; break
            if apply_dir(p["input"], sel, dy, dx) != p["output"]: ok = False; break
        if ok: good = (dy, dx); break
    if not good: return None
    def fn(g, v=good):
        bg, objs = _feats(g); sels = selectors(objs); sel = sels.get(selname)
        if sel is None: return T.gcopy(g)
        return apply_dir(g, sel, v[0], v[1])
    return fn

ACTIONS = {"recolor": _fit_recolor, "delete": _fit_delete, "keep": _fit_keep, "move": _fit_move,
           "outline": _fit_outline, "fillbbox": _fit_fillbbox, "gravity": _fit_gravity}

def _vtrain(fn, task):
    for p in task["train"]:
        try:
            if fn(p["input"]) != p["output"]: return False
        except Exception:
            return False
    return True

def forge_candidates(task):
    """SINTETIZA primitivas (seletor x acao) que casam TODO o treino. Cada uma e uma primitiva nova descoberta."""
    if not task.get("train"): return []
    try:
        bg, objs0 = _feats(task["train"][0]["input"])
    except Exception:
        return []
    selnames = list(selectors(objs0).keys())
    out = []
    for selname in selnames:
        for aname, fit in ACTIONS.items():
            try:
                fn = fit(task, selname)
            except Exception:
                fn = None
            if fn:                                  # SEM filtro de treino-cheio: deixa primitivas PARCIAIS
                out.append((f"forge[{aname}@{selname}]", fn))  # comporem na busca em feixe
    return out
