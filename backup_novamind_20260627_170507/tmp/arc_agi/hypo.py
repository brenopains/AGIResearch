"""
hypo.py — HIPOTESES estilo HUMANO (o raciocinio que o usuario fez de olho, codado).
Em vez de colapsar pro fundo (NCA), GERA e TESTA hipoteses estruturadas vindas da ESTRUTURA dos demos:
  - mark_lines: cada linha/coluna -> cor por um PREDICADO (nº de cores distintas) — "preencher baseado no padrao"
  - periodic_seed_lines: 2 seeds -> linhas cheias ALTERNADAS, tiladas com o espacamento entre eles — "alternar
    cores com espacamento de coluna, preencher como o exemplo" (raciocinio direto do usuario)
Cada hipotese e VERIFICADA nos demos (gate). Sao primitivas que humanos sintetizam e o solver nao tinha.
"""
import tem_arc2 as T

def _lines(g, axis):
    if axis == "row": return [list(r) for r in g]
    return [[g[i][j] for i in range(len(g))] for j in range(len(g[0]))]
def _unlines(ls, axis, H, W):
    if axis == "row": return [list(r) for r in ls]
    out = [[0] * W for _ in range(H)]
    for j, col in enumerate(ls):
        for i in range(H): out[i][j] = col[i]
    return out

def t_mark_lines(axis):
    def induce(task):
        mp = {}                                   # nº de cores distintas na linha -> cor de saida (monocromatica)
        for p in task["train"]:
            gi, go = p["input"], p["output"]
            if T.dims(gi) != T.dims(go): return None
            for a, b in zip(_lines(gi, axis), _lines(go, axis)):
                if len(set(b)) != 1: return None  # linha de saida tem de ser monocromatica
                oc = b[0]; key = len(set(a))
                if key in mp and mp[key] != oc: return None
                mp[key] = oc
        if not mp: return None
        def fn(g):
            H, W = T.dims(g); out = []
            for a in _lines(g, axis):
                out.append([mp.get(len(set(a)), 0)] * len(a))
            return _unlines(out, axis, H, W)
        return fn
    return induce

def _build_periodic(g):
    H, W = T.dims(g); bg = T.bg_color(g)
    cells = [(i, j, g[i][j]) for i in range(H) for j in range(W) if g[i][j] != bg]
    if len(cells) != 2: return None
    (r1, c1, a), (r2, c2, b) = cells
    dr, dc = abs(r1 - r2), abs(c1 - c2)
    out = [[bg] * W for _ in range(H)]
    if dc > 0 and dc <= (dr if dr > 0 else 10 ** 9):          # tila ao longo das COLUNAS -> linhas verticais
        per = dc; first = a if c1 < c2 else b; second = b if c1 < c2 else a
        idx = 0; jj = min(c1, c2)
        while jj < W:
            col = first if idx % 2 == 0 else second
            for i in range(H): out[i][jj] = col
            jj += per; idx += 1
        return out
    if dr > 0:                                                # tila ao longo das LINHAS -> linhas horizontais
        per = dr; first = a if r1 < r2 else b; second = b if r1 < r2 else a
        idx = 0; ii = min(r1, r2)
        while ii < H:
            col = first if idx % 2 == 0 else second
            for j in range(W): out[ii][j] = col
            ii += per; idx += 1
        return out
    return None
def t_periodic_seed_lines(task):
    for p in task["train"]:
        b = _build_periodic(p["input"])
        if b != p["output"]: return None
    return lambda g: (_build_periodic(g) or g)

HYPO = [("mark_lines_row", t_mark_lines("row")), ("mark_lines_col", t_mark_lines("col")),
        ("periodic_seed_lines", t_periodic_seed_lines)]

if __name__ == "__main__":
    import json
    from solver import ARC_DIR
    for tid, name in [("25d8a9c8", "mark_lines"), ("0a938d79", "periodic_seed_lines")]:
        t = json.load(open(ARC_DIR / (tid + ".json"))); t["_id"] = tid
        for nm, ind in HYPO:
            fn = ind(t)
            if fn and T.vtrain(fn, t) and T.vtest(fn, t):
                print(f"{tid}: RESOLVIDO por hipotese '{nm}'"); break
        else:
            print(f"{tid}: nenhuma hipotese fechou")
