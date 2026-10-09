src = open("mundo_fisico.py", encoding="utf-8").read()
a = '''def grandeza_q(molde, fam, p, S0):
    x, y, vx, vy = S0.T'''
b = '''QUANTIDADES = ("rmin", "rmax", "vmin", "vmax", "varre", "r_final", "x_final", "t_peri")
HORIZONTES = (5, 15, 40)


def moldes_gramatica():
    """Moldes LACRAVEIS: 'q:<grandeza>@<horizonte>'. Nenhum coincide com os
    moldes de desenvolvimento (que usam horizonte 3 ou 30)."""
    return [f"q:{q}@{h}" for q in QUANTIDADES for h in HORIZONTES]


def _grandeza_gramatica(molde, fam, p, S0):
    q, h = molde[2:].split("@")
    H = float(h)
    n = int(max(41, H * 8)) + 1
    ts, F = _futuro(fam, p, S0, H, n=n)
    r = np.hypot(F[..., 0], F[..., 1])
    v = np.hypot(F[..., 2], F[..., 3])
    r = np.where(np.isfinite(r), np.minimum(r, 1e3), 1e3)
    v = np.where(np.isfinite(v), np.minimum(v, 1e3), 1e3)
    if q == "rmin":
        return r.min(1)
    if q == "rmax":
        return r.max(1)
    if q == "vmin":
        return v.min(1)
    if q == "vmax":
        return v.max(1)
    if q == "varre":
        ang = np.unwrap(np.arctan2(F[..., 1], F[..., 0]), axis=1)
        return np.abs(ang[:, -1] - ang[:, 0])
    if q == "r_final":
        return r[:, -1]
    if q == "x_final":
        return np.where(np.isfinite(F[:, -1, 0]), F[:, -1, 0], 0.0)
    if q == "t_peri":
        dr = np.diff(r, axis=1)
        out = np.full(len(S0), H)
        for k in range(len(S0)):
            loc = np.where((dr[k, :-1] < 0) & (dr[k, 1:] >= 0))[0]
            if len(loc):
                out[k] = ts[loc[0] + 1]
        return out
    raise ValueError(molde)


def grandeza_q(molde, fam, p, S0):
    if molde.startswith("q:"):
        return _grandeza_gramatica(molde, fam, p, S0)
    x, y, vx, vy = S0.T'''
assert src.count(a) == 1
src = src.replace(a, b)
open("mundo_fisico.py", "w", encoding="utf-8").write(src)
print("ok")
