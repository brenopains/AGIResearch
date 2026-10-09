import time, numpy as np, sys
import mundo_fisico as M, organismo as O, material_fisica as F
orig = O.Organismo._executa
def cron(self, ws, op, e):
    t0 = time.time(); g0 = ws.gasto
    a = orig(self, ws, op, e)
    print(f"   {op:14s} {str(e.id if e else ''):14s} custo {ws.gasto-g0:8.1f} ({time.time()-t0:5.2f}s) valido={ws.valido} gasto={ws.gasto:.0f}", flush=True)
    return a
O.Organismo._executa = cron
rng = np.random.default_rng(11)
org = O.Organismo({}, 1)
for fam, tipo in [("kepler","L"),("spin_orbita","L"),("vortice","L")]:
    P = M.Problema(0, fam, "percebido", tipo, rng, com_traj=True, n_traj=8)
    t0=time.time()
    print(fam, flush=True)
    ok, ws = org.resolve(P, 300.0)
    print(f" -> ok {ok} {time.time()-t0:.1f}s", flush=True)
