"""So o braco E. Dois Pools no mesmo processo quebram no Windows
(PermissionError WinError 5 ao duplicar o handle do pipe), entao cada braco
roda no seu proprio processo. B ja foi medido: 8/400, 5 de 2 passos."""
import time
from multiprocessing import Pool
from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL
from novaphys.pilha import uma

if __name__ == "__main__":
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(10) as p:
        res = p.map(uma, [(n, tr, te, True, 0.85, 40) for n, tr, te in tarefas],
                    chunksize=2)
    ok = {n for n, r, _t in res if r}
    ex = {n: t for n, r, t in res if r}
    p2 = sum(1 for _n, r, _t in res if r == "2 passos")
    print(f"E vocabulario cheio, 2 passos   {len(ok)}/400 "
          f"({len(ok)/400:5.2%})  {p2} de 2 passos  ({time.time()-t0:.0f}s)")
    print(f"\nE - B = {len(ok) - 8:+d}   (o mesmo ganho num passo foi +3)")
    print("\nleis do braco E:")
    for n in sorted(ok):
        print(f"  {n}  {ex[n]}")
