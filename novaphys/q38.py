"""So o braco que faltou: 38 primitivas COM quociente, paralelizado."""
import time
from multiprocessing import Pool
from novaphys.fisica import PASTA_AVAL, carregar, extrair
from novaphys.grupo import busca

DOBRA = None

def uma(args):
    nome, tr, te, col = args
    r, c, ap, _f = busca(tr, te, DOBRA, 14424, 3, col)
    return nome, r, c, ap

def ini(semente):
    # As primitivas extraidas sao CLOSURES e o `spawn` do Windows nao
    # consegue serializa-las. Como `extrair` e deterministico pela semente,
    # cada processo extrai o seu proprio conjunto — identico ao dos outros.
    global DOBRA
    DOBRA = extrair(True, 40, semente)

if __name__ == "__main__":
    dobra = extrair(True, 40, 20260905)
    tarefas = carregar(PASTA_AVAL)
    print(f"{len(dobra)} primitivas extraidas; 38 no total, profundidade 3")
    for col, rot in ((True, "38 prim QUOCIENTE"), (False, "38 prim CRU")):
        t0 = time.time()
        with Pool(9, initializer=ini, initargs=(20260905,)) as p:
            res = p.map(uma, [(n, tr, te, col) for n, tr, te in tarefas],
                        chunksize=2)
        ok = {n for n, r, _c, _a in res if r}
        tc = sum(c for _n, _r, c, _a in res)
        ta = sum(a for _n, _r, _c, a in res)
        n = len(tarefas)
        print(f"  {rot:20s} {len(ok):3d}/{n} ({len(ok)/n:5.2%})  "
              f"{tc//n:>7d} cand/tarefa  {ta//n:>8d} aplic/tarefa  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if col:
            quo = ok, tc, ta
        else:
            cru = ok, tc, ta
    print(f"\nfator de colapso: {cru[1]/max(quo[1],1):.2f}x candidatos, "
          f"{cru[2]/max(quo[2],1):.2f}x aplicacoes")
    print(f"tarefas: CRU {len(cru[0])} -> QUOCIENTE {len(quo[0])}")
    print(f"so o quociente resolve: {sorted(quo[0]-cru[0])}")
