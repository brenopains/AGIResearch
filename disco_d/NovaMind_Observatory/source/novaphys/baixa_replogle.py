"""Stream do pdex do Replogle-Nadig -> dicionario de assinaturas de knockdown.
Guarda tres matrizes: tudo, lotes impares, lotes pares (confiabilidade meio-a-meio)."""
import json, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

BASE = "D:/dados_farmaco"
PATH = "datasets/tahoebio/replogle-nadig-de-rhaister/pdex/all_pdex.parquet"
COLS = ["target", "feature", "fold_change", "cell_line", "gem_group"]

def main():
    genes = json.load(open(f"{BASE}/replogle/feature_names.json"))
    nG = len(genes)
    d = pq.read_table(f"{BASE}/replogle/all_delta.parquet",
                      columns=["cell_line", "gene"]).to_pandas()
    chaves = sorted({(g, c) for g, c in zip(d.gene, d.cell_line)})
    kidx = {k: i for i, k in enumerate(chaves)}
    nR = len(chaves)
    print(f"dicionario: {nR} (gene x linhagem) x {nG} genes de leitura", flush=True)

    S = [np.zeros((nR, nG), np.float32) for _ in range(3)]   # 0=tudo 1=impar 2=par
    C = [np.zeros((nR, nG), np.int16) for _ in range(3)]

    fs = HfFileSystem()
    with fs.open(PATH, "rb") as fh:
        f = pq.ParquetFile(fh)
        n_rg = f.metadata.num_row_groups
        t0 = time.time(); vistos = casados = 0
        for i in range(n_rg):
            t = f.read_row_group(i, columns=COLS).to_pandas()
            vistos += len(t)
            gi = t.feature.to_numpy(np.int32)
            ki = np.array([kidx.get((a, b), -1) for a, b in zip(t.target, t.cell_line)], np.int64)
            v = t.fold_change.to_numpy(np.float32)
            ok = (ki >= 0) & (gi >= 0) & (gi < nG) & np.isfinite(v)
            if ok.any():
                par = (pd.to_numeric(t.gem_group, errors="coerce")
                       .fillna(0).to_numpy(np.int64) % 2)[ok]
                k, g, x = ki[ok], gi[ok], v[ok]
                np.add.at(S[0], (k, g), x); np.add.at(C[0], (k, g), 1)
                for lado, sel in ((1, par == 1), (2, par == 0)):
                    if sel.any():
                        np.add.at(S[lado], (k[sel], g[sel]), x[sel])
                        np.add.at(C[lado], (k[sel], g[sel]), 1)
                casados += int(ok.sum())
            if i % 200 == 0 or i == n_rg - 1:
                dt = time.time() - t0
                print(f"  rg {i+1}/{n_rg}  {vistos/1e6:.0f}M lidas  {casados/1e6:.0f}M casadas  "
                      f"{dt/60:.1f} min  ETA {(n_rg-i-1)*dt/max(i+1,1)/60:.0f} min", flush=True)

    for nome, lado in (("kd", 0), ("kd_impar", 1), ("kd_par", 2)):
        M = np.where(C[lado] > 0, S[lado] / np.maximum(C[lado], 1), np.nan).astype(np.float32)
        np.save(f"{BASE}/{nome}.npy", M)
        print(f"  {nome}: {M.shape} cobertura {np.isfinite(M).mean()*100:.1f}%", flush=True)
    pd.DataFrame(chaves, columns=["gene", "cell_line"]).to_parquet(f"{BASE}/kd_linhas.parquet")
    print("PRONTO", flush=True)

if __name__ == "__main__":
    main()
