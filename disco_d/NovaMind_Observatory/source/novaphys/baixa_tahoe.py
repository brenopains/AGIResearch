"""Stream do pdex do Tahoe com projecao de coluna -> matriz (droga x dose x linhagem) x gene,
no espaco de genes do Replogle. Nada de raw em disco."""
import json, time, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

BASE = "D:/dados_farmaco"
PATH = "datasets/tahoebio/tahoe-de-rhaister/pdex/all_plates_pdex.parquet"
COLS = ["target", "feature", "fold_change", "cell_line"]

def main():
    genes = json.load(open(f"{BASE}/replogle/feature_names.json"))
    gidx = {g: i for i, g in enumerate(genes)}
    rot = pd.read_parquet(f"{BASE}/tahoe/rotulos.parquet")
    chaves = sorted({(t, c) for t, c in zip(rot.treatment, rot.cell_line)})
    kidx = {k: i for i, k in enumerate(chaves)}
    nR, nG = len(chaves), len(genes)
    print(f"matriz alvo: {nR} condicoes x {nG} genes = {nR*nG*4/1e9:.2f} GB", flush=True)

    S = np.zeros((nR, nG), dtype=np.float32)
    C = np.zeros((nR, nG), dtype=np.int16)

    fs = HfFileSystem()
    with fs.open(PATH, "rb") as fh:
        f = pq.ParquetFile(fh)
        n_rg = f.metadata.num_rg if hasattr(f.metadata, "num_rg") else f.metadata.num_row_groups
        t0 = time.time(); vistos = 0; casados = 0; achei = 0
        for i in range(n_rg):
            d = f.read_row_group(i, columns=COLS).to_pandas()
            vistos += len(d)
            gi = d.feature.map(gidx)
            m = gi.notna()
            if m.any():
                d = d[m]; gi = gi[m].to_numpy(np.int32)
                ki = np.array([kidx.get((t, c), -1) for t, c in zip(d.target, d.cell_line)], dtype=np.int64)
                ok = ki >= 0
                v = d.fold_change.to_numpy(np.float32)
                ok &= np.isfinite(v)
                if ok.any():
                    np.add.at(S, (ki[ok], gi[ok]), v[ok])
                    np.add.at(C, (ki[ok], gi[ok]), 1)
                    casados += int(ok.sum()); achei += 1
            if i % 200 == 0 or i == n_rg - 1:
                dt = time.time() - t0
                print(f"  rg {i+1}/{n_rg}  {vistos/1e6:.0f}M lidas  {casados/1e6:.1f}M casadas  "
                      f"{dt/60:.1f} min  ETA {(n_rg-i-1)*dt/max(i+1,1)/60:.0f} min", flush=True)

    M = np.where(C > 0, S / np.maximum(C, 1), np.nan).astype(np.float32)
    np.save(f"{BASE}/tahoe_fc.npy", M)
    pd.DataFrame(chaves, columns=["treatment", "cell_line"]).to_parquet(f"{BASE}/tahoe_fc_linhas.parquet")
    json.dump(genes, open(f"{BASE}/tahoe_fc_genes.json", "w"))
    cob = float(np.isfinite(M).mean())
    print(f"PRONTO. {M.shape}, cobertura {cob*100:.1f}%, {casados/1e6:.0f}M valores", flush=True)

if __name__ == "__main__":
    main()
