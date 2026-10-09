"""G1 — validacao de instrumento: o duto reencontra o alvo ANOTADO sozinho?
Camada 1 DESLIGADA (nao recebe o alvo). Se nao achar o que ja se sabe, nao tem
como achar o que nao se sabe."""
import ast, json, sys
import numpy as np, pandas as pd

sys.path.insert(0, ('C:/Users/usuario/Documents/AGI 2.0').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
from novaphys.residuo_farmaco import z, camada0, _proj_out, varre

B = "D:/dados_farmaco"
K = 24

def main():
    genes = json.load(open(f"{B}/replogle/feature_names.json"))
    lin = pd.read_parquet(f"{B}/kd_linhas.parquet")
    KD = np.load(f"{B}/kd.npy")
    # dicionario: media sobre as 4 linhagens -> um vetor por gene
    gs = sorted(set(lin.gene))
    gi = {g: i for i, g in enumerate(gs)}
    S = np.zeros((len(gs), len(genes)), np.float32)
    C = np.zeros(len(gs), np.int32)
    Z = z(KD)
    for r, g in enumerate(lin.gene):
        S[gi[g]] += Z[r]; C[gi[g]] += 1
    S = z(S / C[:, None])
    print(f"dicionario {S.shape}", flush=True)

    D = np.load(f"{B}/tahoe_fc.npy")
    rot = pd.read_parquet(f"{B}/tahoe_fc_linhas.parquet")
    rot["droga"] = rot.treatment.apply(lambda t: ast.literal_eval(t)[0][0])
    rot["dose"] = rot.treatment.apply(lambda t: float(ast.literal_eval(t)[0][1]))
    print(f"farmaco {D.shape}", flush=True)

    alvo = pd.read_parquet(f"{B}/farmacos_testaveis.parquet")
    mapa = {r.drug: [g for g in r.no_dic if g in gi] for _, r in alvo.iterrows()}
    mapa = {k: v for k, v in mapa.items() if v}
    print(f"farmacos com alvo no dicionario: {len(mapa)}", flush=True)

    # camada 0 estimada em amostra INDEPENDENTE das condicoes pontuadas
    rng = np.random.default_rng(0)
    alvo_rows = np.where(rot.droga.isin(mapa))[0]
    outras = np.setdiff1d(np.arange(len(D)), alvo_rows)
    amostra = rng.choice(outras, size=min(12000, len(outras)), replace=False)
    Dz = z(D[amostra])
    base = camada0(Dz, k=K)
    del Dz
    print(f"camada0 k={K} estimada em {len(amostra)} condicoes independentes", flush=True)

    Sf = z(_proj_out(S, base))
    R = z(_proj_out(z(D[alvo_rows]), base))
    Bt = varre(R, Sf, testaveis=S.shape[0])
    print(f"varredura {Bt.shape}", flush=True)

    sub = rot.iloc[alvo_rows].reset_index(drop=True)
    ordem = np.argsort(-Bt, axis=1)
    res = []
    for i in range(len(sub)):
        alvos = [gi[g] for g in mapa[sub.droga[i]]]
        pos = min(int(np.where(ordem[i] == a)[0][0]) for a in alvos)
        res.append({"droga": sub.droga[i], "dose": sub.dose[i],
                    "linhagem": sub.cell_line[i], "posto": pos,
                    "bits": float(Bt[i].max())})
    R2 = pd.DataFrame(res)
    R2.to_parquet(f"{B}/g1_postos.parquet")
    n = len(R2)
    print("\n=== G1: posto do ALVO ANOTADO entre 2.023 candidatos ===")
    print(f"  condicoes: {n}   acaso top1 = {1/len(gs)*100:.2f}%")
    for t, r in (("top1", 1), ("top10", 10), ("top50", 50), ("top200", 200)):
        print(f"  {t:>6}: {(R2.posto < r).mean()*100:5.1f}%")
    print(f"  posto mediano: {R2.posto.median():.0f} de {len(gs)}")
    print("\n  por dose:")
    print(R2.groupby("dose").posto.agg(["count", "median",
          lambda s: (s < 10).mean()]).rename(columns={"<lambda_0>": "top10"}).to_string())
    print("\n  melhores farmacos (menor posto mediano):")
    m = R2.groupby("droga").posto.median().nsmallest(15)
    for d, p in m.items():
        print(f"    {d:<28} posto {p:6.0f}   alvo {mapa[d]}")

if __name__ == "__main__":
    main()
