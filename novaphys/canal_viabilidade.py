"""novaphys.canal_viabilidade — a SEGUNDA modalidade.

Expressao diz "o farmaco D age pelo gene G". Viabilidade e um canal totalmente
diferente — viver ou morrer — e faz a previsao independente:
  linhagem que DEPENDE de G (CRISPR gene effect negativo) tem que ser mais
  SENSIVEL a D (log-fold-change do PRISM negativo).
Se aparecer nos dois, nao e artefato de compressao de um canal so.
"""
import re
import numpy as np, pandas as pd

DEP = "D:/dados_cancer/depmap/24Q2/CRISPRGeneEffect.csv"
PRIM_L = "D:/dados_cancer/prism/prism-repurposing-20q2-primary-screen-replicate-collapsed-logfold-change.csv"
PRIM_T = "D:/dados_cancer/prism/prism-repurposing-20q2-primary-screen-replicate-collapsed-treatment-info.csv"
PRIM_C = "D:/dados_cancer/prism/prism-repurposing-20q2-primary-screen-cell-line-info.csv"

def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def carrega(genes=None, verbose=True):
    d = pd.read_csv(DEP, index_col=0)
    d.columns = [c.split(" (")[0] for c in d.columns]
    if genes:
        d = d[[g for g in genes if g in d.columns]]
    L = pd.read_csv(PRIM_L, index_col=0)
    T = pd.read_csv(PRIM_T)
    C = pd.read_csv(PRIM_C)
    col = [c for c in C.columns if "depmap" in c.lower() or c.lower() == "row_name"]
    if verbose:
        print(f"  CRISPR  {d.shape[0]} linhagens x {d.shape[1]} genes")
        print(f"  PRISM   {L.shape[0]} linhagens x {L.shape[1]} tratamentos")
    return d, L, T, C

def perfil_farmaco(L, T, nome):
    """log-fold-change por linhagem para um farmaco (media sobre doses)."""
    alvo = norm(nome)
    ids = T[T.name.map(norm) == alvo].column_name
    cols = [c for c in ids if c in L.columns]
    if not cols:
        return None
    return L[cols].mean(axis=1)

def testa_par(d, L, T, farmaco, gene, min_n=25):
    """Correlacao entre dependencia do gene e sensibilidade ao farmaco.
    Negativa esperada? NAO: dependencia mais negativa = mais essencial;
    lfc mais negativo = mais morto. Logo a previsao e correlacao POSITIVA."""
    if gene not in d.columns:
        return None
    p = perfil_farmaco(L, T, farmaco)
    if p is None:
        return None
    j = pd.concat([d[gene].rename("dep"), p.rename("lfc")], axis=1).dropna()
    if len(j) < min_n:
        return None
    r = float(np.corrcoef(j.dep, j.lfc)[0, 1])
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(len(j) - 3)
    return {"farmaco": farmaco, "gene": gene, "n": len(j), "r": r, "z": float(z)}

CONTROLES = [("Encorafenib", "BRAF"), ("Dabrafenib", "BRAF"), ("Vemurafenib", "BRAF"),
             ("Bortezomib", "PSMB5"), ("Selumetinib", "MAP2K1"), ("Trametinib", "MAP2K1"),
             ("Alpelisib", "PIK3CA"), ("Erlotinib", "EGFR"), ("Gefitinib", "EGFR"),
             ("Olaparib", "PARP1"), ("Palbociclib", "CDK4"), ("Venetoclax", "BCL2")]

if __name__ == "__main__":
    genes = sorted({g for _, g in CONTROLES})
    d, L, T, C = carrega(genes)
    print("\nCONTROLE: alvo ANOTADO tem que aparecer no canal de viabilidade")
    print("  previsao: r POSITIVO (gene mais essencial -> farmaco mata mais)\n")
    print(f"  {'farmaco':<14}{'gene':<9}{'n':>5}{'r':>8}{'z':>8}   veredito")
    ok = tot = 0
    for f, g in CONTROLES:
        res = testa_par(d, L, T, f, g)
        if res is None:
            print(f"  {f:<14}{g:<9}{'-':>5}{'-':>8}{'-':>8}   nao testavel")
            continue
        tot += 1
        v = "PEGOU" if res["z"] > 2 else ("sinal fraco" if res["z"] > 0 else "nao")
        ok += res["z"] > 2
        print(f"  {f:<14}{g:<9}{res['n']:5d}{res['r']:8.3f}{res['z']:8.2f}   {v}")
    print(f"\n  controles que o canal pega: {ok}/{tot}")
