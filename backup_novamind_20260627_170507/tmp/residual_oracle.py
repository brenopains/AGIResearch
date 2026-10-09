"""
residual_oracle.py — PASSO 0 do conselho: MEDIR antes de construir, com CRITERIO DE MORTE.
Classifica as ~299 tarefas ARC nao-resolvidas (400 - teto 101) por FAMILIA de transformacao.
Decisao: se replicacao>=20 OU flood-fill>=20 -> ha alvo p/ os passos 1-A/1-B; senao -> saturacao em ~25%.
Honesto: os detectores sao HEURISTICOS e aproximados; servem p/ um histograma de onde mora o residuo, nao p/ resolver.
"""
import os, json, glob, numpy as np
from collections import Counter
try:
    from scipy import ndimage
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "arc_data", "data", "training")

def ceiling():
    s = set()
    for f in ["arc_unified_final.json", "h111_solved.json"]:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            try: s |= set(json.load(open(p)))
            except Exception: pass
    return s

def bg(g):
    v, c = np.unique(g, return_counts=True); return int(v[np.argmax(c)])

def is_global_recolor(train):
    mp = {}
    for i, o in train:
        if i.shape != o.shape: return False
        for a, b in zip(i.ravel(), o.ravel()):
            a, b = int(a), int(b)
            if a in mp and mp[a] != b: return False
            mp[a] = b
    return any(k != v for k, v in mp.items())

def is_fill_enclosed(train):
    if not HAS_SCIPY: return False
    for i, o in train:
        if i.shape != o.shape: return False
        b = bg(i); diff = (i != o)
        if not diff.any(): return False
        # celulas que mudaram eram fundo E estao ENCERRADAS (componente de fundo que nao toca a borda)
        bgmask = (i == b)
        lbl, n = ndimage.label(bgmask)
        border = set(np.unique(np.concatenate([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])))
        enclosed = np.isin(lbl, [k for k in range(1, n + 1) if k not in border])
        # a mudanca tem que estar (majoritariamente) nas celulas encerradas
        if diff.sum() == 0 or (diff & enclosed).sum() < 0.6 * diff.sum(): return False
        if (diff & ~bgmask).sum() > 0.4 * diff.sum(): return False
    return True

def _sym_variants(g):
    return [g, g[:, ::-1], g[::-1, :], g[::-1, ::-1], g.T, g.T[:, ::-1]] if g.shape[0] == g.shape[1] else [g, g[:, ::-1], g[::-1, :], g[::-1, ::-1]]

def is_symmetry(train):
    for i, o in train:
        if i.shape != o.shape: return False
        # out e mais simetrico que in? (out igual a alguma simetrizacao por OR de in com seu espelho onde in era fundo)
        b = bg(i); ok = False
        for axis in [(slice(None), slice(None, None, -1)), (slice(None, None, -1), slice(None))]:
            mir = i[axis]
            comp = np.where(i == b, mir, i)
            if np.array_equal(comp, o): ok = True; break
        if not ok: return False
    return True

def is_move(train):
    for i, o in train:
        if i.shape != o.shape: return False
        if Counter(i.ravel().tolist()) != Counter(o.ravel().tolist()): return False  # mesmas cores/contagens
        if np.array_equal(i, o): return False
    return True

def is_connectivity(train):
    """mudanca cuja COR nova correlaciona com o tamanho/rank do componente conexo (relacional-global)."""
    if not HAS_SCIPY: return False
    for i, o in train:
        if i.shape != o.shape: return False
        b = bg(i); diff = (i != o)
        if not diff.any(): return False
        fg = (i != b); lbl, n = ndimage.label(fg)
        if n < 2: return False
        # objetos inteiros mudam para uma cor (nao celulas soltas)?
        whole = True
        for k in range(1, n + 1):
            cells = (lbl == k)
            outcols = set(o[cells].tolist())
            if len(outcols) != 1: whole = False; break
        if not whole: return False
    return True

def classify(train):
    ish = [tuple(i.shape) for i, o in train]; osh = [tuple(o.shape) for i, o in train]
    same = all(a == b for a, b in zip(ish, osh))
    if not same:
        facs = set()
        for i, o in train:
            ih, iw = i.shape; oh, ow = o.shape
            if ih and iw: facs.add((round(oh / ih, 3), round(ow / iw, 3)))
        if len(facs) == 1:
            rh, rw = facs.pop()
            if rh > 1 and rw > 1 and abs(rh - round(rh)) < 1e-6 and abs(rw - round(rw)) < 1e-6:
                return "replicacao/escala"
            if rh < 1 and rw < 1: return "crop/downscale"
            return "geometria-outra"
        return "shape-variavel"
    if is_global_recolor(train): return "recolor-global"
    if is_fill_enclosed(train): return "flood-fill/conectividade"
    if is_symmetry(train): return "simetria-global"
    if is_connectivity(train): return "conectividade-relacional"
    if is_move(train): return "objeto-movel"
    return "recolor-local/outro"

def main():
    ceil = ceiling()
    files = sorted(glob.glob(os.path.join(DATA, "*.json")))
    hist = Counter(); residual = []; examples = {}
    for fp in files:
        name = os.path.basename(fp)
        if name in ceil: continue
        t = json.load(open(fp))
        train = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        fam = classify(train)
        hist[fam] += 1; residual.append((name, fam))
        examples.setdefault(fam, []).append(name)
    total = len(residual)
    print("=" * 74)
    print(f"ORACULO DE RESIDUO | teto={len(ceil)}/400 ({100*len(ceil)/400:.1f}%) | residuo={total} tarefas nao-resolvidas")
    print(f"scipy={'on' if HAS_SCIPY else 'OFF (flood-fill/conectividade limitados)'}")
    print("-" * 74)
    for fam, n in hist.most_common():
        print(f"  {fam:28s} {n:3d}  ({100*n/total:.0f}%)   ex: {examples[fam][:3]}")
    print("-" * 74)
    repl = hist["replicacao/escala"] + hist["geometria-outra"] + hist["crop/downscale"]
    ff = hist["flood-fill/conectividade"] + hist["conectividade-relacional"]
    print(f"GEOMETRIA-DE-SAIDA (alvo do PASSO 1-A): {repl}")
    print(f"CONECTIVIDADE/FLOOD-FILL (alvo do PASSO 1-B): {ff}")
    alvo = (hist["replicacao/escala"] >= 20) or (ff >= 20)
    print("=" * 74)
    if alvo:
        print(f"VEREDITO: HA ALVO -> prosseguir p/ PASSO 1-A e 1-B.")
    else:
        print(f"VEREDITO (criterio de morte): replicacao={hist['replicacao/escala']}<20 e flood-fill={ff}<20")
        print(f"  -> caminho sem-injecao-de-familia pode estar saturando; 1-A/1-B sao apostas menores.")
    json.dump({fam: examples[fam] for fam in examples}, open(os.path.join(HERE, "residual_families.json"), "w"))
    print("  (famílias por tarefa salvas em residual_families.json)")

if __name__ == "__main__":
    main()
