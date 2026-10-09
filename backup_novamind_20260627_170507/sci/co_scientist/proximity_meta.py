"""PROXIMITY (deduplica/agrupa similares) + META-REVIEW (sintese final)."""
import json, re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from shared.claude_client import ask
from shared.agent_base import Hypothesis

SYSTEM_PROX = """Voce identifica hipoteses que sao essencialmente A MESMA IDEIA (mesmo alvo, mesma via, intervencao quase igual). Recebe lista numerada e retorna grupos.
Saida JSON: {"groups": [[indice,...], [indice,...]]}. Indices base 0. Hipoteses unicas viram grupo de tamanho 1."""

SYSTEM_META = """Voce e o META-REVISOR. Recebe as top hipoteses ranqueadas e produz:
1. Sintese dos achados convergentes (vias/alvos que aparecem multiplas vezes)
2. Ranking final justificado
3. PROXIMO EXPERIMENTO sugerido p/ a top-1 (in silico ou in vitro acessivel)
4. Sinais de alerta (overfit a literatura recente, vies)

Saida: texto markdown claro, ate 600 palavras."""

def cluster(hyps: list) -> list:
    if len(hyps) <= 2: return [[i] for i in range(len(hyps))]
    block = "\n".join(f"[{i}] {h.title} | alvo/mecanismo: {h.mechanism[:200]} | intervencao: {h.proposed_intervention[:120]}" for i,h in enumerate(hyps))
    raw = ask([{"role":"user","content":f"Hipoteses:\n{block}\n\nAgrupe similares. JSON puro."}], system=SYSTEM_PROX, temperature=0.2, max_tokens=600)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m: return [[i] for i in range(len(hyps))]
    try:
        g = json.loads(m.group(0)).get("groups", [])
        return [list(map(int, grp)) for grp in g] or [[i] for i in range(len(hyps))]
    except Exception:
        return [[i] for i in range(len(hyps))]

def dedupe_keep_best(hyps: list, groups: list) -> list:
    out = []
    for grp in groups:
        if not grp: continue
        best = max(grp, key=lambda i: hyps[i].overall + hyps[i].elo/2000)
        out.append(hyps[best])
    return out

def meta_review(disease: str, top: list) -> str:
    block = ""
    for i,h in enumerate(top[:8],1):
        block += f"\n[#{i}] (elo {h.elo:.0f} overall {h.overall:.2f}) {h.title}\n  Mecanismo: {h.mechanism}\n  Intervencao: {h.proposed_intervention}\n  Evidencia PMIDs: {', '.join(h.evidence_pmids[:5])}\n"
    raw = ask([{"role":"user","content":f"DOENCA: {disease}\n\nTOP HIPOTESES:{block}\n\nFaca o meta-review."}], system=SYSTEM_META, temperature=0.5, max_tokens=2000)
    return raw
