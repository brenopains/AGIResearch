"""REFLECTION AGENT: critica cada hipotese (advogado do diabo)."""
import json, re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from shared.claude_client import ask
from shared.agent_base import Hypothesis

SYSTEM = """Voce e um CRITICO cientifico (advogado do diabo). Recebe uma hipotese e produz: falhas mecanisticas, vieses, evidencia contraria conhecida, problemas de seguranca, sinais de overfit a literatura. Seja DURO mas justo.

Tambem pontua de 0 a 1 (numero):
- plausibility: o mecanismo se sustenta com biologia conhecida?
- novelty: o quanto isso ja foi tentado? (1 = nada parecido)
- tractability: existe rota pratica de testar/realizar?
- safety: 1 = baixo risco; 0 = perigoso/toxico/aventureiro demais
- evidence_strength: forca da evidencia citada (1 = robusta, 0 = especulativa)

Formato: JSON puro com chaves: critique (string), plausibility, novelty, tractability, safety, evidence_strength."""

def reflect(h: Hypothesis) -> dict:
    user = f"""HIPOTESE:
Titulo: {h.title}
Afirmacao: {h.statement}
Mecanismo: {h.mechanism}
Intervencao: {h.proposed_intervention}
PMIDs citados: {", ".join(h.evidence_pmids) or "(nenhum)"}
Racional: {h.rationale}

Critique. Saida JSON puro."""
    raw = ask([{"role":"user","content":user}], system=SYSTEM, temperature=0.4, max_tokens=1024)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"critique":"(falha em parsear critica)","plausibility":0.4,"novelty":0.4,"tractability":0.4,"safety":0.5,"evidence_strength":0.3}
    try:
        d = json.loads(m.group(0))
    except Exception:
        return {"critique":raw[:500],"plausibility":0.4,"novelty":0.4,"tractability":0.4,"safety":0.5,"evidence_strength":0.3}
    for k in ["plausibility","novelty","tractability","safety","evidence_strength"]:
        try: d[k] = float(d.get(k, 0.4))
        except Exception: d[k] = 0.4
        d[k] = max(0.0, min(1.0, d[k]))
    return d

def apply(h: Hypothesis):
    d = reflect(h)
    h.critiques.append(d["critique"])
    h.plausibility = d["plausibility"]; h.novelty = d["novelty"]; h.tractability = d["tractability"]
    h.safety = d["safety"]; h.evidence_strength = d["evidence_strength"]
    # overall: combinacao ponderada (testavel, plausivel, com evidencia, segura, com algum nivel de novidade)
    h.overall = 0.30*h.plausibility + 0.20*h.evidence_strength + 0.20*h.tractability + 0.15*h.safety + 0.15*h.novelty
    return h
