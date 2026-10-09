"""EVOLUTION AGENT: refina/combina as melhores hipoteses gerando descendentes."""
import json, re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from shared.claude_client import ask
from shared.agent_base import Hypothesis

SYSTEM_REFINE = """Voce REFINA uma hipotese cientifica enderecando as criticas recebidas SEM mudar a essencia. Pode trocar a intervencao especifica se houver melhor candidata, fortalecer mecanismo, citar evidencia adicional. Mantenha testavel.
Saida JSON: {title, statement, rationale, mechanism, proposed_intervention, evidence_pmids:[...]}."""

SYSTEM_CROSS = """Voce COMBINA duas hipoteses em uma nova que aproveita o que cada uma tem de melhor (mecanismos complementares, sinergia, contexto cruzado). Nao e media — e SINTESE com mecanismo coerente.
Saida JSON: {title, statement, rationale, mechanism, proposed_intervention, evidence_pmids:[...]}."""

def _parse(raw: str, parents: list) -> Hypothesis | None:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m: return None
    try: d = json.loads(m.group(0))
    except Exception: return None
    return Hypothesis(
        title=d.get("title",""), statement=d.get("statement",""), rationale=d.get("rationale",""),
        mechanism=d.get("mechanism",""), proposed_intervention=d.get("proposed_intervention",""),
        evidence_pmids=[str(x) for x in (d.get("evidence_pmids") or [])],
        parent_ids=[p.id for p in parents],
    )

def refine(h: Hypothesis) -> Hypothesis | None:
    crit = "\n- ".join(h.critiques[-2:]) if h.critiques else "(sem criticas)"
    user = f"""HIPOTESE ORIGINAL:
{h.title}
{h.statement}
Mecanismo: {h.mechanism} | Intervencao: {h.proposed_intervention}
Racional: {h.rationale}

CRITICAS A ENDERECAR:
- {crit}

Produza versao refinada (mesma essencia, criticas enderecadas). JSON puro."""
    raw = ask([{"role":"user","content":user}], system=SYSTEM_REFINE, temperature=0.6, max_tokens=1500)
    return _parse(raw, [h])

def cross(a: Hypothesis, b: Hypothesis) -> Hypothesis | None:
    user = f"""HIPOTESE A: {a.title}
{a.statement}
Mecanismo: {a.mechanism} | Intervencao: {a.proposed_intervention}

HIPOTESE B: {b.title}
{b.statement}
Mecanismo: {b.mechanism} | Intervencao: {b.proposed_intervention}

Sintetize em uma nova hipotese melhor. JSON puro."""
    raw = ask([{"role":"user","content":user}], system=SYSTEM_CROSS, temperature=0.75, max_tokens=1500)
    return _parse(raw, [a,b])
