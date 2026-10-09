"""GENERATION AGENT: propoe hipoteses iniciais ancoradas em literatura."""
import json, re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from shared.claude_client import ask, DEFAULT_MODEL
from shared.pubmed import search_and_fetch, format_for_llm
from shared.agent_base import Hypothesis

SYSTEM = """Voce e um agente CIENTIFICO biomedico. Sua tarefa: propor HIPOTESES TESTAVEIS sobre tratamento/cura de uma doenca, ancoradas em mecanismo molecular/celular plausivel e em evidencia da literatura fornecida.

REGRAS DURAS:
- Nada de afirmar cura. Hipotese e POSSIBILIDADE testavel.
- Cada hipotese deve citar PMIDs concretos da literatura fornecida que sustentem o raciocinio.
- Mecanismo claro: alvo molecular, via, efeito esperado.
- Intervencao concreta: composto/biologico/dieta/dispositivo nomeavel.
- Priorize: repurposing de drogas existentes (mais tratavel), inibidores/agonistas de alvos validados, contextos cruzados (insight de outra doenca aplicado).
- Diversidade: produza hipoteses MECANISTICAMENTE DISTINTAS (vias diferentes), nao variantes da mesma ideia.

Formato de saida: JSON puro, array de objetos com chaves: title, statement, rationale, mechanism, proposed_intervention, evidence_pmids (array de strings)."""

def generate(disease: str, n: int = 6, lit_retmax: int = 25) -> list:
    # 1. Busca literatura recente/relevante
    queries = [
        f'{disease} pathogenesis mechanism review',
        f'{disease} treatment trial',
        f'{disease} drug target',
    ]
    papers = []
    seen = set()
    for q in queries:
        for p in search_and_fetch(q, retmax=lit_retmax // len(queries) + 3):
            if p["pmid"] not in seen and p["abstract"]:
                seen.add(p["pmid"]); papers.append(p)
    lit_block = format_for_llm(papers, max_chars_each=1100)

    user = f"""DOENCA-ALVO: {disease}

LITERATURA RECENTE (use estes PMIDs como evidencia):
{lit_block}

Tarefa: proponha {n} hipoteses MECANISTICAMENTE DISTINTAS sobre como tratar/curar esta doenca. Cada uma com mecanismo, intervencao concreta, e PMIDs da literatura acima.

Saida JSON puro, array."""

    raw = ask([{"role":"user","content":user}], system=SYSTEM, temperature=0.85, max_tokens=4096)
    # extrai JSON
    m = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)
    if not m:
        raise RuntimeError(f"Generation nao retornou JSON.\nResposta: {raw[:500]}")
    arr = json.loads(m.group(0))
    out = []
    for h in arr:
        out.append(Hypothesis(
            title=h.get("title",""),
            statement=h.get("statement",""),
            rationale=h.get("rationale",""),
            mechanism=h.get("mechanism",""),
            proposed_intervention=h.get("proposed_intervention",""),
            evidence_pmids=[str(x) for x in (h.get("evidence_pmids") or [])],
        ))
    return out
