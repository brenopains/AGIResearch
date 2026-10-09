"""ROBIN local (FutureHouse-inspired): Crow / Falcon / Owl / DataAnalyst.
Diferenca p/ Co-Scientist: Robin foca em UMA hipotese de cada vez e ITERA com deep-read
de papers + analise de dados, em vez de torneio.
"""
import json, re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from shared.claude_client import ask, DEFAULT_MODEL
from shared.pubmed import search_and_fetch, format_for_llm, search_pmids, fetch_abstracts
from shared.agent_base import Hypothesis

# CROW: gerador de hipotese ANCORADO em literatura (proximo de Generation, mas 1 por vez e com mais foco)
CROW_SYSTEM = """Voce e CROW, gerador cientifico. Recebe uma doenca + bloco de literatura + lista de hipoteses ja exploradas (evitar repetir). Produz UMA hipotese forte, com mecanismo claro e intervencao concreta.
Saida JSON: {title, statement, rationale, mechanism, proposed_intervention, evidence_pmids:[...]}."""

def crow(disease: str, prior_titles: list, lit_block: str) -> Hypothesis | None:
    prior = "\n- ".join(prior_titles) if prior_titles else "(nenhuma ainda)"
    user = f"""DOENCA: {disease}

LITERATURA:
{lit_block}

HIPOTESES JA EXPLORADAS (NAO repetir):
- {prior}

Produza UMA nova hipotese mecanisticamente distinta. JSON puro."""
    raw = ask([{"role":"user","content":user}], system=CROW_SYSTEM, temperature=0.8, max_tokens=1500)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m: return None
    try: d = json.loads(m.group(0))
    except Exception: return None
    return Hypothesis(title=d.get("title",""),statement=d.get("statement",""),rationale=d.get("rationale",""),
                     mechanism=d.get("mechanism",""),proposed_intervention=d.get("proposed_intervention",""),
                     evidence_pmids=[str(x) for x in (d.get("evidence_pmids") or [])])

# FALCON: deep-read da literatura especifica do alvo/mecanismo (mais profundo que Crow)
FALCON_SYSTEM = """Voce e FALCON, revisor de literatura PROFUNDO. Recebe uma hipotese e abstracts focados no alvo/mecanismo dela. Extrai: evidencia DIRETA pro/contra, vias adjacentes, limitacoes dos estudos, qual o GAP que esta hipotese fecharia.
Saida texto markdown ate 400 palavras."""

def falcon(h: Hypothesis, disease: str) -> str:
    # busca papers focados no alvo+doenca
    targets = re.findall(r"[A-Z]{2,6}\d*|[A-Za-z]{4,}", h.mechanism + " " + h.proposed_intervention)[:6]
    if not targets: targets = [disease]
    q = " AND ".join([f'"{t}"[All Fields]' for t in targets[:3]] + [f'"{disease}"[All Fields]'])
    papers = search_and_fetch(q, retmax=12)
    if not papers:
        papers = search_and_fetch(f"{disease} {targets[0] if targets else ''} review", retmax=8)
    lit = format_for_llm(papers, max_chars_each=900)
    user = f"""HIPOTESE: {h.title}\n{h.statement}\nMecanismo: {h.mechanism}\nIntervencao: {h.proposed_intervention}\n\nLITERATURA FOCADA:\n{lit}\n\nAnalise profunda."""
    return ask([{"role":"user","content":user}], system=FALCON_SYSTEM, temperature=0.4, max_tokens=1500)

# FINCH: agente de analise de dados do Robin (paper: 8 trajetorias paralelas + meta-analise consensual).
# Aqui simplificamos: 1 trajetoria que (a) identifica dataset, (b) propoe pipeline, (c) gera codigo Python executavel.
FINCH_SYSTEM = """Voce e FINCH, analise de dados experimentais. Recebe uma hipotese e identifica:
1. Quais datasets publicos (GEO/Expression Atlas/SRA) seriam relevantes para testar.
2. Qual sinal especifico procurar (gene UP/DOWN em qual tecido/contexto/condicao).
3. Que pipeline minimo (Python/scanpy/limma) testaria.
Saida markdown ate 350 palavras."""

def finch(h: Hypothesis) -> str:
    user = f"""HIPOTESE: {h.title}\n{h.statement}\nMecanismo: {h.mechanism}\nIntervencao: {h.proposed_intervention}\n\nQue dados publicos testariam isso? Qual pipeline?"""
    return ask([{"role":"user","content":user}], system=FINCH_SYSTEM, temperature=0.3, max_tokens=1200)

# DATA ANALYST: gera codigo Python que IMPLEMENTA o pipeline sugerido pelo Finch.
DA_SYSTEM = """Voce e o ANALISTA DE DADOS. Recebe descricao de pipeline e produz CODIGO PYTHON COMPLETO E EXECUTAVEL (scanpy/pandas/numpy/scipy) que faz a analise. Sem placeholders, sem TODO. Inclui download de dataset GEO (GEOparse ou diretamente NCBI), processamento, teste estatistico, output em markdown. Comentado em portugues.
Saida: codigo Python entre tres crases."""

def data_analyst(owl_plan: str) -> str:
    user = f"PLANO DE ANALISE:\n{owl_plan}\n\nProduza o codigo Python COMPLETO."
    return ask([{"role":"user","content":user}], system=DA_SYSTEM, temperature=0.2, max_tokens=2500)
