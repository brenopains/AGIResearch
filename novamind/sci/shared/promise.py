"""
PROMISE SCORE — indicador HONESTO p/ leigo: 'vale a pena pessoa-de-verdade investigar esta hipotese?'

Limite honesto reconhecido na cara: nenhum sistema AI consegue dizer 'isto cura X' sem wet-lab.
O que CONSEGUIMOS estimar com PROXIES validaveis:

  (1) Recuperacao de lead  — top hipoteses cujo alvo+intervencao APARECE em ensaio clinico real
                              ou paper recente (= o sistema "puxou da realidade", nao alucinou)
  (2) Convergencia          — varias personas/ciclos convergiram no mesmo ALVO MOLECULAR
                              (= muitos caminhos diferentes chegaram aqui)
  (3) Ganho composicional   — alguma combinacao A>>B teve score MUITO maior que A ou B sozinhos
                              (= o sistema descobriu sinergia, nao soma)
  (4) Concordancia          — juiz independente (LLM com prompt diferente) concorda com top-1
  (5) Novidade contextual   — alvo+intervencao NAO e o primeiro hit trivial do PubMed (nao e copia)
  (6) Diversidade           — top-K cobre vias mecanisticas distintas (nao e mono-tematico)

Score final em [0,1] traduzido p/ semaforo + frase em portugues clara.
"""
import re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from shared.pubmed import search_pmids, fetch_abstracts
from shared.llm_client import ask

JUDGE_SYSTEM = """Voce e juiz cientifico INDEPENDENTE. Recebe a doenca e a hipotese de tratamento. Avalie:
1. promising_for_research (0-1): vale a pena alguem testar em laboratorio?
2. major_red_flags: liste 0-3 problemas serios (seguranca, viabilidade, evidencia fraca). Vazio se nao houver.
3. one_line: 1 frase explicando p/ leigo se isso e promissor ou nao.
JSON puro com essas 3 chaves."""

def independent_judge(disease: str, h: dict) -> dict:
    user = f"DOENCA: {disease}\n\nHIPOTESE:\nTitulo: {h.get('title','')}\nMecanismo: {h.get('mechanism','')}\nIntervencao: {h.get('proposed_intervention','')}\nRacional: {h.get('rationale','')[:600]}\n\nJulgue. JSON puro."
    try:
        raw = ask([{"role":"user","content":user}], system=JUDGE_SYSTEM, temperature=0.1, max_tokens=400, cache=True)
        import json
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m: return {"promising_for_research":0.3, "major_red_flags":["parse_fail"], "one_line":""}
        d = json.loads(m.group(0))
        try: d["promising_for_research"]=float(d.get("promising_for_research",0.3))
        except Exception: d["promising_for_research"]=0.3
        d["promising_for_research"]=max(0.0,min(1.0,d["promising_for_research"]))
        if not isinstance(d.get("major_red_flags"),list): d["major_red_flags"]=[]
        d["one_line"]=str(d.get("one_line",""))[:300]
        return d
    except Exception as e:
        return {"promising_for_research":0.3, "major_red_flags":[f"err:{e}"], "one_line":""}

def lead_in_literature(intervention: str, disease: str) -> bool:
    """Aparece em algum ensaio/paper recente p/ esta doenca? -> sistema 'tracionou com a realidade'."""
    if not intervention or len(intervention.strip())<3: return False
    # extrai palavra-chave principal (1a palavra significativa)
    core = re.sub(r'[\(\),.;:].*$','',intervention).strip().split()[0:3]
    core = " ".join(core)
    try:
        pmids = search_pmids(f'"{core}"[Title/Abstract] AND "{disease}"[Title/Abstract]', retmax=2)
        return len(pmids) > 0
    except Exception:
        return False

def is_trivial_search_hit(intervention: str, disease: str) -> bool:
    """O candidato e o primeiro hit obvio? Se SIM, e copia da literatura, nao novidade."""
    try:
        pmids = search_pmids(f"{disease} treatment review", retmax=3)
        recs = fetch_abstracts(pmids)
        text = " ".join((r["title"]+" "+r["abstract"]) for r in recs).lower()
        core = re.sub(r'[\(\),.;:].*$','',intervention).strip().split()[0]
        return core and core.lower() in text and len(core)>3
    except Exception:
        return False

def target_signature(h: dict) -> str:
    """Heuristica p/ chave de convergencia: sigla mais frequente do mecanismo."""
    txt = (h.get("mechanism","") + " " + h.get("proposed_intervention","")).upper()
    sigs = re.findall(r'\b[A-Z]{2,6}\d*\b', txt)
    if not sigs: return ""
    from collections import Counter
    return Counter(sigs).most_common(1)[0][0]

def evaluate_promise(disease: str, hypotheses: list, compositions_asymmetry: list = None, top_k: int = 6) -> dict:
    """Computa indicador de promessa interpretavel."""
    hyps = sorted(hypotheses, key=lambda h: -(h.get("overall") or h.get("default_score") or 0))[:top_k]
    if not hyps:
        return {"promise":0.0, "label":"vazio", "signals":{}, "summary":"nenhuma hipotese gerada."}
    # 1. RECUPERACAO DE LEAD
    lead_hits = 0
    judge_results = []
    for h in hyps:
        if lead_in_literature(h.get("proposed_intervention",""), disease): lead_hits += 1
        judge_results.append(independent_judge(disease, h))
    lead_rate = lead_hits / len(hyps)
    # 2. CONVERGENCIA: top-K mapeia em quantos alvos distintos? muitos -> divergente; poucos com replicacao -> convergencia
    sigs = [target_signature(h) for h in hyps]
    sigs = [s for s in sigs if s]
    from collections import Counter
    sig_counter = Counter(sigs)
    convergence = (max(sig_counter.values())/len(hyps)) if sig_counter else 0.0  # 1 = todos no mesmo alvo
    convergence_target = sig_counter.most_common(1)[0][0] if sig_counter else ""
    # 3. GANHO COMPOSICIONAL: max assimetria observada
    comp_gain = max(compositions_asymmetry or [0]) if compositions_asymmetry else 0.0
    # 4. CONCORDANCIA: media da promissao do juiz independente p/ top-3
    judge_avg = sum(j["promising_for_research"] for j in judge_results[:3]) / max(1, min(3,len(judge_results)))
    # 5. NOVIDADE CONTEXTUAL: top-1 NAO e hit trivial -> novidade real
    trivial = is_trivial_search_hit(hyps[0].get("proposed_intervention",""), disease)
    novelty_real = 0.0 if trivial else 1.0
    # 6. DIVERSIDADE: nº de alvos distintos no top-K
    diversity = len(set(sigs)) / max(1,len(hyps))   # 1 = totalmente diverso
    # red flags
    flags = []
    for j in judge_results[:3]: flags.extend(j.get("major_red_flags") or [])
    # SCORE COMPOSTO (pesos calibrados p/ leigo: trato lead+concordancia como pilares; convergencia E diversidade balanceados)
    promise = (
        0.25 * lead_rate +
        0.25 * judge_avg +
        0.15 * convergence +           # alguma convergencia ajuda (mostra "puxar do real")
        0.15 * (1.0 if (0.30<=diversity<=0.85) else 0.5) +   # nem mono nem disperso demais
        0.10 * comp_gain +
        0.10 * novelty_real
    )
    promise = max(0.0, min(1.0, promise))
    if promise >= 0.65: label, emoji = "ALTA — vale wet-lab", "🟢"
    elif promise >= 0.40: label, emoji = "MÉDIA — vale revisao manual", "🟡"
    else: label, emoji = "BAIXA — provavel ruido", "🔴"

    summary_parts = []
    summary_parts.append(f"{emoji} Promessa {label} (score {promise:.2f}).")
    summary_parts.append(f"• {lead_rate:.0%} das top-{len(hyps)} tem o tratamento citado em literatura/ensaios reais (sinal de 'tracao com a realidade').")
    if convergence_target:
        summary_parts.append(f"• Convergencia: {sig_counter[convergence_target]}/{len(hyps)} hipoteses miram o alvo '{convergence_target}'.")
    summary_parts.append(f"• Juiz independente: top-3 medio {judge_avg:.0%} de 'vale testar'.")
    if comp_gain > 0.3:
        summary_parts.append(f"• Ganho composicional: A>>B diferiu de B>>A em {comp_gain:.2f} — sistema descobriu assimetria causal.")
    if trivial:
        summary_parts.append(f"• ⚠️ Top-1 e o primeiro hit obvio do PubMed — pode ser copia, nao descoberta.")
    if flags:
        summary_parts.append(f"• Alertas do juiz: {'; '.join(list(set(flags))[:3])}")

    return {
        "promise": round(promise, 3),
        "label": label,
        "emoji": emoji,
        "signals": {
            "lead_recovery_rate": round(lead_rate, 3),
            "convergence": round(convergence, 3),
            "convergence_target": convergence_target,
            "composition_gain_max": round(comp_gain, 3),
            "judge_top3_avg": round(judge_avg, 3),
            "novelty_real": novelty_real,
            "diversity": round(diversity, 3),
        },
        "summary": "\n".join(summary_parts),
        "judge_top3": judge_results[:3],
        "red_flags": list(set(flags))[:5],
        "top_target": convergence_target,
    }
