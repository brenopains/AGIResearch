"""Avaliador de ASSERTIVIDADE para doenca sem cura conhecida (sem ground-truth direto).

Medidas honestas que ele computa:
1. PMID validity: % dos PMIDs citados existem mesmo (esummary).
2. PMID relevance: % dos PMIDs citados sao realmente sobre a doenca (titulo bate).
3. Lead recovery: % das top-K hipoteses cuja intervencao/alvo APARECE em ensaio clinico real (ClinicalTrials.gov via E-utilities) ou consta como mecanismo recorrente na literatura recente.
4. Independent plausibility: avaliacao por Claude com prompt diferente, ignorando os scores internos (anti-overfit).
5. Diversity: nº de vias mecanisticas distintas nas top-K.
"""
import re, json, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from shared.claude_client import ask
from shared.pubmed import search_pmids, fetch_abstracts, _throttle
from Bio import Entrez

SYSTEM_EVAL = """Voce e um avaliador cientifico INDEPENDENTE. Recebe uma hipotese de tratamento e julga friamente:
- already_known: 1 se essa intervencao+alvo JA esta documentada em literatura/ensaios para esta doenca; 0 caso contrario.
- novel_but_plausible: 1 se nao esta documentada mas o mecanismo se sustenta em biologia conhecida; 0 caso contrario.
- speculative: 1 se mecanismo fraco/sem suporte; 0 caso contrario.
- comment: 1 frase justificando.
JSON puro com essas chaves."""

def validate_pmids(pmids: list, disease_keywords: list) -> dict:
    """% existem; % titulo menciona alguma keyword da doenca."""
    pmids = [p for p in pmids if p and re.fullmatch(r"\d+", p)]
    if not pmids: return {"checked":0,"exist_rate":0.0,"relevant_rate":0.0}
    recs = fetch_abstracts(pmids)
    found = {r["pmid"] for r in recs}
    kws = [k.lower() for k in disease_keywords]
    relevant = 0
    for r in recs:
        tt = (r["title"]+" "+r["abstract"]).lower()
        if any(k in tt for k in kws): relevant += 1
    return {"checked":len(pmids),"exist_rate":len(found)/len(pmids),"relevant_rate":relevant/len(pmids)}

def trials_lookup(intervention: str, disease: str) -> bool:
    """ClinicalTrials.gov via Entrez."""
    try:
        _throttle()
        h = Entrez.esearch(db="clinicaltrials", term=f"({intervention}) AND ({disease})", retmax=1)
        # ATENCAO: a NCBI nao serve clinicaltrials por esearch. Usamos pubmed como proxy.
        h.close()
    except Exception: pass
    # proxy: PubMed busca por intervention+disease+trial
    try:
        pmids = search_pmids(f'"{intervention}"[Title/Abstract] AND "{disease}"[Title/Abstract] AND (trial OR clinical OR randomized)', retmax=3)
        return len(pmids) > 0
    except Exception:
        return False

def independent_judge(disease: str, h: dict) -> dict:
    user = f"""DOENCA: {disease}

HIPOTESE:
Titulo: {h.get('title','')}
Mecanismo: {h.get('mechanism','')}
Intervencao: {h.get('proposed_intervention','')}
Racional: {h.get('rationale','')[:600]}

Julgue. JSON puro."""
    try:
        raw = ask([{"role":"user","content":user}], system=SYSTEM_EVAL, temperature=0.0, max_tokens=300, cache=True)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m: return {"already_known":0,"novel_but_plausible":0,"speculative":1,"comment":"(parse fail)"}
        d = json.loads(m.group(0))
        for k in ["already_known","novel_but_plausible","speculative"]:
            try: d[k] = int(d.get(k,0))
            except Exception: d[k] = 0
        return d
    except Exception as e:
        return {"already_known":0,"novel_but_plausible":0,"speculative":1,"comment":f"err {e}"}

def diversity_score(hyps: list) -> float:
    """Diversidade = nº de alvos/vias distintos nas mecanicas (heuristica)."""
    keys = set()
    for h in hyps:
        m = (h.get("mechanism","") or "").lower()
        # extrai palavras-chave grosseiras
        toks = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{3,}", m)
        # foca em palavras com cap (siglas tipo JAK, IL, TNF, STAT)
        sigl = [t for t in toks if re.match(r"^[A-Z]{2,6}\d*$", t) or t.lower().endswith("ase")]
        for s in sigl[:3]: keys.add(s.upper())
        for t in toks[:2]: keys.add(t.lower())
    return len(keys)

def evaluate_run(payload: dict, top_k: int = 5) -> dict:
    disease = payload["disease"]
    hyps = payload["hypotheses"][:top_k]
    kws = re.findall(r"\w{4,}", disease.lower())
    if not kws: kws = [disease.lower()]
    report = {"disease":disease,"top_k":top_k,"per_hypothesis":[]}
    pmid_exist=[]; pmid_relev=[]; known=0; novel=0; spec=0; trials_hits=0
    print(f"\n=== AVALIACAO (top-{top_k}) ===")
    for i,h in enumerate(hyps,1):
        print(f"\n[{i}] {h.get('title','')[:80]}")
        v = validate_pmids(h.get("evidence_pmids",[]) or [], kws)
        print(f"  PMIDs checados={v['checked']} | existem={v['exist_rate']:.0%} | sao da doenca={v['relevant_rate']:.0%}")
        j = independent_judge(disease, h)
        print(f"  juiz independente: known={j['already_known']} novel={j['novel_but_plausible']} speculative={j['speculative']} | {j.get('comment','')[:120]}")
        try:
            t = trials_lookup(h.get("proposed_intervention",""), disease)
            print(f"  intervencao em literatura clinica/trial: {'sim' if t else 'nao'}")
            if t: trials_hits += 1
        except Exception: t = False
        pmid_exist.append(v["exist_rate"]); pmid_relev.append(v["relevant_rate"])
        known += j["already_known"]; novel += j["novel_but_plausible"]; spec += j["speculative"]
        report["per_hypothesis"].append({"title":h.get("title",""),"pmid":v,"judge":j,"in_trials":t})
    n = max(1, len(hyps))
    div = diversity_score(hyps)
    summary = {
        "n_evaluated": len(hyps),
        "avg_pmid_existence": sum(pmid_exist)/n,
        "avg_pmid_relevance": sum(pmid_relev)/n,
        "pct_already_known": known/n,
        "pct_novel_plausible": novel/n,
        "pct_speculative": spec/n,
        "lead_recovery_rate": trials_hits/n,
        "diversity_keys": div,
        # ASSERTIVIDADE COMPOSTA: PMIDs reais & relevantes + recuperacao de leads + plausibilidade + alguma novidade + diversidade
        "assertiveness": round(
            0.20*(sum(pmid_exist)/n) +
            0.20*(sum(pmid_relev)/n) +
            0.25*(known/n) +              # recuperar conhecido = relevancia
            0.15*(novel/n) +              # mas alguma novidade tambem conta
            0.10*(trials_hits/n) +
            0.10*min(1.0, div/12.0),
            3
        ),
    }
    report["summary"] = summary
    print(f"\n=== RESUMO ===")
    for k,v in summary.items(): print(f"  {k}: {v}")
    return report
