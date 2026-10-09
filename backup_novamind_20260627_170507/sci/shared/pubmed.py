"""
Retrieval biomedico on-demand via NCBI E-utilities (decisao do usuario: sem indice local).
esearch -> efetch (abstracts) com throttling educado (3 req/s sem API key, 10 c/ key).
"""
import os, time, re
from Bio import Entrez
from typing import List, Dict

Entrez.email = os.environ.get("ENTREZ_EMAIL", "contributor@example.invalid")
if os.environ.get("ENTREZ_API_KEY"):
    Entrez.api_key = os.environ["ENTREZ_API_KEY"]

_last = [0.0]
_MIN_GAP = 0.34  # ~3 req/s sem key

def _throttle():
    dt = time.time() - _last[0]
    if dt < _MIN_GAP:
        time.sleep(_MIN_GAP - dt)
    _last[0] = time.time()

def search_pmids(query: str, retmax: int = 30, sort: str = "relevance") -> List[str]:
    _throttle()
    h = Entrez.esearch(db="pubmed", term=query, retmax=retmax, sort=sort)
    r = Entrez.read(h); h.close()
    return r.get("IdList", [])

def fetch_abstracts(pmids: List[str]) -> List[Dict]:
    """Retorna [{pmid,title,abstract,year,journal,authors,mesh}]."""
    if not pmids:
        return []
    _throttle()
    h = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="xml")
    recs = Entrez.read(h); h.close()
    out = []
    for art in recs.get("PubmedArticle", []):
        try:
            cit = art["MedlineCitation"]; a = cit["Article"]
            pmid = str(cit.get("PMID", ""))
            title = str(a.get("ArticleTitle", ""))
            abst_parts = a.get("Abstract", {}).get("AbstractText", []) or []
            abst = " ".join(str(x) for x in abst_parts) if isinstance(abst_parts, list) else str(abst_parts)
            year = ""
            try:
                year = str(a["Journal"]["JournalIssue"]["PubDate"].get("Year","")) or str(a["Journal"]["JournalIssue"]["PubDate"].get("MedlineDate",""))[:4]
            except Exception: pass
            journal = str(a.get("Journal",{}).get("Title",""))
            authors = []
            for au in (a.get("AuthorList") or [])[:6]:
                ln, fn = str(au.get("LastName","")), str(au.get("ForeName",""))
                if ln: authors.append(f"{ln} {fn[:1]}")
            mesh = []
            for m in cit.get("MeshHeadingList", []) or []:
                mesh.append(str(m["DescriptorName"]))
            out.append({"pmid":pmid,"title":title,"abstract":abst,"year":year,"journal":journal,"authors":authors,"mesh":mesh})
        except Exception:
            continue
    return out

def _normalize_disease_to_english(disease: str) -> str:
    """Map common PT/ES disease names -> English MeSH-friendly. Se naum bater, deixa como esta."""
    d = disease.lower().strip()
    table = {
        "depressao":"major depressive disorder", "depressão":"major depressive disorder",
        "depressao maior":"major depressive disorder", "depressão maior":"major depressive disorder",
        "depressao resistente":"treatment-resistant depression", "depressão resistente":"treatment-resistant depression",
        "alopecia":"alopecia areata", "vitiligo":"vitiligo",
        "fibrose pulmonar":"idiopathic pulmonary fibrosis", "fibrose pulmonar idiopatica":"idiopathic pulmonary fibrosis",
        "ela":"amyotrophic lateral sclerosis", "esclerose lateral amiotrofica":"amyotrophic lateral sclerosis",
        "alzheimer":"alzheimer disease", "parkinson":"parkinson disease",
        "diabetes tipo 1":"type 1 diabetes", "diabetes tipo 2":"type 2 diabetes",
        "cancer de mama":"breast cancer", "câncer de mama":"breast cancer",
        "esquizofrenia":"schizophrenia", "autismo":"autism spectrum disorder",
        "lupus":"systemic lupus erythematosus", "esclerose multipla":"multiple sclerosis",
        "esclerose múltipla":"multiple sclerosis", "dmri":"age-related macular degeneration",
        "dmri seca":"dry age-related macular degeneration",
    }
    if d in table: return table[d]
    # se contem caractere acentuado, tira acento como aproximacao
    import unicodedata
    nd = "".join(c for c in unicodedata.normalize("NFD", d) if unicodedata.category(c) != "Mn")
    if nd in table: return table[nd]
    return disease

def search_and_fetch(query: str, retmax: int = 30) -> List[Dict]:
    """Busca PubMed; se vier pouca coisa e o termo for portugues, retenta em ingles."""
    pmids = search_pmids(query, retmax=retmax)
    papers = fetch_abstracts(pmids)
    if len(papers) < 8:
        # tenta normalizar termo principal pra ingles + retry
        eng = _normalize_disease_to_english(query.split()[0])
        if eng.lower() != query.split()[0].lower():
            new_q = query.replace(query.split()[0], eng, 1)
            extra = fetch_abstracts(search_pmids(new_q, retmax=retmax))
            seen = {p["pmid"] for p in papers}
            for e in extra:
                if e["pmid"] not in seen: papers.append(e); seen.add(e["pmid"])
    return papers

def format_for_llm(papers: List[Dict], max_chars_each: int = 1200) -> str:
    """Concatena papers num bloco compacto p/ contexto do LLM."""
    out = []
    for p in papers:
        a = (p["abstract"] or "").strip()
        if len(a) > max_chars_each:
            a = a[:max_chars_each] + "..."
        out.append(f"[PMID:{p['pmid']}] ({p['year']}) {p['title']}\n{a}")
    return "\n\n---\n\n".join(out)
