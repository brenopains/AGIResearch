"""
Baixa 5000+ abstracts REAIS do PubMed via NCBI E-utilities API.
Alimenta TODOS os modulos da AGI simultaneamente.

Uso:
    python scripts/fetch_real_pubmed.py
    python scripts/fetch_real_pubmed.py --max 10000 --domains "Alzheimer,cancer,depression"
"""
import sys, os, json, time, re, urllib.request, urllib.parse, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

OUTPUT_DIR = r"D:\novamind_data\pubmed_real"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# Query terms por dominio
DOMAIN_QUERIES = {
    "alzheimer": "Alzheimer disease amyloid tau neurodegeneration",
    "depression": "major depressive disorder serotonin antidepressant",
    "schizophrenia": "schizophrenia dopamine antipsychotic glutamate",
    "autism": "autism spectrum disorder synapse genetic SHANK3",
    "bipolar": "bipolar disorder lithium mood stabilizer circadian",
    "parkinson": "Parkinson disease alpha-synuclein dopamine substantia nigra",
    "cancer": "cancer oncogene tumor suppressor immunotherapy metastasis",
    "immunology": "immune system T cell cytokine inflammation antibody",
    "neuroscience": "neuroplasticity LTP BDNF hippocampus synaptic plasticity",
    "genetics": "CRISPR gene therapy epigenetic genome editing",
    "pharmacology": "drug discovery pharmacokinetics pharmacodynamics CYP450",
    "microbiology": "microbiome gut bacteria probiotics antibiotic resistance",
    "oncology": "EGFR KRAS BRAF checkpoint inhibitor CAR-T",
    "cardiovascular": "heart disease hypertension atherosclerosis cholesterol statin",
    "diabetes": "type 2 diabetes insulin resistance metabolism GLP-1",
    "obesity": "obesity leptin adipokine bariatric surgery microbiome",
    "aging": "aging senescence telomere longevity caloric restriction",
    "pain": "chronic pain opioid nociceptor neuropathic pain management",
    "stroke": "ischemic stroke thrombolysis neuroprotection rehabilitation",
    "epilepsy": "epilepsy seizure antiepileptic GABA glutamate",
    "multiplesclerosis": "multiple sclerosis myelin oligodendrocyte immunomodulation",
    "huntington": "Huntington disease huntingtin CAG repeat striatum",
    "als": "ALS TDP-43 motor neuron neurodegeneration SOD1",
    "retina": "retina photoreceptor macular degeneration vision",
    "hearing": "hearing loss cochlea hair cell tinnitus",
    "liver": "liver cirrhosis hepatitis fibrosis NASH",
    "kidney": "kidney disease nephron dialysis glomerular filtration",
    "lung": "lung disease COPD pulmonary fibrosis asthma",
    "gut": "inflammatory bowel disease Crohn colitis microbiome barrier",
    "bone": "osteoporosis bone fracture osteoblast osteoclast calcium",
    "skin": "skin melanoma psoriasis wound healing keratinocyte",
}


def fetch_pubmed_ids(query, max_results=200, retstart=0):
    """Busca PMIDs no PubMed."""
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "retmax": min(max_results, 10000),
        "retstart": retstart,
        "retmode": "json",
        "sort": "relevance",
    })
    url = f"{NCBI_BASE}esearch.fcgi?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"  [ERRO] esearch: {e}")
        return []


def fetch_pubmed_abstracts(pmids, batch_size=50):
    """Busca detalhes (abstracts) de uma lista de PMIDs."""
    abstracts = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        params = urllib.parse.urlencode({
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract",
        })
        url = f"{NCBI_BASE}efetch.fcgi?{params}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=30)
            xml_data = resp.read()
            root = ET.fromstring(xml_data)

            for article in root.iter("PubmedArticle"):
                try:
                    pmid_elem = article.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else ""

                    title_elem = article.find(".//ArticleTitle")
                    title = "".join(title_elem.itertext()) if title_elem is not None else ""

                    abstract_elem = article.find(".//AbstractText")
                    abstract = "".join(abstract_elem.itertext()) if abstract_elem is not None else ""

                    journal_elem = article.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else ""

                    year_elem = article.find(".//PubDate/Year")
                    year = year_elem.text if year_elem is not None else ""

                    if abstract and len(abstract) > 50:
                        abstracts.append({
                            "pmid": pmid,
                            "title": title,
                            "abstract": abstract,
                            "journal": journal,
                            "year": year,
                        })
                except Exception:
                    continue

            if i % 200 == 0 and abstracts:
                print(f"    Baixados {len(abstracts)} abstracts...")

        except Exception as e:
            print(f"  [ERRO] efetch batch {i}: {e}")
            time.sleep(2)

        time.sleep(0.5)  # NCBI rate limit

    return abstracts


def extract_concepts_from_abstract(title, abstract):
    """Extrai conceitos-chave de um abstract."""
    text = f"{title}. {abstract}"
    concepts = set()

    # Genes maiusculos (EGFR, KRAS, etc)
    for m in re.finditer(r'\b([A-Z]{2,}(?:\d[A-Z]?)?)\b', text):
        c = m.group(1)
        if len(c) >= 2 and not c.isdigit():
            concepts.add(c)

    # Termos compostos com maiuscula + minuscula
    for m in re.finditer(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})', text):
        c = m.group(1).strip()
        if len(c) > 5 and len(c.split()) <= 5:
            concepts.add(c)

    return list(concepts)[:20]


def extract_causal_relations(abstract):
    """Extrai relacoes causais do abstract usando padroes."""
    relations = []
    patterns = [
        (r'(.{10,60}?) (causes|leads to|results in|contributes to|triggers|promotes|induces|drives|mediates|regulates) (.{10,60}?)\.', 1, 3),
        (r'(.{10,60}?) (is associated with|is linked to|correlates with) (.{10,60}?)\.', 1, 3),
        (r'(.{10,60}?) (activates|inhibits|blocks|suppresses|reduces|increases) (.{10,60}?)\.', 1, 3),
    ]
    for pattern, src_idx, tgt_idx in patterns:
        for m in re.finditer(pattern, abstract, re.IGNORECASE):
            src = m.group(src_idx).strip().split(",")[0].strip()
            tgt = m.group(tgt_idx).strip().split(",")[0].strip()
            verb = m.group(2).lower()
            if src and tgt and len(src) > 5 and len(tgt) > 5:
                relations.append({"source": src, "target": tgt, "type": verb})
    return relations[:5]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=5000, help="Max abstracts")
    parser.add_argument("--domains", default=None, help="Comma-separated domains")
    args = parser.parse_args()

    domains_to_fetch = args.domains.split(",") if args.domains else list(DOMAIN_QUERIES.keys())
    total_target = args.max
    per_domain = max(50, total_target // len(domains_to_fetch))

    all_abstracts = []
    all_facts = []

    print(f"PubMed Real Data Fetcher")
    print(f"Dominios: {len(domains_to_fetch)}, alvo: ~{total_target} abstracts")
    print(f"Por dominio: ~{per_domain}")
    print()

    for domain in domains_to_fetch:
        query = DOMAIN_QUERIES.get(domain, domain)
        print(f"\n--- {domain.upper()} ---")
        print(f"  Query: {query}")

        # 1. Busca PMIDs
        pmids = fetch_pubmed_ids(query, max_results=per_domain)
        if not pmids:
            print(f"  0 PMIDs encontrados")
            continue
        print(f"  PMIDs: {len(pmids)}")

        # 2. Busca abstracts
        abstracts = fetch_pubmed_abstracts(pmids)
        print(f"  Abstracts: {len(abstracts)}")

        # 3. Salva
        for ab in abstracts:
            ab["domain"] = domain
            all_abstracts.append(ab)

            # Extrai relacoes causais
            rels = extract_causal_relations(ab["abstract"])
            for rel in rels:
                rel["domain"] = domain
                rel["pmid"] = ab["pmid"]
                all_facts.append(rel)

        # Salva dominio individual
        domain_file = os.path.join(OUTPUT_DIR, f"{domain}.json")
        with open(domain_file, "w", encoding="utf-8") as f:
            json.dump(abstracts, f, indent=2, ensure_ascii=False)

        print(f"  Salvo: {domain_file}")

    # 4. Salva consolidado
    combined_file = os.path.join(OUTPUT_DIR, "all_abstracts.json")
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_abstracts, f, indent=2, ensure_ascii=False)

    facts_file = os.path.join(OUTPUT_DIR, "causal_relations.json")
    with open(facts_file, "w", encoding="utf-8") as f:
        json.dump(all_facts, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"RESUMO")
    print(f"{'='*55}")
    print(f"Total abstracts: {len(all_abstracts)}")
    print(f"Total relacoes causais extraidas: {len(all_facts)}")
    print(f"Arquivo abstracts: {combined_file}")
    print(f"Arquivo relacoes:  {facts_file}")

    # 5. Gera arquivos TXT para o pipeline de ingestao
    for domain in domains_to_fetch:
        domain_abs = [a for a in all_abstracts if a["domain"] == domain]
        if domain_abs:
            txt_path = os.path.join(r"D:\novamind_data\knowledge_raw", f"{domain}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                for ab in domain_abs:
                    f.write(f"{ab['title']}. ABSTRACT: {ab['abstract']}\n")
            print(f"  TXT: {txt_path} ({len(domain_abs)} abstracts)")


if __name__ == "__main__":
    main()
