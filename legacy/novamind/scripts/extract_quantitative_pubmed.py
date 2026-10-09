"""
extract_quantitative_pubmed.py — Extrai dados QUANTITATIVOS dos abstracts do PubMed.

Dados reais para alimentar o ModelComparator:
  "amyloid PET aumentou 20% em pacientes com Alzheimer"
  → biomarker="amyloid_beta", change=+0.20, disease="Alzheimer"

Isso permite comparacao Bayesiana REAL entre modelos rivais.
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

OUTPUT_DIR = r"D:\novamind_data\quantitative_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Padroes para extrair dados quantitativos de abstracts
QUANT_PATTERNS = [
    # "reduced by 20%", "increased by 35%"
    (r'(reduced|decreased|lower|downregulated) by (\d+\.?\d*)%', -1),
    (r'(increased|elevated|higher|upregulated) by (\d+\.?\d*)%', 1),
    (r'(reduction|decrease|decline) of (\d+\.?\d*)%', -1),
    (r'(increase|elevation|rise) of (\d+\.?\d*)%', 1),
    # "35% reduction", "20% increase"
    (r'(\d+\.?\d*)% (reduction|decrease|decline)', -1),
    (r'(\d+\.?\d*)% (increase|elevation|rise)', 1),
    # fold changes
    (r'(\d+\.?\d*)-fold (reduction|decrease)', -1),
    (r'(\d+\.?\d*)-fold (increase|elevation)', 1),
]

# Mapeamento: palavras-chave nos abstracts -> biomarcadores do modelo
BIOMARKER_MAP = {
    "amyloid": "amyloid_beta",
    "Aβ": "amyloid_beta",
    "amyloid-beta": "amyloid_beta",
    "tau": "p_tau",
    "p-tau": "p_tau",
    "neurofibrillary": "p_tau",
    "neurodegeneration": "neurodegeneration",
    "atrophy": "neurodegeneration",
    "neuroinflammation": "neuroinflammation",
    "inflammation": "neuroinflammation",
    "synaptic": "synaptic_function",
    "cognitive": "cognitive_function",
    "cognition": "cognitive_function",
    "MMSE": "cognitive_function",
    "microglial": "microglial_activation",
    "microglia": "microglial_activation",
    "oxidative": "oxidative_stress",
    "mitochondrial": "mitochondrial_function",
    "insulin": "insulin_metabolism",
    "glucose": "insulin_metabolism",
    "dopamine": "dopaminergic_loss",
    "motor": "motor_function",
    "alpha-synuclein": "alpha_synuclein",
    "tumor": "tumor_volume",
    "immune": "immune_activity",
    "angiogenesis": "angiogenesis",
}


def extract_quantitative_data(abstract: str, disease: str) -> list:
    """Extrai dados quantitativos de um abstract do PubMed.
    
    Returns: lista de dicts {biomarker, change, disease, source}
    """
    results = []
    text_lower = abstract.lower()

    # Detecta quais biomarcadores estao presentes
    present_biomarkers = {}
    for keyword, biomarker in BIOMARKER_MAP.items():
        if keyword.lower() in text_lower:
            present_biomarkers[keyword] = biomarker

    if not present_biomarkers:
        return results

    # Extrai valores quantitativos
    for pattern, direction in QUANT_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                # pattern with word + number
                value_str = match[1]
            else:
                value_str = match[0]
            
            try:
                pct = float(value_str)
                if pct > 100:  # provavelmente nao e porcentagem
                    continue
                
                change = direction * pct / 100.0
                
                # Associa a mudanca ao biomarcador mais proximo
                for keyword, biomarker in present_biomarkers.items():
                    # Verifica se a keyword esta proxima do valor no texto
                    idx = text_lower.find(keyword)
                    val_idx = text_lower.find(value_str)
                    if abs(idx - val_idx) < 200:  # mesma frase
                        results.append({
                            "biomarker": biomarker,
                            "change": round(change, 3),
                            "disease": disease,
                            "source": abstract[:100],
                            "confidence": min(0.9, 0.3 + len(results) * 0.1),
                        })
                        break
            except ValueError:
                continue

    return results[:5]


def extract_from_pubmed_cache():
    """Extrai dados quantitativos de TODOS os abstracts do PubMed ja baixados."""
    pubmed_dir = r"D:\novamind_data\pubmed_real"
    all_quant = []
    
    for fname in os.listdir(pubmed_dir):
        if not fname.endswith(".json") or fname in ("all_abstracts.json", "causal_relations.json"):
            continue
        disease = fname.replace(".json", "")
        filepath = os.path.join(pubmed_dir, fname)
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                abstracts = json.load(f)
        except:
            continue
        
        disease_quant = []
        for ab in abstracts:
            data = extract_quantitative_data(
                ab.get("abstract", ""), disease
            )
            disease_quant.extend(data)
            all_quant.extend(data)
        
        print(f"  {disease}: {len(disease_quant)} dados quantitativos de {len(abstracts)} abstracts")
    
    return all_quant


def main():
    print("=" * 60)
    print("Extracao de Dados Quantitativos do PubMed")
    print("=" * 60)
    
    print(f"\nProcessando abstracts do PubMed...")
    quant_data = extract_from_pubmed_cache()
    
    # Salva
    output_path = os.path.join(OUTPUT_DIR, "quantitative_observations.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(quant_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"RESUMO")
    print(f"{'='*60}")
    print(f"  Total dados quantitativos extraidos: {len(quant_data)}")
    print(f"  Salvo em: {output_path}")
    
    # Estatisticas por biomarcador
    by_biomarker = {}
    for d in quant_data:
        b = d["biomarker"]
        if b not in by_biomarker:
            by_biomarker[b] = []
        by_biomarker[b].append(d["change"])
    
    print(f"\n  Biomarcadores encontrados:")
    for b, vals in sorted(by_biomarker.items()):
        mean = sum(vals) / len(vals)
        print(f"    {b:30s}: {len(vals):4d} obs, media={mean:+.3f}")
    
    # Por doenca
    by_disease = {}
    for d in quant_data:
        dis = d["disease"]
        by_disease.setdefault(dis, []).append(d["change"])
    
    print(f"\n  Por doenca:")
    for dis, vals in sorted(by_disease.items()):
        mean = sum(vals) / len(vals) if vals else 0
        print(f"    {dis:20s}: {len(vals):4d} observacoes, media={mean:+.3f}")


if __name__ == "__main__":
    main()
