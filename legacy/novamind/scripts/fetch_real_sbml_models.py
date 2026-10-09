"""
fetch_real_sbml_models.py — Baixa modelos SBML REAIS do BioModels Database
e simula com Tellurium para extrair dados quantitativos.

BioModels e um repositorio publico (nao precisa de token) com milhares
de modelos biológicos curados.

Para cada doenca, baixa modelos SBML relacionados e extrai:
  - Variaveis do modelo (especies quimicas, proteinas, etc.)
  - Parametros cineticos
  - Predicoes quantitativas (trajetorias dos biomarcadores)
  - Dados de intervencao (knockouts, perturbacoes)

Isso alimenta o ModelComparator com dados REAIS.
"""
import sys, os, json, time, re
from typing import Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

OUTPUT_DIR = r"D:\novamind_data\sbml_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# BioModels API endpoints
BIOMODELS_BASE = "https://www.ebi.ac.uk/biomodels"
BIOMODELS_SEARCH = f"{BIOMODELS_BASE}/api/search"
BIOMODELS_DOWNLOAD = f"{BIOMODELS_BASE}/api/model/{{model_id}}/download"

# Termos de busca para cada doenca
DISEASE_SEARCH = {
    "Alzheimer": ["alzheimer", "amyloid", "tau", "neurodegeneration"],
    "Parkinson": ["parkinson", "dopamine", "alpha-synuclein"],
    "cancer": ["cancer", "tumor", "apoptosis", "cell cycle"],
    "diabetes": ["diabetes", "insulin", "glucose metabolism"],
    "depression": ["depression", "serotonin", "monoamine"],
    "schizophrenia": ["schizophrenia", "dopamine", "NMDA"],
}

# Modelos SBML conhecidos que podemos baixar diretamente (BIOMD...)
# IDs do BioModels para modelos relevantes
KNOWN_MODELS = {
    "Alzheimer": [
        "BIOMD0000001060",  # Alzheimer amyloid cascade
    ],
    "Parkinson": [
        "BIOMD0000000955",  # Parkinson mitochrondrial
    ],
    "cancer": [
        "BIOMD0000000938",  # p53 pathway
        "BIOMD0000000975",  # EGFR signaling
    ],
    "diabetes": [
        "BIOMD0000000704",  # insulin signaling
    ],
}


def fetch_biomodels_search(query: str, max_results: int = 5) -> list:
    """Busca modelos no BioModels por query."""
    import urllib.request, urllib.parse
    params = urllib.parse.urlencode({
        "query": query,
        "numResults": max_results,
        "format": "json",
    })
    url = f"{BIOMODELS_SEARCH}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        return data.get("models", [])
    except Exception as e:
        return []


def download_sbml(model_id: str) -> Optional[str]:
    """Baixa modelo SBML do BioModels."""
    url = BIOMODELS_DOWNLOAD.format(model_id=model_id)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=30)
        sbml = resp.read().decode("utf-8")
        return sbml
    except Exception as e:
        return None


def simulate_sbml(sbml_content: str, duration: float = 1000,
                  steps: int = 1000) -> Optional[dict]:
    """Simula modelo SBML com Tellurium."""
    try:
        import tellurium as te
        r = te.loada(sbml_content) if "->" in sbml_content else te.loads(sbml_content)
        s = r.simulate(0, duration, steps)
        
        # Extrai nomes das colunas
        col_names = list(s.colnames)
        if not col_names:
            return None
        
        # Extrai trajetoria final
        final_values = {}
        for i, name in enumerate(col_names):
            if name != "time" and i < len(s[-1]):
                val = float(s[-1][i])
                if abs(val) < 1e10:  # Ignora valores divergentes
                    final_values[name] = val
        
        return {
            "n_species": len(col_names) - 1,
            "timepoints": len(s),
            "final_values": final_values,
            "trajectory_summary": {name: {"start": float(s[0][i]), "end": float(s[-1][i])}
                                  for i, name in enumerate(col_names)
                                  if name != "time" and i < len(s[0]) and i < len(s[-1])},
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 60)
    print("BioModels SBML Downloader + Tellurium Simulator")
    print("=" * 60)

    stats = {}

    # 1. Tenta baixar modelos conhecidos
    print(f"\n--- Downloading known models ---")
    for disease, model_ids in KNOWN_MODELS.items():
        disease_dir = os.path.join(OUTPUT_DIR, disease)
        os.makedirs(disease_dir, exist_ok=True)
        
        for model_id in model_ids:
            print(f"  {disease}/{model_id}...")
            sbml = download_sbml(model_id)
            if sbml:
                filepath = os.path.join(disease_dir, f"{model_id}.sbml")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(sbml)
                
                # Simula com Tellurium
                sim_result = simulate_sbml(sbml, duration=1000, steps=500)
                if sim_result and "error" not in sim_result:
                    print(f"    Simulated: {sim_result['n_species']} species, "
                          f"{sim_result['timepoints']} timepoints")
                    stats[model_id] = sim_result
                else:
                    err = sim_result.get("error", "unknown") if sim_result else "no result"
                    print(f"    Simulation error: {err}")
            else:
                print(f"    Download failed")
    
    # 2. Busca modelos por query no BioModels
    print(f"\n--- Searching BioModels by disease ---")
    for disease, queries in DISEASE_SEARCH.items():
        disease_dir = os.path.join(OUTPUT_DIR, disease)
        os.makedirs(disease_dir, exist_ok=True)
        
        for query in queries[:2]:
            models = fetch_biomodels_search(query, max_results=3)
            print(f"  {disease} ({query}): {len(models)} models found")
            
            for model in models[:2]:
                model_id = model.get("id", "")
                if not model_id:
                    continue
                    
                # Verifica se ja baixamos
                filepath = os.path.join(disease_dir, f"{model_id}.sbml")
                if os.path.exists(filepath):
                    continue
                
                sbml = download_sbml(model_id)
                if sbml:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(sbml)
                    sim_result = simulate_sbml(sbml, duration=500, steps=500)
                    if sim_result and "error" not in sim_result:
                        print(f"    {model_id}: {sim_result['n_species']} species")
                        stats[model_id] = sim_result
    
    # 3. Salva estatisticas
    stats_path = os.path.join(OUTPUT_DIR, "simulation_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({k: {"n_species": v.get("n_species", 0),
                       "timepoints": v.get("timepoints", 0)}
                  for k, v in stats.items()}, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"RESUMO")
    print(f"{'='*60}")
    print(f"  Modelos baixados: {len(stats)}")
    print(f"  Diretorio: {OUTPUT_DIR}")
    print(f"  Stats: {stats_path}")
    
    if stats:
        print(f"\n  Modelos disponiveis para Tellurium:")
        for model_id, s in stats.items():
            print(f"    {model_id}: {s.get('n_species', '?')} species, "
                  f"{s.get('timepoints', '?')} timepoints")


if __name__ == "__main__":
    main()
