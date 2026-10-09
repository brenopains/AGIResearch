"""
test_intervention_discrimination.py — AGI discrimina modelos rivais
comparando PREDICOES DE INTERVENCAO com resultados REAIS de ensaios clinicos.

Cada modelo rival preve um efeito DIFERENTE para a mesma intervencao.
A AGI: 
  1. Simula a intervencao em cada modelo
  2. Extrai a predicao quantitativa
  3. Busca o resultado real no PubMed
  4. Compara — quem acertar vence
  5. Propoe cura baseada no modelo vencedor
"""
import sys, json, re, urllib.request, urllib.parse, time
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))

from core.virtual_lab import VirtualLab
from core.model_library import ModelLibrary


def fetch_pubmed_abstracts(query, max_results=10):
    """Busca abstracts no PubMed."""
    params = urllib.parse.urlencode({
        "db": "pubmed", "term": query, "retmax": max_results,
        "retmode": "xml", "sort": "relevance",
    })
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    import xml.etree.ElementTree as ET
    root = ET.fromstring(resp.read())
    pmids = [id_elem.text for id_elem in root.iter("Id")]
    
    if not pmids:
        return []
    
    fetch_params = urllib.parse.urlencode({
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
    })
    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{fetch_params}"
    req2 = urllib.request.Request(fetch_url, headers={"User-Agent": "Mozilla/5.0"})
    resp2 = urllib.request.urlopen(req2, timeout=30)
    root2 = ET.fromstring(resp2.read())
    
    abstracts = []
    for article in root2.iter("PubmedArticle"):
        title = article.findtext(".//ArticleTitle", "")
        abstract = article.findtext(".//AbstractText", "")
        if abstract and len(abstract) > 50:
            abstracts.append({"title": title, "abstract": abstract})
    
    return abstracts


def extract_quantitative(text, target_concept):
    """Extrai dados quantitativos de resultados de ensaios."""
    text_lower = text.lower()
    target_lower = target_concept.lower()
    
    patterns = [
        (r'(\d+\.?\d*)% (reduction|decrease|improvement)', -1),
        (r'(reduced|decreased|improved) by (\d+\.?\d*)%', -1),
        (r'(\d+\.?\d*)-fold (reduction|decrease)', -1),
        (r'(increase|improvement) of (\d+\.?\d*)%', 1),
    ]
    
    results = []
    for pattern, direction in patterns:
        for match in re.finditer(pattern, text_lower):
            try:
                val = float(match.group(1))
                if val > 100: continue
                change = direction * val / 100.0
                results.append({"change": change, "context": match.group()[:60]})
            except: continue
    
    return results


def main():
    print("="*70)
    print("DISCRIMINACAO DE MODELOS POR PREDICAO DE INTERVENCAO")
    print("="*70)
    
    # Carrega modelos rivais de Alzheimer
    lib = ModelLibrary()
    models = lib.get_models("Alzheimer")
    lab = VirtualLab()
    lab.load_disease_model("Alzheimer")
    
    # Intervencao conhecida: BACE1 inhibition (lecanemab)
    # Cada modelo preve um efeito DIFERENTE
    intervention = "BACE1_inhibition"
    dose = 0.5
    
    print(f"\nIntervencao: {intervention.upper()} (dose={dose})")
    print(f"\n{'Modelo':30s} {'Amyloid':>10s} {'Cognicao':>10s} {'Tau':>10s} {'Inflam':>10s} {'Sinapse':>10s}")
    print(f"{'-'*70}")
    
    model_predictions = {}
    for m in models:
        # Converte para o formato que o VirtualLab entende
        # Cada modelo tem sua propria ODE — usamos o modelo ODE diretamente
        from scipy.integrate import solve_ivp
        import numpy as np
        
        # Simula controle
        sol_ctrl = solve_ivp(m.ode_fn, (0, 365), m.initial_state, method="RK45", max_step=1)
        # Simula com intervencao (usando intervention_fn do modelo)
        if m.intervention_fn:
            y_interv = m.intervention_fn(m.initial_state.copy(), intervention, dose)
        else:
            y_interv = m.initial_state.copy()
        sol_interv = solve_ivp(m.ode_fn, (0, 365), y_interv, method="RK45", max_step=1)
        
        if sol_ctrl.y.shape[1] == 0 or sol_interv.y.shape[1] == 0:
            continue
        
        # Extrai mudancas nos biomarcadores principais
        changes = {}
        for i, var in enumerate(m.variables):
            if i >= sol_ctrl.y.shape[0] or i >= sol_interv.y.shape[0]:
                continue
            ctrl_end = float(sol_ctrl.y[i, -1])
            interv_end = float(sol_interv.y[i, -1])
            if abs(ctrl_end) > 1e-6:
                changes[var] = (interv_end - ctrl_end) / abs(ctrl_end)
            else:
                changes[var] = 0.0
        
        model_predictions[m.name] = changes
        
        amo = changes.get("amyloid_beta", 0)
        cog = changes.get("cognitive_function", 0)
        tau = changes.get("p_tau", 0)
        nfl = changes.get("neuroinflammation", 0)
        syn = changes.get("synaptic_function", 0)
        
        print(f"{m.name:30s} {amo:>10.3f} {cog:>10.3f} {tau:>10.3f} {nfl:>10.3f} {syn:>10.3f}")
    
    # 2. Busca RESULTADOS REAIS de ensaios clinicos no PubMed
    print(f"\n--- Buscando resultados reais de ensaios clinicos ---")
    
    queries = [
        "lecanemab clinical trial cognitive decline 2023",
        "BACE1 inhibitor Alzheimer clinical trial cognitive improvement",
        "anti-amyloid immunotherapy Alzheimer cognition",
        "amyloid reduction cognitive benefit Alzheimer",
    ]
    
    real_outcomes = []
    for q in queries:
        print(f"  PubMed: {q}")
        abstracts = fetch_pubmed_abstracts(q, max_results=5)
        for ab in abstracts:
            quant = extract_quantitative(ab["abstract"], "amyloid")
            for r in quant:
                if "cognit" in ab["abstract"].lower() or "cognit" in ab["title"].lower():
                    real_outcomes.append({
                        "change": r["change"],
                        "context": r["context"],
                        "source": ab["title"][:80],
                    })
        time.sleep(0.5)
    
    if real_outcomes:
        print(f"\n  Resultados reais encontrados: {len(real_outcomes)}")
        for r in real_outcomes[:5]:
            print(f"    {r['change']:+.3f} ({r['context']})")
            print(f"    ↳ {r['source']}")
        
        avg_real = sum(r["change"] for r in real_outcomes) / len(real_outcomes)
        print(f"\n  Efeito medio real: {avg_real:+.3f}")
    else:
        print("  Nenhum resultado quantitativo encontrado (abstracts descritivos)")
        print("  Usando dados conhecidos: lecanemab reducao ~30% declinio cognitivo")
        avg_real = 0.30
    
    # 3. Compara predicoes de cada modelo com a realidade
    print(f"\n--- COMPARACAO: Modelo vs Realidade ---")
    print(f"\n{'Modelo':30s} {'Pred. Cognitivo':>15s} {'Real':>8s} {'Erro':>10s}")
    print(f"{'-'*65}")
    
    best_model = ""
    best_error = 999.0
    
    for m_name, preds in model_predictions.items():
        cog_pred = preds.get("cognitive_function", 0)
        error = abs(cog_pred - avg_real)
        print(f"{m_name:30s} {cog_pred:>+15.3f} {avg_real:>+8.3f} {error:>10.3f}")
        if error < best_error:
            best_error = error
            best_model = m_name
    
    # 4. Modelo VENCEDOR
    print(f"\n{'='*70}")
    print(f"MODELO VENCEDOR: {best_model}")
    print(f"{'='*70}")
    print(f"  Predicao de melhora cognitiva: {model_predictions[best_model]['cognitive_function']:+.3f}")
    print(f"  Real observado: {avg_real:+.3f}")
    print(f"  Erro: {best_error:.3f}")
    
    best_m = next(m for m in models if m.name == best_model)
    print(f"\n  Hipotese do modelo vencedor:")
    print(f"  {best_m.hypothesis}")
    
    # Propoe cura baseada no modelo vencedor — com dose segura
    print(f"\n  --- CURABASEADA NO MODELO VENCEDOR ---")
    
    # Para o modelo tau_first, a cura ideal e uma COMBINACAO:
    # 1. Inibir tau (terapia anti-tau) + inibir BACE1 (anti-amyloid)
    # Porque o modelo vencedor diz que tau e o driver primario
    
    print(f"  Modelo vencedor: {best_model}")
    print(f"  Hipotese: {best_m.hypothesis}")
    
    # Cura proposta: combinacao de anti-tau + anti-inflammatory
    # (ja que tau-first mostra que tau drive neurodegeneracao via inflamacao)
    if best_model == "tau_first":
        cura = ("Combinacao de inibidor de tau (ex: anti-tau oligonucleotideo) "
                "+ anti-inflammatory (ex: minociclina) "
                "+ suporte sinaptico (ex: BDNF mimetico)")
        mecanismo = "tau-first: tau e o driver primario. Anti-amyloid sozinho e insuficiente."
    elif best_model == "amyloid_first":
        cura = ("Inibidor de BACE1 (ex: lecanemab) "
                "+ terapia anti-tau (ex: inibidor de GSK3-beta)")
        mecanismo = "amyloid-first: amyloid dispara tau. Ambos precisam ser inibidos."
    else:
        cura = f"{intervention} em dose {best_dose} como terapia primaria"
        mecanismo = "modelo indica via unica"
    
    print(f"\n  --- CURA PROPOSTA PELA AGI ---")
    print(f"  Doenca: Alzheimer")
    print(f"  Modelo confiavel: {best_model} (erro={best_error:.3f})")
    print(f"  Mecanismo: {mecanismo}")
    print(f"  Terapia: {cura}")
    
    # Predicao quantitativa
    best_pred = model_predictions[best_model]
    print(f"\n  Predicao quantitativa (1 ano):")
    for var, change in sorted(best_pred.items()):
        if var in ["amyloid_beta", "p_tau", "cognitive_function", "neurodegeneration"]:
            print(f"    {var:25s}: {change:+.1%}")
    
    print(f"\n  Experimento para validar:")
    print(f"  1. Testar inibidor de tau em modelo animal")
    print(f"  2. Medir tau PET, MMSE, CDRSB a cada 6 meses")
    print(f"  3. Comparar declinio com grupo controle")
    print(f"  4. Se cognicao declinar 30% menos -> modelo validado")


if __name__ == "__main__":
    main()
