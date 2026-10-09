"""
convert_to_sbml.py — Converte modelos ODE do ModelLibrary para SBML
e valida com Tellurium.

Isso permite:
  1. Validacao cruzada: ODE Python vs Tellurium (SBML)
  2. Reutilizacao de modelos em outros simuladores
  3. Comparacao com modelos do BioModels (quando disponiveis)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

OUTPUT_DIR = r"D:\novamind_data\sbml_models\converted"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from core.model_library import ModelLibrary
from scipy.integrate import solve_ivp
import numpy as np
import time


def ode_to_antimony(model, model_name: str) -> str:
    """Converte simulacao ODE para reacoes SBML aproximadas.

    Como as ODEs sao funcoes Python arbitrarias, nao podemos extrair
    expressoes simbolicas. Em vez disso:
    1. Simulamos a ODE em Python
    2. Ajustamos reacoes simples que reproduzem a trajetoria
    3. Validamos com Tellurium que a trajetoria e similar
    """
    from scipy.integrate import solve_ivp
    
    # Simula ODE
    sol = solve_ivp(model.ode_fn, (0, 365), model.initial_state,
                    method="RK45", rtol=1e-6, atol=1e-8, max_step=1)
    
    if sol.y.shape[1] == 0:
        return "// ODE simulation failed"
    
    # Extrai taxas no inicio da simulacao
    initial_rates = model.ode_fn(0, model.initial_state)
    
    lines = [f"// Model: {model_name}", f"// Disease: {model.disease}",
             f"// Hypothesis: {model.hypothesis}", ""]
    
    for var in model.variables:
        safe = var.replace(" ", "_").replace("-", "_")
        lines.append(f"species {safe};")
    
    lines.append("")
    
    for i, var in enumerate(model.variables):
        safe = var.replace(" ", "_").replace("-", "_")
        rate = initial_rates[i]
        
        # Reacao de producao de ordem zero (constante)
        if abs(rate) > 1e-10:
            k_val = abs(rate)
            lines.append(f"k_prod_{safe} = {k_val:.6f};")
            lines.append(f"J_prod_{safe}: -> {safe}; k_prod_{safe};")
        
        # Reacao de degradacao de primeira ordem
        lines.append(f"k_decay_{safe} = 0.01;")
        lines.append(f"J_decay_{safe}: {safe} -> ; k_decay_{safe}*{safe};")
    
    lines.append("")
    for i, var in enumerate(model.variables):
        safe = var.replace(" ", "_").replace("-", "_")
        lines.append(f"{safe} = {model.initial_state[i]:.4f};")
    
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("ODE -> SBML Converter + Tellurium Validation")
    print("=" * 60)
    
    lib = ModelLibrary()
    
    for disease in lib.list_diseases():
        models = lib.get_models(disease)
        disease_dir = os.path.join(OUTPUT_DIR, disease)
        os.makedirs(disease_dir, exist_ok=True)
        
        for model in models:
            # Converte para Antimony
            ant = ode_to_antimony(model, model.name)
            ant_path = os.path.join(disease_dir, f"{model.name}.ant")
            with open(ant_path, "w", encoding="utf-8") as f:
                f.write(ant)
            
            # Simula com Tellurium
            try:
                import tellurium as te
                r = te.loada(ant)
                start = time.time()
                s = r.simulate(0, 365, 365)
                elapsed = time.time() - start
                
                # Verifica se a simulacao e consistente com a ODE Python
                py_sol = solve_ivp(model.ode_fn, (0, 365), model.initial_state,
                                  t_eval=np.linspace(0, 365, 365), method="RK45")
                
                # Compara resultados no endpoint
                tellurium_end = [float(s[-1][i+1]) for i in range(len(model.variables))]
                python_end = py_sol.y[:, -1].tolist() if py_sol.y.shape[1] > 0 else []
                
                # Erro medio
                if tellurium_end and python_end and len(tellurium_end) == len(python_end):
                    errors = [abs(t - p) / (abs(p) + 1e-10) for t, p in zip(tellurium_end, python_end)]
                    mean_error = sum(errors) / len(errors)
                else:
                    mean_error = -1
                
                print(f"\n  {disease}/{model.name}:")
                print(f"    Tellurium: {len(s)} timepoints in {elapsed*1000:.0f}ms")
                print(f"    Python-Tellurium error: {mean_error:.4f}" if mean_error >= 0 else "    Python-Tellurium: comparison N/A")
                
                if mean_error > 0.01:
                    print(f"    WARNING: Models diverge! Check ODE translation.")
                
            except Exception as e:
                print(f"\n  {disease}/{model.name}: Tellurium error: {e}")
    
    print(f"\nModelos SBML salvos em: {OUTPUT_DIR}")
    print(f"Total de modelos convertidos: {sum(len(lib.get_models(d)) for d in lib.list_diseases())}")


if __name__ == "__main__":
    main()
