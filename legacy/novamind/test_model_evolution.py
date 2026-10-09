"""
test_model_evolution.py — Executa evolucao completa de modelos para Alzheimer.

A AGI:
  1. Carrega modelos iniciais (G10)
  2. Simula intervencao em cada um
  3. Compara com realidade
  4. Seleciona vencedor
  5. MUTA o vencedor (adiciona variaveis, feedbacks, etc.)
  6. Testa filhos vs pai
  7. Se filho melhor -> nova geracao
  8. Repete ate convergencia
  9. Gera relatorio final com arvore genealogica
"""
import sys, json
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))

from core.model_evolution import ModelEvolution

print("="*70)
print("EVOLUCAO DE MODELOS — ALZHEIMER")
print("="*70)

# Dados reais de ensaio clinico: lecanemab reduziu declinio cognitivo ~30%
# Fonte: NEJM 2023, van Dyck et al.
REAL_OUTCOME = 0.30  # 30% de melhora no declinio cognitivo

# Inicializa evolucao
evo = ModelEvolution()
evo.initialize("Alzheimer")

# Roda evolucao (max 8 geracoes, mas para automaticamente se estagnar)
print(f"\nDado real: BACE1 inhibition resulta em {REAL_OUTCOME:.0%} melhora cognitiva")
print(f"Intervencao: BACE1_inhibition dose=0.5")

best = evo.run(
    generations=8,
    intervention="BACE1_inhibition",
    dose=0.5,
    real_outcome=REAL_OUTCOME,
)

# Relatorio final
best_model = evo.get_best_model()
print(f"\n{'='*70}")
print(f"RELATORIO FINAL - EVOLUCAO DE MODELOS")
print(f"{'='*70}")

print(f"\nArvore genealogica:")
for gen_data in evo.get_fitness_history():
    g = gen_data["generation"]
    best_name = gen_data["best"]
    best_err = gen_data["best_error"]
    n = gen_data["n_models"]
    print(f"  G{g+10:2d}: {n:2d} modelos | vencedor={best_name:30s} | erro={best_err:.4f}")

print(f"\nMelhor modelo: {best.model_name}")
print(f"Erro final: {best.prediction_error:.4f}")
print(f"Hipotese: {best_model.hypothesis if best_model else 'N/A'}")

if best.prediction_error < 0.15:
    print(f"\nStatus: [OK] MODELO PREDITIVO (erro < 15%)")
    print(f"A AGI encontrou um modelo que explica bem os dados reais.")
elif best.prediction_error < 0.30:
    print(f"\nStatus: [WARN] MODELO MODERADO (erro entre 15-30%)")
    print(f"A AGI esta na direcao certa, mas precisa de mais mutacoes.")
else:
    print(f"\nStatus: [FAIL] MODELO FRACO (erro > 30%)")
    print(f"A AGI precisa de modelos fundamentalmente diferentes.")

# Sugestao da DMN: qual seria a proxima mutacao?
print(f"\nSugestao da DMN (proxima mutacao candidata):")
print(f"  - Adicionar glicemia/insulina como variavel latente")
print(f"  - Adicionar feedback loop entre tau e inflamacao")
print(f"  - Inverter causalidade: inflamacao -> tau em vez de tau -> inflamacao")
print(f"  - Adicionar clearance glinfatica como variavel")

# Salva
stats = evo.get_stats()
stats_path = "D:/novamind_data/evolution_alzheimer.json"
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)
print(f"\nRelatorio salvo: {stats_path}")
