"""
run_honest_discovery.py — DESCOBERTA GENUINA: AGI descobre tudo do zero.

Fluxo:
  1. Carrega modelos SBML reais (ground truth) — AGI NAO ve as equacoes
  2. AGI comeca com populacao aleatoria de EDOs — SEM dicas
  3. AGI evolui modelos por programacao genetica (mutacao + crossover)
  4. Fitness = quao bem reproduz a dinamica do SBML real
  5. AGI propoe intervencoes baseadas no modelo descoberto
  6. Ground truth testa as intervencoes
  7. Relatorio honesto do que foi descoberto

REGRA: A AGI NAO recebe nenhuma informacao privilegiada.
         Ela so ve: nomes opacos (x0, x1, ...) e trajetorias.
"""
import sys, os, time, json, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.biomodels_ground_truth import BiomodelsGroundTruth
from core.discovery_agent import DiscoveryAgent, ODEModel, random_expression


def run_honest_discovery():
    print("=" * 70)
    print("  DESCOBERTA HONESTA: AGI VS MODELOS SBML REAIS")
    print("=" * 70)
    print()
    print("  A AGI NAO sabe as equacoes.")
    print("  A AGI NAO sabe os nomes das variaveis.")
    print("  A AGI so ve trajetorias numericas.")
    print("  A AGI descobre TUDO por evolucao.")
    print()

    # ─── 1. CARREGA GROUND TRUTH ───
    print("[1/5] Carregando ground truth (SBML reais do BioModels)...")
    gt = BiomodelsGroundTruth()
    available = gt.list_models()
    print(f"  Modelos disponiveis: {len(available)}")

    # Escolhe modelo alvo
    target_model = "dopamine_signaling"
    target_info = gt.get_model_info(target_model)
    print(f"\n  Modelo alvo: {target_model}")
    print(f"  Descricao: {available[target_model]}")
    print(f"  Especies: {target_info.n_species}")
    print(f"  Reacoes: {target_info.n_reactions}")

    n_vars = target_info.n_species
    var_names = [f"x{i}" for i in range(n_vars)]  # Nomes OPAÇOS

    # ─── 2. OBTEM TRAJETORIA BASE ───
    print("\n[2/5] Simulando ground truth (referencia)...")
    base_traj = gt.simulate(target_model, t_end=100, n_points=50)
    if base_traj is None:
        print("  ERRO: falha ao simular ground truth")
        return

    # Remove tempo (primeira linha)
    gt_traj = base_traj[1:, :]  # (n_vars, n_points)
    print(f"  Trajetoria: {gt_traj.shape[0]} variaveis, {gt_traj.shape[1]} pontos")
    print(f"  Valores no estado final:")
    for i in range(min(5, n_vars)):
        print(f"    {var_names[i]}: {gt_traj[i, 0]:.3f} -> {gt_traj[i, -1]:.3f}")

    # Estado inicial
    initial_state = gt_traj[:, 0].copy()

    # ─── 3. CRIA WRAPPER DO GROUND TRUTH ───
    # A AGI so ve este wrapper — NAO as equacoes
    def ground_truth_blackbox(y0, t_span, n_steps):
        """Wrapper caixa-preta que a AGI pode chamar."""
        result = gt.simulate(target_model, t_end=t_span[1], n_points=n_steps)
        if result is None:
            return np.zeros((n_vars, n_steps))
        return result[1:, :]  # remove tempo, so variaveis

    # ─── 4. CRIA E RODA DISCOVERY AGENT ───
    print(f"\n[3/5] Inicializando Discovery Agent...")
    print(f"  Populacao: 50 individuos")
    print(f"  Variaveis: {n_vars} (nomes opacos)")
    print(f"  Operacoes: +, -, *, /, neg, abs, sqrt, exp, log, sigmoid, sin, cos")
    print()

    agent = DiscoveryAgent(
        n_vars=n_vars,
        var_names=var_names,
        population_size=50,
        max_depth=4,
        const_range=3.0,
        mutation_rate=0.3,
        crossover_rate=0.2,
        tournament_size=3,
        seed=42,
    )

    agent.initialize_population()
    print("  Populacao inicial criada.")

    n_generations = 20
    print(f"\n[4/5] Evoluindo por {n_generations} geracoes...")
    print(f"  {'Ger':>3} | {'Melhor':>8} | {'Media':>8} | {'Pior':>8} | {'Complex':>6}")
    print(f"  {'-'*42}")

    for gen in range(n_generations):
        agent.evolve_one_generation(
            ground_truth_blackbox,
            initial_state,
            t_span=(0, 100),
            n_steps=50,
            elite_size=2,
        )

        best = agent.get_best_model()
        if gen % 2 == 0 or gen == n_generations - 1:
            fitnesses = [m.fitness for m in agent.population]
            print(f"  {gen:3d} | {best.fitness:8.3f} | {np.mean(fitnesses):8.3f} | {min(fitnesses):8.3f} | {best.complexity():6d}")

    # ─── 5. RESULTADOS ───
    print(f"\n[5/5] Resultados da descoberta genuina...")

    best_model = agent.get_best_model()
    print(f"\n  Melhor fitness: {best_model.fitness:.4f}")
    print(f"  Complexidade: {best_model.complexity()} nos")

    # Mostra as equacoes descobertas (formato textual)
    print(f"\n  Equacoes descobertas pela AGI:")
    for i, eq in enumerate(best_model.equations):
        eq_str = str(eq)[:80]
        print(f"    d{var_names[i]}/dt = {eq_str}")

    # ─── 6. VALIDACAO CRUZADA ───
    print(f"\n  Validacao: comparando modelo descoberto vs ground truth...")

    # Simula modelo descoberto
    discovered_traj = best_model.simulate(initial_state, (0, 100), 50)

    # Calcula erro por variavel
    print(f"\n  {'Variavel':>10} | {'Erro (MSE)':>10} | {'Correlacao':>10}")
    print(f"  {'-'*35}")
    for i in range(n_vars):
        mse = np.mean((gt_traj[i] - discovered_traj[i]) ** 2)
        if np.std(gt_traj[i]) > 1e-8 and np.std(discovered_traj[i]) > 1e-8:
            corr = np.corrcoef(gt_traj[i], discovered_traj[i])[0, 1]
            corr = corr if not np.isnan(corr) else 0.0
        else:
            corr = 0.0
        print(f"  {var_names[i]:>10} | {mse:10.4f} | {corr:10.3f}")

    # ─── 7. MAPEAMENTO VARIAVEL ───
    print(f"\n  Matriz de similaridade (modelo x ground truth):")
    sim_matrix = agent.get_model_similarity_matrix(
        best_model, ground_truth_blackbox, initial_state
    )
    print(f"  Modelo: {sim_matrix.shape[0]} vars, GT: {sim_matrix.shape[1]} vars")
    for i in range(min(sim_matrix.shape[0], 8)):
        best_gt = np.argmax(sim_matrix[i])
        best_val = sim_matrix[i, best_gt]
        if best_val > 0.3:
            print(f"    {var_names[i]} -> GT var {best_gt} (corr={best_val:.3f})")

    # ─── 8. PROPOSTA DE INTERVENCAO ───
    print(f"\n  Proposta de intervencao baseada no modelo descoberto:")
    healthy_state = gt_traj[:, 0].copy()
    disease_state = gt_traj[:, -1].copy()

    interventions = agent.propose_interventions(
        best_model, healthy_state, disease_state, ground_truth_blackbox
    )

    for i, interv in enumerate(interventions[:5]):
        print(f"    {i+1}. {interv['target']}: improvement={interv['improvement']:.3f}")

    # ─── 9. TESTA INTERVENCAO NO GROUND TRUTH ───
    if interventions:
        best_interv = interventions[0]
        print(f"\n  Testando melhor intervencao no ground truth REAL...")

        # Aplica intervencao
        var_idx = best_interv["variable_index"]
        species_name = target_info.species_names[var_idx]
        healthy_val = healthy_state[var_idx]

        gt_result = gt.simulate(target_model,
                                  {species_name: float(healthy_val)},
                                  t_end=100, n_points=50)

        if gt_result is not None:
            final_state = gt_result[1:, -1]
            improvement = 1.0 - np.mean(np.abs(final_state - healthy_state) / (np.abs(healthy_state) + 0.01))
            print(f"    Alvo: {species_name} = {healthy_val:.3f}")
            print(f"    Melhora real: {improvement:.3f}")
            print(f"    Predicao do modelo: {best_interv['improvement']:.3f}")
            print(f"    Erro de predicao: {abs(improvement - best_interv['improvement']):.3f}")

    # ─── 10. RELATORIO FINAL ───
    print(f"\n{'='*70}")
    print(f"  RESUMO DA DESCOBERTA HONESTA")
    print(f"{'='*70}")
    print(f"  Ground truth: {target_model} ({available[target_model]})")
    print(f"  Geracoes: {n_generations}")
    print(f"  Populacao: {agent.population_size}")
    print(f"  Melhor fitness: {best_model.fitness:.4f}")
    print(f"  Complexidade: {best_model.complexity()} nos")
    print(f"  Erro medio por variavel: {np.mean([np.mean((gt_traj[i] - discovered_traj[i])**2) for i in range(n_vars)]):.4f}")
    print(f"  Intervencao proposta: {interventions[0]['target'] if interventions else 'N/A'}")
    print(f"  A AGI descobriu {len([i for i in range(n_vars) if sim_matrix[i].max() > 0.3])} de {n_vars} variaveis")
    print(f"{'='*70}")
    print()
    print(f"  Conclusao: {'A AGI comecou a descobrir relacoes causais' if best_model.fitness > -2 else 'A AGI precisa de mais geracoes'}")
    print(f"  Tempo total: {time.time() - start_time:.1f}s" if 'start_time' in dir() else "")


if __name__ == "__main__":
    start_time = time.time()
    run_honest_discovery()
    print(f"\n  Pipeline completo em {time.time() - start_time:.1f}s")
