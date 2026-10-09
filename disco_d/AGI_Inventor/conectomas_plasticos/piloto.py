"""Piloto de infraestrutura multiespécie, não teste de AGI ou de eficiência humana.

Dados de experiência sintéticos explícitos: pixels de órbita e amostras de áudio.
Conectividade é real. Genoma selecionado somente em desenvolvimento.
"""
import runtime
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import time
import numpy as np
from circuito import Circuito, Genoma, mutar
from contratos import Diario, Feedback, digest
from importar import fuse, rewire
from limites import Budget, constrain
from ponte import Ponte

HERE = Path(__file__).resolve().parent


def experience(domain, seed, frames):
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi)
    speed = rng.uniform(.07, .19)
    observations = []
    for t in range(frames):
        if domain == 'pixels_orbita':
            frame = np.full((16, 16), -1.)
            x = int(round(7.5 + 5 * np.cos(phase + speed * t)))
            y = int(round(7.5 + 4 * np.sin(phase + speed * t)))
            frame[max(0, y-1):min(16, y+2), max(0, x-1):min(16, x+2)] = 1
            observations.append(frame.ravel())
        elif domain == 'audio_tons':
            samples = np.arange(256) / 256
            amplitude = .5 + .2 * np.sin(phase + t * speed)
            wave = amplitude * np.sin(2*np.pi * (3.0 + .3 * np.cos(t * speed)) * samples + phase + t * speed)
            wave += .2 * np.cos(2*np.pi * 7 * samples + phase - t * speed)
            observations.append(wave)
        else:
            raise ValueError(domain)
    return np.asarray(observations)


def train(circuit, observations, budget, ponte=None, plastic=True, growth=True, communicating=True):
    circuit.reset()
    states, losses = [], []
    for current, following in zip(observations[:-1], observations[1:]):
        budget.check()
        circuit.step(current, communicating=communicating)
        states.append(circuit.x.copy())
        if ponte:
            losses.append(ponte.realimentar(circuit, Feedback(following.tolist(), None, 'observed_next_raw_sample'),
                                          plastic=plastic, growth=growth))
        else:
            losses.append(circuit.feedback(following, plastic=plastic, growth=growth))
    return np.asarray(states), losses


def evaluate(circuit, observations, budget, communicating=True):
    circuit.reset()
    states, errors = [], []
    for current, following in zip(observations[:-1], observations[1:]):
        budget.check()
        prediction = circuit.step(current, communicating=communicating)
        states.append(circuit.x.copy())
        errors.append(float(np.mean((prediction - following)**2)))
        circuit.pending = False  # Sem aprendizagem; previsão já foi medida pelo avaliador.
    return float(np.mean(errors)), np.asarray(states)


def evolve(graph, budget, journal):
    rng = np.random.default_rng(20261003)
    population = [Genoma()] + [mutar(Genoma(), rng) for _ in range(2)]
    history = []
    best = population[0]
    for generation in range(2):
        scored = []
        for index, genome in enumerate(population):
            errors, costs = [], []
            for domain in ('pixels_orbita', 'audio_tons'):
                budget.check()
                circuit = Circuito(graph, 256, 256, genome, seed=17)
                train(circuit, experience(domain, 31, 49), budget)
                loss, _ = evaluate(circuit, experience(domain, 43, 25), budget)
                errors.append(loss)
                costs.append(circuit.cost)
            fitness = float(np.mean(errors))
            record = {'generation': generation, 'index': index, 'mse': fitness, 'domain_mse': errors,
                      'genome': asdict(genome), 'cost': costs, 'split': 'development_only'}
            history.append(record)
            journal.event('genome_evaluated', **record)
            scored.append((fitness, index, genome))
        scored.sort(key=lambda x: (x[0], x[1]))
        best = scored[0][2]
        journal.event('genome_selected', generation=generation, development_mse=scored[0][0], genome=asdict(best))
        # Elitismo + um indivíduo de diversidade, mesmo sem ganho imediato.
        population = [best, mutar(best, rng), mutar(Genoma(), rng)]
    return best, history


def run(output, seconds=120):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    limits = constrain()
    budget = Budget(seconds)
    journal = Diario(output / 'eventos.jsonl')
    source_files = sorted(HERE.glob('*.py'))
    files = [HERE/'grafos'/f'{s}.json' for s in ('human_h01', 'mouse_microns', 'flywire')]
    manifest = {'schema': 1, 'scope': 'infrastructure_pilot_not_AGI_confirmation', 'limits': limits,
                'deadline_seconds': seconds, 'sources': {str(p): digest(p) for p in source_files + files},
                'development_seeds': [31, 43], 'evaluation_seeds': [191, 193],
                'synthetic_experience': True, 'biological_connectivity': True,
                'structural_families_held_out': False, 'cost_matched_species_comparison': False,
                'dynamics_biologically_calibrated': False,
                'stage': 'build_and_wiring_check_not_generalization_gate'}
    (output/'manifesto.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    result = {'status': 'running', 'human_level_demonstrated': False, 'rows': []}
    start = time.monotonic()
    try:
        graphs = [json.loads(p.read_text(encoding='utf-8')) for p in files]
        fused = fuse(graphs, bridge_pairs=8, seed=2)
        (output/'grafo_acoplado.json').write_text(json.dumps(fused, indent=2), encoding='utf-8')
        genome, history = evolve(fused, budget, journal)
        frozen = {'genome': asdict(genome), 'history': history, 'evaluation_started': False}
        genome_file = output/'genoma_congelado.json'
        genome_file.write_text(json.dumps(frozen, indent=2), encoding='utf-8')
        genome_sha = digest(genome_file)
        journal.event('genome_frozen', sha256=genome_sha)
        # A partir deste ponto a seleção externa não recebe resultados de avaliação.
        ponte = Ponte(journal)
        variants = [(name, g, True, True, True) for name, g in zip(('human_h01', 'mouse_microns', 'flywire'), graphs)]
        variants += [('coupled', fused, True, True, True), ('no_plasticity', fused, False, False, True),
                     ('no_growth', fused, True, False, True), ('no_communication', fused, True, True, False),
                     ('rewired', rewire(fused, seed=13), True, True, True)]
        for domain, variant in itertools.product(('pixels_orbita', 'audio_tons'), variants):
            budget.check()
            name, graph, plastic, growth, communicating = variant
            circuit = Circuito(graph, 256, 256, genome, seed=17, diario=journal)
            train_obs = experience(domain, 191, 97)
            test_obs = experience(domain, 193, 49)
            train_states, losses = train(circuit, train_obs, budget, ponte, plastic, growth, communicating)
            checkpoint = output / f'{domain}_{name}.npz'
            circuit.save(checkpoint)
            mse, test_states = evaluate(circuit, test_obs, budget, communicating)
            persistence = float(np.mean((test_obs[1:] - test_obs[:-1])**2))
            row = {'domain': domain, 'variant': name, 'mse': mse, 'persistence_mse': persistence,
                   'last_training_mse': float(np.mean(losses[-16:])), 'nodes': circuit.n, 'edges': len(circuit.w),
                   'grown': circuit.grown_total, 'cost': circuit.cost, 'checkpoint_sha256': digest(checkpoint)}
            if name == 'coupled':
                # Sonda pública, não estado oculto: previsão do sinal de uma projeção de amostras futuras observadas.
                projection = np.random.default_rng(83).normal(0, 1, 256)
                values = train_obs[1:] @ projection
                threshold = float(np.median(values))
                labels = np.where(values > threshold, 1, -1)
                test_labels = np.where(test_obs[1:] @ projection > threshold, 1, -1)
                # Portas fixas neste primeiro piloto; não afirmar alinhamento semântico aprendido.
                ports = np.linspace(0, circuit.n-1, 8).astype(int)
                proposal = ponte.propor(train_states[:, ports], labels, timeout=min(15, max(1, budget.deadline-time.monotonic())))
                predicted = ponte.executar(proposal, test_states[:, ports])
                row['motor_probe_accuracy'] = float(np.mean(np.asarray(predicted)[2:] == test_labels[2:]))
                row['majority_probe_accuracy'] = float(np.mean(test_labels[2:] == (1 if np.sum(labels == 1) >= np.sum(labels == -1) else -1)))
                row['motor_probe_scope'] = 'fixed_random_observable_projection_not_game_solving'
                ponte.guardar(f'{domain}_representation', train_states[:, ports], {'checkpoint': str(checkpoint), 'proposal': proposal})
                recovered = ponte.recuperar(test_states[:, ports])
                row['retrieved_items'] = len(recovered)
                # Mesmo executor reaplicado ao outro domínio, sem reescrita nem atualização.
                other = 'audio_tons' if domain == 'pixels_orbita' else 'pixels_orbita'
                transfer = Circuito.load(checkpoint)
                before_hash = digest(checkpoint)
                row['cross_domain_reuse_mse'], _ = evaluate(transfer, experience(other, 197, 33), budget)
                journal.event('executor_reused', executor_sha256=before_hash, source_domain=domain, target_domain=other,
                              body_rewritten=False, outcome_mse=row['cross_domain_reuse_mse'], demonstrates_transfer_gain=False)
                original = Circuito.load(checkpoint)
                original_mse, _ = evaluate(original, test_obs, budget)
                removed = Circuito.load(checkpoint)
                removed.remove_module(0)
                removed_mse, _ = evaluate(removed, test_obs, budget)
                restored = Circuito.load(checkpoint)
                restored_mse, _ = evaluate(restored, test_obs, budget)
                if original_mse != restored_mse or digest(checkpoint) != before_hash:
                    raise AssertionError('Restauração não reproduziu executor')
                row['module_0_removed_mse'] = removed_mse
                row['restored_mse'] = restored_mse
                journal.event('module_intervention', domain=domain, removed=0, original_mse=original_mse,
                              removed_mse=removed_mse, restored_mse=restored_mse)
            result['rows'].append(row)
            journal.event('pilot_row', **row)
            (output/'checkpoint_resultado.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(json.dumps({'domain': domain, 'variant': name, 'mse': mse, 'persistence_mse': persistence, 'grown': circuit.grown_total}), flush=True)
        if digest(genome_file) != genome_sha:
            raise AssertionError('Genoma mudou depois do congelamento')
        result['status'] = 'complete'
        result['genome_sha256'] = genome_sha
    except TimeoutError as e:
        result['status'] = 'inconclusive_budget_exhausted'
        result['error'] = str(e)
    except Exception as e:
        result['status'] = 'failed_infrastructure'
        result['error'] = type(e).__name__ + ': ' + str(e)
        raise
    finally:
        result['wall_seconds'] = time.monotonic() - start
        journal.event('pilot_end', status=result['status'], wall_seconds=result['wall_seconds'], human_level_demonstrated=False)
        journal.close()
        result['events_sha256'] = digest(output/'eventos.jsonl')
        result['manifest_sha256'] = digest(output/'manifesto.json')
        (output/'resultado.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=120)
    a = p.parse_args()
    result = run(a.out, a.seconds)
    print(json.dumps({'status': result['status'], 'rows': len(result['rows']), 'wall_seconds': result['wall_seconds'],
                      'human_level_demonstrated': False}), flush=True)
