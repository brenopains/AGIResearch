"""Diagnóstico posterior. Leitores ajustados só no treino; não nova confirmação.

Com corpo neural congelado, distinguir limitação do leitor online de informação
acessível linearmente ao leitor. Ausência de ganho não prova ausência de informação.
"""
import runtime
import json
from pathlib import Path
import sys
import numpy as np
from circuito import Circuito
from contratos import digest
from piloto import experience


def states(body, observations):
    body.reset()
    result = []
    for current in observations[:-1]:
        body.step(current)
        result.append(body.x.copy())
        body.pending = False
    return np.asarray(result)


def fit_predict(train, target, test, ridge=.1):
    # Ridge dual, computado com exemplos de treino apenas; mesmo lambda declarado.
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.)
    a = np.c_[(train-mu)/sd, np.ones(len(train))]
    b = np.c_[(test-mu)/sd, np.ones(len(test))]
    beta = a.T @ np.linalg.solve(a@a.T + ridge*np.eye(len(a)), target)
    return b @ beta


def run(output):
    output = Path(output)
    frozen = json.loads((output/'manifesto.json').read_text(encoding='utf-8'))
    for name, expected in frozen['sources'].items():
        if digest(name) != expected:
            raise AssertionError('Código/circuitos do piloto alterados')
    rows = []
    for domain in ('pixels_orbita', 'audio_tons'):
        checkpoint = output/f'{domain}_coupled.npz'
        body = Circuito.load(checkpoint)
        training = experience(domain, 191, 97)
        evaluation = experience(domain, 193, 49)
        train_states = states(body, training)
        test_states = states(body, evaluation)
        baseline = float(np.mean((evaluation[1:]-evaluation[:-1])**2))
        for representation, train, test in [('neural', train_states, test_states), ('raw', training[:-1], evaluation[:-1])]:
            for residual in (False, True):
                target = training[1:] - training[:-1] if residual else training[1:]
                pred = fit_predict(train, target, test)
                if residual:
                    pred += evaluation[:-1]
                rows.append({'domain': domain, 'representation': representation, 'residual_reader': residual,
                             'mse': float(np.mean((pred-evaluation[1:])**2)), 'persistence_mse': baseline,
                             'training_observations': 96, 'evaluation_observations': 48})
    result = {'scope': 'post_hoc_linear_reader_diagnostic_not_confirmation', 'ridge_fixed': .1,
              'held_out_outcomes_used_for_fitting': False, 'world_families_held_out': False,
              'human_level_demonstrated': False, 'sources': {str(Path(__file__)): digest(__file__)}, 'rows': rows}
    with (output/'diagnostico_leitor.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run(sys.argv[1])
