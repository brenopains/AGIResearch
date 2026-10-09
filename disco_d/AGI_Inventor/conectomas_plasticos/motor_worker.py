"""Processo separado: busca existente, apenas exemplos disponíveis e estado neural."""
import runtime
import hashlib
import json
import sys
from pathlib import Path
from contratos import digest


def verify():
    base = Path(__file__).resolve().parent / 'motor_snapshot'
    manifest = json.loads((base / 'manifesto.json').read_text(encoding='utf-8'))
    for name, info in manifest['files'].items():
        if digest(base / name) != info['sha256']:
            raise ValueError('Componente do motor alterado')
    return manifest


def main():
    verify()
    from motor_snapshot.substrato import resolver
    request = json.load(sys.stdin)
    x, y = request['observations'], request['outcomes']
    if len(x) != len(y) or not 6 <= len(x) <= 512 or not 1 <= len(x[0]) <= 16:
        raise ValueError('Experiência fora dos limites')
    if any(len(row) != len(x[0]) for row in x) or any(value not in (-1, 1) for value in y):
        raise ValueError('Exemplos inválidos')
    result = resolver(x, y, len(x[0]), [-.5, 0., .5], [], t0=2)
    program = result['programa']
    def encode(p):
        return {'op': p.op, 'args': [encode(a) if hasattr(a, 'op') else a for a in p.args]}
    body = encode(program)
    body_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    print(json.dumps({'program': body, 'sha256': body_hash, 'description': str(program),
                      'training_accuracy': result['ajuste'], 'candidates_evaluated': result['n_avaliados'],
                      'cost': result['custo'], 'scope': 'existing_signal_grammar_not_general_program_invention'}))


if __name__ == '__main__':
    main()
