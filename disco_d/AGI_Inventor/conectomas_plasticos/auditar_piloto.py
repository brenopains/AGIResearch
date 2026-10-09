"""Recalcular procedência e contagens; não é revisão independente por outro agente."""
import runtime
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from contratos import digest
from importar import rows
from circuito import Circuito

HERE = Path(__file__).resolve().parent


def audit(output):
    output = Path(output)
    manifest = json.loads((output/'manifesto.json').read_text(encoding='utf-8'))
    result = json.loads((output/'resultado.json').read_text(encoding='utf-8'))
    if digest(output/'manifesto.json') != result['manifest_sha256'] or digest(output/'eventos.jsonl') != result['events_sha256']:
        raise AssertionError('Resultado não corresponde ao manifesto/eventos')
    for name, expected in manifest['sources'].items():
        if digest(name) != expected:
            raise AssertionError('Fonte alterada: ' + name)
    checks = []
    specs = {'human_h01': 'h01_c3_shard0.avro', 'mouse_microns': 'microns_v117_prefix.csv', 'flywire': 'flywire_v783.feather'}
    for source, filename in specs.items():
        graph = json.loads((HERE/'grafos'/f'{source}.json').read_text(encoding='utf-8'))
        path = HERE/'dados'/filename
        receipt = json.loads(Path(str(path)+'.receipt.json').read_text(encoding='utf-8'))
        if digest(path) != receipt['sha256'] or receipt['sha256'] != graph['provenance']['file_sha256']:
            raise AssertionError('Procedência não confere')
        selected = {int(x.split(':')[1]) for x in graph['node_ids']}
        recomputed = Counter()
        for pre, post, count in rows(path, source, graph['provenance']['row_limit']):
            if pre in selected and post in selected and pre > 0 and post > 0 and count > 0:
                recomputed[(pre, post)] += count
        expected = {(int(graph['node_ids'][a].split(':')[1]), int(graph['node_ids'][b].split(':')[1])): c
                    for a, b, c in zip(graph['pre'], graph['post'], graph['counts'])}
        if dict(recomputed) != expected:
            raise AssertionError('Arestas importadas não correspondem aos dados reais')
        checks.append({'source': source, 'nodes': len(selected), 'edges_recomputed': len(recomputed),
                       'synapses': sum(recomputed.values()), 'verified_against_raw': True})
    events = [json.loads(s) for s in (output/'eventos.jsonl').read_text(encoding='utf-8').splitlines()]
    if [x['seq'] for x in events] != list(range(len(events))):
        raise AssertionError('Linha do tempo quebrada')
    for row in result['rows']:
        path = output/f"{row['domain']}_{row['variant']}.npz"
        if digest(path) != row['checkpoint_sha256']:
            raise AssertionError('Executor trocado')
        body = Circuito.load(path)
        if body.grown_total != row['grown']:
            raise AssertionError('Contagem de conexões fictícia')
        if row['variant'] == 'coupled' and row['restored_mse'] != row['mse']:
            raise AssertionError('Restauração alterou o resultado')
    out = {'passed': True, 'status': result['status'], 'scope': 'source_and_event_consistency_audit',
           'independent_reviewer': False, 'graphs': checks, 'event_counts': dict(Counter(x['kind'] for x in events)),
           'human_level_demonstrated': False}
    with (output/'auditoria.json').open('x', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    audit(sys.argv[1])
