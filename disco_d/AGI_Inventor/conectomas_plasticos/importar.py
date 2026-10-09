"""Importar conectividade real com limites explícitos de leitura e seleção."""
import runtime
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from contratos import digest

HERE = Path(__file__).resolve().parent


def rows(path, source, limit):
    if source == 'human_h01':
        from fastavro import reader
        with path.open('rb') as f:
            for i, row in enumerate(reader(f)):
                if i >= limit:
                    break
                # Parceiros não associados a um segmento não viram neurônios fictícios.
                yield int(row['pre_synaptic_site']['neuron_id'] or 0), int(row['post_synaptic_partner']['neuron_id'] or 0), 1
    elif source == 'mouse_microns':
        # v117 oficial é CSV SEM cabeçalho; índices documentados no repositório estático.
        with path.open(encoding='utf-8', newline='') as f:
            for i, row in enumerate(csv.reader(f)):
                if i >= limit:
                    break
                if len(row) != 16:
                    raise ValueError(f'Linha MICrONS incompatível: {len(row)} colunas')
                if row[1] != 't':
                    continue
                yield int(row[6]), int(row[11]), 1
    elif source == 'flywire':
        import pyarrow as pa
        with pa.memory_map(str(path), 'r') as file:
            reader = pa.ipc.open_file(file)
            columns = ['pre_pt_root_id', 'post_pt_root_id', 'syn_count']
            if not set(columns) <= set(reader.schema.names):
                raise ValueError('Esquema FlyWire incompatível')
            remaining = limit
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i).slice(0, min(remaining, reader.get_batch(i).num_rows))
                values = [batch.column(reader.schema.get_field_index(k)).to_pylist() for k in columns]
                for pre, post, count in zip(*values):
                    yield int(pre), int(post), int(count)
                remaining -= batch.num_rows
                if remaining <= 0:
                    break
    else:
        raise ValueError(source)


def build_graph(records, source, node_limit, provenance):
    if not 2 <= node_limit <= 4096:
        raise ValueError('Limite de nós inválido')
    edges = Counter()
    scanned = invalid = 0
    for pre, post, count in records:
        scanned += 1
        if pre <= 0 or post <= 0 or count <= 0:
            invalid += 1
            continue
        edges[(pre, post)] += count
    if not edges:
        raise ValueError('Nenhuma conexão válida')
    neighbors = defaultdict(set)
    degree = Counter()
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
        degree[a] += 1
        degree[b] += 1
    # Escolha determinística de um componente a partir do maior grau; não amostra aleatória.
    start = min(degree, key=lambda x: (-degree[x], x))
    selected, frontier = set(), {start}
    while frontier and len(selected) < node_limit:
        node = min(frontier, key=lambda x: (-degree[x], x))
        frontier.remove(node)
        selected.add(node)
        frontier.update(neighbors[node] - selected)
    ids = sorted(selected)
    index = {v: k for k, v in enumerate(ids)}
    kept = sorted((a, b, c) for (a, b), c in edges.items() if a in selected and b in selected)
    cut = sum(c for (a, b), c in edges.items() if (a in selected) != (b in selected))
    return {'schema': 1, 'node_ids': [f'{source}:{x}' for x in ids],
            'pre': [index[a] for a, b, c in kept], 'post': [index[b] for a, b, c in kept],
            'counts': [c for a, b, c in kept], 'modules': [0] * len(ids), 'edge_measured': [True] * len(kept),
            'provenance': {**provenance, 'source': source, 'scanned_rows': scanned, 'invalid_rows': invalid,
                           'requested_node_limit': node_limit, 'selected_nodes': len(ids), 'selected_edges': len(kept), 'selected_synapses': sum(c for _, _, c in kept),
                           'cut_synapses_observed_in_sample': cut, 'all_external_connections_known': False,
                           'selection': 'highest_degree_weak_component_within_biased_row_prefix',
                           'measured_neuron_dynamics': False, 'node_identity_scope': 'reconstruction_root_or_segment'}}


def fuse(graphs, bridge_pairs=8, seed=0):
    if not graphs:
        raise ValueError('Lista vazia')
    out = {'schema': 1, 'node_ids': [], 'pre': [], 'post': [], 'counts': [], 'modules': [], 'edge_measured': [],
           'provenance': {'source': 'coupled', 'components': [g['provenance'] for g in graphs],
                          'bridge_initialization': 'seeded_sparse_ports_artificial_and_plastic'}}
    offsets, offset = [], 0
    for module, g in enumerate(graphs):
        offsets.append(offset)
        out['node_ids'].extend(g['node_ids'])
        out['pre'].extend(offset + p for p in g['pre'])
        out['post'].extend(offset + p for p in g['post'])
        out['counts'].extend(g['counts'])
        out['modules'].extend([module] * len(g['node_ids']))
        out['edge_measured'].extend(g.get('edge_measured', [True] * len(g['pre'])))
        offset += len(g['node_ids'])
    if len(set(out['node_ids'])) != len(out['node_ids']):
        raise ValueError('Identidades sobrepostas; usar namespaces distintos')
    rng = np.random.default_rng(seed)
    bridges = set()
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            for _ in range(bridge_pairs):
                a = offsets[i] + int(rng.integers(len(graphs[i]['node_ids'])))
                b = offsets[j] + int(rng.integers(len(graphs[j]['node_ids'])))
                bridges.update([(a, b), (b, a)])
    for a, b in sorted(bridges):
        out['pre'].append(a)
        out['post'].append(b)
        out['counts'].append(1)
        out['edge_measured'].append(False)
    out['provenance']['artificial_bridges'] = len(bridges)
    return out


def rewire(graph, seed=0, attempts_per_edge=5):
    """Trocas dirigidas preservam graus de entrada/saída; não preservam força ponderada."""
    out = json.loads(json.dumps(graph))
    rng = np.random.default_rng(seed)
    pairs = list(zip(out['pre'], out['post']))
    present = set(pairs)
    swaps = 0
    for _ in range(len(pairs) * attempts_per_edge):
        if len(pairs) < 2:
            break
        i, j = rng.choice(len(pairs), 2, replace=False)
        a, b = pairs[i]
        c, d = pairs[j]
        if a == c or b == d or a == d or c == b or (a, d) in present or (c, b) in present:
            continue
        present.remove((a, b))
        present.remove((c, d))
        pairs[i], pairs[j] = (a, d), (c, b)
        present.update([pairs[i], pairs[j]])
        swaps += 1
    out['pre'] = [a for a, b in pairs]
    out['post'] = [b for a, b in pairs]
    out['edge_measured'] = [False] * len(pairs)
    out['provenance'] = {'source': 'degree_preserving_rewired_control', 'original': graph['provenance'], 'swaps': swaps,
                         'weighted_strength_preserved': False}
    return out


def main(node_limit=128, row_limit=200000):
    specs = {'flywire': ('flywire_v783.feather', '783'), 'human_h01': ('h01_c3_shard0.avro', 'c3_20210729_shard0'),
             'mouse_microns': ('microns_v117_prefix.csv', '117')}
    output = HERE / 'grafos'
    output.mkdir(exist_ok=True)
    for source, (name, version) in specs.items():
        path = HERE / 'dados' / name
        receipt = json.loads(Path(str(path) + '.receipt.json').read_text(encoding='utf-8'))
        if digest(path) != receipt['sha256']:
            raise ValueError('Arquivo real alterado: ' + name)
        existing = output / (source + '.json')
        if existing.exists():
            g = json.loads(existing.read_text(encoding='utf-8'))
            if g['provenance']['file_sha256'] != receipt['sha256'] or g['provenance']['row_limit'] != row_limit or g['provenance'].get('requested_node_limit', 128) != node_limit:
                raise ValueError('Grafo existente tem configuração diferente; usar outra pasta')
            print(json.dumps({'source': source, 'status': 'existing_preserved', 'nodes': len(g['node_ids'])}), flush=True)
            continue
        graph = build_graph(rows(path, source, row_limit), source, node_limit,
                            {'file_sha256': receipt['sha256'], 'receipt': receipt, 'version': version,
                             'row_limit': row_limit, 'sampling_bias': 'first_rows_not_representative_of_species'})
        with (output / (source + '.json')).open('x', encoding='utf-8') as f:
            json.dump(graph, f, indent=2)
        print(json.dumps({'source': source, 'nodes': len(graph['node_ids']), 'edges': len(graph['pre']),
                          'synapses': graph['provenance']['selected_synapses'], 'scanned': graph['provenance']['scanned_rows']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--nodes', type=int, default=128)
    p.add_argument('--rows', type=int, default=200000)
    a = p.parse_args()
    main(a.nodes, a.rows)
