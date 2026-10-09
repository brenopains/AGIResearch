"""Read-only integration of the copied Acervo implementation with real events.

Each execution owns a fresh memory store. Event persistence is an audit trail,
not consolidation into a shared agent memory. No experiment entrypoint is run.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys
from typing import Any, Callable

MODULE_ID = 'novaphys.acervo'
SOURCE_ROOT = Path(__file__).resolve().parents[1] / 'source'
DIMENSION = 2000
MAX_ITEMS = 6
MAX_CONTENT = 600
VECTOR_SAMPLE = 32

CAPABILITY = {
    'id': 'acervo.memory',
    'label': 'Memória associativa',
    'description': (
        'Guarda conteúdo e léxico em um Acervo real e recupera as memórias por '
        'similaridade de trigramas. Cada execução começa com memória vazia.'
    ),
    'module_id': MODULE_ID,
    'input_schema': {
        'type': 'object',
        'properties': {
            'items': {
                'type': 'string',
                'title': 'Memórias (uma por linha, rótulo | conteúdo)',
                'maxLength': 4500,
                'description': 'Até 6 memórias; conteúdo de 3 a 600 caracteres por linha.',
            },
            'query': {
                'type': 'string',
                'title': 'Consulta',
                'minLength': 3,
                'maxLength': 600,
            },
        },
        'required': ['items', 'query'],
        'additionalProperties': False,
    },
}


def _validate(input_data: Any) -> tuple[list[tuple[str, str]], str]:
    if not isinstance(input_data, dict):
        raise ValueError('A entrada precisa conter memórias e consulta.')
    if set(input_data) - {'items', 'query'}:
        raise ValueError('Campos aceitos: items e query.')
    raw, query = input_data.get('items'), input_data.get('query')
    if not isinstance(raw, str) or not isinstance(query, str):
        raise ValueError('Memórias e consulta precisam ser texto.')
    if len(raw) > 4500 or len(query) > MAX_CONTENT:
        raise ValueError('Entrada acima do limite: 4500 caracteres de memórias e 600 na consulta.')
    query = query.strip()
    if len(query) < 3:
        raise ValueError('A consulta precisa ter ao menos 3 caracteres para gerar trigramas.')
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not 1 <= len(lines) <= MAX_ITEMS:
        raise ValueError('Informe de 1 a 6 memórias, uma por linha.')
    parsed = []
    for index, line in enumerate(lines, 1):
        if '|' not in line:
            raise ValueError(f'Memória {index}: use rótulo | conteúdo.')
        label, content = (part.strip() for part in line.split('|', 1))
        if not 1 <= len(label) <= 120:
            raise ValueError(f'Memória {index}: o rótulo deve ter de 1 a 120 caracteres.')
        if not 3 <= len(content) <= MAX_CONTENT:
            raise ValueError(f'Memória {index}: o conteúdo deve ter de 3 a 600 caracteres.')
        parsed.append((label, content))
    return parsed, query


def _load_real_modules():
    source = SOURCE_ROOT.resolve()
    if not (source / 'novaphys' / 'acervo.py').is_file():
        raise RuntimeError('A cópia de novaphys.acervo não está disponível.')
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    acervo = importlib.import_module('novaphys.acervo')
    hdc = importlib.import_module('novaphys.hdc_indice')
    for module in (acervo, hdc):
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(source):
            raise RuntimeError('O módulo carregado não pertence à cópia isolada do projeto.')
    return acervo, hdc


def _source(module, symbol: str) -> dict:
    path = Path(module.__file__).resolve()
    return {
        'mode': 'runtime',
        'path': path.relative_to(SOURCE_ROOT.resolve()).as_posix(),
        'symbol': symbol,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'adapter': 'acervo.memory',
    }


def _vector(vector) -> dict:
    sample = vector[:VECTOR_SAMPLE]
    return {
        'dim': int(vector.size),
        'dtype': str(vector.dtype),
        'sha256': hashlib.sha256(vector.tobytes(order='C')).hexdigest(),
        'sample': [int(value) for value in sample],
        'sampling': {
            'policy': 'first_dimensions',
            'captured_count': len(sample),
            'total_count': int(vector.size),
            'meaning': 'Amostra das coordenadas do vetor; não é uma projeção 3D.',
        },
    }


def run(input_data: dict, emit: Callable, check_cancel: Callable) -> dict:
    """Store and rank bounded user content using Acervo's actual methods."""
    parsed, query = _validate(input_data)
    check_cancel()
    acervo_module, hdc = _load_real_modules()
    space = hdc.Espaco(dim=DIMENSION)
    memory = acervo_module.Acervo(space)
    source_store = _source(acervo_module, 'Acervo.guarda')
    source_rank = _source(acervo_module, 'Acervo.ordena')
    source_vector = _source(hdc, 'de_texto / sim')
    scope = (
        'Acervo local desta execução. Os eventos serão preservados para inspeção; '
        'isso não treina nem atualiza uma memória global do agente.'
    )
    start = emit(
        'memory.started', 'Acervo iniciado', scope,
        data={'scope': 'run', 'items_requested': len(parsed), 'dimension': DIMENSION,
              'extraction': {'rounds': 20, 'candidates': 8},
              'persistence': 'events_only'},
        module_id=MODULE_ID, source=source_store,
    )
    parent = start['id']
    snapshots, ids = [], {}
    for index, (label, content) in enumerate(parsed, 1):
        check_cancel()
        memory.guarda(label, content, rodadas=20, cand=8)
        check_cancel()
        item = memory.itens[-1]
        item_id = f'memory:{index}'
        ids[id(item)] = item_id
        snapshot = {
            'id': item_id,
            'rotulo': item['rotulo'],
            'bruto': item['bruto'],
            'lexico': sorted(item['lexico']),
            'conta': {token: int(count) for token, count in sorted(item['conta'].items())},
            'vector': _vector(item['v']),
        }
        snapshots.append(snapshot)
        emit(
            'memory.stored', f'Memória guardada: {label}',
            f'{len(content)} caracteres; {len(snapshot["lexico"])} entradas compostas no léxico.',
            data=snapshot,
            metrics={'characters': len(content), 'lexicon_entries': len(snapshot['lexico']),
                     'tokens': sum(snapshot['conta'].values()), 'vector_dimensions': DIMENSION},
            module_id=MODULE_ID, parent_id=parent, source=source_store,
        )
    check_cancel()
    query_vector = hdc.de_texto(space, query)
    query_event = emit(
        'memory.query', 'Consulta codificada',
        'A consulta é representada por trigramas de caracteres e comparada aos vetores originais.',
        data={'id': 'query', 'text': query, 'encoder': 'character_trigrams',
              'ngram_size': 3, 'vector': _vector(query_vector)},
        module_id='novaphys.hdc_indice', parent_id=parent, source=source_vector,
    )
    ranked = memory.ordena(query)
    ranking = []
    for rank, item in enumerate(ranked, 1):
        check_cancel()
        similarity = float(hdc.sim(query_vector, item['v']))
        record = {
            'rank': rank, 'memory_id': ids[id(item)], 'rotulo': item['rotulo'],
            'similarity': similarity, 'metric': 'bipolar_cosine',
            'characters': len(item['bruto']), 'lexicon_entries': len(item['lexico']),
        }
        ranking.append(record)
        emit(
            'relation.observed', f'{rank}. {item["rotulo"]}',
            f'Similaridade cosseno no vetor bipolar original: {similarity:.4f}.',
            data={'source': 'query', 'target': ids[id(item)], 'relation': 'similar_to',
                  'rank': rank, 'score': similarity, 'score_name': 'bipolar_cosine',
                  'range': [-1, 1], 'formula': 'dot(a, b) / dimension',
                  'is_probability': False},
            metrics={'similarity': similarity},
            module_id='novaphys.hdc_indice', parent_id=query_event['id'], source=source_vector,
        )
    explanation = (
        f'A consulta ordenou {len(ranking)} memórias usando trigramas de caracteres. '
        f'A primeira foi “{ranking[0]["rotulo"]}”, com similaridade '
        f'{ranking[0]["similarity"]:.4f} no espaço bipolar de {DIMENSION} dimensões. '
        'Esse valor mede sobreposição na representação textual; não é probabilidade '
        'nem demonstra compreensão semântica. ' + scope
    )
    emit(
        'artifact.created', 'Ranking de memórias', explanation,
        data={'artifact': {'type': 'table', 'title': 'Recuperação associativa',
                           'columns': ['rank', 'rotulo', 'similarity', 'characters', 'lexicon_entries'],
                           'rows': ranking},
              'metric': {'name': 'bipolar_cosine', 'unit': 'dimensionless', 'range': [-1, 1],
                         'is_probability': False, 'projection_used': False}},
        module_id=MODULE_ID, parent_id=query_event['id'], source=source_rank,
    )
    result = {
        'summary': explanation, 'output': explanation, 'scope': 'run',
        'persistence': 'events_only', 'memory_count': len(snapshots),
        'dimension': DIMENSION, 'query': query, 'memories': snapshots,
        'ranking': ranking, 'metric': 'bipolar_cosine',
    }
    emit(
        'state.snapshot', 'Estado final do acervo', scope,
        data=result, metrics={'memory_items': len(snapshots)},
        module_id=MODULE_ID, parent_id=parent, source=source_rank,
    )
    return result
