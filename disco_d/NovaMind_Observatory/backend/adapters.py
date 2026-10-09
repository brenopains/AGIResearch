"""Allowlisted adapters executing original copied functions, plus explicit historical import."""
from __future__ import annotations
import hashlib
import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from .tracing import SOURCE, SourceTrace, provenance


def object_schema(properties, required=()):
    return dict(type='object',properties=properties,required=list(required),additionalProperties=False)

def integer(title, low, high, default):
    return dict(type='integer',title=title,minimum=low,maximum=high,default=default)

def string(title, maximum=80):
    return dict(type='string',title=title,minLength=1,maxLength=maximum)

CAPABILITIES = [
    dict(id='lingua.segmenta',title='Descobrir unidades em uma sequência',name='Descobrir unidades em uma sequência',
         module_id='novaphys.lingua',mode='live',
         description='Executa segmenta do projeto: propõe fusões de símbolos e aceita quando o custo MDL diminui. Não é conversa nem compreensão de linguagem.',
         input_schema=object_schema({'text':dict(type='string',title='Texto ou sequência de símbolos',minLength=2,maxLength=12000,format='textarea',description='Espaços removidos pelo adaptador; o texto original é preservado.')},['text']),
         parameter_schema=object_schema({'rodadas':integer('Máximo de rodadas',1,100,40),'max_len':integer('Tamanho máximo de unidade',2,24,12),'cand':integer('Candidatos por rodada',1,16,8)}),
         observability=dict(level='instrumented',description='Chamadas originais, candidatos, custos, tokens, léxico, aceites e rejeições observados. Amostras e limites são identificados.')),
    dict(id='cena.hdc',title='Construir e consultar uma cena associativa',name='Construir e consultar uma cena associativa',
         module_id='novaphys.cena',mode='live',
         description='Associa papéis a objetos com hipervetores e recupera um objeto do codebook. A cena é um conjunto de relações, não uma imagem mental produzida pelo sistema.',
         input_schema=object_schema({'pairs':dict(type='array',title='Pares papel / objeto',minItems=1,maxItems=32,
                                                  items=object_schema({'role':string('Papel'),'object':string('Objeto')},['role','object'])),
                                     'query_role':string('Qual papel consultar?'),
                                     'codebook':dict(type='array',title='Objetos candidatos (opcional)',maxItems=128,items=string('Objeto'))},['pairs','query_role']),
         parameter_schema=object_schema({'dim':integer('Dimensões do hipervetor',128,16384,2048)}),
         observability=dict(level='instrumented',description='Pares, amostra explícita e hash do vetor, átomos e ranking completo real. Similaridade não é probabilidade.')),
    dict(id='archive.traco',title='Abrir registro histórico do projeto',name='Abrir registro histórico do projeto',
         module_id='novaphys.nucleo',mode='import',
         description='Importa uma tarefa de traco.json. Registro parcial de busca ARC: não executa novamente a tarefa; o tempo de cada passo é desconhecido.',
         input_schema=object_schema({'task_index':integer('Índice da tarefa gravada',0,100000,0)},['task_index']),
         parameter_schema=object_schema({}),
         observability=dict(level='imported_partial',description='Ordem, número de estados avaliados e gravados. Sem timestamps inventados.')),
]


def get_capabilities():
    result = json.loads(json.dumps(CAPABILITIES))
    path = SOURCE/'traco.json'
    if path.is_file():
        archive = json.loads(path.read_text(encoding='utf-8-sig'))
        field = result[2]['input_schema']['properties']['task_index']
        field.update(maximum=max(0,len(archive)-1),enum=list(range(len(archive))),
                     enumNames=[f"{i}: {item.get('tarefa','sem nome')} · {item.get('gravados',len(item.get('busca',[])))} estados" for i,item in enumerate(archive)])
    else:
        result[2]['available'] = False
    return result


def validate_schema(value,schema,path='input'):
    kind = schema.get('type')
    check = {'object':isinstance(value,dict),'array':isinstance(value,list),'string':isinstance(value,str),
             'integer':isinstance(value,int) and not isinstance(value,bool)}
    if kind in check and not check[kind]:
        raise ValueError(f'{path}: tipo esperado {kind}')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{path}: opção desconhecida')
    if kind=='object':
        props = schema.get('properties',{})
        if schema.get('additionalProperties') is False and set(value)-set(props):
            raise ValueError(f'{path}: campos desconhecidos')
        for key in schema.get('required',[]):
            if key not in value:
                raise ValueError(f'{path}.{key}: obrigatório')
        for key,item in value.items():
            if key in props:
                validate_schema(item,props[key],f'{path}.{key}')
    elif kind=='array':
        if not schema.get('minItems',0)<=len(value)<=schema.get('maxItems',100000):
            raise ValueError(f'{path}: número de itens fora do limite')
        for index,item in enumerate(value):
            validate_schema(item,schema['items'],f'{path}[{index}]')
    elif kind=='string':
        if not schema.get('minLength',0)<=len(value)<=schema.get('maxLength',100000):
            raise ValueError(f'{path}: tamanho fora do limite')
    elif kind=='integer':
        if not schema.get('minimum',-1e15)<=value<=schema.get('maximum',1e15):
            raise ValueError(f'{path}: valor fora do limite')


def prepare(capability_id,input_data,parameters=None):
    capability = next((c for c in get_capabilities() if c['id']==capability_id),None)
    if capability is None:
        raise ValueError('Capacidade desconhecida.')
    parameters = {} if parameters is None else parameters
    validate_schema(input_data,capability['input_schema'])
    validate_schema(parameters,capability['parameter_schema'],'parameters')
    resolved = {key:s['default'] for key,s in capability['parameter_schema']['properties'].items() if 'default' in s}
    resolved.update(parameters)
    if capability_id=='lingua.segmenta' and len(''.join(input_data['text'].split()))<2:
        raise ValueError('A sequência sem espaços precisa conter dois ou mais símbolos.')
    return capability,resolved


def vector_info(vector):
    import numpy as np
    flat = np.asarray(vector).ravel()
    indices = np.linspace(0,max(0,len(flat)-1),min(96,len(flat)),dtype=int)
    return dict(type='vector',shape=list(vector.shape),dtype=str(vector.dtype),sample=flat[indices].tolist(),
                sample_indices=indices.tolist(),sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
                sampled=len(flat)>96,total_values=int(len(flat)))


def execute(capability_id,input_data,parameters,emit,check_cancel):
    capability,resolved = prepare(capability_id,input_data,parameters)
    if str(SOURCE) not in sys.path:
        sys.path.insert(0,str(SOURCE))
    check_cancel()
    return RUNNERS[capability_id](input_data,resolved,emit,check_cancel)


def load_source(name):
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(SOURCE.resolve()):
        raise RuntimeError(f'{name}: módulo carregado fora da cópia isolada; reinicie o servidor.')
    return module


def run_segment(input_data,parameters,emit,check_cancel):
    module = load_source('novaphys.lingua')
    original = input_data['text']
    sequence = ''.join(original.split())
    source = provenance('novaphys.lingua',function='segmenta')
    emit('input','Sequência recebida','Espaços removidos pelo adaptador; conteúdo original preservado.',
         data=dict(original=original,sequence=sequence,length=len(sequence)),module_id='novaphys.lingua',source=source)
    with SourceTrace(emit,check_cancel,{'novaphys.lingua'},mdl=True):
        tokens,lexicon,initial,final = module.segmenta(sequence,**parameters)
    check_cancel()
    result = dict(tokens=tokens,lexicon=sorted(lexicon),cost_initial_bits=initial,cost_final_bits=final,
                  compression_gain_bits=initial-final,sequence=sequence,segmented=' · '.join(tokens),
                  units=[dict(token=t,count=c) for t,c in Counter(tokens).most_common()])
    emit('memory.snapshot','Léxico produzido nesta execução',
         'Unidades aprendidas por compressão; não implica compreensão semântica nem persistência na memória original da AGI.',
         data=result,metrics=dict(lexicon_size=len(lexicon),description_bits=final,compression_gain_bits=initial-final),
         module_id='novaphys.lingua',source=source)
    return result


def run_scene(input_data,parameters,emit,check_cancel):
    scene_module = load_source('novaphys.cena')
    hdc = load_source('novaphys.hdc_indice')
    pairs = [(p['role'],p['object']) for p in input_data['pairs']]
    codebook = list(dict.fromkeys(input_data.get('codebook') or [o for _,o in pairs]))
    query = input_data['query_role']
    source = provenance('novaphys.cena')
    emit('input','Papéis e objetos recebidos','Relações fornecidas pelo usuário; a cena é um hipervetor associativo.',
         data=dict(pairs=input_data['pairs'],codebook=codebook,query_role=query),module_id='novaphys.cena',source=source)
    with SourceTrace(emit,check_cancel,{'novaphys.cena','novaphys.hdc_indice'}):
        space = hdc.Espaco(dim=parameters['dim'])
        vector = scene_module.monta_cena(space,pairs)
        emit('scene.snapshot','Cena associativa construída',
             'Grafo dos pares reais. As posições 3D da interface são uma disposição visual, não coordenadas mentais medidas.',
             data=dict(pairs=input_data['pairs'],vector=vector_info(vector),dim=space.dim,atom_count=len(space._cache)),
             module_id='novaphys.cena',source={**source,'function':'monta_cena'})
        selected = scene_module.consulta(space,vector,query,codebook)
    check_cancel()
    # Observer computes all scores with the original sim; consulta's return is authoritative.
    probe = space.bind(space.atomo(f'papel:{query}'),vector)
    scores = [dict(object=obj,similarity=hdc.sim(probe,space.atomo(f'obj:{obj}'))) for obj in codebook]
    scores.sort(key=lambda row:-row['similarity'])
    result = dict(pairs=input_data['pairs'],query_role=query,selected=selected,scores=scores,dim=space.dim,
                  codebook=codebook,expected_objects=[o for p,o in pairs if p==query],vector=vector_info(vector),
                  score_origin='Ranking recalculado com sim original para expor o cleanup; similaridade não é probabilidade.')
    emit('query.result',f'Objeto recuperado: {selected}',
         'O retorno é o de consulta: escolhe entre objetos do codebook, sem gerar um objeto novo.',
         data=result,metrics=dict(best_similarity=scores[0]['similarity'],codebook_size=len(codebook)),
         module_id='novaphys.cena',source={**source,'function':'consulta'})
    emit('memory.snapshot','Átomos presentes neste espaço',
         'Cache real do Espaco criado nesta execução; não representa a memória global da AGI.',
         data=dict(atoms=sorted(space._cache),atom_count=len(space._cache),dim=space.dim,
                   vector_bytes=sum(v.nbytes for v in space._cache.values())),
         module_id='novaphys.hdc_indice',source=provenance('novaphys.hdc_indice',function='Espaco.atomo'))
    return result


def run_archive(input_data,parameters,emit,check_cancel):
    path = SOURCE/'traco.json'
    blob = path.read_bytes()
    source = dict(path='traco.json',sha256=hashlib.sha256(blob).hexdigest(),mode='imported',function=None,line=None)
    entries = json.loads(blob.decode('utf-8-sig'))
    item = entries[input_data['task_index']]
    task = item.get('tarefa','desconhecida')
    context = {k:v for k,v in item.items() if k not in {'busca','estagios'}}
    context.update(recorded_count=len(item.get('busca',[])),evaluated_count=item.get('avaliados'),
                   recording_partial=len(item.get('busca',[]))!=item.get('avaliados'),timestamps_available=False)
    root = emit('archive.context',f'Gravação: {task}',
                'Arquivo histórico do projeto. Ordem conhecida; tempo de cada passo desconhecido.',
                data=context,module_id='novaphys.nucleo',source=source,imported=True)['id']
    for row in item.get('busca',[]):
        check_cancel()
        emit('hypothesis.evaluated',f"Candidato registrado {row.get('i')}",
             'Registro histórico: apenas uma parte da busca pode estar gravada.',data=row,
             metrics=dict(match_score=row['casa']) if 'casa' in row else {},
             module_id='novaphys.nucleo',parent_id=root,source=source,imported=True)
    for row in item.get('estagios',[]):
        check_cancel()
        emit('operator.applied',str(row.get('op','Etapa gravada')),'Estado registrado após operação.',
             data=row,module_id='novaphys.nucleo',parent_id=root,source=source,imported=True)
    result = dict(task=task,solved=item.get('resolvida'),solution=item.get('solucao'),recorded_count=context['recorded_count'],
                  evaluated_count=context['evaluated_count'],recording_partial=context['recording_partial'],timestamps_available=False)
    emit('archive.result','Resultado da gravação','Conclusão do arquivo: não foi reavaliada nem executada agora.',
         data=result,module_id='novaphys.nucleo',parent_id=root,source=source,imported=True)
    return result


RUNNERS = {'lingua.segmenta':run_segment,'cena.hdc':run_scene,'archive.traco':run_archive}

CAPABILITIES.append(dict(
    id='import.events',title='Importar eventos de outro módulo',name='Importar eventos de outro módulo',
    module_id='observatory.import',mode='import',
    description='Preserva eventos estruturados fornecidos pelo usuário. Não executa código e não confirma a autenticidade do produtor; o tempo original, se informado, é preservado como dado.',
    input_schema=object_schema({'events':dict(type='array',title='Eventos JSON',minItems=1,maxItems=500,
        items=dict(type='object',additionalProperties=True,required=['protocol_version','module_id','kind','title'],
                   properties={'protocol_version':dict(type='string',enum=['1.0']),
                               'module_id':string('Módulo',160),'kind':string('Tipo de evento',120),
                               'title':string('Título',240)}))},['events']),
    parameter_schema=object_schema({}),
    observability=dict(level='external_import',description='Conteúdo externo preservado, origem não verificada. Replay por ordem dos registros, sem tempos inventados.')
))


def run_import(input_data,parameters,emit,check_cancel):
    mapping = {}
    count = 0
    for item in input_data['events']:
        check_cancel()
        parent = mapping.get(item.get('parent_id')) if isinstance(item.get('parent_id'),str) else None
        imported_source = item.get('source') if isinstance(item.get('source'),dict) else {}
        source = {**imported_source,'mode':'imported','verification':'external_unverified'}
        payload = item.get('data') if isinstance(item.get('data'),dict) else {}
        # Every unknown field and the claimed original time remain inspectable.
        data = {**payload,'_imported_event':item}
        metrics = item.get('metrics') if isinstance(item.get('metrics'),dict) else {}
        record = emit(item['kind'],item['title'],'Evento externo importado; origem não verificada.',
                      data=data,metrics=metrics,module_id=item['module_id'],parent_id=parent,source=source,imported=True)
        if isinstance(item.get('id'),str):
            mapping[item['id']] = record['id']
        count += 1
    return dict(imported_count=count,verification='external_unverified',time_basis='ordinal',
                summary='Conteúdo preservado. Importação não confirma que o produtor executou as operações descritas.')


RUNNERS['import.events'] = run_import
