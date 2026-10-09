"""Local evidence server. Discovery uses AST; execution uses explicit adapters."""
from __future__ import annotations
import json, os, sys, threading, time, subprocess, shutil, traceback, uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
import psutil
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, ConfigDict
from .catalog import Catalog
from .events import EventStore, Cancelled, TERMINAL, PROTOCOL, utcnow

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source'
sys.path.insert(0, str(SOURCE))
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

class RunRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    capability_id: str = Field(min_length=1, max_length=100)
    input: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)

class ExternalStart(BaseModel):
    title: str = Field(min_length=1,max_length=180)
    producer: str = Field(min_length=1,max_length=150)
    source: dict[str, Any] = Field(default_factory=dict)

class IngestRequest(BaseModel):
    token: str
    events: list[dict[str, Any]] = Field(max_length=100)
    finish: bool = False
    status: str = 'completed'
    result: Any = None

def capabilities():
    from .adapters import get_capabilities
    result = get_capabilities()
    try:
        from .acervo_adapter import CAPABILITY
        result.append(dict(CAPABILITY, mode='live', title=CAPABILITY['label']))
    except ImportError:
        pass
    return result

def validate_schema(value, schema, path='input'):
    kind=schema.get('type')
    if kind=='object':
        if not isinstance(value,dict): raise ValueError(f'{path}: esperado objeto.')
        for key in schema.get('required',[]):
            if key not in value: raise ValueError(f'{path}.{key}: campo obrigatório.')
        props=schema.get('properties',{})
        if schema.get('additionalProperties') is False and set(value)-set(props):
            raise ValueError(f'{path}: campo desconhecido.')
        for key,item in value.items():
            if key in props: validate_schema(item,props[key],f'{path}.{key}')
    elif kind=='string':
        if not isinstance(value,str): raise ValueError(f'{path}: esperado texto.')
        if len(value)<schema.get('minLength',0) or len(value)>schema.get('maxLength',30000):
            raise ValueError(f'{path}: comprimento fora do limite.')
    elif kind in ('integer','number'):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or (kind=='integer' and not isinstance(value,int)):
            raise ValueError(f'{path}: esperado número.')
        if value<schema.get('minimum',-1e12) or value>schema.get('maximum',1e12): raise ValueError(f'{path}: valor fora do limite.')
    elif kind=='array':
        if not isinstance(value,list): raise ValueError(f'{path}: esperada lista.')
        if not schema.get('minItems',0)<=len(value)<=schema.get('maxItems',1000): raise ValueError(f'{path}: número de itens fora do limite.')
        for item in value: validate_schema(item,schema.get('items',{}),path+'[]')
    if 'enum' in schema and value not in schema['enum']: raise ValueError(f'{path}: opção inválida.')

def create_app(data_dir=None, source_dir=None, dist_dir=None):
    app=FastAPI(title='NovaMind · Observatório',version='1.0.0')
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','testserver'])
    store=EventStore(Path(data_dir or ROOT/'data'/'runs'))
    catalog=Catalog(Path(source_dir or SOURCE))
    dist=Path(dist_dir or ROOT/'frontend'/'dist')
    lock=threading.RLock()
    active={'id':None,'cancel':None,'thread':None}
    external={}
    app.state.store,app.state.catalog,app.state.active=store,catalog,active
    gpu_cache={'at':0,'value':None}
    psutil.cpu_percent()

    @app.middleware('http')
    async def local_only(request:Request,call_next):
        origin=request.headers.get('origin')
        if origin:
            from urllib.parse import urlsplit
            parsed=urlsplit(origin)
            if parsed.scheme not in ('http','https') or parsed.hostname not in ('localhost','127.0.0.1'):
                return JSONResponse({'detail':'Origem não autorizada.'},status_code=403)
        if request.method in ('POST','PUT','PATCH'):
            if int(request.headers.get('content-length','0'))>2_000_000:
                return JSONResponse({'detail':'Entrada excede 2 MB.'},status_code=413)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        if request.url.path.startswith('/api/'): response.headers['Cache-Control']='no-store'
        return response

    def getrun(run_id):
        try:return store.get(run_id)
        except KeyError:raise HTTPException(404,'Execução não encontrada.')

    @app.get('/api/health')
    def health():return dict(ok=True,protocol_version=PROTOCOL,service='novamind-observatory')

    @app.get('/api/catalog')
    def get_catalog():
        with lock:
            result=catalog.data()
            mids=set()
            for cap in capabilities():
                if cap.get('mode')=='live':mids.update(cap.get('module_ids',[cap.get('module_id')]))
                if cap['id']=='cena.hdc':mids.add('novaphys.hdc_indice')
            for node in result['nodes']:node['instrumented']=node['id'] in mids
            result['summary']['instrumented_modules']=sum(n['instrumented'] for n in result['nodes'])
            result['source_root']=str(catalog.source)
            result['scope']='Módulos Python da cópia de trabalho; catálogo estático não implica atividade.'
            return result

    @app.get('/api/capabilities')
    def get_capabilities():return dict(protocol_version=PROTOCOL,capabilities=capabilities())

    @app.get('/api/runs')
    def list_runs():return dict(runs=store.list())

    @app.get('/api/runs/{run_id}')
    def read_run(run_id:str):return getrun(run_id)

    @app.get('/api/runs/{run_id}/events')
    def read_events(run_id:str,after:int=Query(0,ge=0),limit:int=Query(1000,ge=1,le=5000)):
        getrun(run_id)
        return store.read(run_id,after,limit)

    @app.get('/api/runs/{run_id}/export')
    def export_run(run_id:str):
        meta=getrun(run_id)
        return JSONResponse(dict(protocol_version=PROTOCOL,run=meta,events=store.read(run_id,0,max(meta['event_count'],1))['events']),
                            headers={'Content-Disposition':f'attachment; filename="novamind-{run_id[:8]}.json"'})

    def worker(run, cancel):
        run_id=run['id']; start=time.monotonic(); count=0
        def check_cancel():
            if cancel.is_set():raise Cancelled('Cancelamento solicitado.')
            if time.monotonic()-start>90:raise TimeoutError('Orçamento de 90 segundos atingido.')
        def emit(*args,**kwargs):
            nonlocal count
            check_cancel();count+=1
            if count>12000:raise RuntimeError('Limite de 12.000 eventos atingido; execução interrompida e cobertura parcial.')
            return store.emit(run_id,*args,**kwargs)
        try:
            store.emit(run_id,'run.started','Execução iniciada',run['title'],data={'input':run['input'],'parameters':run['parameters']})
            if run['capability_id']=='acervo.memory':
                from .acervo_adapter import run as run_acervo
                result=run_acervo(run['input'],emit,check_cancel)
            else:
                from .adapters import execute
                result=execute(run['capability_id'],run['input'],run['parameters'],emit,check_cancel)
            check_cancel()
            store.emit(run_id,'run.completed','Execução concluída','Resultado produzido pelo adaptador real.',data={'result':result})
            store.finish(run_id,'completed',result=result)
        except Cancelled as exc:
            store.emit(run_id,'run.cancelled','Execução cancelada',str(exc))
            store.finish(run_id,'cancelled',error=str(exc))
        except Exception as exc:
            store.emit(run_id,'run.failed','Execução interrompida',str(exc),data={'error_type':type(exc).__name__})
            store.finish(run_id,'failed',error=str(exc))
        finally:
            with lock:
                if active['id']==run_id:active.update(id=None,cancel=None,thread=None)

    @app.post('/api/runs',status_code=201)
    def start_run(req:RunRequest):
        by_id={c['id']:c for c in capabilities()}
        cap=by_id.get(req.capability_id)
        if not cap:raise HTTPException(422,'Capacidade desconhecida. Consulte o catálogo.')
        try:
            if req.capability_id=='acervo.memory':
                from .acervo_adapter import _validate
                _validate(req.input)
                validate_schema(req.parameters,dict(type='object',properties={},additionalProperties=False),'parameters')
            else:
                from .adapters import prepare
                _,req.parameters=prepare(req.capability_id,req.input,req.parameters)
            if len(json.dumps(req.input))>100000:raise ValueError('Entrada muito grande.')
        except ValueError as exc:raise HTTPException(422,str(exc))
        with lock:
            if active['id'] is not None:raise HTTPException(409,'Já existe uma execução local em andamento.')
            run=store.create(req.capability_id,cap.get('title',cap.get('label',cap['id'])),
                             'imported' if req.capability_id in ('archive.traco','import.events') else 'live',req.input,req.parameters)
            cancel=threading.Event()
            thread=threading.Thread(target=worker,args=(run,cancel),daemon=True,name='novamind-run')
            active.update(id=run['id'],cancel=cancel,thread=thread)
            thread.start()
            return run

    @app.post('/api/runs/{run_id}/cancel')
    def cancel_run(run_id:str):
        meta=getrun(run_id)
        with lock:
            if active['id']==run_id and active['cancel']:
                active['cancel'].set()
                return dict(id=run_id,status='cancelling')
            if run_id in external:
                external.pop(run_id)
                store.emit(run_id,'run.cancelled','Captura externa encerrada','O produtor externo não foi interrompido.')
                store.finish(run_id,'cancelled')
                return store.get(run_id)
        return meta

    @app.get('/api/source')
    def read_source(module_id:str):
        try:
            with lock:return catalog.source_for(module_id)
        except KeyError:raise HTTPException(404,'Módulo não catalogado.')

    @app.get('/api/metrics')
    def metrics():
        mem=psutil.virtual_memory()
        if time.monotonic()-gpu_cache['at']>5:
            gpu_cache['at']=time.monotonic()
            exe=shutil.which('nvidia-smi')
            gpu=None
            if exe:
                try:
                    proc=subprocess.run([exe,'--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,timeout=2,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    if proc.returncode==0:
                        row=[s.strip() for s in proc.stdout.splitlines()[0].split(',')]
                        gpu=dict(name=row[0],utilization_percent=float(row[1]),memory_used_mib=float(row[2]),memory_total_mib=float(row[3]))
                except (OSError,ValueError,IndexError,subprocess.SubprocessError):pass
            gpu_cache['value']=gpu
        return dict(timestamp=utcnow(),cpu_percent=psutil.cpu_percent(interval=0.1),memory_percent=mem.percent,
                    memory_used_bytes=mem.used,memory_total_bytes=mem.total,gpu=gpu_cache['value'],
                    scope='Máquina inteira; não atribuir estes valores exclusivamente à AGI.')

    @app.post('/api/external-runs',status_code=201)
    def external_start(req:ExternalStart):
        run=store.create('external',req.title,'external',{'producer':req.producer}, {})
        token=uuid.uuid4().hex+uuid.uuid4().hex
        external[run['id']]=dict(token=token,producer=req.producer,source=req.source,last_seq=0,parents={})
        store.emit(run['id'],'run.started','Produtor conectado',req.producer,data={'producer':req.producer},source={'mode':'external','producer':req.producer})
        return dict(**run,ingest_token=token)

    @app.post('/api/external-runs/{run_id}/events')
    def ingest(run_id:str,req:IngestRequest):
        import secrets
        with lock:
            producer=external.get(run_id)
            if not producer or not secrets.compare_digest(req.token,producer['token']):raise HTTPException(403,'Token de captura inválido.')
            if req.status not in ('completed','failed','cancelled'):raise HTTPException(422,'Status terminal inválido.')
            accepted=[]
            # Validate the entire batch before appending; transport retries remain idempotent.
            expected=producer['last_seq']
            for ev in req.events:
                if ev.get('protocol_version')!=PROTOCOL:raise HTTPException(422,'Versão de evento incompatível.')
                if type(ev.get('seq')) is not int or ev['seq']<1:raise HTTPException(422,'Sequência do produtor inválida.')
                if ev['seq']<=producer['last_seq']:continue
                if ev['seq']!=expected+1:raise HTTPException(409,'Lacuna na sequência do produtor.')
                expected=ev['seq']
                if expected>12000:raise HTTPException(422,'Limite de captura: 12.000 eventos.')
                for key in ('source','metrics','data'):
                    if key in ev and not isinstance(ev[key],dict):raise HTTPException(422,f'{key}: esperado objeto.')
                for key in ('id','parent_id','timestamp'):
                    if ev.get(key) is not None and not isinstance(ev[key],str):raise HTTPException(422,f'{key}: esperado texto.')
                if not isinstance(ev.get('module_id','external'),str):raise HTTPException(422,'module_id: esperado texto.')
                if not isinstance(ev.get('kind'),str) or not isinstance(ev.get('title'),str) or not ev['kind'] or not ev['title'] or len(ev['kind'])>120 or len(ev['title'])>250:raise HTTPException(422,'Evento inválido.')
                try:json.dumps(ev,allow_nan=False)
                except (ValueError,TypeError):raise HTTPException(422,'Evento contém número ou valor inválido.')
            for ev in req.events:
                if ev.get('protocol_version')!=PROTOCOL:raise HTTPException(422,'Versão de evento incompatível.')
                if not isinstance(ev.get('seq'),int):raise HTTPException(422,'Sequência do produtor obrigatória.')
                if ev['seq']<=producer['last_seq']:continue
                if ev['seq']!=producer['last_seq']+1:raise HTTPException(409,'Lacuna na sequência do produtor.')
                kind=ev.get('kind');title=ev.get('title')
                if not isinstance(kind,str) or not isinstance(title,str) or len(kind)>120 or len(title)>250:raise HTTPException(422,'Evento inválido.')
                source={**producer['source'],**(ev.get('source') or {}),'mode':'external','producer':producer['producer']}
                data={**ev.get('data',{}),'_producer':{'seq':ev['seq'],'occurred_at':ev.get('timestamp'),
                      'original_id':ev.get('id'),'original_parent_id':ev.get('parent_id')}}
                parent=producer['parents'].get(ev.get('parent_id'))
                result=store.emit(run_id,kind,title,str(ev.get('summary',''))[:3000],data=data,
                                  metrics=ev.get('metrics',{}),module_id=str(ev.get('module_id','external'))[:200],parent_id=parent,source=source)
                producer['last_seq']=ev['seq']
                if ev.get('id'):producer['parents'][ev['id']]=result['id']
                accepted.append(result['id'])
            if req.finish:
                store.emit(run_id,'run.'+req.status,'Captura externa finalizada','Estado declarado pelo produtor.',source={'mode':'external','producer':producer['producer']})
                store.finish(run_id,req.status,result=req.result)
                external.pop(run_id)
            return dict(accepted=accepted,last_producer_seq=producer['last_seq'],run_id=run_id)

    @app.get('/{path:path}')
    def frontend(path:str):
        if path.startswith('api/'):raise HTTPException(404,'Endpoint não encontrado.')
        target=(dist/path).resolve()
        if not target.is_relative_to(dist.resolve()):raise HTTPException(404,'Arquivo não encontrado.')
        if target.is_file():return FileResponse(target)
        if (dist/'index.html').exists():return FileResponse(dist/'index.html')
        return JSONResponse({'detail':'Frontend ainda não compilado. Execute iniciar.ps1.'},status_code=503)
    return app

app=create_app()
