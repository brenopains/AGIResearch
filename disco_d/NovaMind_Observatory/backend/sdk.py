"""Producer SDK: sends actual emitted observations; never infers mental states."""
from __future__ import annotations
import json, uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.request import Request,urlopen
class Recorder:
    def __init__(self,title,producer,source=None,url='http://127.0.0.1:8765'):
        self.url=url.rstrip('/')
        self.seq=0;self.parents=[];self.closed=False;self.pending=None
        run=self._post('/api/external-runs',{'title':title,'producer':producer,'source':source or {}})
        self.run_id=run['id'];self.token=run['ingest_token']
    def _post(self,path,value):
        request=Request(self.url+path,json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),{'Content-Type':'application/json'},method='POST')
        with urlopen(request,timeout=10) as response:return json.load(response)
    def _send_pending(self):
        if self.pending:
            self._post(f'/api/external-runs/{self.run_id}/events',{'token':self.token,'events':[self.pending]})
            self.pending=None
    def event(self,kind,title,data=None,summary='',module_id='external',metrics=None,source=None):
        if self.closed:raise RuntimeError('Captura encerrada.')
        self._send_pending()
        self.seq+=1
        event={'protocol_version':'1.0','id':uuid.uuid4().hex,'seq':self.seq,
               'timestamp':datetime.now(timezone.utc).isoformat(),'parent_id':self.parents[-1] if self.parents else None,
               'kind':kind,'title':title,'summary':summary,'module_id':module_id,'data':data or {},
               'metrics':metrics or {},'source':source or {}}
        self.pending=event
        self._send_pending()
        return event['id']
    @contextmanager
    def span(self,title,module_id='external',data=None,source=None):
        id=self.event('span.started',title,data,module_id=module_id,source=source)
        self.parents.append(id)
        try:
            yield self
        except Exception as exc:
            self.event('span.failed',title,{'error':str(exc)},module_id=module_id,source=source)
            raise
        else:self.event('span.completed',title,module_id=module_id,source=source)
        finally:self.parents.pop()
    def close(self,result=None,status='completed'):
        if not self.closed:
            self._send_pending()
            self._post(f'/api/external-runs/{self.run_id}/events',{'token':self.token,'events':[],'finish':True,'status':status,'result':result})
            self.closed=True
    def __enter__(self):return self
    def __exit__(self,typ,exc,tb):
        self.close({'error':str(exc)} if exc else None,'failed' if exc else 'completed')
