"""Integration tests. All run data and discovery fixtures use temporary folders."""
import hashlib
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.events import TERMINAL
from backend.server import create_app


class ObservatoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='novamind-integration-')
        self.root = Path(self.temp.name)
        self.data = self.root / 'runs'
        self.app = create_app(data_dir=self.data)
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.client.__enter__()

    def tearDown(self):
        active = self.app.state.active
        if active.get('cancel'): active['cancel'].set()
        if active.get('thread'): active['thread'].join(timeout=3)
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def start(self, capability, input_data, parameters=None):
        r = self.client.post('/api/runs', json={'capability_id':capability,'input':input_data,'parameters':parameters or {}})
        self.assertEqual(r.status_code,201,r.text)
        return r.json()['id']

    def finish(self, run_id):
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            meta=self.client.get(f'/api/runs/{run_id}').json()
            if meta['status'] in TERMINAL: return meta
            time.sleep(0.01)
        self.fail('Run did not terminate')

    def events(self, run_id, after=0, limit=5000):
        r=self.client.get(f'/api/runs/{run_id}/events',params={'after':after,'limit':limit})
        self.assertEqual(r.status_code,200,r.text)
        return r.json()

    def external(self):
        r=self.client.post('/api/external-runs',json={'title':'External','producer':'test.producer','source':{'path':'producer.py'}})
        self.assertEqual(r.status_code,201,r.text)
        return r.json()

    def ingest(self, run, events, **kwargs):
        return self.client.post(f'/api/external-runs/{run["id"]}/events',json={'token':run['ingest_token'],'events':events,**kwargs})

    @staticmethod
    def event(seq, **kwargs):
        return {'protocol_version':'1.0','seq':seq,'id':f'producer:{seq}','module_id':'future.unknown','kind':'custom.observed','title':'Observed','data':{'value':seq},**kwargs}

    def test_health_and_local_origin(self):
        r=self.client.get('/api/health')
        self.assertTrue(r.json()['ok'])
        self.assertEqual(r.headers['cache-control'],'no-store')
        self.assertEqual(self.client.post('/api/runs',headers={'origin':'https://untrusted.example'},json={}).status_code,403)
        self.assertEqual(self.client.get('/api/health',headers={'host':'untrusted.example'}).status_code,400)

    def test_real_catalog_source_and_static_edges(self):
        catalog=self.client.get('/api/catalog').json()
        ids={n['id'] for n in catalog['nodes']}
        self.assertTrue({'novaphys.lingua','novaphys.cena','novaphys.acervo','novaphys.autonomo'}<=ids)
        self.assertGreater(len(ids),100)
        self.assertTrue(catalog['edges'])
        self.assertTrue(all(e['kind']=='static_import' for e in catalog['edges']))
        source=self.client.get('/api/source',params={'module_id':'novaphys.acervo'})
        self.assertIn('class Acervo',source.json()['content'])
        self.assertEqual(self.client.get('/api/source',params={'module_id':'../../server'}).status_code,404)

    def test_ast_never_executes_and_refreshes(self):
        source=self.root/'source'; package=source/'new_domain'; package.mkdir(parents=True)
        (package/'__init__.py').write_text('',encoding='utf-8')
        (package/'peer.py').write_text('VALUE = 1\n',encoding='utf-8')
        dangerous=package/'novel.py'
        dangerous.write_text('raise RuntimeError("must not execute")\nfrom .peer import VALUE\nclass Brain:\n def observe(self, data):\n  return data\n',encoding='utf-8')
        with TestClient(create_app(data_dir=self.root/'ast-runs',source_dir=source)) as client:
            catalog=client.get('/api/catalog').json()
            node=next(n for n in catalog['nodes'] if n['id']=='new_domain.novel')
            self.assertFalse(node['instrumented'])
            self.assertEqual(node['sha256'],hashlib.sha256(dangerous.read_bytes()).hexdigest())
            self.assertEqual(node['classes'][0]['methods'][0]['name'],'observe')
            self.assertTrue(any(e['source']=='new_domain.novel' and e['target']=='new_domain.peer' for e in catalog['edges']))
            (package/'added.py').write_text('def later(x):\n return x\n',encoding='utf-8')
            self.assertIn('new_domain.added',{n['id'] for n in client.get('/api/catalog').json()['nodes']})

    def test_invalid_http_run_rejected_before_creation(self):
        cases=[
            {'capability_id':'shell.exec','input':{}},
            {'capability_id':'lingua.segmenta','input':{'text':'abab'},'parameters':{'rodadas':101}},
            {'capability_id':'lingua.segmenta','input':{'text':'abab'},'parameters':{'unknown':1}},
            {'capability_id':'lingua.segmenta','input':{'text':'   '}},
            {'capability_id':'cena.hdc','input':{'pairs':[],'query_role':'role'}},
            {'capability_id':'cena.hdc','input':{'pairs':[{'role':'r','object':'o'}],'query_role':'r'},'parameters':{'dim':True}},
            {'capability_id':'archive.traco','input':{'task_index':100000}},
            {'capability_id':'import.events','input':{'events':[]}},
            {'capability_id':'acervo.memory','input':{'items':'label | content','query':'x'}},
        ]
        for body in cases:
            with self.subTest(body=body):
                r=self.client.post('/api/runs',json=body)
                self.assertEqual(r.status_code,422,r.text)
        self.assertEqual(self.client.get('/api/runs').json()['runs'],[])

    def test_sequence_persistence_and_replay_prefix(self):
        run_id=self.start('lingua.segmenta',{'text':'abcabcabcabcabcabc'},{'rodadas':5,'cand':3})
        meta=self.finish(run_id)
        self.assertEqual(meta['status'],'completed',meta)
        self.assertEqual(''.join(meta['result']['tokens']),'abcabcabcabcabcabc')
        self.assertLessEqual(meta['result']['cost_final_bits'],meta['result']['cost_initial_bits'])
        full=self.events(run_id)['events']
        self.assertGreater(len(full),3)
        self.assertEqual([e['seq'] for e in full],list(range(1,len(full)+1)))
        prefix=self.events(run_id,limit=3)
        self.assertEqual(prefix['events']+self.events(run_id,after=prefix['cursor'])['events'],full)
        self.assertEqual(self.events(run_id,after=2,limit=2)['events'],full[2:4])
        self.assertEqual(self.events(run_id)['events'],full)
        self.assertEqual(self.client.get(f'/api/runs/{run_id}/export').json()['events'],full)
        observed=[e for e in full if e['module_id']=='novaphys.lingua']
        self.assertTrue(observed)
        self.assertTrue(all(e['source'].get('sha256') for e in observed))
        with TestClient(create_app(data_dir=self.data)) as client:
            self.assertEqual(client.get(f'/api/runs/{run_id}').json()['result'],meta['result'])
            self.assertEqual(client.get(f'/api/runs/{run_id}/events').json()['events'],full)

    def test_real_scene_and_memory(self):
        run_id=self.start('cena.hdc',{'pairs':[{'role':'estrela','object':'sol'},{'role':'planeta','object':'terra'}],'query_role':'estrela'},{'dim':512})
        meta=self.finish(run_id)
        self.assertEqual(meta['status'],'completed',meta)
        self.assertEqual(meta['result']['selected'],'sol')
        self.assertEqual(meta['result']['dim'],512)
        self.assertTrue(all(-1<=r['similarity']<=1 for r in meta['result']['scores']))
        run_id=self.start('acervo.memory',{'items':'Energia | energia solar energia solar energia solar\nMovimento | movimento orbital movimento orbital movimento orbital','query':'energia solar'})
        meta=self.finish(run_id)
        self.assertEqual(meta['status'],'completed',meta)
        self.assertEqual(meta['result']['ranking'][0]['rotulo'],'Energia')
        self.assertEqual(meta['result']['memory_count'],2)
        self.assertEqual(meta['result']['persistence'],'events_only')
        next_id=self.start('acervo.memory',{'items':'Nova | outra memória outra memória','query':'outra memória'})
        second=self.finish(next_id)
        self.assertEqual(second['status'],'completed',second)
        self.assertEqual(second['result']['memory_count'],1)

    def test_cooperative_cancellation(self):
        entered=threading.Event()
        def cancellable(capability_id,input_data,parameters,emit,check_cancel):
            emit('test.started','Cooperative worker'); entered.set()
            while True:
                check_cancel(); time.sleep(0.005)
        with patch('backend.adapters.execute',side_effect=cancellable):
            run_id=self.start('lingua.segmenta',{'text':'abab'})
            self.assertTrue(entered.wait(3))
            self.assertEqual(self.client.post(f'/api/runs/{run_id}/cancel').status_code,200)
            meta=self.finish(run_id)
        self.assertEqual(meta['status'],'cancelled')
        kinds=[e['kind'] for e in self.events(run_id)['events']]
        self.assertIn('run.cancelled',kinds); self.assertNotIn('run.completed',kinds)
        self.assertIsNone(meta['result'])

    def test_archive_ordinal_partial_trace(self):
        run_id=self.start('archive.traco',{'task_index':0}); meta=self.finish(run_id)
        self.assertEqual(meta['status'],'completed',meta)
        imported=[e for e in self.events(run_id)['events'] if e['source'].get('mode')=='imported']
        self.assertTrue(imported)
        self.assertTrue(all(e['timestamp'] is None and e['elapsed_ms'] is None and e['time_basis']=='ordinal' for e in imported))
        context=next(e for e in imported if e['kind']=='archive.context')['data']
        self.assertIn('recorded_count',context); self.assertIn('evaluated_count',context)
        self.assertFalse(context['timestamps_available'])

    def test_external_dedup_order_parents_and_provenance(self):
        run=self.external(); timestamp='2020-01-01T00:00:00Z'
        first=self.event(1,timestamp=timestamp,source={'mode':'observed'})
        second=self.event(2,parent_id=first['id'],data={'new_shape':[1,2,3]})
        r=self.ingest(run,[first,second]); self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(len(r.json()['accepted']),2)
        again=self.ingest(run,[first,second]); self.assertEqual(again.status_code,200,again.text)
        self.assertEqual(again.json()['accepted'],[])
        full=self.events(run['id'])['events']; self.assertEqual(len(full),3)
        self.assertEqual(full[2]['parent_id'],full[1]['id'])
        self.assertEqual(full[1]['source']['mode'],'external')
        self.assertEqual(full[1]['source']['producer'],'test.producer')
        self.assertEqual(full[1]['data']['_producer']['occurred_at'],timestamp)
        self.assertEqual(full[2]['data']['new_shape'],[1,2,3])
        self.assertEqual(self.ingest(run,[self.event(4)]).status_code,409)
        self.assertEqual(len(self.events(run['id'])['events']),3)
        finished=self.ingest(run,[self.event(3)],finish=True,result={'answer':'producer result'})
        self.assertEqual(finished.status_code,200,finished.text)
        self.assertEqual(self.client.get(f'/api/runs/{run["id"]}').json()['result'],{'answer':'producer result'})
        self.assertEqual(self.ingest(run,[self.event(4)]).status_code,403)

    def test_external_malformed_shapes_return_422_without_append(self):
        bad=[self.event(True),self.event(1,source='string'),self.event(1,parent_id=['bad']),self.event(1,metrics='string'),self.event(1,id=['bad']),self.event(1,title='')]
        for event in bad:
            with self.subTest(event=event):
                run=self.external(); r=self.ingest(run,[event])
                self.assertEqual(r.status_code,422,r.text)
                self.assertEqual(len(self.events(run['id'])['events']),1)
                self.client.post(f'/api/runs/{run["id"]}/cancel')

    def test_external_batch_is_atomic(self):
        run=self.external()
        r=self.ingest(run,[self.event(1),self.event(3)])
        self.assertEqual(r.status_code,409)
        self.assertEqual(len(self.events(run['id'])['events']),1)
        self.assertEqual(self.ingest(run,[self.event(1)]).status_code,200)

    def test_external_wrong_token_cancel_and_restart(self):
        run=self.external()
        r=self.client.post(f'/api/external-runs/{run["id"]}/events',json={'token':'wrong','events':[self.event(1)]})
        self.assertEqual(r.status_code,403)
        self.assertEqual(self.ingest(run,[self.event(1)]).status_code,200)
        full=self.events(run['id'])['events']
        with TestClient(create_app(data_dir=self.data)) as client:
            self.assertEqual(client.get(f'/api/runs/{run["id"]}').json()['status'],'interrupted')
            self.assertEqual(client.get(f'/api/runs/{run["id"]}/events').json()['events'],full)
        self.assertEqual(self.client.post(f'/api/runs/{run["id"]}/cancel').json()['status'],'cancelled')
        self.assertEqual(self.ingest(run,[self.event(2)]).status_code,403)

    def test_unknown_imported_payload_preserved_unverified(self):
        event=self.event(1,source={'mode':'observed'},custom_future_field={'a':[1,2]})
        run_id=self.start('import.events',{'events':[event]}); meta=self.finish(run_id)
        self.assertEqual(meta['status'],'completed',meta)
        imported=next(e for e in self.events(run_id)['events'] if e['module_id']=='future.unknown')
        self.assertEqual(imported['source']['verification'],'external_unverified')
        self.assertEqual(imported['data']['_imported_event']['custom_future_field'],{'a':[1,2]})
        self.assertEqual(imported['time_basis'],'ordinal'); self.assertIsNone(imported['timestamp'])

    def test_pagination_boundaries(self):
        run=self.external()
        for params in ({'after':-1},{'limit':0},{'limit':5001}):
            self.assertEqual(self.client.get(f'/api/runs/{run["id"]}/events',params=params).status_code,422)
        self.assertEqual(self.client.get('/api/runs/absent').status_code,404)
        page=self.events(run['id'],after=999)
        self.assertEqual(page['events'],[]); self.assertEqual(page['cursor'],1)


if __name__=='__main__': unittest.main()
