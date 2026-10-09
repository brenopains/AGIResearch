from __future__ import annotations
import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROTOCOL = '1.0'
TERMINAL = {'completed', 'failed', 'cancelled', 'interrupted'}

def utcnow():
    return datetime.now(timezone.utc).isoformat()

class Cancelled(Exception):
    pass

class EventStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.runs = {}
        self.events = {}
        self.started = {}
        for path in sorted(self.root.glob('*/run.json')):
            try:
                meta = json.loads(path.read_text(encoding='utf-8'))
                run_id = meta['id']
                if not re.fullmatch(r'[a-f0-9]{32}', run_id) or path.parent.name != run_id:
                    continue
                records = []
                event_file = path.parent / 'events.jsonl'
                if event_file.exists():
                    for line in event_file.read_text(encoding='utf-8').splitlines():
                        try:
                            event = json.loads(line)
                            if event['seq'] == len(records) + 1 and event['run_id'] == run_id:
                                records.append(event)
                            else:
                                break
                        except (ValueError, KeyError):
                            break
                self.events[run_id] = records
                self.runs[run_id] = meta
                meta['event_count'] = len(records)
                if meta['status'] not in TERMINAL:
                    meta.update(status='interrupted', finished_at=utcnow(), error='Servidor reiniciado antes de concluir.')
                    self._save(meta)
            except (OSError, ValueError, KeyError):
                continue

    def _save(self, meta):
        path = self.root / meta['id'] / 'run.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(meta, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        temp.replace(path)

    def create(self, capability_id, title, mode, input_data, parameters):
        with self.lock:
            run_id = uuid.uuid4().hex
            (self.root / run_id).mkdir()
            meta = dict(id=run_id, protocol_version=PROTOCOL, capability_id=capability_id, title=title,
                        mode=mode, status='running', created_at=utcnow(), finished_at=None,
                        event_count=0, input=input_data, parameters=parameters, result=None, error=None)
            self.runs[run_id], self.events[run_id], self.started[run_id] = meta, [], time.perf_counter()
            self._save(meta)
            return self.get(run_id)

    def emit(self, run_id, kind, title, summary='', data=None, metrics=None, module_id='observatory',
             parent_id=None, source=None, imported=False):
        with self.lock:
            meta = self.runs[run_id]
            seq = len(self.events[run_id]) + 1
            record = dict(protocol_version=PROTOCOL, id=f'{run_id}:{seq}', run_id=run_id, seq=seq,
                          timestamp=None if imported else utcnow(),
                          elapsed_ms=None if imported else round((time.perf_counter() - self.started.get(run_id, time.perf_counter())) * 1000, 3),
                          time_basis='ordinal' if imported else 'clock', parent_id=parent_id,
                          module_id=module_id, kind=kind, title=title, summary=summary,
                          data=data or {}, metrics=metrics or {}, source=source or dict(mode='observatory'))
            encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
            with (self.root / run_id / 'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(encoded + '\n')
                stream.flush()
            self.events[run_id].append(record)
            meta['event_count'] = seq
            self._save(meta)
            return record

    def finish(self, run_id, status, result=None, error=None):
        with self.lock:
            meta = self.runs[run_id]
            meta.update(status=status, result=result, error=error, finished_at=utcnow())
            self._save(meta)

    def get(self, run_id):
        with self.lock:
            if run_id not in self.runs:
                raise KeyError(run_id)
            return json.loads(json.dumps(self.runs[run_id], ensure_ascii=False))

    def list(self):
        with self.lock:
            return [self.get(key) for key in sorted(self.runs, key=lambda k: self.runs[k]['created_at'], reverse=True)]

    def read(self, run_id, after=0, limit=1000):
        with self.lock:
            meta = self.get(run_id)
            records = self.events[run_id][after:after + limit]
            return dict(events=records, cursor=records[-1]['seq'] if records else min(after, meta['event_count']),
                        status=meta['status'], total=meta['event_count'])
