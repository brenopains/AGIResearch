"""Bounded source tracing; observation never replaces an algorithm."""
from __future__ import annotations
import ast
import hashlib
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'source'


def provenance(module_id, function=None, line=None, mode='observed'):
    path = SOURCE.joinpath(*module_id.split('.')).with_suffix('.py')
    if not path.is_file():
        path = SOURCE.joinpath(*module_id.split('.'), '__init__.py')
    return dict(path=path.relative_to(SOURCE).as_posix() if path.is_file() else None,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
                function=function, line=line, mode=mode)


class SourceTrace:
    def __init__(self, emit, check_cancel, modules, mdl=False, call_limit=1200):
        self.emit, self.check_cancel, self.modules = emit, check_cancel, set(modules)
        self.mdl, self.call_limit = mdl, call_limit
        self.frames, self.sources = {}, {}
        self.recorded_calls, self.omitted_calls = 0, 0
        self.pending, self.round, self.lines = None, 0, {}
        if mdl:
            code = (SOURCE/'novaphys'/'lingua.py').read_text(encoding='utf-8-sig')
            function = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name=='segmenta')
            for node in ast.walk(function):
                if isinstance(node, ast.Assign):
                    targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                    if 't2' in targets:
                        self.lines['proposal'] = node.lineno
                    if 'pares' in targets:
                        self.lines['round'] = node.lineno
                if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id=='d2':
                    if not (len(node.test.ops)==1 and isinstance(node.test.ops[0],ast.Lt) and len(node.test.comparators)==1 and isinstance(node.test.comparators[0],ast.Name) and node.test.comparators[0].id=='dl'):
                        raise RuntimeError('Critério de aceitação MDL mudou; adaptador precisa ser revisado.')
                    self.lines['evaluation'] = node.lineno
                    self.lines['accepted'] = next(n.lineno for n in node.body if isinstance(n, ast.Break))
            if len(self.lines)!=4:
                raise RuntimeError('segmenta mudou: o adaptador semântico precisa ser revisado.')

    def event(self, module, function, line, kind, title, summary='', data=None, metrics=None, parent_id=None):
        if module not in self.sources:
            self.sources[module] = provenance(module)
        return self.emit(kind, title, summary, data=data, metrics=metrics, module_id=module,
                         parent_id=parent_id, source={**self.sources[module], 'function':function,'line':line})

    def __enter__(self):
        self.previous = sys.gettrace()
        sys.settrace(self.trace)
        return self

    def __exit__(self, *args):
        sys.settrace(self.previous)
        self.emit('telemetry.coverage','Cobertura desta observação',
                  'Chamadas Python dos módulos selecionados. Bibliotecas externas e valores internos não instrumentados não estão representados.',
                  data=dict(modules=sorted(self.modules),recorded_calls=self.recorded_calls,
                            omitted_calls=self.omitted_calls,call_limit=self.call_limit,
                            call_trace_truncated=self.omitted_calls>0,semantic_mdl_events=self.mdl))

    def trace(self, frame, event, arg):
        module = frame.f_globals.get('__name__','')
        if module not in self.modules or frame.f_code.co_name.startswith('<'):
            return None
        self.check_cancel()
        name = frame.f_code.co_qualname
        is_segment = self.mdl and name=='segmenta'
        if event=='call':
            frame.f_trace_lines = is_segment
            parent, prior = None, frame.f_back
            while parent is None and prior is not None:
                parent = self.frames.get(id(prior))
                prior = prior.f_back
            if self.recorded_calls < self.call_limit:
                item = self.event(module,name,frame.f_code.co_firstlineno,'function.enter',f'{name}()',
                                  'Chamada observada no código copiado.',
                                  dict(function=name,caller_module=frame.f_back.f_globals.get('__name__') if frame.f_back else None),parent_id=parent)
                self.frames[id(frame)] = item['id']
                self.recorded_calls += 1
            else:
                self.omitted_calls += 1
        elif event=='return':
            parent = self.frames.pop(id(frame),None)
            if parent:
                self.event(module,name,frame.f_lineno,'function.exit',f'{name} retornou',
                           'Retorno da função observada.',dict(function=name),parent_id=parent)
        elif event=='line' and is_segment:
            local = frame.f_locals
            parent = self.frames.get(id(frame))
            if frame.f_lineno == self.lines['round']:
                self.round += 1
            elif frame.f_lineno == self.lines['proposal']:
                data = dict(candidate=local['novo'],pair=[local['x'],local['y']],frequency=local['_n'],
                            cost_before_bits=local['dl'],round=self.round)
                self.pending = self.event(module,name,frame.f_lineno,'hypothesis.proposed',
                                          f"Testar unidade «{local['novo']}»",'Fusão de pares adjacentes frequentes.',
                                          data,parent_id=parent)['id']
            elif frame.f_lineno == self.lines['evaluation']:
                accepted = local['d2'] < local['dl']
                data = dict(candidate=local['novo'],pair=[local['x'],local['y']],round=self.round,
                            cost_before_bits=local['dl'],cost_after_bits=local['d2'],accepted=accepted,
                            tokens=local['t2'][:300],tokens_total=len(local['t2']),tokens_sampled=len(local['t2'])>300)
                metrics = dict(cost_before_bits=local['dl'],cost_after_bits=local['d2'],gain_bits=local['dl']-local['d2'])
                self.event(module,name,frame.f_lineno,'hypothesis.evaluated',f"Custo de «{local['novo']}»",
                           'Custo real de léxico mais sequência codificada; menor é melhor.',data,metrics,self.pending)
                if not accepted:
                    self.event(module,name,frame.f_lineno,'hypothesis.rejected',f"«{local['novo']}» não reduziu o custo",
                               'A condição original d2 < dl foi falsa.',data,metrics,self.pending)
            elif frame.f_lineno == self.lines['accepted']:
                self.event(module,name,frame.f_lineno,'hypothesis.accepted',f"Unidade «{local['novo']}» incorporada",
                           'A atribuição original atualizou tokens, léxico e custo.',
                           dict(candidate=local['novo'],lexicon=sorted(local['lex']),tokens=local['toks'][:300],
                                tokens_total=len(local['toks']),tokens_sampled=len(local['toks'])>300,round=self.round),
                           dict(description_bits=local['dl'],lexicon_size=len(local['lex'])),self.pending)
        return self.trace
