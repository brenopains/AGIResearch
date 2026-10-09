from __future__ import annotations
import ast
import hashlib
from pathlib import Path

PROTOCOL = '1.0'
INSTRUMENTED = {'novaphys.lingua', 'novaphys.cena', 'novaphys.hdc_indice'}

class Catalog:
    """Read Python syntax only. Never import a module to discover its structure."""
    def __init__(self, source: Path):
        self.source = source.resolve()
        self.paths = {}
        self.nodes = []
        self.edges = []
        self.fingerprint = None

    def refresh(self):
        files = sorted(self.source.rglob('*.py'))
        stamp = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
        if stamp == self.fingerprint:
            return
        self.fingerprint = stamp
        nodes, paths, trees = [], {}, {}
        for path in files:
            rel = path.relative_to(self.source).as_posix()
            module_id = rel[:-3].replace('/', '.')
            if module_id.endswith('.__init__'):
                module_id = module_id[:-9]
            blob = path.read_bytes()
            content = blob.decode('utf-8-sig', errors='replace')
            node = dict(id=module_id, name=module_id.rsplit('.', 1)[-1], label=module_id.rsplit('.', 1)[-1],
                        kind='module', relative_path=rel, sha256=hashlib.sha256(blob).hexdigest(),
                        line_count=len(content.splitlines()), functions=[], classes=[],
                        instrumented=module_id in INSTRUMENTED, description='', docstring='', parse_error=None)
            try:
                tree = ast.parse(content)
                doc = ast.get_docstring(tree) or ''
                node['docstring'] = doc
                node['description'] = doc.splitlines()[0] if doc else rel
                for member in tree.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        entry = dict(name=member.name, line=member.lineno, end_line=member.end_lineno,
                                     docstring=ast.get_docstring(member) or '')
                        if isinstance(member, ast.ClassDef):
                            entry['methods'] = [dict(name=m.name, line=m.lineno, end_line=m.end_lineno, docstring=ast.get_docstring(m) or '')
                                                for m in member.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                            node['classes'].append(entry)
                        else:
                            entry['parameters'] = [a.arg for a in member.args.args]
                            node['functions'].append(entry)
                trees[module_id] = tree
            except SyntaxError as exc:
                node['parse_error'] = str(exc)
            nodes.append(node)
            paths[module_id] = path
        edges = {}
        for module_id, tree in trees.items():
            path = paths[module_id]
            package = module_id if path.name == '__init__.py' else module_id.rsplit('.', 1)[0]
            for item in ast.walk(tree):
                targets = []
                if isinstance(item, ast.Import):
                    targets.extend(a.name for a in item.names)
                elif isinstance(item, ast.ImportFrom):
                    if item.level:
                        parts = package.split('.')
                        prefix = '.'.join(parts[:len(parts) - item.level + 1])
                        base = '.'.join(filter(None, [prefix, item.module]))
                    else:
                        base = item.module or ''
                    targets.append(base)
                    targets.extend(base + '.' + a.name for a in item.names)
                for target in targets:
                    if target in paths and target != module_id:
                        key = (module_id, target)
                        if key not in edges:
                            edges[key] = dict(id=f'{module_id}->{target}', source=module_id, target=target,
                                              kind='static_import', evidence='AST import declaration', lines=[])
                        edges[key]['lines'].append(item.lineno)
        self.nodes, self.paths, self.edges = nodes, paths, list(edges.values())

    def data(self):
        self.refresh()
        return dict(protocol_version=PROTOCOL, nodes=self.nodes, edges=self.edges,
                    summary=dict(modules=len(self.nodes), functions=sum(len(n['functions']) for n in self.nodes),
                                 classes=sum(len(n['classes']) for n in self.nodes), static_edges=len(self.edges),
                                 instrumented_modules=sum(n['instrumented'] for n in self.nodes)),
                    limitations=['Importação estática mostra dependência declarada, não uma chamada observada.',
                                 'Presença no código não demonstra integração nem atividade em uma execução.'])

    def source_for(self, module_id):
        self.refresh()
        path = self.paths.get(module_id)
        if path is None:
            raise KeyError(module_id)
        blob = path.read_bytes()
        return dict(module_id=module_id, path=path.relative_to(self.source).as_posix(),
                    sha256=hashlib.sha256(blob).hexdigest(), content=blob.decode('utf-8-sig', errors='replace'))

    def provenance(self, module_id, function=None, line=None, mode='observed'):
        path = self.paths.get(module_id)
        return dict(path=path.relative_to(self.source).as_posix() if path else None,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
                    function=function, line=line, mode=mode)
