"""Copiar somente dois componentes agnósticos a domínio; registrar hashes."""
import json
from pathlib import Path
import shutil
from contratos import digest

HERE = Path(__file__).resolve().parent
ORIGINAL = Path(('C:\\Users\\usuario\\Documents\\AGI 2.0').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))


def main():
    destination = HERE / 'motor_snapshot'
    destination.mkdir(exist_ok=True)
    manifest = {'scope': 'two_components_not_entire_AGI', 'files': {}}
    for name in ('substrato.py', 'hdc_indice.py'):
        source, target = ORIGINAL / 'novaphys' / name, destination / name
        if target.exists():
            raise FileExistsError(target)
        before = digest(source)
        shutil.copyfile(source, target)
        if digest(target) != before or digest(source) != before:
            raise RuntimeError('Cópia não confere')
        manifest['files'][name] = {'source': str(source), 'sha256': before}
    manifest['architecture_sha256'] = digest(ORIGINAL / 'ARQUITETURA.md')
    (destination / '__init__.py').write_text('', encoding='utf-8')
    with (destination / 'manifesto.json').open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
