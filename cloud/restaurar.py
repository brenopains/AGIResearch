"""Verify release assets and restore their original repository-relative paths."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def restore(assetdir, target, check_only=False):
    manifest = json.loads((ROOT / 'cloud/arquivos.json').read_text(encoding='utf-8'))
    for archive in manifest['archives']:
        parts = []
        for asset in archive['assets']:
            p = assetdir / asset['name']
            if p.stat().st_size != asset['bytes']:
                raise ValueError('Tamanho incorreto: ' + p.name)
            with p.open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            if digest != asset['sha256']:
                raise ValueError('SHA-256 incorreto: ' + p.name)
            parts.append(p)
        with tempfile.TemporaryFile() as joined:
            if len(parts) > 1:
                for p in parts:
                    with p.open('rb') as f:
                        while b := f.read(8 * 1024 * 1024): joined.write(b)
                joined.seek(0)
                source = joined
            else:
                source = parts[0]
            with zipfile.ZipFile(source) as z:
                for info in z.infolist():
                    dest = (target / info.filename).resolve()
                    if not dest.is_relative_to(target.resolve()):
                        raise ValueError('Caminho fora do destino: ' + info.filename)
                if check_only:
                    bad = z.testzip()
                    if bad: raise ValueError('Arquivo corrompido: ' + bad)
                else:
                    z.extractall(target)
            print(archive['archive'] + ': ' + ('verificado' if check_only else 'restaurado'), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assets', type=Path, help='Pasta com os anexos baixados do release')
    parser.add_argument('--destino', type=Path, default=ROOT)
    parser.add_argument('--verificar', action='store_true')
    args = parser.parse_args()
    restore(args.assets, args.destino, args.verificar)
