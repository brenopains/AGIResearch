"""Inventário e downloads públicos limitados; sem autenticação ou microscopia."""
import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = 'AGI-Inventor-research/0.1'


def fetch(url, limit=4 * 1024 * 1024):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=30) as r:
        data = r.read(limit + 1)
        if len(data) > limit:
            raise ValueError('Resposta excedeu limite do inventário')
        return data


def listing(base, prefix, maximum=5):
    url = base + '?' + urllib.parse.urlencode({'prefix': prefix, 'max-keys': maximum})
    root = ET.fromstring(fetch(url))
    out = []
    for c in root:
        if c.tag.endswith('Contents'):
            row = {x.tag.split('}')[-1]: x.text for x in c}
            out.append({'key': row['Key'], 'bytes': int(row['Size']), 'etag': row.get('ETag'),
                        'url': base + '/' + urllib.parse.quote(row['Key'], safe='/')})
    return {'listing_url': url, 'objects': out}


def inventory():
    def fly():
        doc = json.loads(fetch('https://zenodo.org/api/records/10676866'))
        return {'source': 'FlyWire', 'reference': 'https://zenodo.org/records/10676866',
                'license': doc.get('metadata', {}).get('license'),
                'objects': [{'key': f['key'], 'bytes': f['size'], 'checksum': f.get('checksum'),
                             'url': f['links']['self']} for f in doc['files']]}
    jobs = {
        'flywire': fly,
        'human_h01': lambda: listing('https://storage.googleapis.com/h01-release', 'data/20210729/c3/synapses/exported/'),
        'mouse_microns': lambda: listing('https://bossdb-open-data.s3.amazonaws.com', 'iarpa_microns/minnie/minnie65/'),
    }
    out = {'time_unix': time.time(), 'scope': 'metadata_only', 'sources': {}}
    with ThreadPoolExecutor(max_workers=3) as pool:
        fs = {name: pool.submit(fn) for name, fn in jobs.items()}
        for name, f in fs.items():
            try:
                out['sources'][name] = f.result()
            except Exception as e:
                out['sources'][name] = {'error': type(e).__name__ + ': ' + str(e)}
    return out


def download(url, path, limit_bytes, expected_checksum=None):
    path = Path(path)
    if path.exists() or Path(str(path) + '.receipt.json').exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + '.partial')
    sha, md5, count = hashlib.sha256(), hashlib.md5(), 0
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=30) as r:
        size = r.headers.get('Content-Length')
        if size and int(size) > limit_bytes:
            raise ValueError(f'Arquivo de {size} bytes excede limite {limit_bytes}')
        with partial.open('xb') as f:
            while True:
                chunk = r.read(min(1024 * 1024, limit_bytes - count + 1))
                if not chunk:
                    break
                count += len(chunk)
                if count > limit_bytes:
                    raise ValueError('Download excedeu o limite; parcial preservado')
                f.write(chunk)
                sha.update(chunk)
                md5.update(chunk)
        if size and count != int(size):
            raise ValueError('Download incompleto; parcial preservado')
        if expected_checksum:
            algo, value = expected_checksum.split(':', 1)
            measured = {'md5': md5.hexdigest(), 'sha256': sha.hexdigest()}.get(algo)
            if measured != value:
                raise ValueError('Checksum publicado não confere; parcial preservado')
        receipt = {'url': url, 'final_url': r.url, 'bytes': count, 'sha256': sha.hexdigest(),
                   'published_checksum': expected_checksum, 'time_unix': time.time(), 'etag': r.headers.get('ETag')}
    partial.replace(path)
    Path(str(path) + '.receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    inv = sub.add_parser('inventory')
    inv.add_argument('--out', type=Path, default=HERE / 'dados' / 'inventario.json')
    dl = sub.add_parser('download')
    dl.add_argument('--url', required=True)
    dl.add_argument('--out', type=Path, required=True)
    dl.add_argument('--max-mib', type=int, default=512)
    dl.add_argument('--checksum')
    a = p.parse_args()
    if a.cmd == 'inventory':
        result = inventory()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
    else:
        result = download(a.url, a.out, a.max_mib * 1024 * 1024, a.checksum)
    print(json.dumps(result, indent=2))
