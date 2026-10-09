"""Aquisição inicial (~1.1 GiB). MICrONS: prefixo explícito, não mapa completo."""
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fontes import HERE, download, UA


def csv_prefix(url, path, max_bytes=32 * 1024 * 1024):
    path = Path(path)
    receipt_path = Path(str(path) + '.receipt.json')
    if path.exists() or receipt_path.exists():
        raise FileExistsError(path)
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Range': f'bytes=0-{max_bytes-1}'})
    with urllib.request.urlopen(req, timeout=30) as r:
        if r.status != 206 or not r.headers.get('Content-Range', '').startswith('bytes 0-'):
            raise ValueError('Fonte não confirmou intervalo parcial')
        raw = r.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError('Intervalo excedeu limite')
        cut = raw.rfind(b'\n') + 1
        if not cut:
            raise ValueError('Nenhuma linha completa')
        raw = raw[:cut]
        receipt = {'url': url, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                   'content_range': r.headers['Content-Range'], 'etag': r.headers.get('ETag'),
                   'scope': 'biased_prefix_complete_csv_lines', 'time_unix': time.time()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(raw)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


def main():
    base = HERE / 'dados'
    inventory = json.loads((base / 'inventario.json').read_text(encoding='utf-8'))
    fly = next(o for o in inventory['sources']['flywire']['objects'] if o['key'] == 'proofread_connections_783.feather')
    human = next(o for o in inventory['sources']['human_h01']['objects'] if o['key'].endswith('export000000000000'))
    jobs = {
        'flywire': lambda: download(fly['url'], base / 'flywire_v783.feather', 900 * 1024**2, fly['checksum']),
        'human_h01': lambda: download(human['url'], base / 'h01_c3_shard0.avro', 210 * 1024**2,
                                       'md5:' + human['etag'].strip('"')),
        'mouse_microns': lambda: csv_prefix('https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/synapse_graph/synapses_pni_2.csv',
                                            base / 'microns_v117_prefix.csv'),
    }
    results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        fs = {k: pool.submit(fn) for k, fn in jobs.items()}
        for name, f in fs.items():
            try:
                results[name] = f.result()
                print(json.dumps({'source': name, 'download': 'complete', 'receipt': results[name]}), flush=True)
            except Exception as e:
                results[name] = {'error': type(e).__name__ + ': ' + str(e)}
                print(json.dumps({'source': name, **results[name]}), flush=True)
    with (base / 'aquisicao_inicial.json').open('x', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    return int(any('error' in r for r in results.values()))


if __name__ == '__main__':
    raise SystemExit(main())
