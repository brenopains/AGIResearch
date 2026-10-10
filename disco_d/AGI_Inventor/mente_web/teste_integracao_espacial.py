"""Entrada de video real do corpo -> memoria 3D, sem seeding no renderizador."""
import json
from pathlib import Path
import tempfile

from corpo import Pagina
from mente import Mente
from teste_espacial import video

frames,K = video()
class CorpoControle:
    assistiu_info = {}
    def assistir(self, u, segundos):
        for k,q in enumerate(frames):yield k*.9,q,None

with tempfile.TemporaryDirectory() as d:
    m=Mente(9,d,registrar=False)
    m.corpo=CorpoControle()
    pg=Pagina('teste://multivista','controle.local',midias=[('video','video://controle',b'')])
    m._assistir(pg)
    p=Path(d)/'espacial/indice.json'
    assert p.exists(), 'Video nao chegou ao orgao espacial'
    index=json.loads(p.read_text(encoding='utf-8'))
    assert len(index['cenas'])==1,index
    assert index['cenas'][0]['pontos']>500
    print('OK: video do corpo chegou ao orgao espacial com resolucao maior que a retina 64x64.')
