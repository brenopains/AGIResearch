"""
teste_compat — a mente que vai ficar viva acorda com o codigo novo? (sem rede, sem tocar no diario dela)
Uso: python teste_compat.py vidas/viva v2     -> imprime OK ou o erro; codigo de saida 0/1
"""
import os
import sys
import tempfile
import traceback

import numpy as np

pasta, versao = sys.argv[1], sys.argv[2]
try:
    from corpo import Pagina
    from mente import Mente
    from mente2 import Mente2
    m = (Mente2 if versao == "v2" else Mente).carrega(pasta, 1, log=False)
    m.pasta = tempfile.mkdtemp()                      # telemetria de teste vai para longe da vida real
    rng = np.random.default_rng(0)

    class CorpoFalso:
        n_pedidos = 0

        def _pg(self, img=False, vid=False):
            pg = Pagina(f"x://{rng.integers(1e9)}", f"dom{rng.integers(9)}.org",
                        texto=(b"the quick brown fox jumps over the lazy dog " * 120)[: 3000 + int(rng.integers(900))])
            pg.links = [(b"um link", f"x://{rng.integers(1e9)}") for _ in range(10)]
            if img:
                pg.midias.append(("imagem", "img://1", b"Mapa"))
            if vid:
                pg.midias.append(("video", "vid://1", b"Video"))
            return pg

        def aleatorio(self, porta): return self._pg(porta == "imagem", porta == "video")
        def abrir(self, u, **k): return self._pg()
        def pesquisar(self, b, q): return self._pg()
        def usar_formulario(self, f, q): return self._pg()
        def voltar(self): return self._pg()
        def pode_voltar(self): return True
        def ver(self, u): return rng.random((32, 32)).astype(np.float32)

        def assistir(self, u, s):
            for k in range(10):
                yield k * .2, rng.random((32, 32)).astype(np.float32), rng.normal(0, .1, 3200).astype(np.float32)

    m.corpo = CorpoFalso()
    m.pagina = m.corpo._pg(True)
    for _ in range(15):
        m.passo()
    if versao == "v2":
        m.sonha(10)
    m._det_t = 0
    m.telemetria(forcar=True)
    print(f"OK: acordou no passo {m.t}, {len(m.E)} estruturas, {len(m.fusoes)} fusoes, 15 decisoes sem erro")
    sys.exit(0)
except SystemExit:
    raise
except Exception:
    print("FALHOU:\n" + traceback.format_exc()[-1500:])
    sys.exit(1)
