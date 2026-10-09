"""
teste_laboratorio — o transplante 10 numa mente de teste, sem rede (corpo falso)
==================================================================================
Uso:  python teste_laboratorio.py [pasta_de_uma_vida_para_copiar]
Sem argumento: mente recem-nascida numa pasta temporaria. Com argumento: COPIA a pasta
(nunca toca a vida real) e acorda a copia com o codigo novo.
Confere: a porta arena aparece; ir a arena = assistir; o laboratorio percebe objetos,
cria teoria, salva em laboratorio.pkl; a mente dorme, salva, ACORDA e continua.
"""
import json
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

os.environ["MENTE_ARENA_RAPIDA"] = "1"
from corpo import Pagina  # noqa: E402
from mente import Mente  # noqa: E402

try:
    pasta = tempfile.mkdtemp()
    if len(sys.argv) > 1:
        shutil.copytree(sys.argv[1], pasta, dirs_exist_ok=True)
        for f in ("PARAR", "REINICIAR"):
            if os.path.exists(os.path.join(pasta, f)):
                os.remove(os.path.join(pasta, f))
    rng = np.random.default_rng(0)

    class CorpoFalso:
        n_pedidos = 0

        def _pg(self, img=False, vid=False):
            pg = Pagina(f"x://{rng.integers(1e9)}", f"dom{rng.integers(9)}.org",
                        texto=(b"the quick brown fox jumps over the lazy dog " * 120)[: 3000 + int(rng.integers(900))])
            pg.links = [(b"um link", f"x://{rng.integers(1e9)}") for _ in range(10)]
            if vid:
                pg.midias.append(("video", "vid://1", b"Video"))
            return pg

        def aleatorio(self, porta): return self._pg(False, porta == "video")
        def abrir(self, u, **k): return self._pg()
        def pesquisar(self, b, q): return self._pg()
        def usar_formulario(self, f, q): return self._pg()
        def voltar(self): return self._pg()
        def pode_voltar(self): return True
        def ver(self, u): return rng.random((32, 32)).astype(np.float32)

        def assistir(self, u, s):
            for k in range(10):
                yield k * .2, rng.random((32, 32)).astype(np.float32), rng.normal(0, .1, 3200).astype(np.float32)

    m = Mente.carrega(pasta, 1)
    m.corpo = CorpoFalso()
    m.pagina = m.corpo._pg()
    assert "porta:arena" in m.disponiveis(m.pagina), "porta arena nao aparece"
    escolhas = []
    for _ in range(8):                                  # ela escolhe livremente
        m.passo()
        escolhas.append(m._ultima_acao["estrategia"])
    origem = Mente.escolhe
    for _ in range(6):                                  # e 6 visitas FORCADAS a arena
        m.escolhe = lambda pg: ("porta:arena", None, origem(m, pg)[2])
        m.passo()
    del m.escolhe
    m.dorme()
    m.salva()
    lab = m.lab
    assert lab is not None and lab.contas["episodios"] >= 4, f"laboratorio nao aprendeu: {lab and lab.contas}"
    assert os.path.exists(os.path.join(pasta, "laboratorio.pkl")), "laboratorio.pkl nao foi salvo"
    m2 = Mente.carrega(pasta, 2)
    m2.corpo = CorpoFalso()
    m2.pagina = m2.corpo._pg()
    assert m2._lab().contas["episodios"] == lab.contas["episodios"], "laboratorio nao acordou igual"
    m2.escolhe = lambda pg: ("porta:arena", None, origem(m2, pg)[2])
    m2.passo()
    del m2.escolhe
    m2.salva()
    m2.telemetria(forcar=True)
    m2._det_t = 0
    m2.detalhe()
    det = json.load(open(os.path.join(pasta, "detalhe.json"), encoding="utf-8"))
    assert "laboratorio" in det, "painel sem o laboratorio"
    for arq in ("laboratorio_vivo.json", "holograma.json", "lab_quadro.jpg"):      # telas LABORATORIO e HOLOGRAMA
        assert os.path.exists(os.path.join(pasta, arq)), f"painel sem {arq}"
    h = json.load(open(os.path.join(pasta, "holograma.json"), encoding="utf-8"))
    assert h["teorias"] and h["teorias"][0]["cenas"], "holograma sem cena de teoria"
    erros = [ln for ln in open(os.path.join(pasta, "ontogenia.jsonl"), encoding="utf-8")
             if json.loads(ln)["ev"] in ("erro", "erro_orgao") or json.loads(ln)["ev"].endswith("_falhou")]
    assert not erros, "erros no diario:\n" + "".join(erros[:5])
    print(f"OK: escolhas livres {escolhas}; laboratorio {m2.lab.contas}; teorias {len(m2.lab.teorias)}; "
          f"arena.json visitas {m2._arena().st['visitas']}")
    print("ultimos:", list(m2.lab_ultimos)[-3:])
    sys.exit(0)
except SystemExit:
    raise
except Exception:
    print("FALHOU:\n" + traceback.format_exc()[-2000:])
    sys.exit(1)
