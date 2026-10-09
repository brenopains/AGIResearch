"""
mente_web.viver — deixa a mente viver por um tempo de relogio, salvando e retomando
=====================================================================================
Uso:  python viver.py <pasta> <minutos> [semente]
Se a pasta ja tem uma mente salva, ela ACORDA de onde parou (continuidade).
Nao recebe tarefa, URL de conteudo nem lingua. So nasce numa porta aleatoria.
"""
import os
import sys
import time
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from mente import Mente  # noqa: E402

if __name__ == "__main__":
    pasta = sys.argv[1]
    minutos = float(sys.argv[2])
    sem = int(sys.argv[3]) if len(sys.argv) > 3 else 20261005
    m = Mente.carrega(pasta, sem)
    if m.pagina is None:
        porta = list(__import__("corpo").PORTAS)[int(m.rng.integers(3))]
        m.pagina = m.corpo.aleatorio(porta)
        m.ev("nasceu", porta=porta, url=m.pagina.url)
        m.ler(m.pagina)
    fim = time.time() + minutos * 60
    ultimo_salvo = time.time()
    erros = 0
    while time.time() < fim:
        try:
            m.passo()
            erros = 0
        except KeyboardInterrupt:
            break
        except Exception as e:
            erros += 1
            m.ev("erro", tipo=type(e).__name__, msg=str(e)[:200], pilha=traceback.format_exc()[-600:])
            if erros >= 5:
                m.pagina = m.corpo.aleatorio("wiki_en")
                erros = 0
        if time.time() - ultimo_salvo > 900:
            m.salva()
            ultimo_salvo = time.time()
    m.dorme()
    m.salva()
    print(f"viveu {m.t} passos, {m.n_pag} paginas lidas, {len(m.E)} estruturas, "
          f"{m.texto.bytes_lidos/1e6:.1f} MB lidos, {m.corpo.n_pedidos} pedidos")
