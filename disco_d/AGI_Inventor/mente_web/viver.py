"""
mente_web.viver — deixa a mente viver por um tempo de relogio, salvando e retomando
=====================================================================================
Uso:  python viver.py <pasta> <minutos> [semente] [v1|v2]
Se a pasta ja tem uma mente salva, ela ACORDA de onde parou (continuidade).
Nao recebe tarefa, URL de conteudo nem lingua. So nasce numa porta aleatoria.

v1: tira uma FOTOGRAFIA completa da mente nos marcos da ontogenia (tempo de vida
acumulado): 30 min, 1 h, 2 h, 4 h, 6 h — em <pasta>/foto_<segundos>. As fotos
servem para perguntar o que a mente das 6 h faz que a das 30 min nao fazia.
"""
import os
import sys
import time
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from mente import Mente  # noqa: E402
from mente2 import Mente2  # noqa: E402

# marcos da ontogenia (tempo de vida acumulado) + uma fotografia por DIA de vida
# (diaria no 1o mes, semanal depois: as fotos sao o que mais ocupa disco)
MARCOS = (1800, 3600, 7200, 14400, 21600) + tuple(86400 * d for d in range(1, 31)) +     tuple(86400 * 7 * w for w in range(5, 53))

if __name__ == "__main__":
    pasta = sys.argv[1]
    minutos = float(sys.argv[2])
    sem = int(sys.argv[3]) if len(sys.argv) > 3 else 20261006
    classe = Mente2 if (len(sys.argv) > 4 and sys.argv[4] == "v2") else Mente
    m = classe.carrega(pasta, sem)
    m.ev("versao", versao=m.VERSAO)
    if m.pagina is None:
        porta = list(__import__("corpo").PORTAS)[int(m.rng.integers(3))]
        m.pagina = m.corpo.aleatorio(porta)
        m.ev("nasceu", porta=porta, url=m.pagina.url)
        m.ler(m.pagina)
    fim = time.time() + minutos * 60
    ultimo_salvo = time.time()
    erros = 0
    while time.time() < fim:
        antes = m.vida_s
        try:
            m.passo()
            erros = 0
        except KeyboardInterrupt:
            break
        except Exception as e:
            erros += 1
            m._congeladas = set()
            m.ev("erro", tipo=type(e).__name__, msg=str(e)[:200], pilha=traceback.format_exc()[-800:])
            if erros >= 5:
                m.pagina = m.corpo.aleatorio("wiki_en")
                erros = 0
        for mc in MARCOS:
            if antes < mc <= m.vida_s:
                m.dorme()
                m.salva(sub=f"foto_{mc}")
        if time.time() - ultimo_salvo > 900:
            m.salva()
            ultimo_salvo = time.time()
        # parar com calma (PARAR) ou recomecar o bloco com codigo novo (REINICIAR): salva antes
        reinicia = os.path.join(pasta, "REINICIAR")
        if os.path.exists(os.path.join(pasta, "PARAR")) or os.path.exists(reinicia):
            if os.path.exists(reinicia):
                os.remove(reinicia)
            m.ev("parada_calma")
            break
    m.dorme()
    m.salva()
    print(f"viveu {m.t} passos ({m.vida_s / 60:.0f} min de vida), {m.n_pag} paginas lidas, "
          f"{len(m.E)} estruturas, {len(m.fusoes)} fusoes, {len(m.fechadas)} lacunas fechadas, "
          f"{m.texto.bytes_lidos / 1e6:.1f} MB lidos, {m.corpo.n_pedidos} pedidos")
