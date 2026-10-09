"""
novaphys.coleta — trazer mundo de fora com TETO e PROCEDENCIA
===============================================================

DUAS RESTRICOES DO USUARIO, E AS DUAS VIRAM CODIGO AQUI
---------------------------------------------------------
"so cuidado com o armazenamento"  -> TETO EM BYTES, conferido antes de cada
    arquivo e abortado no limite. Nao e recomendacao num comentario: o laco
    para. C: tem 31 GB livres e esta a 94%; tudo vai para D:, que tem 65 GB.

"tipo uns 5 gb de cada, nossa AGI nao precisaria de tantos dados assim de cara"
    -> e isso esta certo, e agora tem respaldo medido: `modelo_mundo` mediu que
    o gargalo NAO e volume. Com 5 mundos e 1146 fragmentos o operador ja
    atravessa a 3,9x o acaso. O que faltou la foi SINAL DE DESFECHO, nao dado.

O CRITERIO DE ESCOLHA, QUE MUDOU DEPOIS DO RESULTADO DE ONTEM
---------------------------------------------------------------
[[modelo-mundo-credito]] mediu que o credito por dominio protege contra fonte
que mente, MAS exige sinal de desfecho: sondas com rotulo do que de fato
aconteceu. Entao a ordem de coleta nao e por tamanho nem por fama — e por
quanto de desfecho a fonte carrega junto:

  VPT (Minecraft)   acao por frame, sincronizada com o video. Desfecho TOTAL.
                    5096 episodios de ~3,4 MB. E gameplay, que e o que o
                    usuario pediu, e e acao->consequencia de verdade.
  PDEBench          simulador exato: o estado seguinte E o rotulo. Desfecho
                    TOTAL, e e fisica de verdade em vez de operador sintetico.
  OC20              energia e forca por configuracao. Desfecho TOTAL, quimica.
  Something-Somethg 174 templates que SAO operadores ("empurrando algo da
                    esquerda para a direita"). Exige cadastro.
  Ego4D             exige licenca assinada e revisada. Ultimo, e nao por
                    qualidade: por porta administrativa.

O MANIFESTO NAO E BUROCRACIA
------------------------------
[[procedencia-atravessa-recombinando]] mediu que a cadeia fonte -> executavel
so atravessa quando da para RECOMBINAR, e para isso tem que dar para dizer de
onde cada peca veio. Cada arquivo baixado entra no manifesto com URL de origem,
tamanho e data. Sem isso o dominio nao pode receber credito nem ser rebaixado,
e o mecanismo inteiro de [[modelo-mundo-credito]] fica sem chao.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence, Tuple

DESTINO = r"D:\agi_dados"
MANIFESTO = os.path.join(DESTINO, "manifesto.json")
GB = 1024 ** 3


def _livre(caminho: str) -> int:
    import shutil
    return shutil.disk_usage(os.path.splitdrive(caminho)[0] + os.sep).free


def manifesto_carrega() -> Dict:
    if os.path.exists(MANIFESTO):
        try:
            return json.load(open(MANIFESTO, encoding="utf-8"))
        except Exception:
            pass
    return {}


def manifesto_grava(m: Dict) -> None:
    os.makedirs(DESTINO, exist_ok=True)
    json.dump(m, open(MANIFESTO, "w", encoding="utf-8"), indent=1)


def baixa(url: str, alvo: str, tempo: int = 120) -> int:
    """Baixa um arquivo. Devolve bytes escritos, 0 se falhou."""
    os.makedirs(os.path.dirname(alvo), exist_ok=True)
    tmp = alvo + ".parcial"
    try:
        socket.setdefaulttimeout(tempo)
        req = urllib.request.Request(url, headers={"User-Agent": "novaphys"})
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
            n = 0
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
                n += len(b)
        os.replace(tmp, alvo)
        return n
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass          # no Windows o `.parcial` pode estar preso por outro
        return 0          # processo; o tratador nao pode ser quem derruba tudo


def coleta(dominio: str, itens: Sequence[Tuple[str, str]], teto_gb: float,
           margem_gb: float = 10.0, fios: int = 16, verbose: bool = True) -> Dict:
    """
    Baixa (url, caminho_relativo) ate estourar `teto_gb` — ou ate o disco
    chegar em `margem_gb` livres, o que vier primeiro.

    EM PARALELO, e isso nao e enfeite: medido sequencial, 15 MB em 170 s deu
    0,09 MB/s, o que poria 5 GB em ~17 horas. O gargalo e latencia por arquivo
    (sao milhares de arquivos de 3 MB), nao banda — entao `fios` resolve.

    O teto e conferido a cada arquivo concluido e o laco PARA de submeter. Como
    ha ate `fios` downloads em voo quando isso acontece, o estouro maximo e
    `fios` arquivos — aqui ~50 MB, e eu prefiro isso a serializar tudo.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    raiz = os.path.join(DESTINO, dominio)
    m = manifesto_carrega()
    d = m.setdefault(dominio, {"arquivos": {}, "bytes": 0, "origem": ""})
    teto = int(teto_gb * GB)
    baixados, novos = d["bytes"], 0
    inicio = baixados
    t0 = time.time()

    pendentes = [(u, r) for u, r in itens
                 if not (r in d["arquivos"]
                         and os.path.exists(os.path.join(raiz,
                                                         r.replace("/", os.sep))))]
    if verbose:
        print("  %d arquivos pendentes, %d fios" % (len(pendentes), fios))

    fila = iter(pendentes)
    parar = False

    def _um(par):
        u, r = par
        alvo = os.path.join(raiz, r.replace("/", os.sep))
        return r, u, baixa(u, alvo)

    with ThreadPoolExecutor(max_workers=fios) as ex:
        voando = {}
        for _ in range(fios):
            try:
                p = next(fila)
            except StopIteration:
                break
            voando[ex.submit(_um, p)] = p
        while voando:
            for fut in as_completed(list(voando)):
                voando.pop(fut, None)
                try:
                    r, u, n = fut.result()
                except Exception:
                    n = 0
                if n:
                    d["arquivos"][r] = {"url": u, "bytes": n,
                                        "data": time.strftime("%Y-%m-%d")}
                    d["bytes"] = baixados = baixados + n
                    novos += 1
                    manifesto_grava(m)   # a CADA arquivo: matar o processo
                    if verbose and novos % 100 == 0:   # nao pode custar a procedencia
                        dt = max(time.time() - t0, 1e-9)
                        print("    %5d arq | %.2f de %.1f GB | %.1f MB/s"
                              % (novos, baixados / GB, teto_gb,
                                 (baixados - inicio) / 1e6 / dt), flush=True)
                        manifesto_grava(m)
                if baixados >= teto:
                    parar = True
                if _livre(DESTINO) < margem_gb * GB:
                    if verbose:
                        print("  MARGEM DE DISCO: parando.")
                    parar = True
                if not parar:
                    try:
                        p = next(fila)
                        voando[ex.submit(_um, p)] = p
                    except StopIteration:
                        pass
                break
    if parar and verbose:
        print("  TETO de %.1f GB atingido — parando." % teto_gb)
    manifesto_grava(m)
    if verbose:
        dt = max(time.time() - t0, 1e-9)
        print("  %s: %d arquivos, %.2f GB, %.1f MB/s (disco livre %.1f GB)"
              % (dominio, len(d["arquivos"]), d["bytes"] / GB,
                 (baixados - inicio) / 1e6 / dt, _livre(DESTINO) / GB))
    return d


# ------------------------------------------------------------------- VPT
IDX_VPT = ("https://openaipublic.blob.core.windows.net/minecraft-rl/"
           "snapshots/all_8xx_Jun_29.json")


def vpt(teto_gb: float = 5.0, so_acao: bool = True, verbose: bool = True) -> Dict:
    """
    Gameplay de Minecraft com a ACAO de cada frame num .jsonl ao lado do .mp4.

    E a fonte que melhor responde a queixa do usuario sobre as colecoes velhas
    ("nao tem experiencia temporal, nao tem sincronia de eventos, nao tem
    multimodalidade"): cada linha do jsonl traz mouse, teclas, xpos/ypos/zpos,
    yaw/pitch, inventario, tick e milissegundo. Acao no tick t, consequencia no
    tick t+1, relogio partilhado com o video.

    ERRO MEU, CORRIGIDO COM AMOSTRA: eu estimei "3,4 MB por episodio" sondando
    UM arquivo — o primeiro do indice, que e um cotoco. Amostra de 20:
        mp4    media 141 MB (mediana 164)
        jsonl  media  15 MB (mediana  14)
    A colecao inteira e 776 GB, nao 17. Em 5 GB cabem ~32 episodios completos.

    `so_acao` e a consequencia de desenho disso, e e o que respeita o teto: o
    jsonl custa um DECIMO do mp4 e ja traz a consequencia (posicao, inventario,
    stats). Em 5 GB isso da ~335 episodios, ~1,8 milhao de passos de acao, em
    vez de 32 episodios. O video so passa a ser necessario quando entrar o
    leitor visual — e ai se baixa video para um SUBCONJUNTO.
    """
    socket.setdefaulttimeout(60)
    req = urllib.request.Request(IDX_VPT, headers={"User-Agent": "novaphys"})
    idx = json.load(urllib.request.urlopen(req))
    base, rel = idx["basedir"], idx["relpaths"]
    if verbose:
        print("  indice VPT: %d episodios | medido: ~141 MB de video, ~15 MB"
              % len(rel))
        print("  de acao por episodio | modo: %s"
              % ("SO ACAO (jsonl)" if so_acao else "acao + video"))
    itens: List[Tuple[str, str]] = []
    for p in rel:
        s = p[:-4] if p.endswith(".mp4") else p
        itens.append((base + s + ".jsonl", s + ".jsonl"))
        if not so_acao:
            itens.append((base + s + ".mp4", s + ".mp4"))
    m = manifesto_carrega()
    m.setdefault("vpt", {"arquivos": {}, "bytes": 0})["origem"] = IDX_VPT
    manifesto_grava(m)
    return coleta("vpt", itens, teto_gb, verbose=verbose)


# ------------------------------------------------ mundo humano, sem cadastro
# 24/09: o Ego4D travou no cadastro. Tudo abaixo foi conferido por HEAD antes de
# entrar aqui — abre sem conta, aceita Range, e cada fonte traz junto uma ANCORA
# HUMANA (nomes que pessoas deram ao que acontece) que a digestora NUNCA ve, so
# o experimento, para medir se o que ela extraiu coincide com o que humanos
# nomeiam — o mesmo desenho de [[descobre-andar]].

def xadrez(teto_gb: float = 0.5, verbose: bool = True) -> Dict:
    """
    Pessoas jogando xadrez (Lichess, CC0). Partidas cruas de 2013 (~18 MB/mes)
    e o banco de PROBLEMAS com TEMAS nomeados por humanos (garfo, cravada,
    espeto, ataque descoberto, mate no corredor...) — a ancora.
    """
    base = "https://database.lichess.org/"
    itens = [(base + "lichess_db_puzzle.csv.zst", "lichess_db_puzzle.csv.zst")]
    for mes in range(1, 7):
        n = "lichess_db_standard_rated_2013-%02d.pgn.zst" % mes
        itens.append((base + "standard/" + n, n))
    m = manifesto_carrega()
    m.setdefault("xadrez", {"arquivos": {}, "bytes": 0})["origem"] = base + " (CC0)"
    manifesto_grava(m)
    return coleta("xadrez", itens, teto_gb, fios=4, verbose=verbose)


ATARI_JOGOS = ("breakout", "freeway", "montezuma_revenge", "ms_pacman",
               "space_invaders", "seaquest")


def atari(teto_gb: float = 2.5, verbose: bool = True) -> Dict:
    """
    Pessoas jogando Atari com o OLHAR gravado (Atari-HEAD, Zenodo, CC-BY 4.0):
    frame, acao e ponto de fixacao do olho. O olhar e sinal de RELEVANCIA — o
    que faltou no HAR ([[protocolo-nao-percepcao]]).
    """
    base = "https://zenodo.org/records/3451402/files/"
    itens = [(base + "action_enums.txt", "action_enums.txt")]
    itens += [(base + j + ".zip", j + ".zip") for j in ATARI_JOGOS]
    m = manifesto_carrega()
    m.setdefault("atari_head", {"arquivos": {}, "bytes": 0})["origem"] = \
        "zenodo.org/records/3451402 (CC-BY 4.0)"
    manifesto_grava(m)
    return coleta("atari_head", itens, teto_gb, fios=3, verbose=verbose)


EPIC = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/"
EPIC_ROTULOS = ("https://raw.githubusercontent.com/epic-kitchens/"
                "epic-kitchens-55-annotations/master/EPIC_train_action_labels.csv")


def _tamanho(url: str) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "novaphys"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return int(r.headers.get("Content-Length", 0))
    except Exception:
        return 0


def epic(teto_gb: float = 4.0, max_mb: int = 700, verbose: bool = True) -> Dict:
    """
    Video EGOCENTRICO com AUDIO (EPIC-KITCHENS, Bristol, sem cadastro): maos,
    objetos, cozinha, som. Substitui o Ego4D. Sonda os tamanhos, prefere videos
    curtos e no maximo dois por pessoa (variedade paga: [[variedade-paga]]).
    Rotulos verbo/substantivo sao a ancora humana.
    """
    from concurrent.futures import ThreadPoolExecutor
    cands = ["P%02d/P%02d_%02d.MP4" % (p, p, v) for p in range(1, 32)
             for v in range(1, 11)]
    with ThreadPoolExecutor(16) as ex:
        tams = dict(zip(cands, ex.map(lambda c: _tamanho(EPIC + c), cands)))
    ok = sorted((t, c) for c, t in tams.items() if 0 < t <= max_mb * 1e6)
    por_pessoa: Dict[str, int] = {}
    escolhidos, soma = [], 0
    for t, c in ok:
        p = c[:3]
        if por_pessoa.get(p, 0) >= 2 or soma + t > teto_gb * GB:
            continue
        por_pessoa[p] = por_pessoa.get(p, 0) + 1
        escolhidos.append(c)
        soma += t
    if verbose:
        print("  EPIC: %d videos existem ate %d MB; escolhidos %d de %d pessoas, "
              "%.2f GB" % (len(ok), max_mb, len(escolhidos), len(por_pessoa),
                           soma / GB))
    itens = [(EPIC_ROTULOS, "EPIC_train_action_labels.csv")]
    itens += [(EPIC + c, c) for c in escolhidos]
    m = manifesto_carrega()
    m.setdefault("epic", {"arquivos": {}, "bytes": 0})["origem"] = \
        "data.bris.ac.uk EPIC-KITCHENS-55 (CC BY-NC 4.0)"
    manifesto_grava(m)
    return coleta("epic", itens, teto_gb + 0.1, fios=4, verbose=verbose)


class _Remoto:
    """Arquivo HTTP legivel com seek, por Range: deixa o `zipfile` ler um zip de
    16 GB tirando SO os membros escolhidos, sem baixar o resto."""

    def __init__(self, url: str, bloco: int = 1 << 20) -> None:
        self.url, self.pos, self.bloco = url, 0, bloco
        self.n = _tamanho(url)
        self._ini, self._buf = -1, b""

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, de: int = 0) -> int:
        self.pos = off if de == 0 else self.pos + off if de == 1 else self.n + off
        return self.pos

    def read(self, k: int = -1) -> bytes:
        if k < 0:
            k = self.n - self.pos
        if not (self._ini <= self.pos and self.pos + k <= self._ini + len(self._buf)):
            fim = min(self.n, self.pos + max(k, self.bloco)) - 1
            req = urllib.request.Request(self.url, headers={
                "User-Agent": "novaphys", "Range": "bytes=%d-%d" % (self.pos, fim)})
            with urllib.request.urlopen(req, timeout=120) as r:
                self._buf = r.read()
            self._ini = self.pos
        a = self.pos - self._ini
        out = self._buf[a:a + k]
        self.pos += len(out)
        return out


CHARADES = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/"


def charades(teto_gb: float = 3.0, verbose: bool = True) -> Dict:
    """
    Pessoas em casa fazendo coisas do dia a dia (Charades 480p, AllenAI, com
    audio do celular): 157 acoes rotuladas por humanos como ancora. Tira um
    SUBCONJUNTO do zip de 16 GB por Range, na ordem dos nomes (sem escolher a
    dedo), ate o teto.
    """
    import zipfile
    raiz = os.path.join(DESTINO, "charades")
    os.makedirs(raiz, exist_ok=True)
    m = manifesto_carrega()
    d = m.setdefault("charades", {"arquivos": {}, "bytes": 0})
    d["origem"] = CHARADES + "Charades_v1_480.zip (AllenAI, uso nao comercial)"
    anot = os.path.join(raiz, "Charades.zip")
    if not os.path.exists(anot):
        n = baixa(CHARADES + "Charades.zip", anot)
        d["arquivos"]["Charades.zip"] = {"url": CHARADES + "Charades.zip", "bytes": n,
                                         "data": time.strftime("%Y-%m-%d")}
        d["bytes"] += n
    z = zipfile.ZipFile(_Remoto(CHARADES + "Charades_v1_480.zip", 8 << 20))
    membros = sorted((i for i in z.infolist() if i.filename.endswith(".mp4")),
                     key=lambda i: i.filename)
    if verbose:
        print("  Charades: %d videos no zip remoto" % len(membros))
    for k, info in enumerate(membros):
        if d["bytes"] >= teto_gb * GB or _livre(DESTINO) < 10 * GB:
            break
        nome = os.path.basename(info.filename)
        if nome in d["arquivos"]:
            continue
        try:
            dado = z.read(info)
        except Exception:
            continue
        open(os.path.join(raiz, nome), "wb").write(dado)
        d["arquivos"][nome] = {"url": "Charades_v1_480.zip:" + info.filename,
                               "bytes": len(dado), "data": time.strftime("%Y-%m-%d")}
        d["bytes"] += len(dado)
        if k % 50 == 0:
            manifesto_grava(m)
            if verbose:
                print("    %d videos | %.2f GB" % (len(d["arquivos"]), d["bytes"] / GB),
                      flush=True)
    manifesto_grava(m)
    if verbose:
        print("  charades: %d arquivos, %.2f GB" % (len(d["arquivos"]), d["bytes"] / GB))
    return d


FONTES: Dict[str, Callable] = {"vpt": vpt, "xadrez": xadrez, "atari": atari,
                               "epic": epic, "charades": charades}


if __name__ == "__main__":
    import sys
    nome = sys.argv[1] if len(sys.argv) > 1 else "vpt"
    teto = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    print("=" * 78)
    print("COLETA — %s, teto %.1f GB, destino %s" % (nome, teto, DESTINO))
    print("=" * 78)
    print("  disco livre antes: %.1f GB" % (_livre(DESTINO) / GB))
    FONTES[nome](teto)
