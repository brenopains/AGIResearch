"""
painel.servidor — o painel da mente (so local: http://127.0.0.1:8765)
=====================================================================
Le os registros da vida (ontogenia.jsonl) de forma incremental e serve:
  /            a pagina
  /api/estado  tudo o que a pagina mostra (JSON)
  /api/fala    o resumo falado ("desde a ultima vez...")
Nao escreve nada na mente; so le. Uso: pythonw servidor.py [pasta_da_vida]
"""
from __future__ import annotations

import collections
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psutil

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
import controle  # noqa: E402
sys.path.insert(0, AQUI)
RAIZ = os.path.dirname(AQUI)
PORTA = 8765

NOME_ACAO = {
    "seguir_novo": "seguindo um link para um lugar onde nunca esteve",
    "seguir_familiar": "seguindo um link que lhe parece familiar",
    "seguir_acaso": "seguindo um link ao acaso",
    "seguir_lacuna": "seguindo um link por causa de uma pergunta",
    "assistir": "assistindo a um vídeo (imagem e som juntos)",
    "ver": "olhando imagens",
    "voltar": "voltando à página anterior",
    "porta:wiki_pt": "abrindo uma página aleatória da Wikipédia em português",
    "porta:wiki_en": "abrindo uma página aleatória da Wikipédia em inglês",
    "porta:pequena_web": "pulando para um site pequeno qualquer da web",
}
NOME_BARRA = {"enciclopedia_pt": "Wikipédia (pt)", "enciclopedia_en": "Wikipédia (en)",
              "pequena_web": "pequena web", "midia": "Wikimedia Commons", "arquivo": "Internet Archive"}
NOME_LACUNA = {"depois": "não sabe o que vem depois de", "inesperado": "não consegue antecipar",
               "divergente": "vê usos diferentes de", "ligacao": "quer saber se combina com a imagem:"}


def acha_vida():
    if len(sys.argv) > 1:
        return os.path.abspath(sys.argv[1])
    for nome in ("viva", "gemea_v2"):                 # a vida continua; antes dela, a gemea v2
        p = os.path.join(RAIZ, "vidas", nome)
        if os.path.exists(os.path.join(p, "ontogenia.jsonl")):
            return p
    cands = [os.path.join(RAIZ, "vidas", d) for d in os.listdir(os.path.join(RAIZ, "vidas"))]
    cands = [c for c in cands if os.path.exists(os.path.join(c, "ontogenia.jsonl"))]
    return max(cands, key=lambda c: os.path.getmtime(os.path.join(c, "ontogenia.jsonl")))


class Leitor:
    """Agrega a ontogenia sem reler tudo: guarda onde parou."""

    def __init__(self, pasta):
        self.pasta = pasta
        self.arq = os.path.join(pasta, "ontogenia.jsonl")
        self.pos = 0
        self.lock = threading.Lock()
        self.c = collections.Counter()
        self.acoes = collections.Counter()
        self.paginas = self.bytes_lidos = self.imagens = self.videos = 0
        self.dominios = set()
        self.feed = collections.deque(maxlen=40)
        self.pesquisas = collections.deque(maxlen=12)
        self.altas = collections.deque(maxlen=40)
        self.nivel_max = 1
        self.curva = []                       # (vida_s, bits/byte no 1o contato)
        self._balde = []
        self.ultimo_sono = {}
        self.ultimo_sonho = {}
        self.ultima_acao = None
        self.vida_s = 0.0
        self.t = 0
        self.versao = "v1"
        self.sem_rede = False
        self.barras = []
        self.fechadas = 0
        self.imagin = {"real": 0, "nulo": 0}
        self.construcoes = []
        self.inicio_parede = None
        self.seq = 0
        self.recentes = collections.deque(maxlen=400)   # para animar o organismo

    def atualiza(self):
        if not os.path.exists(self.arq):
            return
        with self.lock, open(self.arq, "rb") as f:
            f.seek(self.pos)
            bloco = f.read()
            fim = bloco.rfind(b"\n")
            if fim < 0:
                return
            self.pos += fim + 1
            for linha in bloco[:fim].split(b"\n"):
                try:
                    self._um(json.loads(linha))
                except Exception:
                    continue

    LEVES = {"leu": ("dominio",), "acao": ("estrategia", "pagou"), "pesquisa": ("consulta", "barra"),
             "nascimento": ("nivel", "pedaco"), "lacuna": ("tipo", "alvo"), "fechou": ("alvo", "como"),
             "sonho": ("ligou", "deprimiu", "rejeitou"), "imaginou_e_viu": ("par",), "viu": ("n",),
             "assistiu": ("passos",), "sem_rede": (), "rede_voltou": (), "deprimiu": ("unidade",),
             "construcao": ("molde",), "barra_descoberta": ("dominio",), "aterramento": ("pedaco",),
             "ligacao_av": (), "transplante": ("orgaos",)}

    def _um(self, e):
        ev = e.get("ev")
        if ev in self.LEVES and not (ev == "nascimento" and (e.get("nivel") or 1) < 2):
            self.seq += 1
            self.recentes.append({"seq": self.seq, "ev": ev, "vida_s": e.get("vida_s", 0),
                                  **{k: e.get(k) for k in self.LEVES[ev]}})
        self.c[ev] += 1
        self.vida_s = max(self.vida_s, e.get("vida_s", 0.0))
        self.t = max(self.t, e.get("t", 0))
        if ev == "versao":
            self.versao = e.get("versao", "v1")
        elif ev == "leu":
            self.url_atual = e.get("url")
            self.paginas += 1
            self.bytes_lidos += e.get("lidos") or 0
            self.dominios.add(e.get("dominio"))
            b = e.get("bits_primeiro_bloco")
            if b:
                self._balde.append(b)
                if len(self._balde) >= 20:
                    self.curva.append((round(self.vida_s), round(sum(self._balde) / len(self._balde), 3)))
                    self._balde = []
                    if len(self.curva) > 400:
                        self.curva = self.curva[::2]
        elif ev == "acao":
            self.acoes[e["estrategia"]] += 1
            self.ultima_acao = e
            self.sem_rede = False
        elif ev == "viu":
            self.imagens += e.get("n", 0)
        elif ev == "assistiu":
            self.videos += 1
            self._feed(e, f"assistiu a um vídeo ({e.get('passos', 0)} instantes de imagem e som)")
        elif ev == "pesquisa":
            b = e["barra"]
            onde = NOME_BARRA.get(b, b[1:] if b.startswith("@") else b)
            self.pesquisas.append({"consulta": e["consulta"], "onde": onde, "porque": e.get("alvo"),
                                   "tipo": e.get("lacuna")})
        elif ev == "nascimento" and (e.get("nivel") or 1) >= 2:
            n = e["nivel"]
            self.nivel_max = max(self.nivel_max, n)
            self.altas.append({"u": e["pedaco"], "n": n})
            if n >= 3:
                self._feed(e, f"formou uma unidade de nível {n}: “{e['pedaco'].strip()}”")
        elif ev == "fechou":
            self.fechadas += 1
            if e.get("como") == "perseguida":
                self._feed(e, f"resolveu uma pergunta que perseguiu: “{e['alvo'].strip()}”")
        elif ev == "imaginou_e_viu":
            self.imagin["real"] = e.get("confirmadas", self.imagin["real"])
            self.imagin["nulo"] = e.get("placebo", self.imagin["nulo"])
            self._feed(e, f"imaginou “{''.join(e['par']).strip()}” no sono e depois encontrou no mundo", "ouro")
        elif ev == "placebo_viu":
            self.imagin["nulo"] = e.get("placebo", self.imagin["nulo"])
        elif ev == "construcao":
            self.construcoes.append(" ".join(e["molde"]))
            self._feed(e, f"criou uma construção com lacuna: “{' '.join(e['molde'])}”", "ouro")
        elif ev == "barra_descoberta":
            self.barras.append(e["dominio"])
            self._feed(e, f"descobriu sozinha uma barra de busca em {e['dominio']}")
        elif ev == "aterramento":
            self._feed(e, f"ligou “{e['pedaco'].strip()}” a um padrão visual que se repete", "ouro")
        elif ev == "ligacao_av":
            self._feed(e, "ligou um padrão de imagem a um padrão de som (confirmado em vídeo novo)", "ouro")
        elif ev == "deprimiu":
            self._feed(e, f"desfez “{e['unidade'].strip()}”: re-testou e não pagava mais")
        elif ev == "sono":
            self.ultimo_sono = e
        elif ev == "sonho":
            self.ultimo_sonho = e
        elif ev == "sem_rede":
            self.sem_rede = True
            self._feed(e, "a internet sumiu: ela parou de agir e está consolidando o que já viveu", "alerta")
        elif ev == "rede_voltou":
            self.sem_rede = False
            self._feed(e, "a internet voltou: ela acordou e seguiu de onde estava")
        elif ev == "acordou":
            self._feed(e, f"acordou com toda a memória (passo {e.get('passos')})")

    def _feed(self, e, texto, tipo="normal"):
        self.feed.append({"vida_s": e.get("vida_s", 0), "texto": texto, "tipo": tipo})

    def pensamento(self):
        a = self.ultima_acao
        if a is None:
            return "acordando…"
        if self.sem_rede:
            return "sem internet — consolidando o que já viveu"
        e = a["estrategia"]
        if e.startswith("pesquisar:"):
            b = e.split(":", 1)[1]
            onde = NOME_BARRA.get(b, b[1:] if b.startswith("@") else b)
            q = a.get("consulta") or "…"
            alvo = (a.get("alvo_lacuna") or "").strip()
            porque = NOME_LACUNA.get(a.get("lacuna"), "quer entender")
            return f"pesquisando “{q}” em {onde}, porque {porque} “{alvo}”"
        return NOME_ACAO.get(e, e)

    def processo(self):
        agora = time.time()
        if agora - getattr(self, "_proc_t", 0) < 10:
            return self._proc
        self._proc_t, self._proc = agora, self._processo()
        return self._proc

    def _processo(self):
        for p in psutil.process_iter(["cmdline", "memory_info", "cpu_percent"]):
            try:
                cl = " ".join(p.info["cmdline"] or [])
            except Exception:
                continue
            nome = os.path.basename(self.pasta)
            if "viver.py" in cl and nome in cl:
                try:
                    return {"viva": True, "ram_mb": round(p.memory_info().rss / 2 ** 20),
                            "cpu": round(p.cpu_percent(interval=0.1), 1)}
                except Exception:
                    pass
        return {"viva": False, "ram_mb": 0, "cpu": 0.0}

    def disco_mb(self):
        agora = time.time()
        if agora - getattr(self, "_disco_t", 0) < 60:
            return self._disco
        self._disco_t, self._disco = agora, self._disco_mb()
        return self._disco

    def _disco_mb(self):
        tot = 0
        for raiz, _d, fs in os.walk(self.pasta):
            for f in fs:
                try:
                    tot += os.path.getsize(os.path.join(raiz, f))
                except OSError:
                    pass
        return round(tot / 2 ** 20)

    def estado(self):
        self.atualiza()
        with self.lock:
            s = self.ultimo_sono
            pr = self.processo()
            status = "SEM INTERNET" if (pr["viva"] and self.sem_rede) else ("VIVA" if pr["viva"] else "DORMINDO")
            pz = controle.estado(self.pasta)
            if pz:
                status = "PAUSANDO" if pz == "pausando" else "PAUSADA"
            return {
                "status": status, "versao": self.versao, "vida_s": self.vida_s, "passos": self.t,
                "url_atual": getattr(self, "url_atual", None),
                "pensamento": self.pensamento(),
                "consumo": {"paginas": self.paginas, "mb_lidos": round(self.bytes_lidos / 1e6, 1),
                            "sites": len(self.dominios), "pedidos": s.get("pedidos", 0),
                            "imagens": self.imagens, "videos": self.videos,
                            "ram_mb": pr["ram_mb"], "cpu": pr["cpu"], "disco_mb": self.disco_mb()},
                "mente": {"estruturas": s.get("estruturas", 0), "fusoes": s.get("fusoes", 0),
                          "niveis": s.get("niveis", {}), "nivel_max": self.nivel_max,
                          "prototipos_visuais": s.get("prototipos_g", 0),
                          "prototipos_sonoros": s.get("prototipos_a", 0),
                          "aterrados": s.get("aterrados", 0), "aterrados_nulo": s.get("aterrados_nulo", 0),
                          "lacunas": s.get("lacunas_abertas", {}), "fechadas": self.fechadas,
                          "perguntas": s.get("perguntas", [])[:6],
                          "altas": sorted({a["u"].strip(): a["n"] for a in self.altas}.items(),
                                          key=lambda z: (-z[1], -len(z[0])))[:10]},
                "imaginacao": {"confirmadas": self.imagin["real"], "placebo": self.imagin["nulo"],
                               "imaginadas": self.ultimo_sonho.get("imaginadas"),
                               "construcoes": self.construcoes[-6:],
                               "sonhos": self.c.get("sonho", 0)},
                "acoes": self.acoes.most_common(8), "barras": self.barras[-12:],
                "pesquisas": list(self.pesquisas)[::-1],
                "feed": list(self.feed)[::-1][:16], "curva": self.curva[-160:],
            }


def _n(x):
    return f"{int(x):,}".replace(",", ".")


def fala(L, est):
    """O relatorio falado, no jeito de um mordomo digital (Jarvis): formal, curto, preciso,
    'senhor', humor seco SO quando um fato pede. Tudo o que diz e medido; nada inventado."""
    arq = os.path.join(AQUI, "visto.json")
    antes = json.load(open(arq, encoding="utf-8")) if os.path.exists(arq) else None
    c, m = est["consumo"], est["mente"]
    olhadas = 0
    try:
        olhadas = json.load(open(os.path.join(L.pasta, "olho.json"), encoding="utf-8")).get("olhadas", 0)
    except Exception:
        pass
    agora = {"paginas": c["paginas"], "sites": c["sites"], "estruturas": m["estruturas"],
             "fechadas": m["fechadas"], "confirmadas": est["imaginacao"]["confirmadas"],
             "vida_s": est["vida_s"], "imagens": c.get("imagens", 0), "videos": c.get("videos", 0),
             "olhadas": olhadas}
    h = est["vida_s"] / 3600
    idade = f"{h / 24:.1f} dias".replace(".", ",") if h >= 48 else f"{h:.1f} horas".replace(".", ",")
    hora = time.localtime().tm_hour
    partes = [("Bom dia" if 5 <= hora < 12 else "Boa tarde" if 12 <= hora < 18 else "Boa noite") + ", senhor."]
    if est["status"] in ("PAUSADA", "PAUSANDO"):
        partes.append(f"A mente está em pausa, como o senhor pediu. Vida acumulada: {idade}. "
                      "Ela retoma de onde parou quando o senhor mandar.")
    elif est["status"] == "DORMINDO":
        partes.append(f"A mente está em repouso. Vida acumulada: {idade}.")
    elif est["status"] == "SEM INTERNET":
        partes.append("Estamos sem conexão. Ela está consolidando o que já viu; volto a informar quando a rede retornar.")
    else:
        partes.append(f"Todos os sistemas operando. Vida acumulada: {idade}.")
    if antes:
        dp, ds = agora["paginas"] - antes["paginas"], agora["sites"] - antes["sites"]
        de = agora["estruturas"] - antes["estruturas"]
        partes.append(f"Desde a sua última visita: {_n(dp)} páginas, {_n(ds)} sites novos "
                      f"e {'mais' if de >= 0 else 'menos'} {_n(abs(de))} estruturas na memória.")
        di, dv = agora["imagens"] - antes.get("imagens", 0), agora["videos"] - antes.get("videos", 0)
        if di or dv:
            partes.append(f"Ela olhou {_n(di)} imagens e assistiu {_n(dv)} vídeos.")
        dol = agora["olhadas"] - antes.get("olhadas", 0)
        if dol > 0:
            partes.append(f"O córtex visual registrou {_n(dol)} olhadas.")
        dc = agora["confirmadas"] - antes["confirmadas"]
        if dc > 0:
            partes.append(f"Durante o sono, imaginou {_n(dc)} combinações que o mundo depois confirmou. Nada mal.")
        if dp > 30 and ds == 0:
            partes.append("Ela parece ter se afeiçoado aos mesmos sites.")
    else:
        partes.append(f"Até o momento: {_n(c['paginas'])} páginas em {_n(c['sites'])} sites, "
                      f"cerca de {_n(c['mb_lidos'])} megabytes de texto.")
    if m["altas"]:
        partes.append(f"A construção mais alta que ela formou sozinha: {m['altas'][0][0].strip(' ,.;:-')}.")
    if est["status"] in ("VIVA", "SEM INTERNET"):
        partes.append(f"Neste momento, ela está {est['pensamento']}.")
    partes.append("Deseja ver o que ela está vendo, senhor?")
    json.dump(agora, open(arq, "w", encoding="utf-8"))
    return " ".join(partes)


L = None
_checou = 0.0


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _envia(self, corpo, tipo):
        b = corpo.encode("utf-8") if isinstance(corpo, str) else corpo
        self.send_response(200)
        self.send_header("Content-Type", tipo)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        """PAUSAR / RETOMAR (so do proprio PC: o painel escuta apenas em 127.0.0.1)."""
        if self.path.startswith("/api/pausar"):
            r = controle.pausar(L.pasta)
        elif self.path.startswith("/api/retomar"):
            r = controle.retomar(L.pasta)
        else:
            self.send_response(404)
            self.end_headers()
            return
        L._proc_t = 0
        self._envia(json.dumps({"estado": r or "viva"}), "application/json; charset=utf-8")

    def do_GET(self):
        global L, _checou
        if len(sys.argv) <= 1 and time.time() - _checou > 60:     # a vencedora virou "viva"?
            _checou = time.time()
            p = acha_vida()
            if p != L.pasta:
                L = Leitor(p)
        if self.path.startswith("/api/estado"):
            self._envia(json.dumps(L.estado(), ensure_ascii=False), "application/json; charset=utf-8")
        elif self.path.startswith("/api/eventos"):
            L.atualiza()
            try:
                desde = int(self.path.split("desde=")[1].split("&")[0]) if "desde=" in self.path else 0
            except ValueError:
                desde = 0
            with L.lock:
                evs = [e for e in L.recentes if e["seq"] > desde][-200:]
                ult = L.seq
            self._envia(json.dumps({"ultimo": ult, "eventos": evs}, ensure_ascii=False),
                        "application/json; charset=utf-8")
        elif self.path.split("?")[0] in ("/api/olho.jpg", "/api/palavras_visuais.jpg", "/api/verbos.jpg"):
            nome = self.path.split("?")[0].rsplit("/", 1)[1]
            try:
                self._envia(open(os.path.join(L.pasta, nome), "rb").read(), "image/jpeg")
            except Exception:
                self.send_response(404)
                self.end_headers()
        elif self.path.startswith("/api/olho"):
            try:
                v = json.load(open(os.path.join(L.pasta, "olho.json"), encoding="utf-8"))
                v["atraso_s"] = round(time.time() - v.get("quando", 0), 1)
            except Exception:
                v = {"indisponivel": True}
            self._envia(json.dumps(v, ensure_ascii=False), "application/json; charset=utf-8")
        elif self.path.startswith("/api/tela.jpg"):
            try:
                self._envia(open(os.path.join(L.pasta, "tela.jpg"), "rb").read(), "image/jpeg")
            except Exception:
                self.send_response(404)
                self.end_headers()
        elif self.path.startswith("/api/tela"):
            try:
                v = json.load(open(os.path.join(L.pasta, "tela.json"), encoding="utf-8"))
                v["atraso_s"] = round(time.time() - v.get("quando", 0), 1)
            except Exception:
                v = {"indisponivel": True}
            self._envia(json.dumps(v, ensure_ascii=False), "application/json; charset=utf-8")
        elif self.path.startswith("/api/detalhe"):
            try:
                v = json.load(open(os.path.join(L.pasta, "detalhe.json"), encoding="utf-8"))
                v["atraso_s"] = round(time.time() - v.get("quando", 0), 1)
            except Exception:
                v = {"indisponivel": True}
            self._envia(json.dumps(v, ensure_ascii=False), "application/json; charset=utf-8")
        elif self.path.startswith("/api/visao"):
            try:
                v = json.load(open(os.path.join(L.pasta, "visao.json"), encoding="utf-8"))
                v["atraso_s"] = round(time.time() - v.get("quando", 0), 1)
            except Exception:
                v = {"indisponivel": True}
            self._envia(json.dumps(v), "application/json; charset=utf-8")
        elif self.path.startswith("/api/fala_audio"):
            # ENCAIXE DE VOZ: se existir painel/voz.json com {"comando": [...]} (um programa de
            # voz LOCAL que le o texto na entrada e grava um WAV em {saida}), o relatorio sai
            # nessa voz; sem ele, o painel usa a voz do proprio navegador.
            cfg = os.path.join(AQUI, "voz.json")
            if not os.path.exists(cfg):
                self.send_response(404)
                self.end_headers()
                return
            import subprocess
            import tempfile
            saida = os.path.join(tempfile.gettempdir(), "mente_relatorio.wav")
            cmd = [x.replace("{saida}", saida).replace("{python}", sys.executable).replace("{voz}", os.path.join(RAIZ, "..", "voz"))
                   for x in json.load(open(cfg, encoding="utf-8"))["comando"]]
            subprocess.run(cmd, input=fala(L, L.estado()).encode("utf-8"), timeout=120,
                           creationflags=0x08000000 if os.name == "nt" else 0)
            self._envia(open(saida, "rb").read(), "audio/wav")
        elif self.path.startswith("/api/organismo"):
            arq = os.path.join(L.pasta, "telemetria.json")
            try:
                tel = json.load(open(arq, encoding="utf-8"))
                tel["atraso_s"] = round(time.time() - tel.get("parede", 0), 1)
                tel["vida_pasta"] = os.path.basename(L.pasta)
            except Exception:
                tel = {"indisponivel": True, "vida_pasta": os.path.basename(L.pasta)}
            self._envia(json.dumps(tel, ensure_ascii=False), "application/json; charset=utf-8")
        elif self.path.startswith("/api/fala"):
            self._envia(fala(L, L.estado()), "text/plain; charset=utf-8")
        else:
            self._envia(open(os.path.join(AQUI, "index.html"), "rb").read(), "text/html; charset=utf-8")


def vigia():
    """Quando a internet volta (a mente registra 'rede_voltou'), o painel pula na tela."""
    import janela
    ultimo, pasta = None, None
    while True:
        try:
            atual = L
            atual.atualiza()
            n = atual.c.get("rede_voltou", 0)
            if pasta == atual.pasta and ultimo is not None and n > ultimo:
                janela.abrir()
            ultimo, pasta = n, atual.pasta
        except Exception:
            pass
        time.sleep(10)


if __name__ == "__main__":
    L = Leitor(acha_vida())
    L.atualiza()
    threading.Thread(target=vigia, daemon=True).start()
    porta = int(sys.argv[2]) if len(sys.argv) > 2 else PORTA
    ThreadingHTTPServer(("127.0.0.1", porta), H).serve_forever()
