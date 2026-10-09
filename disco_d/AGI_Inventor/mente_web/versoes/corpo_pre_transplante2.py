"""
mente_web.corpo — o corpo: um navegador de verdade, so-leitura
================================================================
O organismo recebe portas, nao conteudo. Ninguem lhe da uma URL de conteudo:
ele sabe que existem barras de busca e portas aleatorias, e decide.

Acoes:
  aleatorio(porta)          nascer/voltar a um lugar qualquer
  pesquisar(barra, texto)   barras: pequena_web (wiby), enciclopedia_pt,
                            enciclopedia_en (API da Wikipedia), midia (API do
                            Commons: video/audio/imagem), arquivo (Internet Archive)
  abrir(url)                pagina: bytes do texto visivel, links (ancora, url),
                            midias (tipo, url)
  assistir(url, segundos)   video COM audio, no mesmo relogio (PyAV), em fluxo:
                            nada e gravado; so passa pelos sentidos
  ver(url)                  imagem
  voltar()
  usar_formulario(f, texto) v1: barras DESCOBERTAS nas proprias paginas — formulario
                            GET com exatamente um campo de texto (como o rastreador
                            do Google faz desde 2008, Madhavan et al.); robots vale
                            para a URL de destino

Seguranca (corpo limitado, pedido do usuario e da outra IA):
  - so GET; nada de formulario, login, cookie guardado, JavaScript, upload;
  - robots.txt respeitado para paginas; APIs publicas seguem a regra delas
    (User-Agent identificado, no maximo 1 pedido por segundo);
  - intervalo minimo por dominio; tamanho maximo por pagina;
  - nunca baixa executavel/arquivo compactado; midia so em fluxo, com limite
    de tempo, e descartada depois dos sentidos.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass, field

import numpy as np
import requests

# Politica da Wikimedia (mediawiki.org/wiki/Wikimedia_APIs/Rate_limits): sem contato
# no User-Agent o cliente e "nao identificado" e o limite e 10 pedidos/min. Contato so
# entra se o usuario autorizar (variavel de ambiente MENTE_CONTATO); sem ele, 1 pedido a
# cada 6,5 s para TODO o grupo Wikimedia.
def _contato():
    """Contato autorizado pelo usuario em 08/10 (contato.txt ao lado deste arquivo)."""
    v = os.environ.get("MENTE_CONTATO", "")
    if v:
        return v
    arq = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contato.txt")
    return open(arq, encoding="utf-8").read().strip() if os.path.exists(arq) else ""


CONTATO = _contato()
UA = ("MenteOrganismoPesquisa/0.1 (somente leitura; experimento de pesquisa nao comercial"
      + (f"; {CONTATO}" if CONTATO else "") + ") python-requests")
# duas vidas ao mesmo tempo (gemeas) dividem o limite: MENTE_PARTES=2 -> 13 s cada
PARTES = max(1, int(os.environ.get("MENTE_PARTES", "1")))
INTERVALO_WIKIMEDIA = (0.5 if CONTATO else 6.5) * PARTES
WIKIMEDIA = ("wikipedia.org", "wikimedia.org", "wiktionary.org", "wikidata.org", "mediawiki.org",
             "wikibooks.org", "wikisource.org", "wikiquote.org", "wikinews.org", "wikiversity.org",
             "wikivoyage.org")
MAX_BYTES = 3_000_000
INTERVALO_DOMINIO = 3.0
INTERVALO_API = 1.0
PROIBIDO_EXT = (".exe", ".msi", ".zip", ".rar", ".7z", ".gz", ".tar", ".dmg", ".apk",
                ".iso", ".bin", ".jar", ".deb", ".rpm", ".pdf", ".doc", ".docx", ".xls")
PROIBIDO_FORM = ("donat", "login", "logon", "signin", "sign-in", "signup", "sign-up", "register",
                 "account", "checkout", "cart", "pay", "subscri", "newsletter", "contact", "mail",
                 "password", "auth", "comment", "post", "order", "buy", "shop")
VIDEO_EXT = (".webm", ".ogv", ".mp4", ".mov", ".mkv")
AUDIO_EXT = (".ogg", ".oga", ".mp3", ".wav", ".flac", ".opus", ".m4a")
IMG_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp")

# APIs publicas com politica propria (nao sao paginas rastreadas)
APIS = ("pt.wikipedia.org/w/api.php", "en.wikipedia.org/w/api.php",
        "commons.wikimedia.org/w/api.php", "archive.org/advancedsearch.php")

PORTAS = {
    "wiki_pt": "https://pt.wikipedia.org/wiki/Especial:Aleat%C3%B3ria",
    "wiki_en": "https://en.wikipedia.org/wiki/Special:Random",
    "pequena_web": "https://wiby.me/surprise/",
    "imagem": "https://commons.wikimedia.org/wiki/Special:Random/File",   # pela API (abaixo)
}
BARRAS = ("pequena_web", "enciclopedia_pt", "enciclopedia_en", "midia", "arquivo")
_META_REFRESH = re.compile(rb"""http-equiv=["']?refresh["']?[^>]*url=([^"'>\s]+)""", re.I)


@dataclass
class Pagina:
    url: str
    dominio: str
    texto: bytes = b""
    links: list = field(default_factory=list)      # (ancora bytes, url)
    midias: list = field(default_factory=list)     # (tipo, url, legenda bytes)
    tipo: str = "pagina"                           # pagina | busca | erro
    origem: str = ""
    formularios: list = field(default_factory=list)  # (acao, campo, ocultos)


def dominio(url):
    return urllib.parse.urlparse(url).netloc.lower()


class Corpo:
    def __init__(self, log=None):
        self.sess = requests.Session()
        self.sess.headers.update({"User-Agent": UA, "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8"})
        self.robots = {}
        self.ultimo = {}
        self.bloqueado = {}
        self._rede = (0.0, True)          # (quando checou, havia rede?)
        self._offline = False
        self.pilha = []
        self.log = log
        self.n_pedidos = 0
        self.bytes_vistos = 0

    # ── regras do corpo ──────────────────────────────────────────────────────
    def _eh_api(self, url):
        return any(a in url for a in APIS)

    def rede_ok(self):
        """O MUNDO sumiu ou so este lugar falhou? Pergunta ao endereco que o proprio
        Windows usa para isso (NCSI); resposta guardada por 20 s."""
        t, ok = self._rede
        if time.time() - t < 20:
            return ok
        try:
            r = requests.get("http://www.msftconnecttest.com/connecttest.txt", timeout=5)
            ok = r.status_code == 200 and "Microsoft" in r.text
        except Exception:
            ok = False
        self._rede = (time.time(), ok)
        return ok

    def _permitido(self, url):
        if self._eh_api(url):
            return True
        p = urllib.parse.urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        if p.path.lower().endswith(PROIBIDO_EXT):
            return False
        base = f"{p.scheme}://{p.netloc}"
        rp = self.robots.get(base)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            try:
                if not self._espera(base + "/robots.txt"):
                    return False                   # nao decide nada enquanto bloqueado
                r = self.sess.get(base + "/robots.txt", timeout=10, allow_redirects=True)
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
                elif r.status_code in (401, 403):
                    rp.parse(["User-agent: *", "Disallow: /"])
                else:
                    rp.parse([])
            except Exception:
                if not self.rede_ok():            # sem internet: nao decide nada sobre o site
                    self._offline = True
                    return False
                return False                      # site fora do ar: hoje nao; tenta outro dia
            self.robots[base] = rp
        return rp.can_fetch(UA, url)

    def _espera(self, url):
        """Respeita o intervalo; se o lugar pediu para esperar (429), NAO fica
        parado: devolve False e a mente vai fazer outra coisa."""
        d = dominio(url)
        lim = INTERVALO_API if self._eh_api(url) else INTERVALO_DOMINIO
        if d.endswith(WIKIMEDIA):
            d, lim = "wikimedia", max(lim, INTERVALO_WIKIMEDIA)
        agora = time.time()
        if self.bloqueado.get(d, 0.0) > agora:
            return False
        dt = agora - self.ultimo.get(d, 0.0)
        if dt < lim:
            time.sleep(lim - dt)
        self.ultimo[d] = time.time()
        return True

    def _get(self, url, **kw):
        self._offline = False
        if not self._permitido(url):
            return None, ("sem_rede" if self._offline else "robots")
        if not self._espera(url):
            return None, "esperando"
        try:
            r = self.sess.get(url, timeout=15, stream=True, allow_redirects=True, **kw)
        except Exception as e:
            if not self.rede_ok():
                return None, "sem_rede"
            return None, type(e).__name__
        self.n_pedidos += 1
        if r.status_code == 429:
            # pediram para esperar: o grupo inteiro fica parado o tempo pedido (>= 60 s)
            try:
                ra = float(r.headers.get("retry-after", 60))
            except ValueError:
                ra = 60.0
            g = "wikimedia" if dominio(url).endswith(WIKIMEDIA) else dominio(url)
            self.bloqueado[g] = time.time() + min(max(ra, 60.0), 3600.0)
            r.close()
            return None, "429"
        ct = r.headers.get("content-type", "").lower()
        if not any(t in ct for t in ("text/html", "text/plain", "json", "xml")):
            r.close()
            return None, f"tipo {ct[:30]}"
        corpo = b""
        for parte in r.iter_content(65536):
            corpo += parte
            if len(corpo) > MAX_BYTES:
                break
        r.close()
        self.bytes_vistos += len(corpo)
        return (r, corpo), None

    # ── acoes ────────────────────────────────────────────────────────────────
    def aleatorio(self, porta):
        if porta == "imagem":
            # uma imagem qualquer do mundo (Commons, API publica): a porta da visao
            url = ("https://commons.wikimedia.org/w/api.php?action=query&generator=random"
                   "&grnnamespace=6&grnlimit=6&prop=imageinfo&iiprop=url|mime&iiurlwidth=320&format=json")
            res, erro = self._get(url)
            pg = Pagina(url, "commons.wikimedia.org", tipo="busca", origem="porta:imagem")
            if res is None:
                pg.tipo, pg.origem = "erro", erro or ""
                return pg
            try:
                for p in json.loads(res[1]).get("query", {}).get("pages", {}).values():
                    ii = (p.get("imageinfo") or [{}])[0]
                    if ii.get("mime", "").startswith("image/") and ii.get("thumburl"):
                        t = p.get("title", "").encode("utf-8")
                        pg.midias.append(("imagem", ii["thumburl"], t))
                        pg.texto += t + b"\n"
            except Exception:
                pg.tipo = "erro"
            return pg
        if porta == "wiki_en":
            # Special:Random e proibido pelo robots.txt da Wikipedia em ingles; a
            # API publica da o mesmo sorteio e o artigo em si e permitido
            res, _e = self._get("https://en.wikipedia.org/w/api.php?action=query&list=random"
                                "&rnnamespace=0&rnlimit=1&format=json")
            try:
                t = json.loads(res[1])["query"]["random"][0]["title"]
            except Exception:
                return Pagina(PORTAS[porta], "en.wikipedia.org", tipo="erro", origem=_e or "porta")
            return self.abrir("https://en.wikipedia.org/wiki/" + urllib.parse.quote(t.replace(" ", "_")),
                              empilha=False, origem=f"porta:{porta}")
        return self.abrir(PORTAS[porta], empilha=False, origem=f"porta:{porta}")

    def abrir(self, url, empilha=True, origem=""):
        res, erro = self._get(url)
        if res is None:
            return Pagina(url, dominio(url), tipo="erro", origem=erro or "")
        r, corpo = res
        final = r.url
        meta = _META_REFRESH.search(corpo[:4000])
        if meta and len(corpo) < 3000 and origem != "refresh":
            alvo = urllib.parse.urljoin(final, meta.group(1).decode("utf-8", "ignore"))
            return self.abrir(alvo, empilha=empilha, origem="refresh")
        pg = _analisa_html(final, corpo) if b"<" in corpo[:2000] else \
            Pagina(final, dominio(final), texto=corpo[:200000])
        pg.origem = origem
        if empilha:
            self.pilha.append(final)
        return pg

    def pode_voltar(self):
        return len(self.pilha) >= 2

    def voltar(self):
        if len(self.pilha) >= 2:
            self.pilha.pop()
            return self.abrir(self.pilha.pop(), origem="voltar")
        return None

    def usar_formulario(self, f, texto):
        """So GET (seguro por definicao do HTTP), um campo de texto, sem senha/
        arquivo/area de texto; o texto e sempre uma estrutura que ela leu em
        pagina publica (nunca dado pessoal)."""
        acao, campo, ocultos = f
        q = urllib.parse.urlencode(list(ocultos) + [(campo, texto[:120])])
        sep = "&" if "?" in acao else "?"
        return self.abrir(acao + sep + q, origem=f"form:{dominio(acao)}")

    def pesquisar(self, barra, texto):
        q = urllib.parse.quote(texto[:120])
        if barra == "pequena_web":
            return self.abrir(f"https://wiby.me/?q={q}", origem=f"busca:{barra}")
        if barra in ("enciclopedia_pt", "enciclopedia_en"):
            lang = "pt" if barra.endswith("pt") else "en"
            url = (f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch={q}"
                   f"&srlimit=15&format=json&utf8=1")
            res, erro = self._get(url)
            pg = Pagina(url, f"{lang}.wikipedia.org", tipo="busca", origem=f"busca:{barra}")
            if res is None:
                pg.tipo, pg.origem = "erro", erro or ""
                return pg
            try:
                dados = json.loads(res[1])
                for h in dados.get("query", {}).get("search", []):
                    t = h["title"]
                    sn = re.sub("<[^>]+>", "", h.get("snippet", ""))
                    pg.links.append((f"{t} {sn}".encode("utf-8"),
                                     f"https://{lang}.wikipedia.org/wiki/" + urllib.parse.quote(t.replace(" ", "_"))))
                    pg.texto += (f"{t}. {sn}\n").encode("utf-8")
            except Exception:
                pg.tipo = "erro"
            return pg
        if barra == "midia":
            url = ("https://commons.wikimedia.org/w/api.php?action=query&generator=search"
                   f"&gsrsearch={q}%20filetype:video|audio|bitmap&gsrnamespace=6&gsrlimit=12"
                   "&prop=imageinfo&iiprop=url|mime|size&iiurlwidth=320&format=json")
            res, erro = self._get(url)
            pg = Pagina(url, "commons.wikimedia.org", tipo="busca", origem=f"busca:{barra}")
            if res is None:
                pg.tipo, pg.origem = "erro", erro or ""
                return pg
            try:
                dados = json.loads(res[1])
                for p in dados.get("query", {}).get("pages", {}).values():
                    ii = (p.get("imageinfo") or [{}])[0]
                    mime = ii.get("mime", "")
                    tipo = "video" if mime.startswith("video") else \
                        "audio" if mime.startswith("audio") else "imagem"
                    u = ii.get("thumburl") if tipo == "imagem" else ii.get("url")
                    if u:
                        leg = p.get("title", "").encode("utf-8")
                        pg.midias.append((tipo, u, leg))
                        pg.texto += leg + b"\n"
            except Exception:
                pg.tipo = "erro"
            return pg
        if barra == "arquivo":
            url = (f"https://archive.org/advancedsearch.php?q={q}&fl[]=identifier&fl[]=title"
                   "&fl[]=mediatype&rows=12&output=json")
            res, erro = self._get(url)
            pg = Pagina(url, "archive.org", tipo="busca", origem=f"busca:{barra}")
            if res is None:
                pg.tipo, pg.origem = "erro", erro or ""
                return pg
            try:
                dados = json.loads(res[1])
                for d in dados.get("response", {}).get("docs", []):
                    t = d.get("title", "")
                    t = t if isinstance(t, str) else " ".join(t)
                    pg.links.append((t.encode("utf-8"), "https://archive.org/details/" + d["identifier"]))
                    pg.texto += t.encode("utf-8") + b"\n"
            except Exception:
                pg.tipo = "erro"
            return pg
        raise ValueError(barra)

    def assistir(self, url, segundos, passo=0.2):
        """Video (ou audio) em FLUXO. Gera (t, quadro 32x32 em [0,1] ou None,
        audio mono 16 kHz do intervalo ou None). Nada e gravado em disco."""
        import av
        if not self._permitido(url) and not ("upload.wikimedia.org" in url or "archive.org" in url):
            return
        if not self._espera(url):
            return
        try:
            cont = av.open(url, timeout=15, options={"user_agent": UA})
        except Exception:
            return
        try:
            vs = cont.streams.video[0] if cont.streams.video else None
            as_ = cont.streams.audio[0] if cont.streams.audio else None
            streams = [s for s in (vs, as_) if s is not None]
            if not streams:
                return
            reamostra = av.AudioResampler(format="s16", layout="mono", rate=16000) if as_ else None
            quadro, aud = None, []
            prox = 0.0
            for pacote in cont.demux(*streams):
                for f in pacote.decode():
                    t = float(f.pts * f.time_base) if f.pts is not None else prox
                    if t > segundos:
                        return
                    if isinstance(f, av.VideoFrame):
                        img = f.to_ndarray(format="gray")
                        h, w = img.shape
                        img = img[: h - h % 32, : w - w % 32]
                        q = img.reshape(32, img.shape[0] // 32, 32, img.shape[1] // 32).mean((1, 3))
                        quadro = (q / 255.0).astype(np.float32)
                    else:
                        for rf in reamostra.resample(f):
                            aud.append(rf.to_ndarray().ravel().astype(np.float32) / 32768.0)
                    if t >= prox:
                        a = np.concatenate(aud) if aud else None
                        yield prox, quadro, a
                        aud = []
                        prox += passo
        except Exception:
            return
        finally:
            cont.close()

    def ver(self, url):
        from PIL import Image
        if not ("upload.wikimedia.org" in url or self._permitido(url)):
            return None
        if not self._espera(url):
            return None
        try:
            r = self.sess.get(url, timeout=15)
            if not r.headers.get("content-type", "").startswith("image"):
                return None
            im = Image.open(io.BytesIO(r.content)).convert("L").resize((32, 32))
            self.bytes_vistos += len(r.content)
            return np.asarray(im, np.float32) / 255.0
        except Exception:
            return None


def _analisa_html(url, corpo):
    from bs4 import BeautifulSoup
    try:
        s = BeautifulSoup(corpo, "lxml")
    except Exception:
        return Pagina(url, dominio(url), tipo="erro", origem="html")
    for t in s(["script", "style", "noscript", "svg", "template"]):
        t.decompose()
    pg = Pagina(url, dominio(url))
    links, midias = [], []
    for a in s.find_all("a", href=True):
        u = urllib.parse.urljoin(url, a["href"]).split("#")[0]
        if not u.startswith("http"):
            continue
        anc = a.get_text(" ", strip=True)[:200]
        low = u.lower()
        if low.endswith(VIDEO_EXT):
            midias.append(("video", u, anc.encode("utf-8")))
        elif low.endswith(AUDIO_EXT):
            midias.append(("audio", u, anc.encode("utf-8")))
        elif low.endswith(PROIBIDO_EXT):
            continue
        elif anc:
            links.append((anc.encode("utf-8"), u))
    for tag in s.find_all(["video", "audio", "source"]):
        src = tag.get("src")
        if src:
            u = urllib.parse.urljoin(url, src)
            tp = "audio" if tag.name == "audio" or u.lower().endswith(AUDIO_EXT) else "video"
            midias.append((tp, u, b""))
    for im in s.find_all("img", src=True)[:30]:
        u = urllib.parse.urljoin(url, im["src"])
        try:
            w = int(im.get("width", 0) or 0)
        except ValueError:
            w = 0
        if (w == 0 or w >= 80) and u.lower().split("?")[0].endswith(IMG_EXT):
            midias.append(("imagem", u, (im.get("alt") or "").encode("utf-8")))
    forms = []
    for f in s.find_all("form")[:10]:
        if (f.get("method") or "get").strip().lower() != "get" or f.find("textarea"):
            continue
        ins = f.find_all("input")
        tipos = [(i.get("type") or "text").lower() for i in ins]
        if any(t in ("password", "file", "email", "tel") for t in tipos):
            continue
        txt_in = [i for i, t in zip(ins, tipos) if t in ("text", "search") and i.get("name")]
        if len(txt_in) != 1:
            continue
        acao = urllib.parse.urljoin(url, f.get("action") or url).split("#")[0]
        if not acao.startswith("http"):
            continue
        # seguranca do corpo (nao e semantica para a mente): nunca formulario de
        # doacao, login, cadastro, conta, pagamento, assinatura, contato ou comentario
        if any(w in acao.lower() for w in PROIBIDO_FORM):
            continue
        ocultos = tuple((i["name"], i.get("value", "")) for i, t in zip(ins, tipos)
                        if t == "hidden" and i.get("name"))
        forms.append((acao, txt_in[0]["name"], ocultos))
    pg.formularios = forms
    pg.texto = s.get_text(" ", strip=True).encode("utf-8")[:400000]
    pg.links = links[:400]
    pg.midias = midias[:40]
    return pg
