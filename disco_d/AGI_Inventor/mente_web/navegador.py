"""
mente_web.navegador — o corpo 2: um NAVEGADOR DE VERDADE (Edge invisivel)
==========================================================================
O corpo 1 lia o HTML como um leitor de tela: texto ja separado das tags, sem
JavaScript, sem pagina montada; imagem so se ela pedisse para baixar. Nao era ver.

Aqui a pagina e RENDERIZADA como para um humano, e ela:
  - OLHA a tela inteira em cores (1024 x 700), em resolucao total; para onde os
    olhos vao (a fovea) e decisao dela, nao do HTML (ver mente._ler_olhando);
  - ROLA tela por tela; o texto que ela le e o do bloco para onde ela olhou
    (como o dedo no braile: o texto vem do lugar do olhar);
  - CLICA nos links pelas coordenadas na tela (se o link estiver ali);
  - ASSISTE videos DENTRO da pagina: clica no video, ele toca em tempo real,
    ela recebe 5 quadros por segundo e OUVE o som (16 kHz) do que esta tocando.

Seguranca (corpo limitado): perfil temporario isolado, sem downloads, sem
permissoes (camera, local, notificacoes), janelas extras e alertas fechados na hora;
NENHUM pedido que nao seja de leitura (POST/PUT/DELETE bloqueados na rede: ela nao
consegue enviar, comprar, comentar, cadastrar nada); video so carrega quando ela
decide assistir; nenhuma caixa de som toca (navegador mudo, o som vai so para ela);
robots.txt e intervalo por dominio como no corpo 1. Se o navegador falhar, cai no corpo 1.
"""
from __future__ import annotations

import io
import json
import os
import time
import urllib.parse

import numpy as np

from corpo import Corpo, Pagina, dominio

LARG, ALT = 1024, 700
MAX_TELAS = 10
REINICIA_A_CADA = 120          # paginas: o navegador e reaberto (memoria nao cresce)
_SENTIDOS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentidos.js"), encoding="utf-8").read()

_JS_MAPA = """() => {
  const sy = scrollY, out = {blocos: [], imagens: [], links: [], videos: []};
  const cache = new Map();
  const blocoDe = (el) => {
    let e = el;
    while (e && e !== document.body) {
      if (cache.has(e)) return cache.get(e);
      const d = getComputedStyle(e).display;
      if (d !== "contents" && !d.startsWith("inline")) { cache.set(el, e); return e; }
      e = e.parentElement;
    }
    return document.body;
  };
  const bl = new Map();
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n, tot = 0;
  while ((n = w.nextNode()) && tot < 60000) {
    const t = n.textContent.replace(/\\s+/g, " ").trim();
    if (t.length < 2) continue;
    const el = n.parentElement; if (!el) continue;
    const cs = getComputedStyle(el); if (cs.display === "none" || cs.visibility === "hidden") continue;
    const rg = document.createRange(); rg.selectNodeContents(n); const r = rg.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    const b = blocoDe(el);
    let a = bl.get(b);
    if (!a) { a = {x0: r.left, y0: r.top + sy, x1: r.right, y1: r.bottom + sy, t: []}; bl.set(b, a); }
    a.x0 = Math.min(a.x0, r.left); a.y0 = Math.min(a.y0, r.top + sy);
    a.x1 = Math.max(a.x1, r.right); a.y1 = Math.max(a.y1, r.bottom + sy);
    a.t.push(t); tot += t.length;
  }
  for (const a of bl.values())
    out.blocos.push([Math.round(a.x0), Math.round(a.y0), Math.round(a.x1 - a.x0), Math.round(a.y1 - a.y0), a.t.join(" ")]);
  out.blocos.sort((p, q) => (p[1] - q[1]) || (p[0] - q[0]));
  for (const im of document.images) {
    const r = im.getBoundingClientRect();
    if (r.width >= 64 && r.height >= 48) out.imagens.push([Math.round(r.left), Math.round(r.top + sy), Math.round(r.width), Math.round(r.height), (im.alt || "").slice(0, 120)]);
  }
  for (const a of document.links) {
    const r = a.getBoundingClientRect(), t = (a.innerText || "").trim();
    if (r.width > 4 && r.height > 4 && t) out.links.push([Math.round(r.left), Math.round(r.top + sy), Math.round(r.width), Math.round(r.height), t.slice(0, 200), a.href]);
  }
  [...document.querySelectorAll("video,audio")].forEach((v, i) => {
    const r = v.getBoundingClientRect();
    const s = v.currentSrc || v.src || (v.querySelector("source") || {}).src || "";
    if ((r.width >= 80 && r.height >= 30) || v.tagName === "AUDIO")
      out.videos.push([Math.round(r.left), Math.round(r.top + sy), Math.round(r.width), Math.round(r.height), v.tagName.toLowerCase(), i, s,
                       (v.getAttribute("aria-label") || v.title || "").slice(0, 120)]);
  });
  out.forms = [];
  for (const f of document.forms) {
    if ((f.method || "get").toLowerCase() !== "get" || f.querySelector("textarea")) continue;
    const ins = [...f.querySelectorAll("input")], tipos = ins.map((i) => (i.type || "text").toLowerCase());
    if (tipos.some((t) => ["password", "file", "email", "tel"].includes(t))) continue;
    const txt = ins.filter((i, k) => ["text", "search"].includes(tipos[k]) && i.name);
    if (txt.length !== 1) continue;
    out.forms.push([f.action, txt[0].name, ins.filter((i, k) => tipos[k] === "hidden" && i.name).map((i) => [i.name, i.value || ""])]);
  }
  out.altura = document.documentElement.scrollHeight;
  return out;
}"""


def _rgb(jpg):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB"))


class CorpoNavegador(Corpo):
    def __init__(self, log=None):
        super().__init__(log)
        self._pw = self._nav = self._ctx = self._pg = None
        self._mapa = None
        self._paginas = 0
        self.pasta_painel = None                  # a mente diz onde mostrar a tela dela
        self.falhas_navegador = 0
        self.ultimo_clique = None
        self._deixa_midia = False                 # video so carrega quando ELA decide assistir
        self.bloqueados_envio = 0                 # pedidos de envio barrados na rede
        self.assistiu_info = {}

    # ── ciclo de vida do navegador ───────────────────────────────────────────
    @staticmethod
    def _limpa_orfaos():
        """Navegadores invisiveis de uma vida anterior (PC desligado no meio) nao podem
        ficar acumulando memoria. So mata os do Playwright em modo invisivel — nunca o
        Edge do usuario."""
        import psutil
        for p in psutil.process_iter(["name", "cmdline"]):
            try:
                cl = " ".join(p.info["cmdline"] or [])
                if (p.info["name"] or "").lower() == "msedge.exe" and "--headless" in cl and "playwright" in cl:
                    p.kill()
            except Exception:
                pass

    def _rota(self, rota):
        rq = rota.request
        try:
            if rq.method not in ("GET", "HEAD", "OPTIONS"):        # o corpo dela nunca ENVIA nada
                self.bloqueados_envio += 1
                return rota.abort()
            if rq.resource_type == "media" and not self._deixa_midia:
                return rota.abort()
            return rota.continue_()
        except Exception:
            pass

    def _liga(self):
        if self._pg is not None and self._paginas < REINICIA_A_CADA:
            return
        self._desliga()
        if not getattr(CorpoNavegador, "_limpou", False):
            CorpoNavegador._limpou = True
            self._limpa_orfaos()
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._nav = self._pw.chromium.launch(
            channel="msedge" if os.name == "nt" else None, headless=True,
            args=["--disable-gpu", "--mute-audio", "--no-first-run", "--disable-extensions",
                  "--disable-background-networking", "--disable-component-update", "--disable-default-apps",
                  "--disable-features=Translate,MediaRouter,OptimizationHints", "--renderer-process-limit=2",
                  "--js-flags=--max-old-space-size=256", "--autoplay-policy=user-gesture-required"])
        self._ctx = self._nav.new_context(viewport={"width": LARG, "height": ALT}, accept_downloads=False,
                                          user_agent=self.sess.headers["User-Agent"], permissions=[],
                                          locale="pt-BR")
        self._ctx.add_init_script(_SENTIDOS)
        self._ctx.route("**/*", self._rota)
        self._pg = self._ctx.new_page()
        self._ctx.on("page", lambda p: p.close() if p != self._pg else None)       # nada de janela extra
        self._pg.on("dialog", lambda d: d.dismiss())                               # nada de alerta
        self._paginas = 0

    def _desliga(self):
        for f in (lambda: self._ctx.close(), lambda: self._nav.close(), lambda: self._pw.stop()):
            try:
                f()
            except Exception:
                pass
        self._pw = self._nav = self._ctx = self._pg = None

    # ── portas: imagem e video abrem a PAGINA do arquivo, no navegador ───────
    def aleatorio(self, porta):
        if porta not in ("imagem", "video"):
            return super().aleatorio(porta)
        api = super().aleatorio(porta)                 # sorteio pela API publica do Commons
        if api.tipo == "erro" or not api.midias:
            return api
        _tipo, _u, titulo = api.midias[int(np.random.default_rng().integers(len(api.midias)))]
        url = "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(titulo.decode("utf-8", "ignore").replace(" ", "_"))
        pg = self.abrir(url, empilha=True, origem=f"porta:{porta}")
        if pg.tipo == "erro" or not getattr(pg, "navegador", False):
            return api
        return pg

    # ── abrir = ver a pagina montada ─────────────────────────────────────────
    def abrir(self, url, empilha=True, origem=""):
        if not self._permitido(url):
            return Pagina(url, dominio(url), tipo="erro", origem="sem_rede" if self._offline else "robots")
        if not self._espera(url):
            return Pagina(url, dominio(url), tipo="erro", origem="esperando")
        try:
            self._liga()
            self._deixa_midia = False
            self.ultimo_clique = None
            clicou = False
            if self._mapa:                                   # o link esta na tela? entao CLICA nele
                for (x, y, w, h, t, href) in self._mapa["links"]:
                    if href == url:
                        self._pg.evaluate(f"window.scrollTo(0, {max(0, y - ALT // 3)})")
                        self._pg.wait_for_timeout(250)
                        sy = self._pg.evaluate("scrollY")                 # a rolagem REAL
                        cx, cy = x + w / 2, y - sy + h / 2
                        if not (0 <= cy < ALT):
                            break
                        self._mostra(foco=None, clique=(cx, cy), nota=f"clicando em “{t[:60]}”")
                        try:
                            with self._pg.expect_navigation(timeout=12000):
                                self._pg.mouse.click(cx, cy)
                            clicou = True
                            self.ultimo_clique = {"x": cx, "y": cy, "texto": t[:80]}
                        except Exception:
                            clicou = False
                        break
            if not clicou:
                self._pg.goto(url, wait_until="domcontentloaded", timeout=20000)
            self._pg.wait_for_timeout(900)
            self.n_pedidos += 1
            self._paginas += 1
            final = self._pg.url
            if not final.startswith("http") or not self._permitido(final):
                return Pagina(final, dominio(final), tipo="erro", origem="robots")
            mapa = self._pg.evaluate(_JS_MAPA)
            self._mapa = mapa
            texto = self._pg.evaluate("() => document.body ? document.body.innerText : ''").encode("utf-8")[:400000]
            pg = Pagina(final, dominio(final), texto=texto, origem=origem)
            pg.links = [(t.encode("utf-8"), href) for (_x, _y, _w, _h, t, href) in mapa["links"]
                        if href.startswith("http")][:400]
            # cada video/audio da pagina e um objeto que ela pode decidir assistir
            pg.midias = [("audio" if v[4] == "audio" else "video", f"el:{v[5]}", (v[7] or "").encode("utf-8"))
                         for v in mapa["videos"][:6]]
            from corpo import PROIBIDO_FORM
            pg.formularios = [(a, n, tuple(tuple(h) for h in hs)) for (a, n, hs) in mapa.get("forms", [])
                              if a.startswith("http") and not any(w in a.lower() for w in PROIBIDO_FORM)]
            pg.telas = self._telas(mapa)
            pg.navegador = True
            if empilha:
                self.pilha.append(final)
            self.falhas_navegador = 0
            return pg
        except Exception as e:
            self.falhas_navegador += 1
            if self.falhas_navegador >= 3:
                self._desliga()
            if not self.rede_ok():
                return Pagina(url, dominio(url), tipo="erro", origem="sem_rede")
            # o navegador falhou: o corpo 1 continua funcionando
            pg = super().abrir(url, empilha=empilha, origem=origem)
            pg.origem = (pg.origem or "") + f" (navegador falhou: {type(e).__name__})"
            return pg

    def _telas(self, mapa):
        """Agrupa os blocos de texto, imagens e videos da pagina em TELAS (o que cabe na
        janela de cada vez). Coordenadas em relacao ao topo da pagina."""
        altura = max(mapa.get("altura", ALT), ALT)
        telas = []
        for i in range(min(MAX_TELAS, int(np.ceil(altura / ALT)))):
            y0, y1 = i * ALT, (i + 1) * ALT
            bls = [b for b in mapa["blocos"] if y0 <= b[1] < y1]
            ims = [m for m in mapa["imagens"] if y0 <= m[1] + m[3] / 2 < y1]
            vds = [v for v in mapa["videos"] if y0 <= v[1] + v[3] / 2 < y1 and v[4] == "video"]
            if not bls and not ims and not vds:
                continue
            telas.append({"y": y0, "texto": " ".join(b[4] for b in bls).encode("utf-8"),
                          "blocos": bls, "imagens": ims, "videos": vds})
        return telas

    def olha_tela(self, tela):
        """Rola ate a tela e OLHA: a tela inteira em cores, resolucao total (RGB uint8)."""
        self._pg.evaluate(f"window.scrollTo(0, {tela['y']})")
        self._pg.wait_for_timeout(350)
        jpg = self._pg.screenshot(type="jpeg", quality=80)
        sy = self._pg.evaluate("scrollY")
        self._ultima_jpg, self._ultima_sy = jpg, sy
        return _rgb(jpg), sy

    # ── assistir e ouvir, dentro da pagina, em tempo real ────────────────────
    def assistir(self, chave, segundos, passo=0.2):
        """Gera (t do video, quadro RGB do video na tela, audio mono 16 kHz do intervalo
        ou None) em TEMPO REAL, enquanto o video toca na pagina. Quem consome decide
        quando parar (basta parar de pedir: o video e pausado)."""
        if not str(chave).startswith("el:"):
            yield from super().assistir(chave, segundos, passo)      # corpo 1: arquivo direto
            return
        if self._pg is None:
            return
        i = int(chave[3:])
        pg = self._pg
        self.assistiu_info = {"clicou": False, "mao": False, "ouviu": 0, "trechos": 0, "sem_som_do_site": False}
        try:
            box = pg.evaluate(f"""() => {{ const v = document.querySelectorAll('video,audio')[{i}]; if (!v) return null;
                v.scrollIntoView({{block: 'center'}}); return 1; }}""")
            if not box:
                return
            pg.wait_for_timeout(300)
            box = pg.evaluate(f"() => {{ const r = document.querySelectorAll('video,audio')[{i}].getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }}")
            self._deixa_midia = True
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            if 0 <= cy < ALT and box[2] > 0:
                self._mostra(clique=(cx, cy), nota="clicando no video para assistir")
                pg.mouse.click(cx, cy)                                   # a mao dela aperta o play
                self.assistiu_info["clicou"] = True
            pg.wait_for_timeout(1500)
            st = pg.evaluate("() => { const v = window.__vivo(); return v ? [v.paused, v.currentTime] : null }")
            if st is None:
                return
            if st[0] or st[1] == 0:                                     # o player nao reagiu ao clique
                pg.evaluate("() => { const v = window.__vivo(); v.play && v.play().catch(() => {}); }")
                self.assistiu_info["mao"] = True
                pg.wait_for_timeout(1200)
            pg.evaluate("() => window.__ouve(window.__vivo())")
            t0 = time.time()
            parado = 0
            while time.time() - t0 < segundos:
                a = time.time()
                r = pg.evaluate("""() => { const v = window.__vivo(); if (!v) return null; const r = v.getBoundingClientRect();
                    return [r.left, r.top, r.width, r.height, v.currentTime, v.ended, v.paused, v.tagName, !!v.__semCors] }""")
                if r is None or r[5]:
                    break
                parado = parado + 1 if r[6] else 0
                if parado > 10:
                    break
                quadro = None
                if r[7] == "VIDEO" and r[2] >= 16 and r[3] >= 16:
                    clip = {"x": max(0, r[0]), "y": max(0, r[1]), "width": min(LARG - max(0, r[0]), r[2]),
                            "height": min(ALT - max(0, r[1]), r[3])}
                    if clip["width"] >= 16 and clip["height"] >= 16:
                        jpg = pg.screenshot(type="jpeg", quality=75, clip=clip)
                        quadro = _rgb(jpg)
                        self._ultima_jpg = jpg
                        self.video_caixa = [clip["x"], clip["y"], clip["width"], clip["height"]]
                e = pg.evaluate("() => window.__escuta(window.__vivo())")
                audio = None
                self.assistiu_info["trechos"] += 1
                self.assistiu_info["sem_som_do_site"] = bool(r[8])
                if e and e["n"] >= 256:
                    audio = np.array(e["amostras"], np.float32) / 32767.0
                    if float(np.abs(audio).max()) > 1e-4:
                        self.assistiu_info["ouviu"] += 1
                yield float(r[4]), quadro, audio
                dt = time.time() - a
                if dt < passo:
                    time.sleep(passo - dt)
        except GeneratorExit:
            raise
        except Exception:
            return
        finally:
            try:
                pg.evaluate("() => document.querySelectorAll('video,audio').forEach(v => { try { v.pause() } catch (e) {} })")
            except Exception:
                pass
            self._deixa_midia = False

    # ── o painel: o que ela esta vendo AGORA ─────────────────────────────────
    def _mostra(self, foco=None, clique=None, nota="", imagens=(), tela_i=None, n_telas=None, reconhece=None,
                extra=None, foto_nova=False):
        """Grava o que ela esta vendo AGORA para o painel (tela.jpg + tela.json)."""
        if not self.pasta_painel or self._pg is None:
            return
        try:
            if clique is not None or foto_nova or not hasattr(self, "_ultima_jpg"):
                self._ultima_jpg = self._pg.screenshot(type="jpeg", quality=60)
                self._ultima_sy = self._pg.evaluate("scrollY")
            arq = os.path.join(self.pasta_painel, "tela.jpg")
            with open(arq + ".tmp", "wb") as f:
                f.write(self._ultima_jpg)
            os.replace(arq + ".tmp", arq)
            info = {"quando": time.time(), "url": self._pg.url, "largura": LARG, "altura": ALT,
                    "rolagem": getattr(self, "_ultima_sy", 0), "altura_pagina": (self._mapa or {}).get("altura", ALT),
                    "foco": foco, "clique": clique, "imagens": list(imagens), "nota": nota,
                    "tela": tela_i, "telas": n_telas, "reconhece": reconhece, "modo": "pagina",
                    "envios_barrados": self.bloqueados_envio}
            if extra:
                info.update(extra)
            with open(os.path.join(self.pasta_painel, "tela.json.tmp"), "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False)
            os.replace(os.path.join(self.pasta_painel, "tela.json.tmp"), os.path.join(self.pasta_painel, "tela.json"))
        except Exception:
            pass
