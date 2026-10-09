"""
mente_web.sonda — a bateria FIXA do avaliador (a mente nunca le isto)
=======================================================================
Uso: python sonda.py            (uma vez, ANTES da vida longa)

Busca pelo mesmo corpo so-leitura: 25 paginas aleatorias da Wikipedia em
portugues, 15 em ingles, 15 da pequena web (wiby, sites que a mente quase
certamente nunca vai visitar) e 40 imagens aleatorias do Commons com o titulo.
Guarda em sondas/bateria.pkl com o sha256. Serve para comparar as fotografias
da mente (30 min, 1 h, 2 h, 4 h, 6 h) no MESMO material.
"""
import hashlib
import json
import os
import pickle
import time

from corpo import Corpo

if __name__ == "__main__":
    os.makedirs("sondas", exist_ok=True)
    arq = "sondas/bateria.pkl"
    if os.path.exists(arq):
        raise SystemExit("bateria ja existe; nao refazer (seria trocar a regua)")
    c = Corpo()
    textos = []
    for porta, n in (("wiki_pt", 25), ("wiki_en", 15), ("pequena_web", 15)):
        k = tent = 0
        while k < n and tent < n * 5:
            tent += 1
            pg = c.aleatorio(porta)
            if pg.tipo == "erro" or len(pg.texto) < 1500:
                continue
            if any(t["url"] == pg.url for t in textos):
                continue
            textos.append(dict(porta=porta, url=pg.url, dominio=pg.dominio, texto=pg.texto[:8192]))
            k += 1
        print(porta, k, flush=True)
    imagens = []
    tent = 0
    while len(imagens) < 40 and tent < 30:
        tent += 1
        url = ("https://commons.wikimedia.org/w/api.php?action=query&generator=random"
               "&grnnamespace=6&grnlimit=10&prop=imageinfo&iiprop=url|mime&iiurlwidth=320&format=json")
        res, _e = c._get(url)
        if res is None:
            continue
        for p in json.loads(res[1]).get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [{}])[0]
            if not ii.get("mime", "").startswith("image/") or not ii.get("thumburl"):
                continue
            q = c.ver(ii["thumburl"])
            if q is None:
                continue
            imagens.append(dict(titulo=p.get("title", "").encode("utf-8"), url=ii["thumburl"], quadro=q))
            if len(imagens) >= 40:
                break
        print("imagens", len(imagens), flush=True)
    dados = dict(criada=time.strftime("%Y-%m-%d %H:%M:%S"), textos=textos, imagens=imagens)
    b = pickle.dumps(dados)
    open(arq, "wb").write(b)
    h = hashlib.sha256(b).hexdigest().upper()
    open("sondas/bateria.sha256", "w").write(h + "\n")
    print("bateria:", len(textos), "textos,", len(imagens), "imagens; sha256", h[:16])
