"""
teste_mundos — a aba MUNDOS mostra EXATAMENTE o que ela tem, e nao mexe nela (transplante 12)
==============================================================================================
Uso:  python teste_mundos.py [pasta_de_uma_mente_com_conceitos]
Sem argumento: so a parte de mundos fisicos (arena). Com argumento: COPIA a pasta (nunca toca
a vida real) e testa tambem os conceitos.

  W0 FIDELIDADE      cada caminho publicado = a conta do laboratorio refeita agora; cada pixel
                     das camadas = a cena guardada (fundo, frente, recortes), bit a bit; nenhum
                     numero de profundidade/distancia publicado (so a ordem)
  M3 NAO MEXE NELA   publicar 3 vezes nao muda o estado salvo, o sorteio, o cortex, os
                     vocabularios nem o laboratorio
  M1 PROVENIENCIA    com o cortex trocado por pesos aleatorios, as imagens DECODIFICADAS mudam e
                     as camadas da cena (pixels guardados) nao mudam
  M2 IDA E VOLTA     (descritivo) palavra visual decodificada -> cortex.ve -> palavra mais proxima:
                     volta a mesma mais que uma palavra sorteada?
"""
import copy
import io
import json
import os
import pickle
import shutil
import sys
import tempfile
import traceback

import numpy as np

os.environ["MENTE_ARENA_RAPIDA"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena as AR  # noqa: E402
import cortex as CX  # noqa: E402
import laboratorio as LB  # noqa: E402
import mundos as MU  # noqa: E402
from mente import Mente  # noqa: E402

CACHES = {"_val_set", "_val_t", "_mundos_cache", "_mundos_t"}


def png(arq):
    from PIL import Image
    return np.asarray(Image.open(arq).convert("RGBA"))


def estado(m):
    d = {k: v for k, v in m.__dict__.items() if k not in ("corpo", "texto", "log", "cortex", "verbos", "lab")
         and k not in CACHES and not k.startswith("_imag") and k not in ("_arena_mundo", "_exp_pendente", "_quadros_recentes")}
    return pickle.dumps(d, protocol=4)


def lab_arena(sem=101, n=10):
    cam = AR.camera(np.random.default_rng(sem))
    lab = LB.Laboratorio(sem)
    for k in range(n):
        m_ = "giro" if k < n // 2 else "mola"
        ep = AR.episodio(m_, np.random.default_rng([sem, 400 + k]), cam)
        per = LB.percebe(ep.quadros, ep.tempos)
        r = lab.observa(None, None, fonte="arena", url=f"arena://{k}", percepcao=per)
        if r.get("episodio") is not None:
            lab.guarda_cena(r["episodio"], LB.cena(ep.quadros, per))
    return lab


def confere_fisicos(lab, pasta, out):
    erros = []
    for f in out["fisicos"]:
        T = lab.teorias[f["id"]]
        c = lab.cenas[f["cena_do_video"]]
        A = png(os.path.join(pasta, "mundos", f["atlas"]))
        x, y, w, h = f["rects"]["fundo"]
        if not np.array_equal(A[y:y + h, x:x + w, :3], c["fundo"]):
            erros.append(f"T{f['id']}: fundo publicado difere da cena guardada")
        if c.get("frente") is not None:
            x, y, w, h = f["rects"]["frente"]
            if not np.array_equal(A[y:y + h, x:x + w, 3] > 0, c["frente"]):
                erros.append(f"T{f['id']}: camada da frente publicada difere da mascara deduzida")
        elif f["rects"]["frente"] is not None:
            erros.append(f"T{f['id']}: publicou frente sem ela ter deduzido")
        for k, (rec, mask) in enumerate(c["sprites"]):
            x, y, w, h = f["rects"]["coisas"][k]
            if not (np.array_equal(A[y:y + h, x:x + w, :3][mask], rec[mask]) and np.array_equal(A[y:y + h, x:x + w, 3] > 0, mask)):
                erros.append(f"T{f['id']}: recorte {k} difere")
        ep = lab.eps[f["cena_do_video"]]
        i = 0
        for tr in ep["trilhas"][:6]:
            Zc, okc = LB.estados_causais(tr)
            ii = np.nonzero(okc)[0]
            if not len(ii):
                continue
            i0 = int(ii[0])
            t = tr["t"].astype(float)[i0:] - float(tr["t"][i0])
            sel = np.linspace(0, len(t) - 1, min(len(t), 40)).astype(int)
            if np.abs(np.asarray(f["reais"][i]["p"]) - tr["p"][i0:][sel]).max() > 6e-4:
                erros.append(f"T{f['id']}: trilha real {i} difere da memoria")
            if np.abs(np.asarray(f["imaginados_do_real"][i]) - lab._caminho(T["lei"], Zc[i0], t[sel])).max() > 6e-4:
                erros.append(f"T{f['id']}: caminho imaginado {i} difere da lei dela")
            if np.abs(np.asarray(f["retos"][i]) - lab._caminho(None, Zc[i0], t[sel])).max() > 6e-4:
                erros.append(f"T{f['id']}: caminho reto {i} difere")
            i += 1
    txt = json.dumps(out, ensure_ascii=False).lower()
    for proib in ('"profundidade"', '"distancia"', '"z":', '"depth"'):
        if proib in txt:
            erros.append(f"publicou numero de profundidade: {proib}")
    return erros


def main():
    resultados = {}
    pasta = tempfile.mkdtemp()
    m = Mente(7, pasta)
    m.lab = lab_arena()
    out = MU.publica(m, pasta, {})
    assert out["fisicos"], "nenhum mundo fisico publicado"
    e = confere_fisicos(m.lab, pasta, out)
    resultados["W0_fidelidade"] = "OK" if not e else e
    # M3: nao mexe nela
    antes = (estado(m), pickle.dumps(m.lab, protocol=4), copy.deepcopy(m.rng.bit_generator.state))
    for _ in range(3):
        MU.publica(m, pasta, {})
    depois = (estado(m), pickle.dumps(m.lab, protocol=4), m.rng.bit_generator.state)
    resultados["M3_nao_mexe_nela_fisicos"] = "OK" if antes[0] == depois[0] and antes[1] == depois[1] and antes[2] == depois[2] \
        else [n for n, a, b in zip(("estado", "laboratorio", "sorteio"), antes, depois) if a != b]
    if len(sys.argv) > 1:
        p2 = tempfile.mkdtemp()
        shutil.copytree(sys.argv[1], p2, dirs_exist_ok=True)
        m2 = Mente.carrega(p2, 9)
        m2._cortex()
        m2._lab()
        cache = {}
        out2 = MU.publica(m2, p2, cache)
        resultados["conceitos_publicados"] = len(out2["conceitos"])
        if getattr(m2, "lab", None) is not None:
            e = confere_fisicos(m2.lab, p2, out2)
            resultados["W0_fidelidade_mente"] = "OK" if not e else e
        cx = m2.cortex
        params = {k: v.copy() for k, v in cx.p.items()}
        olhoP, audP = m2.olho.P.copy(), m2.aud.P.copy()
        antes = (estado(m2), copy.deepcopy(m2.rng.bit_generator.state), cx.n_mem, cx.olhadas)
        for _ in range(3):
            MU.publica(m2, p2, cache)
        mudou = [k for k in params if not np.array_equal(params[k], cx.p[k])]
        depois = (estado(m2), m2.rng.bit_generator.state, cx.n_mem, cx.olhadas)
        ok = not mudou and antes == depois and np.array_equal(olhoP, m2.olho.P) and np.array_equal(audP, m2.aud.P)
        resultados["M3_nao_mexe_nela_conceitos"] = "OK" if ok else {"params": mudou, "estado": antes[0] != depois[0],
                                                                     "sorteio": antes[1] != depois[1]}
        if out2["conceitos"]:
            # M1 proveniencia: cortex aleatorio -> decodificados mudam; camadas de cena nao
            folhas = {c["folha"]: png(os.path.join(p2, "mundos", c["folha"])) for c in out2["conceitos"]}
            atlas = {f["atlas"]: png(os.path.join(p2, "mundos", f["atlas"])) for f in out2["fisicos"]}
            rng = np.random.default_rng(0)
            salvo = {k: v.copy() for k, v in cx.p.items()}
            for k in cx.p:
                cx.p[k] = rng.normal(0, float(np.std(salvo[k]) + 1e-3), cx.p[k].shape).astype(np.float32)
            out3 = MU.publica(m2, p2, {})
            dif = []
            for c in out3["conceitos"]:
                if c["folha"] in folhas and c["rects"].get("arquetipo"):
                    x, y, w, h = c["rects"]["arquetipo"]
                    a0 = folhas[c["folha"]][y:y + h, x:x + w, :3].astype(float)
                    a1 = png(os.path.join(p2, "mundos", c["folha"]))[y:y + h, x:x + w, :3].astype(float)
                    dif.append(float(np.abs(a0 - a1).mean()))
            iguais = all(np.array_equal(v, png(os.path.join(p2, "mundos", k))) for k, v in atlas.items())
            for k in cx.p:
                cx.p[k] = salvo[k]
            resultados["M1_proveniencia"] = {"arquetipo_muda_px_medio": round(float(np.mean(dif)), 1) if dif else None,
                                             "camadas_da_cena_iguais": iguais,
                                             "veredito": "OK" if dif and np.mean(dif) > 10 and iguais else "FALHA"}
            # M2 ida e volta das palavras visuais
            fundo = MU._mapa_neutro(cx)
            ids = sorted({v["id"] for c in out2["conceitos"] for v in c["visuais"]})[:40]
            volta = sorteio = 0
            for j in ids:
                im = MU.palavra_decodificada(cx, m2.olho, j, fundo)
                z = cx.ve(im[None])[0][0]
                c0 = CX.GRADE // 2 - 1
                cel = z[c0:c0 + 2, c0:c0 + 2].reshape(-1, CX.CANAIS)
                w = m2.olho.codifica(cel, 0, aprende=False)
                volta += int(j in w)
                sorteio += int(int(rng.integers(len(m2.olho.P))) in w)
            resultados["M2_ida_e_volta"] = {"palavras": len(ids), "voltou_a_mesma": volta, "controle_sorteada": sorteio}
    print(json.dumps(resultados, ensure_ascii=False, indent=1))
    falhas = [k for k, v in resultados.items() if (isinstance(v, list) and v) or (isinstance(v, dict) and v.get("veredito") == "FALHA")
              or (k.startswith("M3") and v != "OK")]
    print("FALHOU:" if falhas else "OK", falhas)
    return 1 if falhas else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        print("FALHOU:\n" + traceback.format_exc()[-2500:])
        sys.exit(1)
