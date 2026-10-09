"""Testes de contrato do organismo. Rodar: python testes.py"""
import json
import os
import shutil

import numpy as np

import hdc
import material_fisica as F
import mundo_fisico as M
import organismo as O
import vida

OK = []


def caso(f):
    f()
    OK.append(f.__name__)
    print("ok", f.__name__)


@caso
def t_hdc_semelhanca():
    a = hdc.codifica({"a": 1, "b": 2, "c": 3})
    b = hdc.codifica({"a": 1, "b": 2, "c": 4})
    c = hdc.codifica({"x": 9})
    assert hdc.sim(a, b) > 0.5 > abs(hdc.sim(a, c))


@caso
def t_busca_ortogonal_acha_forma_verdadeira():
    rng = np.random.default_rng(0)
    acertos = 0
    for _ in range(10):
        P = M.Problema(0, "oblato", "percebido", "L", rng, n_lab=48)
        leis, _b, _c = F.ajusta_lei([(P.S, P.A)], {}, [])
        acertos += P.verifica(leis[0])[0]
    assert acertos >= 9, acertos


@caso
def t_spin_orbita_exige_h():
    rng = np.random.default_rng(1)
    h = F.Conceito([("x", "vy"), ("y", "vx")], np.array([1., -1.]), 1.0, {}, "h")
    b = o = 0
    for _ in range(8):
        P = M.Problema(0, "spin_orbita", "percebido", "L", rng)
        b += P.verifica(F.ajusta_lei([(P.S, P.A)], {}, [])[0][0])[0]
        o += P.verifica(F.ajusta_lei([(P.S, P.A)], {"h": h}, [])[0][0])[0]
    assert b == 0 and o >= 7, (b, o)


@caso
def t_invencao_acha_h_e_E():
    rng = np.random.default_rng(2)
    p = M.params("kepler", rng)
    Ttr, Tho = M.trajetorias("kepler", p, rng, 6), M.trajetorias("kepler", p, rng, 6)
    out, _c = F.inventa(Ttr, Tho, {})
    S = M.estados("kepler", p, rng, 300)
    x, y, vx, vy = S.T
    h, E = x * vy - y * vx, 0.5 * (vx ** 2 + vy ** 2) - p["mu"] / np.hypot(x, y)
    cs = [F.Conceito(s, w, e, pa, "q")(S) for s, w, _g, e, pa in out]
    assert any(abs(np.corrcoef(c, h)[0, 1]) > 0.999 for c in cs)
    assert any(abs(np.corrcoef(c, E)[0, 1]) > 0.999 for c in cs)


@caso
def t_limite_interrompe_busca():
    rng = np.random.default_rng(3)
    P = M.Problema(0, "einstein", "percebido", "L", rng)
    _l, _b, c = F.ajusta_lei([(P.S, P.A)], {}, [], limite=1.0)
    _l, _b, c2 = F.ajusta_lei([(P.S, P.A)], {}, [])
    assert c < c2


@caso
def t_organismo_sem_tetos():
    """O braco ORGANISMO nao liga nenhum MAX_*; TETOS e PIPELINE ligam."""
    for b, fl in O.BRACOS.items():
        f = dict(O.FLAGS_ORGANISMO)
        f.update(fl)
        assert f["tetos"] == (b in ("TETOS", "PIPELINE")), b


@caso
def t_cada_ablacao_muda_uma_chave():
    """Vida 1: cada ablacao muda UMA chave do ORGANISMO. Vida 2: cada braco
    difere do LOCAL2 SO na largura do contexto (escopo/beta_fixo)."""
    vida2 = ("LOCAL2", "ESCOPO", "LARGURA_3", "GLOBAL")
    base2 = dict(O.FLAGS_ORGANISMO)
    base2.update(O.BRACOS["LOCAL2"])
    for b, fl in O.BRACOS.items():
        if b in ("ORGANISMO", "PIPELINE", "LOCAL2"):
            continue
        if b in vida2:
            f = dict(O.FLAGS_ORGANISMO)
            f.update(fl)
            difs = {k for k in f if f[k] != base2[k]}
            assert difs <= {"escopo", "beta_fixo"} and difs, (b, difs)
        else:
            assert len(fl) == 1, b


@caso
def t_lacrados_fora_do_desenvolvimento():
    lac = json.load(open("LACRE.json"))["lacrados"]
    assert all(m.startswith("q:") for m in lac)
    assert not set(lac) & set(M.MOLDES_DEV + M.MOLDES_RESERVA)


@caso
def t_fluxo_identico_entre_bracos():
    a, s1 = vida.agenda(1001, ("q:x",))
    b, s2 = vida.agenda(1001, ("q:x",))
    assert a == b and s1 == s2


@caso
def t_vida_deterministica_e_ablacao():
    pasta = "resultados/_teste"
    shutil.rmtree(pasta, ignore_errors=True)
    r1 = vida.vive("ORGANISMO", 7, ("periodo_longo",), pasta + "/a", n_max=25)
    r2 = vida.vive("ORGANISMO", 7, ("periodo_longo",), pasta + "/b", n_max=25)
    assert [l["ok"] for l in r1["linhas"]] == [l["ok"] for l in r2["linhas"]]
    shutil.rmtree(pasta, ignore_errors=True)


@caso
def t_refutacao_e_confianca_existem():
    org = O.Organismo({}, 0)
    assert "op:confiar" in org.E
    assert hasattr(org, "_refutada")


print(f"\n{len(OK)} testes passaram")
