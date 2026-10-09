"""
mirror.py — NEURONIOS ESPELHO: a maquina aprende OBSERVANDO a solucao de OUTRAS tarefas analogas.
Mantem um repertorio persistente (fingerprint estrutural -> regra que funcionou). Diante de uma tarefa NOVA,
recupera as tarefas ESTRUTURALMENTE ANALOGAS ja resolvidas e TENTA a regra delas aqui (imitacao da
transformacao bem-sucedida de outro). Fundamento neuro: o sistema espelho dispara tanto ao AGIR quanto ao
VER outro agir -> aprendizado por observacao/analogia. Honesto: so reusa regras nomeadas (reconstrutiveis
do pool da tarefa atual); o ganho e transferencia por analogia, nao uma regra nova inventada do zero.
"""
import json, pathlib
import tem_arc2 as T

STORE = pathlib.Path(__file__).parent / "mirror_store.json"

def _struct_fp(task):
    """assinatura ESTRUTURAL (ignora cores/conteudo; foca na forma da transformacao)."""
    p = task["train"][0]; gi, go = p["input"], p["output"]
    Hi, Wi = T.dims(gi); Ho, Wo = T.dims(go)
    same = (Hi, Wi) == (Ho, Wo)
    bg = T.bg_color(gi); nobj = len(T.components(gi, bg))
    rr = round((Ho / Hi) if Hi else 0, 1); rc = round((Wo / Wi) if Wi else 0, 1)
    return f"{int(same)}|{min(nobj,6)}|{rr}|{rc}"

def load():
    if STORE.exists():
        try: return json.load(open(STORE))
        except Exception: return {}
    return {}

def remember(task, prog):
    """grava: esta classe estrutural foi resolvida por ESTA regra (nome reconstrutivel)."""
    d = load(); fp = _struct_fp(task)
    d.setdefault(fp, [])
    if prog not in d[fp]: d[fp].append(prog); d[fp] = d[fp][-12:]
    try: json.dump(d, open(STORE, "w"))
    except Exception: pass

def mirror_recall(task):
    """retorna nomes de regras de tarefas ANALOGAS (mesma classe estrutural) p/ imitar aqui."""
    d = load(); fp = _struct_fp(task)
    out = list(d.get(fp, []))
    for k, progs in d.items():       # analogos: mesma forma+ratio (conteudo pode diferir)
        if k != fp and k.split("|")[0] == fp.split("|")[0] and k.split("|")[2:] == fp.split("|")[2:]:
            out += progs
    seen = set(); uniq = []
    for r in out:
        if r not in seen: seen.add(r); uniq.append(r)
    return uniq[:20]
