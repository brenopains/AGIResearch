"""
mente_web.decide_e_solta — espera as gemeas, mede, escolhe pela regra registrada,
copia a vencedora para vidas/viva e a solta em vida continua (guardiao).
"""
import json
import os
import shutil
import subprocess
import sys
import time

import psutil

RAIZ = os.path.dirname(os.path.abspath(__file__))
os.chdir(RAIZ)
PYW = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
PY = os.path.join(os.path.dirname(sys.executable), "python.exe")
SEM_JANELA = 0x08000000 | 0x00000008          # CREATE_NO_WINDOW | DETACHED_PROCESS
BAIXA = 0x00004000
diario = open(os.path.join(RAIZ, "vidas", "decisao.log"), "a", encoding="utf-8")


def log(m):
    diario.write(time.strftime("%Y-%m-%d %H:%M:%S ") + m + "\n")
    diario.flush()


def gemeas_vivas():
    n = 0
    for p in psutil.process_iter(["cmdline"]):
        cl = " ".join(p.info["cmdline"] or [])
        if "viver.py" in cl and "gemea_v" in cl:
            n += 1
    return n


if os.path.exists(os.path.join(RAIZ, "vidas", "viva", "estado.pkl")):
    log("ja existe vidas/viva: nada a decidir")
    sys.exit(0)
log("esperando as gemeas terminarem")
while gemeas_vivas():
    time.sleep(60)
log("gemeas terminaram; medindo")
res = {}
for v in ("gemea_v1", "gemea_v2"):
    with open(os.path.join(RAIZ, "vidas", v + "_marco.txt"), "w", encoding="utf-8") as out:
        subprocess.run([PY, "marco.py", os.path.join("vidas", v)], stdout=out, stderr=subprocess.STDOUT,
                       creationflags=SEM_JANELA | BAIXA, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    res[v] = json.load(open(os.path.join(RAIZ, "vidas", v, "marcos.json"), encoding="utf-8"))
f1, f2 = res["gemea_v1"]["fotos"]["final"], res["gemea_v2"]["fotos"]["final"]


def palavra(r):
    j = r["janelas"][-1] if r["janelas"] else {}
    cc = j.get("consultas_classe", {})
    return cc.get("palavra", 0) / max(sum(cc.values()), 1)


conf = {
    "mdl": (f1["mdl_bits_por_byte"], f2["mdl_bits_por_byte"], "menor"),
    "f1": (f1["f1_fronteira"], f2["f1_fronteira"], "maior"),
    "pequena_web": (f1["bits_por_byte"].get("pequena_web"), f2["bits_por_byte"].get("pequena_web"), "menor"),
    "consultas_palavra": (palavra(res["gemea_v1"]), palavra(res["gemea_v2"]), "maior"),
}
v1 = v2 = 0
for k, (a, b, sentido) in conf.items():
    if a is None or b is None or a == b:
        continue
    if (a < b) == (sentido == "menor"):
        v1 += 1
    else:
        v2 += 1
venc = "gemea_v1" if v1 > v2 else "gemea_v2"
versao = "v1" if venc == "gemea_v1" else "v2"
log(f"confrontos {conf} -> v1 {v1} x v2 {v2}: segue viva {venc}")
shutil.copytree(os.path.join(RAIZ, "vidas", venc), os.path.join(RAIZ, "vidas", "viva"))
open(os.path.join(RAIZ, "vidas", "viva", "versao.txt"), "w").write(versao)
json.dump({"confrontos": conf, "placar": {"v1": v1, "v2": v2}, "vencedora": venc},
          open(os.path.join(RAIZ, "vidas", "viva", "decisao.json"), "w", encoding="utf-8"), indent=1)
# a vencedora acorda com o codigo novo? (sem rede, sem tocar no diario dela)
r = subprocess.run([PY, "teste_compat.py", os.path.join("vidas", "viva"), versao], cwd=RAIZ,
                   capture_output=True, text=True, creationflags=SEM_JANELA, timeout=1800)
log("teste de compatibilidade: " + (r.stdout or "").strip()[-1200:] + (r.stderr or "").strip()[-600:])
if os.path.exists(os.path.join(RAIZ, "vidas", "SEGURAR")):
    log("vidas/SEGURAR existe: vencedora pronta, mas NAO solta (sentidos novos sendo instalados)")
    sys.exit(0)
subprocess.Popen([PYW, "guardiao.py", os.path.join("vidas", "viva"), versao], cwd=RAIZ,
                 creationflags=SEM_JANELA)
log("guardiao ligado em vidas/viva")
