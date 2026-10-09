"""
mente_web.guardiao — a vida CONTINUA: roda a mente em blocos, reinicia se cair
================================================================================
Uso:  python guardiao.py <pasta> <v1|v2> [semente]
- cada bloco e `viver.py <pasta> 360 ...`; a mente ACORDA de onde parou (o estado
  salvo a cada 15 min e nas fotos), entao cair custa no maximo 15 min de vida;
- prioridade abaixo do normal (o PC do usuario continua dele);
- para parar com calma: criar o arquivo <pasta>/PARAR (ela termina o bloco salvando).
"""
import datetime
import os
import subprocess
import sys
import time

pasta, versao = sys.argv[1], sys.argv[2]
sem = sys.argv[3] if len(sys.argv) > 3 else "20261008"
os.makedirs(pasta, exist_ok=True)
diario = open(os.path.join(pasta, "guardiao.log"), "a", encoding="utf-8")
BAIXA = 0x00004000 if os.name == "nt" else 0          # BELOW_NORMAL_PRIORITY_CLASS
quedas = 0
while not os.path.exists(os.path.join(pasta, "PARAR")):
    t0 = time.time()
    diario.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} bloco inicia\n")
    diario.flush()
    with open(os.path.join(pasta, "bloco.out"), "a", encoding="utf-8") as out:
        r = subprocess.run([sys.executable, "viver.py", pasta, "360", sem, versao],
                           stdout=out, stderr=subprocess.STDOUT, creationflags=BAIXA)
    dur = time.time() - t0
    diario.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} bloco terminou codigo={r.returncode} "
                 f"duracao={dur / 60:.0f} min\n")
    diario.flush()
    quedas = quedas + 1 if (r.returncode != 0 and dur < 600) else 0
    if quedas >= 5:                                    # caindo em seguida: para e espera alguem olhar
        diario.write("5 quedas seguidas: guardiao parou\n")
        break
    time.sleep(30 if r.returncode == 0 else 120)
