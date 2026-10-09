"""Grava o hash de cada fonte (e documentos de pre-registro) num manifesto.
Uso: python congelar.py MANIFESTO_vida2.json [arquivos extras...]"""
import glob
import hashlib
import json
import sys

saida = sys.argv[1]
fontes = sorted(glob.glob("*.py")) + ["LACRE.json"] + sys.argv[2:]
man = {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in fontes}
json.dump(man, open(saida, "w"), indent=1)
for f, h in man.items():
    print(f"{h[:16]}  {f}")
