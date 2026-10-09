"""Sorteia 3 moldes da gramatica e grava o lacre (hash) SEM imprimir quais."""
import hashlib, json
import numpy as np
import mundo_fisico as M
frase = "lacre organismo 2026-10-05 portao decisivo"
sem = int.from_bytes(hashlib.sha256(frase.encode()).digest()[:8], "little")
g = M.moldes_gramatica()
idx = np.random.default_rng(sem).choice(len(g), 3, replace=False)
lacrados = sorted(g[i] for i in idx)
corpo = json.dumps(lacrados)
h = hashlib.sha256(corpo.encode()).hexdigest()
json.dump({"frase": frase, "lacrados": lacrados, "sha256": h}, open("LACRE.json", "w"))
print("lacre gravado; sha256 dos moldes:", h)
