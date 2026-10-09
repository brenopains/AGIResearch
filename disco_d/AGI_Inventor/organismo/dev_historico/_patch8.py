m = open("material_fisica.py", encoding="utf-8").read()
def rep(s, a, b):
    assert s.count(a) == 1, a[:200]
    return s.replace(a, b)
m = rep(m, '''def ajusta_lei(dados, conceitos, extras, forma=None):''', '''def ajusta_lei(dados, conceitos, extras, forma=None, limite=float("inf")):''')
m = rep(m, '''    termos = []
    for _ in range(SEGURANCA_TERMOS):
        R, Qs = residuo_e_base(termos)''', '''    termos = []
    for _ in range(SEGURANCA_TERMOS):
        if estado["custo"] >= limite:
            break
        R, Qs = residuo_e_base(termos)''')
m = rep(m, '''    for _passada in range(3):
        mudou = False
        for pos in range(len(termos)):
            if pos >= len(termos):
                break''', '''    for _passada in range(3):
        mudou = False
        for pos in range(len(termos)):
            if pos >= len(termos) or estado["custo"] >= limite:
                break''')
open("material_fisica.py", "w", encoding="utf-8").write(m)
o = open("organismo.py", encoding="utf-8").read()
o = rep(o, '''            conc, ext, usa = ws.vocab_para(op, self)
            ws.assinaturas.add(ws.assinatura(op, usa))
            leis, bits, c = F.ajusta_lei(ws.dados, conc, ext)''', '''            conc, ext, usa = ws.vocab_para(op, self)
            ws.assinaturas.add(ws.assinatura(op, usa))
            leis, bits, c = F.ajusta_lei(ws.dados, conc, ext,
                                         limite=max(ws.orcamento - ws.gasto, 1.0))''')
open("organismo.py", "w", encoding="utf-8").write(o)
print("ok")
