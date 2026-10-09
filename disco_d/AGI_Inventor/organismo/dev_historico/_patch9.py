src = open("organismo.py", encoding="utf-8").read()
n = 0
def rep(a, b):
    global src, n
    assert src.count(a) == 1, a[:200]
    src = src.replace(a, b); n += 1
# TETOS: so os numeros magicos que o PROJETO usou (MAX_LEIS=6 em mente.py,
# K_CONC=3 em conceito_util.py, 3 direcoes de base na gramatica, RAG k=3);
# o teto de 'candidatas por decisao' era invencao minha e sai.
rep('''TETOS = dict(conceito=6, direcao=4, lei=6, aberto=6, episodio=12, coalizao=4,
             vocab=3, candidatas=4, rag=3, admitidos=3)''', '''TETOS = dict(conceito=3, direcao=3, lei=6, aberto=6, episodio=12, coalizao=4,
             vocab=3, rag=3, admitidos=3)''')
rep('''        if self.f["tetos"] and len(out) > TETOS["candidatas"]:
            out.sort(key=lambda c: -(self.ativacao(c[1]) if c[1] is not None else 0))
            out = out[:TETOS["candidatas"]]
        return out''', '''        return out''')
# SEM_META: agenda fixa retoma so o aberto mais crivel (sem k magico, sem
# inundar o orcamento antes de inventar)
rep('''        T = self._temperatura(ws)
        fontes = self._de_tipo("aberto") if self.f["meso"] else self._de_tipo("episodio")
        out = []
        for e in fontes:
            xe = ws.x_reat if e.tipo == "aberto" else ws.x
            if self.media(e, xe, T) > 0.5:
                out.append(e)
        return out''', '''        T = self._temperatura(ws)
        fontes = self._de_tipo("aberto") if self.f["meso"] else self._de_tipo("episodio")
        melhor, mm = None, 0.5
        for e in fontes:
            xe = ws.x_reat if e.tipo == "aberto" else ws.x
            m = self.media(e, xe, T)
            if m > mm:
                melhor, mm = e, m
        return [melhor] if melhor is not None else []''')
open("organismo.py", "w", encoding="utf-8").write(src)
print("trocas", n)
