s = open("mente.py", encoding="utf-8").read()
n = 0
def rep(a, b, c=1):
    global s, n
    assert s.count(a) == c, (s.count(a), a[:120]); s = s.replace(a, b); n += 1
rep('''        estr_pag = self._estruturas_da_pagina(pg)''', '''        estr_pag = self._estruturas_texto(pg)''')
rep('''    def _estruturas_da_pagina(self, pg):
        return [p for p in set(self.lex) & set()] if False else []

''', '')
rep('''        if modo == "seguir_novo":
            novos = [u for a, u in L if (G_dom := u.split("/")[2] if "//" in u else "") not in self.dom]''', '''        if modo == "seguir_novo":
            from corpo import dominio
            novos = [u for a, u in L if dominio(u) not in self.dom]''')
rep('''        if modo == "seguir_familiar":
            sc = [(sum(1 for p in self.E if len(p) >= 3 and p in a) if self.E else 0, u) for a, u in L[:150]]''', '''        if modo == "seguir_familiar":
            sc = [(sum(1 for p in self._pedacos(a) if p in self.E), u) for a, u in L[:150]]''')
rep('''            alvo = [p for p in self.E if len(p) >= 3 and p in alt][:5] + estr_pag[:5]''', '''            alvo = [p for p in self._pedacos(alt) if p in self.E and len(p) >= 3][:5] + estr_pag[:5]''')
rep('''    def _estruturas_texto(self, pg):''', '''    def _pedacos(self, dados: bytes):
        if len(dados) < 3:
            return []
        sur = self.texto.surpresa(dados, False)
        return [dados[a:b] for a, b in G.segmenta(dados, sur)]

    def _estruturas_texto(self, pg):''')
# total por prototipo visual mantido (em vez de varrer self.av)
rep('''        self.ligacoes_av = set()''', '''        self.ligacoes_av = set()
        self.av_v = collections.Counter()''')
rep('''                    nv = sum(n for (vv, _a), (n, _s) in self.av.items() if vv == v) if len(self.av) < 3000 else e[0] + 1
                    surp.append(-math.log2((e[0] + 1) / (nv + 2)))
                    e[0] += 1''', '''                    nv = self.av_v[v]
                    surp.append(-math.log2((e[0] + 1) / (nv + 2)))
                    e[0] += 1
                    self.av_v[v] += 1''')
open("mente.py", "w", encoding="utf-8").write(s)
c = open("corpo.py", encoding="utf-8").read()
a = '''        pg = _analisa_html(final, corpo) if b"<" in corpo[:2000] else \
            Pagina(final, dominio(final), texto=corpo[:200000])'''
b = '''        meta = re.search(rb'http-equiv=["\']?refresh["\']?[^>]*url=([^"\'>\s]+)', corpo[:4000], re.I)
        if meta and len(corpo) < 3000 and origem != "refresh":
            alvo = urllib.parse.urljoin(final, meta.group(1).decode("utf-8", "ignore"))
            return self.abrir(alvo, empilha=empilha, origem="refresh")
        pg = _analisa_html(final, corpo) if b"<" in corpo[:2000] else \
            Pagina(final, dominio(final), texto=corpo[:200000])'''
assert c.count(a) == 1
c = c.replace(a, b)
open("corpo.py", "w", encoding="utf-8").write(c)
print("trocas mente", n, "corpo ok")
