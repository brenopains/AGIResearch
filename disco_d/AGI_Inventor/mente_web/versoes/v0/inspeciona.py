"""Resumo rapido de uma vida: python inspeciona.py <pasta>"""
import collections
import json
import sys

ev = [json.loads(l) for l in open(sys.argv[1] + "/ontogenia.jsonl", encoding="utf-8")]
c = collections.Counter(e["ev"] for e in ev)
print("eventos:", dict(c))
for e in ev:
    if e["ev"] == "erro":
        print("ERRO", e["tipo"], e["msg"][:150])
        print(e["pilha"][-500:])
        break
acs = collections.Counter(e["estrategia"] for e in ev if e["ev"] == "acao")
pag = collections.Counter(e["estrategia"] for e in ev if e["ev"] == "acao" and e["pagou"])
print("acoes:", acs.most_common())
print("pagou:", pag.most_common())
print("\nbuscas proprias:")
for e in [e for e in ev if e["ev"] == "pesquisa"][:20]:
    print(f"  t={e['t']:5d} [{e['barra']}] {e['consulta']!r}")
print("\nreusos entre sites (amostra):")
for e in [e for e in ev if e["ev"] == "reuso"][:12]:
    print(f"  {e['pedaco']!r:20s} nasceu em {e['de']:25s} -> {e['em']} util={e['util']}")
for tipo in ("ligacao_av", "aterramento", "reativacao", "pergunta_depois"):
    xs = [e for e in ev if e["ev"] == tipo]
    print(f"\n{tipo}: {len(xs)}", xs[:4])
lidos = [e for e in ev if e["ev"] == "leu"]
if lidos:
    n = len(lidos)
    for i in range(0, n, max(1, n // 6)):
        bl = [e["bits_primeiro_bloco"] for e in lidos[i:i + max(1, n // 6)] if e["bits_primeiro_bloco"]]
        if bl:
            print(f"  paginas {i:4d}-{i + max(1, n // 6):4d}: bits/byte no 1o contato {sum(bl) / len(bl):.2f}")
s = [e for e in ev if e["ev"] == "sono"]
if s:
    u = s[-1]
    print("\nultimo sono:", {k: v for k, v in u.items() if k not in ("mais_frequentes", "perguntas")})
    print("perguntas abertas:", u["perguntas"])
    print("mais frequentes:", u["mais_frequentes"])
