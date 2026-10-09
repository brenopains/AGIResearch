import os, json, time
from dotenv import load_dotenv; load_dotenv('.env')
from shared.llm_client import _openai_compat_post, VENICE_URL, VENICE_KEY
from shared.pubmed import search_and_fetch, format_for_llm
M="qwen3-235b-a22b-instruct-2507"
def call(sysp, usr, mx=900):
    for a in range(3):
        try: return _openai_compat_post(VENICE_URL, VENICE_KEY, M, [{"role":"system","content":sysp},{"role":"user","content":usr}], 0.6, mx, is_venice=True)
        except Exception as e: time.sleep(2)
    return ""
print("Buscando literatura ELA..."); lit=format_for_llm(search_and_fetch("amyotrophic lateral sclerosis mechanism therapeutic target review", retmax=20))[:14000]
PERSONAS={
 "geometra":"Voce busca SIMETRIAS e estruturas profundas em vias biologicas. Hipoteses mecanisticas elegantes.",
 "objetista":"Voce foca em ALVOS moleculares concretos e drogas existentes reposicionaveis.",
 "relacional":"Voce conecta vias APARENTEMENTE distintas (cross-talk inesperado entre sistemas).",
}
hyps={}
for name,sysp in PERSONAS.items():
    r=call(sysp+" Responda em 3-4 frases: 1 hipotese terapeutica para ELA (alvo + mecanismo + porque plausivel).",
           "LITERATURA:\n"+lit+"\n\nSua hipotese para ELA:")
    hyps[name]=r.strip(); print(f"\n=== HIPOTESE [{name}] ===\n{r.strip()[:500]}")
# COMPOSICAO NAO-COMUTATIVA (o mecanismo da arquitetura)
a,b=hyps["objetista"],hyps["relacional"]
print("\n"+"="*60+"\nCOMPOSICAO NAO-COMUTATIVA (A>>B vs B>>A):")
ab=call("Voce COMPOE duas hipoteses na ordem A-entao-B: use A como base e B como modulador/potenciador. Seja especifico e nao-obvio.",
        f"A:\n{a}\n\nB:\n{b}\n\nHipotese COMPOSTA A>>B (combinacao sinergica concreta, 3 frases):", 700)
ba=call("Voce COMPOE duas hipoteses na ordem B-entao-A: use B como base e A como modulador/potenciador. Seja especifico e nao-obvio.",
        f"A:\n{a}\n\nB:\n{b}\n\nHipotese COMPOSTA B>>A (combinacao sinergica concreta, 3 frases):", 700)
print(f"\n--- A>>B ---\n{ab.strip()[:500]}")
print(f"\n--- B>>A ---\n{ba.strip()[:500]}")
# JUIZ: ancoragem + novidade + assimetria
judge=call("Voce e um revisor cetico. Avalie em PT-BR curto.",
  f"Hipotese A>>B:\n{ab}\n\nHipotese B>>A:\n{ba}\n\n1) Os mecanismos citados sao REAIS/conhecidos na ELA (ancoragem)? 2) A composicao e NAO-OBVIA (vai alem de A e B isolados)? 3) A>>B e B>>A sao DIFERENTES (assimetria de ordem)? Responda os 3 itens em 1-2 frases cada.", 600)
print("\n"+"="*60+"\nVEREDITO DO REVISOR:\n"+judge.strip()[:800])
json.dump({"hyps":hyps,"AB":ab,"BA":ba,"judge":judge}, open("als_direct_result.json","w"), ensure_ascii=False, indent=1)
print("\n[salvo als_direct_result.json]")
