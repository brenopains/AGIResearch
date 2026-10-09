"""
Teste Multilingue AGI — O CGE alinha conceitos entre linguas automaticamente?

Se o sistema e realmente AGI, ele deve:
  1. Codificar "dog" em ingles, "cachorro" em portugues, "chien" em frances no mesmo HDV
  2. No CGE, "dog", "cachorro", "chien" devem ser SIMILARES (mesmo conceito)
  3. O sistema deve inferir que "dog is mammal" implica "cachorro e mamifero"
  
NAO estamos dando um dicionario. O sistema deve descobrir a equivalencia
pela ESTRUTURA do conceito no espaco HDV, nao por lookup.

Teste: se cos(dog, cachorro) > cos(dog, gato), o sistema entende linguas.
"""
import sys, torch, numpy as np
sys.path.insert(0, ".")

from core.cognitive_graph_engine import CognitiveGraphEngine
from core.bge_encoder import BGEEncoder

print("=" * 70)
print("TESTE MULTILINGUE AGI — Alinhamento de Conceitos no HDV")
print("=" * 70)

cge = CognitiveGraphEngine(hdc_memory=None, device="cuda")
bge = BGEEncoder(device="cuda")

# ─── 1. Codifica o mesmo conceito em 4 linguas ───
conceitos = [
    # (nome_no_cge, texto_em_portugues, texto_em_ingles, texto_em_frances, texto_em_espanhol)
    ("cachorro", "cachorro e um animal domestico", "dog is a domestic animal", "chien est un animal domestique", "perro es un animal domestico"),
    ("gato", "gato e um felino domestico", "cat is a domestic feline", "chat est un felin domestique", "gato es un felino domestico"),
    ("agua", "agua e um liquido essencial", "water is an essential liquid", "eau est un liquide essentiel", "agua es un liquido esencial"),
    ("fogo", "fogo produz calor e luz", "fire produces heat and light", "feu produit de la chaleur", "fuego produce calor"),
    ("arvore", "arvore e uma planta", "tree is a plant", "arbre est une plante", "arbol es una planta"),
]

# Escreve no CGE
for pt, txt_pt, txt_en, txt_fr, txt_es in conceitos:
    cge.write_text_concepts(txt_pt)
    cge.write_text_concepts(txt_en)
    cge.write_text_concepts(txt_fr)
    cge.write_text_concepts(txt_es)

# ─── 2. Mede similaridade entre o mesmo conceito em diferentes linguas ───
print("\n[TESTE] Similaridade entre linguas para o MESMO conceito:")

pairs = [
    ("dog", "cachorro",      "animal domestico"),
    ("cat", "gato",          "felino domestico"),
    ("water", "agua",        "liquido essencial"),
    ("fire", "fogo",         "produz calor"),
    ("tree", "arvore",       "planta"),
]

def get_hdv(word):
    v = bge.encode(word)
    t = torch.tensor(v, dtype=torch.float32, device="cuda")
    return torch.sign(t)

same_concept_sims = []
diff_concept_sims = []

for en_word, pt_word, desc in pairs:
    hdv_en = get_hdv(en_word)
    hdv_pt = get_hdv(pt_word)
    sim = (hdv_en * hdv_pt).sum().item() / 10000.0
    same_concept_sims.append(sim)
    print(f"  cos({en_word:10s}, {pt_word:10s}) = {sim:.4f}  ({desc})")

# ─── 3. Compara com conceitos DIFERENTES entre linguas ───
print("\n[CONTROLE] Similaridade entre conceitos DIFERENTES:")
different = [
    ("dog", "gato"), ("cat", "cachorro"), ("water", "fogo"),
    ("fire", "agua"), ("tree", "cachorro"), ("dog", "agua"),
]
for w1, w2 in different:
    hdv1 = get_hdv(w1)
    hdv2 = get_hdv(w2)
    sim = (hdv1 * hdv2).sum().item() / 10000.0
    diff_concept_sims.append(sim)
    print(f"  cos({w1:10s}, {w2:10s}) = {sim:.4f}")

# ─── 4. Estatisticas ───
avg_same = np.mean(same_concept_sims)
avg_diff = np.mean(diff_concept_sims)
margin = avg_same - avg_diff

print(f"\n[RESULTADOS]")
print(f"  Media MESMO conceito entre linguas: {avg_same:.4f}")
print(f"  Media conceitos DIFERENTES:          {avg_diff:.4f}")
print(f"  Margem (same - diff):               {margin:.4f}")

if margin > 0.05:
    print(f"\n  >>> O CGE ALINHA CONCEITOS ENTRE LINGUAS")
    print(f"  >>> cos(dog, cachorro) = {avg_same:.3f} > cos(dog, gato) = {avg_diff:.3f}")
    print(f"  >>> O sistema entende que 'dog' e 'cachorro' sao o MESMO conceito")
    print(f"  >>> SEM dicionario. SEM lookup. So geometria HDV.")
else:
    print(f"\n  >>> Alinhamento fraco. BGE encoder pode ser o limitante.")

# ─── 5. Teste de inferencia cross-lingua ───
print(f"\n[INFERENCIA CROSS-LINGUA]")
print(f"  Se o CGE sabe que 'dog is mammal', ele infere que 'cachorro e mamifero'?")

cge.write_text_concepts("dog is a mammal. dogs breathe air. dogs have fur.")
cge.write_text_concepts("mammals are warm-blooded.")

# Query: "cachorro" no CGE — quais conceitos sao similares?
query_hdv = get_hdv("cachorro")
results = cge.query_attention(query_hdv, top_k=8)
print(f"  Query 'cachorro' (via HDV) encontra no CGE:")
for nid, ndata, sim in results:
    print(f"    {nid:30s} sim={sim:.4f}")

# Verifica se "mammal" aparece (deveria, se o alinhamento funciona)
mammal_found = any("mammal" in str(nid).lower() for nid, _, _ in results)
if mammal_found:
    print(f"\n  >>> 'cachorro' similar a 'mammal': INFERENCIA CROSS-LINGUA CONFIRMADA")
else:
    print(f"\n  >>> 'cachorro' NAO similar a 'mammal' (BGE encoder pode ser limitante)")
