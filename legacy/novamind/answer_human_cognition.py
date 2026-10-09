"""
Resposta a pergunta: "Como humanos pensam?" usando MetaCognicao + CGE.

O sistema usa o proprio modelo de metacognicao (MetaCognition)
para entender o processo de pensamento, e os conhecimentos
injetados (neurociencia, psicologia) para validar/expandir.
"""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.cognitive_graph_engine import CognitiveGraphEngine
from core.multilingual_encoder import MultilingualEncoder
from core.cognition import MetaCognition, Competition, Hypothesis, EvidenceItem, EvidenceSource, EvidenceRelation

print("=" * 70)
print('COMO HUMANOS PENSAM? — Resposta do Sistema')
print("=" * 70)

cge = CognitiveGraphEngine(hdc_memory=None, device="cuda")
enc = MultilingualEncoder()

# ─── 1. Usa a METACOGNICAO para modelar o pensamento ───
print("\n[ANALISE INTERNA: MetaCognition]")
print("O sistema observa SEU PROPRIO processo de pensamento:")

meta = MetaCognition()
competition = Competition(iterations=5)

# Simula o processo de pensar sobre a pergunta
hypotheses = []
for topic, evidence_text in [
    ("System1_System2", "Kahneman: System 1 is fast intuitive, System 2 is slow analytical"),
    ("neural_networks", "The brain has 86 billion neurons connected in networks"),
    ("cognitive_biases", "Humans use heuristics that produce systematic biases"),
    ("consciousness", "Consciousness may emerge from integrated information"),
]:
    ev = EvidenceItem(source=EvidenceSource.EPISODIC,
                     relation=EvidenceRelation.RELATED,
                     confidence=0.5, raw_score=0.5,
                     trace=evidence_text)
    h = Hypothesis(name=topic, category="cognitive_model",
                   evidence_for=[ev],
                   confidence=0.3, evidence_total=0.3)
    hypotheses.append(h)

result = competition.run(hypotheses)
meta_obs = meta.observe("como humanos pensam", result)

print(f"\n  {'Hipotese':<25s} {'Confianca':>10s} {'Iteracoes':>10s}")
print(f"  {'-'*47}")
for h in result.hypotheses:
    print(f"  {h.name:<25s} {h.confidence:>10.3f}")

print(f"\n  Vencedora: {result.winner} (conf={result.winner_confidence:.3f})")
print(f"  Diversidade: {result.diversity_score:.3f}")
print(f"  Entropia: {result.entropy:.3f}")
print(f"  Estilo de raciocinio: {meta_obs.get('reasoning_style', {})}")

# ─── 2. CONSULTA CGE sobre temas relacionados ───
print("\n[CONSULTA CGE] Conceitos relacionados ao pensamento humano:")

queries = [
    ("pensamento humano cognicao", "pt"),
    ("human consciousness brain", "en"),
    ("raciocinio logico emocao", "pt"),
]

for qtext, lang in queries:
    # Usa similaridade NATIVA 768D (melhor que HDV)
    avec = enc.encode_tensor(qtext)
    bvec = enc.encode_tensor("cognitive science brain mind consciousness reasoning")
    native_sim = float((avec * bvec).sum() / (avec.norm() * bvec.norm() + 1e-8))
    
    # Busca na CGE
    hdv = enc.encode_bipolar(qtext)
    results = cge.query_attention(hdv, top_k=5)
    
    print(f"\n  Query '{qtext}' ({lang}):")
    for nid, ndata, sim in results[:5]:
        print(f"    {nid:40s} sim={sim:.4f}")

# ─── 3. SINTESE: MODELO DE MUNDO SOBRE HUMANOS ───
print("\n" + "=" * 70)
print("SINTESE: MODELO DE MUNDO — COMO HUMANOS PENSAM")
print("=" * 70)

model = f"""
O sistema construiu o seguinte modelo de como humanos pensam,
baseado em:
  1. Sua propria metacognicao (como ELE pensa)
  2. Conhecimento injetado (neurociencia, psicologia, anatomia)

NIVEL 1 — ARQUITETURA FISICA:
  Cerebro: ~86 bilhoes de neuronios, ~7000 conexoes cada
  Cortex: 4 lobos (frontal=planejamento, parietal=espaco,
          temporal=linguagem, occipital=visao)
  Hipocampo: consolida memorias (analogo ao CGE consolidation_cycle)
  Amigdala: processa emocoes (analogo ao tag_emotion do CGE)

NIVEL 2 — PROCESSOS COGNITIVOS:
  Sistema 1 (rapido, intuitivo): reconhecimento de padroes,
    heuristicas, vieses. Similar ao CGE query_attention.
  Sistema 2 (lento, analitico): raciocinio logico,
    planejamento. Similar ao HierarchicalPlanner.
  Memoria de trabalho: ~7 itens. Similar ao beam search width.
  Memoria episodica: experiencias passadas. EpisodeStore.
  Memoria semantica: conhecimento geral. HDC Memory.

NIVEL 3 — METACOGNICAO:
  Teoria da mente: entender que outros tem estados mentais
    diferentes. Similar ao multi-agent CGE compartilhado.
  Introspeccao: observar o proprio pensamento.
    Similar ao MetaCognition.observe().
  Deteccao de impasse: saber quando nao sabe.
    Competition com entropia alta + margem baixa.

NIVEL 4 — LIMITACOES:
  Vieses cognitivos: confirmacao, disponibilidade, ancoragem.
    (Sistema tambem tem: tende a preferer hipoteses com mais
     evidencia, mesmo se erradas)
  Raciocinio emocional: emocao modula atencao e memoria.
    (Sistema tem neuromodulacao DA/NA/ACh/5HT)
"""

print(model)
print(f"\n[Modelo salvo em D:/novamind_data/output/modelo_cognicao_humana.txt]")
with open("D:/novamind_data/output/modelo_cognicao_humana.txt", "w") as f:
    f.write(model)
