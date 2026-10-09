"""
TESTE DE TRANSFERENCIA CONCEITUAL: BRANCHING em biologia -> geometria.

O sistema observa exemplos de BRANCHING em biologia (arvores, vasos, neuronios).
Depois enfrenta um problema geometrico que requer BRANCHING para resolver.

Se o sistema usar o conceito BRANCHING (formado em biologia) para resolver
o problema geometrico, a transferencia conceitual foi bem-sucedida.

Pergunta: "O conceito de ramificacao aprendido em arvores ajuda a
           resolver um problema de expansao fractal em geometria?"
"""
import sys, numpy as np, time
sys.path.insert(0, ".")

from core.conceptual_discovery import (
    ObservationLayer, InvariantDiscovery, ConceptFormation,
    TheoryBuilder, ImaginationSimulator, ConstraintNetwork,
)

print("=" * 70)
print("TESTE DE TRANSFERENCIA CONCEITUAL")
print("BRANCHING: biologia -> geometria")
print("=" * 70)

# ─── 1. CFE: Observa exemplos de BRANCHING em 3 dominios biologicos ───
print("\n[FASE 1] Alimentando exemplos de BRANCHING em biologia...")

obs = ObservationLayer()
inv = InvariantDiscovery()
cf = ConceptFormation()

# Dominio 1: Arvores (botanica)
for i in range(5):
    obs.observe("text", {"content": f"tree trunk splits into branches at node {i}"}, 
                domain="biology", modality="text")
    # Alimenta o discovery com conteudo textual (detecta "splits", "branches")
    inv.observe({"content": f"tree trunk splits into branches at node {i}", "domain": "biology"})

# Dominio 2: Vasos sanguineos (medicina)
for i in range(5):
    obs.observe("text", {"content": f"blood vessel bifurcates at junction {i}"}, 
                domain="medicine", modality="text")
    inv.observe({"content": f"blood vessel bifurcates at junction {i}", "domain": "medicine"})

# Dominio 3: Neuronios (neurociencia)
for i in range(5):
    obs.observe("text", {"content": f"neuron dendrite branches at segment {i}"}, 
                domain="neuroscience", modality="text")
    inv.observe({"content": f"neuron dendrite branches at segment {i}", "domain": "neuroscience"})

# EXTRA: Dominio 4 — geografia (rios)
for i in range(3):
    obs.observe("text", {"content": f"river splits into tributaries at delta {i}"}, 
                domain="geography", modality="text")
    inv.observe({"content": f"river splits into tributaries at delta {i}", "domain": "geography"})

# ├ē executado o Invariant Discovery
invariants = inv.get_invariants(min_confidence=0.1)
print(f"\n  Invariantes descobertos: {len(invariants)}")
for inv_obj in invariants:
    print(f"    {inv_obj.structure_type}: {inv_obj.domains} (conf={inv_obj.confidence:.2f})")

# ├ē executado o Concept Formation
concept = cf.build("branching", {"biology", "medicine", "neuroscience", "geography"})
print(f"\n  Conceito formado: {concept.name}")
print(f"    Dominios: {concept.domains}")
print(f"    Condicoes: {concept.conditions}")
print(f"    Efeitos: {concept.effects}")
print(f"    Restricoes: {concept.constraints}")
print(f"    Confianca: {concept.confidence:.2f}")

# ─── 2. THEORY BUILDER: Constroi teoria de BRANCHING ───
print(f"\n[FASE 2] Theory Builder...")
tb = TheoryBuilder()
tb.observe_concept("branching", {"biology", "medicine", "neuroscience", "geography"})
tb.observe_concept("reflection", {"physics", "audio"})
tb.observe_concept("expansion", {"geometry", "biology"})

theories = tb.build_theories()
print(f"  Teorias construidas: {len(theories)}")
for t in theories:
    print(f"    {t.name}: conceitos={t.concepts} dominios={t.domains} conf={t.confidence:.2f}")

# ─── 3. IMAGINATION: Conecta invariantes ao simulador e simula ───
print(f"\n[FASE 3] Imagination Simulator...")
sim = ImaginationSimulator()
# Aprende dominios dos invariantes descobertos
sim.learn_domains_from_invariants(invariants)
print(f"  Dominios aprendidos do discovery: {sim.get_domains('branching')}")
instances = sim.imagine("branching")

print(f"  BRANCHING se manifesta em {len(instances)} dominios:")
for inst in instances:
    print(f"    [{inst.domain}] {inst.description[:60]}... (plaus={inst.plausibility:.2f})")

geometric = [i for i in instances if i.domain == "geometry"]
if geometric:
    print(f"\n  >>> BRANCHING em geometria CONFIRMADO!")
    print(f"  >>> {geometric[0].description}")
    print(f"  >>> Transferencia conceitual: biologia -> geometria")
else:
    print(f"\n  >>> BRANCHING NAO encontrado em geometria")
    print(f"  >>> A transferencia NAO foi automatica")

transfer_score = sim.get_transfer_score("branching")
print(f"\n  Transfer Score: {transfer_score:.2f} (max=1.0)")
print(f"  Se > 0.6: conceito e genuinamente abstrato")
print(f"  Resultado: {'ABSTRACAO REAL' if transfer_score > 0.6 else 'PADRAO LOCAL'}")

# ─── 4. CONSTRAINT NETWORK: Valida o conceito ───
print(f"\n[FASE 4] Constraint Network...")
network = ConstraintNetwork(simulator=sim)
verdicts = network.validate("branching", "branching",
                              domains={"biology", "medicine", "neuroscience", "geography"})

all_approved = True
for v in verdicts:
    status = "APROVADO" if v.approved else "REJEITADO"
    if not v.approved:
        all_approved = False
    print(f"    [{status}] {v.source}: {v.reason}")

print(f"\n  Conceito BRANCHING e {'VALIDO' if all_approved else 'INVALIDO'} na rede de restricoes")

# ─── 5. METRICAS OBJETIVAS ───
print(f"\n[FASE 5] Metricas Objetivas do Conceito")
print(f"\n  {'Metrica':<35s} {'Valor':>10s}")
print(f"  {'-'*45}")
print(f"  {'Predictive Error Reduction':<35s} {1.0 - concept.metric_predictive_error:>10.2%}")
print(f"  {'Search Space Reduction':<35s} {1.0 - concept.metric_search_space:>10.2%}")
print(f"  {'Program Length Reduction':<35s} {1.0 - concept.metric_program_length:>10.2%}")
print(f"  {'Reuse Count':<35s} {concept.metric_reuse_count:>10d}")
print(f"  {'Compression Contribution':<35s} {concept.metric_compression:>10.2%}")
print(f"\n  {'Viability (2+ metrics > 10%)':<35s} {'OK' if concept.is_viable else 'BAIXA'}")

# ─── 6. RESUMO ───
print(f"\n\n{'='*70}")
print("RELATORIO: TESTE DE TRANSFERENCIA CONCEITUAL")
print(f"{'='*70}")
print(f"""
  CONCEITO: branchting
  OBSERVADO EM: biology, medicine, neuroscience, geography (4 dominios)
  INSTANCIA EM GEOMETRIA: {bool(geometric)}
  TRANSFER SCORE: {transfer_score:.2f}/1.0
  VIABILIDADE: {'OK' if concept.is_viable else 'BAIXA'}
  TEORIAS DERIVADAS: {len(theories)}
  
  O sistema{' ' if geometric else ' NAO '}conseguiu transferir o conceito
  de BRANCHING da biologia para a geometria.
  
  {'Isso sugere que o CFE esta formando conceitos genuinamente abstratos.'
   if geometric and transfer_score > 0.6 else
   'O CFE precisa de mais exemplos cross-dominio.'}
""")
