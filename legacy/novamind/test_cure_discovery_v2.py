"""
test_cure_discovery_v2.py — Pipeline completo: exploracao → multi-linha → adversario → convergencia.

Todas as quantidades sao DINAMICAS — nao ha numeros fixos.
O sistema decide quando comecar, quando explorar, quando parar.
"""
import sys, json, time
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))

from core.knowledge_graph import KnowledgeGraph
from core.inference_engine import InferenceEngine
from core.hypothesis_filter import HypothesisFilter
from core.explorer import KnowledgeExplorer
from core.multi_line import MultiLineInvestigator
from core.adversarial_tester import AdversarialTester
from core.virtual_experiment import VirtualExperimenter
from core.meta_executive import MetaExecutive, ExecutiveDecision
from core.cure_reasoner import CureReasoner

# 1. Load
kg = KnowledgeGraph()
with open("D:/novamind_data/knowledge_graph/concepts.json", encoding="utf-8") as f:
    data = json.load(f)
for name, info in data.items():
    kg.add_concept(name, domain=info.get("domain", ""))
with open("D:/novamind_data/knowledge_graph/relations.json", encoding="utf-8") as f:
    rels = json.load(f)
for r in rels:
    kg.add_causal_relation(r.get("source",""), r.get("target",""),
                           r.get("type",""), domain=r.get("domain",""))
print(f"Base KG: {kg.total_ingested} conceitos, {kg.total_relations} relacoes")

# 2. Infer + Filter
engine = InferenceEngine(kg)
raw = engine.run_all(target_diseases=[])
filt = HypothesisFilter()
filtered = filt.process(raw, min_quality=0.15)
print(f"Hipoteses: {len(raw)} brutas → {len(filtered)} filtradas")
print(f"  Terapeuticas: {len(filt.get_therapeutic_candidates())}")

# 3. Para cada doenca, executa pipeline completo
CureReasoner._disease_mechanisms = CureReasoner.DISEASE_MECHANISMS

diseases = ["Alzheimer", "Parkinson", "depression", "schizophrenia", "cancer", "diabetes"]
all_results = {}

for disease in diseases:
    print(f"\n{'='*60}")
    print(f"{disease}")
    print(f"{'='*60}")

    # Mecanismos conhecidos para esta doenca
    mechanisms = CureReasoner.DISEASE_MECHANISMS.get(disease, [])

    # 3a. MultiLineInvestigator
    inv = MultiLineInvestigator()
    n_lines = inv.initialize(disease, filtered, mechanisms)
    print(f"  Linhas iniciais: {n_lines}")

    # 3b. Explorer
    explorer = KnowledgeExplorer(kg=kg)

    # 3c. Tester
    tester = AdversarialTester(kg=kg)

    # 3d. MetaExecutive
    meta = MetaExecutive()

    # 3e. VirtualExperimenter (so no final)
    vex = VirtualExperimenter(kg=kg)

    # 3f. Loop DINAMICO — nao ha numero fixo de rodadas
    round_num = 0
    while not inv.should_stop() and round_num < 50:  # 50 e teto de seguranca, nao regra
        round_num += 1

        # EXPLORE: a cada 3 rodadas, vagueia
        if round_num % 3 == 0:
            explorer_findings = explorer.explore(
                start_concept=disease, max_steps=None  # sem maximo! para quando estagnar
            )
        else:
            explorer_findings = []

        # INVESTIGATE: atualiza linhas
        events = inv.run_round(filtered, explorer_findings)

        # TEST: testa cada linha ativa adversarialmente
        adv_stats = {"survived": 0, "refuted": 0}
        for line in inv.lines:
            if line.is_active:
                result = tester.test(line, disease=disease,
                                     filtered_hypotheses=filtered)
                if result.survived:
                    line.refutations_survived += 1
                    line.score = min(1.0, line.score + 0.05)
                    adv_stats["survived"] += 1
                else:
                    line.is_active = False
                    adv_stats["refuted"] += 1

        # META: decide o que fazer
        ml_stats = inv.get_stats()
        exp_stats = explorer.get_stats() if explorer_findings else {"steps_taken": 0}
        decision = meta.tick(ml_stats, adv_stats, exp_stats)

        # LOG
        best = inv.get_best_line()
        best_info = f"best={best.mechanism[:30]}({best.score:.2f})" if best else "no_best"
        score_dist = ml_stats.get("score_distribution", {})
        print(f"  R{round_num:2d}: {decision.action:10s} | "
              f"lines={ml_stats.get('active_lines',0)}/{ml_stats.get('total_lines',0)} | "
              f"{best_info} | "
              f"scores={score_dist.get('min',0):.2f}-{score_dist.get('max',0):.2f} | "
              f"survived={adv_stats['survived']}")

        # Se meta decidiu convergir → para
        if decision.action == "converge":
            print(f"  → CONVERGIU na rodada {round_num}")
            break

        if decision.action == "abandon":
            print(f"  → ABANDONOU na rodada {round_num}")
            break

    # 4. Proposta final
    best_line = inv.get_best_line()
    if best_line:
        # Experimento virtual
        pred = vex.design_experiment(disease, best_line.intervention, best_line.target)

        print(f"\n  ✅ PROPOSTA DE CURA")
        print(f"     Intervencao: {best_line.intervention}")
        print(f"     Mecanismo:   {best_line.mechanism}")
        print(f"     Score:       {best_line.score:.3f}")
        print(f"     Rodadas:     {round_num}")
        print(f"     Refutacoes:  {best_line.refutations_survived}")
        print(f"     Predicao:    {pred.estimated_effect}")
        print(f"     Experimento: {pred.dose_response}")

        all_results[disease] = {
            "intervention": best_line.intervention,
            "mechanism": best_line.mechanism,
            "score": best_line.score,
            "rounds": round_num,
            "survived_refutations": best_line.refutations_survived,
            "prediction": pred.estimated_effect,
            "meta_decisions": [d.to_dict() for d in meta.decisions],
        }
    else:
        print(f"\n  ❌ Nenhuma hipotese sobreviveu")
        all_results[disease] = {"error": "no hypothesis survived"}

# 5. Summary
print(f"\n{'='*60}")
print(f"RESUMO FINAL")
print(f"{'='*60}")
for disease, result in all_results.items():
    if "error" in result:
        print(f"  {disease:20s}: ❌ {result['error']}")
    else:
        print(f"  {disease:20s}: ✅ {result['intervention'][:45]}")
        print(f"  {'':20s}   score={result['score']:.3f}, rounds={result['rounds']}, "
              f"refutations={result['survived_refutations']}")
        print(f"  {'':20s}   ├ meta: {sum(1 for d in result['meta_decisions'] if d['action']=='explore')} explore, "
              f"{sum(1 for d in result['meta_decisions'] if d['action']=='refine')} refine, "
              f"{sum(1 for d in result['meta_decisions'] if d['action']=='converge')} converge")
        print(f"  {'':20s}   └ pred: {result['prediction'][:60]}")
