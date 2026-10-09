"""
run_study.py — Modo estudo autônomo.

O sistema entra em modo ESTUDO:
  1. Carrega KnowledgeGraph de D:\novamind_data\knowledge_graph\
  2. A cada ciclo: seleciona um conceito/artigo, processa, gera hipóteses
  3. Alimenta ConceptAgents, WorldModel, LawRegistry
  4. Gera hipóteses causais sobre relacoes entre conceitos
  5. Log de rastreabilidade completo

Uso:
    python run_study.py --cycles 1000 --domain medicine
    python run_study.py --cycles 10000 --focus "Alzheimer"
"""
import sys, os, time, json, random
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from core.knowledge_graph import KnowledgeGraph
from core.hypothesis_engine_causal import HypothesisEngineCausal
from core.conceptual_discovery.concept_agent import ConceptAgent, ConceptAgentRegistry
from core.conceptual_discovery.world_laws import WorldLawRegistry
from core.conceptual_discovery.concept_economy import ConceptEconomy
from core.world_model import WorldModel
from core.traceability import TraceabilityAnalyzer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Study Mode")
    parser.add_argument("--cycles", type=int, default=5000)
    parser.add_argument("--snapshot", type=int, default=100)
    parser.add_argument("--focus", default=None, help="Conceito foco (ex: Alzheimer)")
    parser.add_argument("--domain", default=None, help="Dominio foco")
    parser.add_argument("--load-cge", action="store_true", help="Carregar CGE real")
    args = parser.parse_args()

    print("=" * 60)
    print(f"ESTUDO AUTONOMO — {args.cycles} ciclos")
    if args.focus:
        print(f"Foco: {args.focus}")
    if args.domain:
        print(f"Dominio: {args.domain}")
    print("=" * 60)

    # ─── Carrega conhecimento ───
    kg_path = r"D:\novamind_data\knowledge_graph"
    if not os.path.exists(os.path.join(kg_path, "concepts.json")):
        print("ERRO: Rode scripts/ingest_knowledge.py primeiro")
        return

    kg = KnowledgeGraph()
    with open(os.path.join(kg_path, "concepts.json"), "r", encoding="utf-8") as f:
        concepts_data = json.load(f)
    for name, data in concepts_data.items():
        kg.add_concept(
            name=name,
            domain=data.get("domain", "general"),
            description=data.get("description", ""),
            confidence=data.get("confidence", 0.5),
        )
    print(f"KnowledgeGraph: {len(kg.concepts)} conceitos, {len(kg.relations)} relacoes")

    # Carrega relacoes
    rel_path = os.path.join(kg_path, "relations.json")
    if os.path.exists(rel_path):
        with open(rel_path, "r", encoding="utf-8") as f:
            relations_data = json.load(f)
        for r in relations_data:
            kg.add_causal_relation(
                source=r["source"], target=r["target"],
                relation_type=r["type"],
                confidence=r.get("confidence", 0.5),
                evidence=r.get("evidence", ""),
                domain=r.get("domain", ""),
            )
        print(f"  Relacoes carregadas: {len(relations_data)}")

    # ─── Setup ───
    concept_registry = ConceptAgentRegistry()
    law_registry = WorldLawRegistry()
    concept_economy = ConceptEconomy()
    world_model = WorldModel()
    hyp_causal = HypothesisEngineCausal(knowledge_graph=kg)

    # Cria ConceptAgents dos dominios principais
    domains = set()
    for name, c in kg.concepts.items():
        if c.domain:
            domains.add(c.domain)

    for domain in sorted(domains):
        agent = ConceptAgent(
            name=f"study_{domain}",
            invariant_type="knowledge",
            description=f"studies {domain}",
        )
        agent.add_prediction("domain_knowledge", True)
        agent.observe(domain)
        agent.confidence = 0.5
        concept_registry.register(agent)

    print(f"ConceptAgents: {concept_registry.count} (dominios: {len(domains)})")

    # Traceability
    analyzer = TraceabilityAnalyzer()

    # ─── Loop de Estudo ───
    concept_names = list(kg.concepts.keys())
    start_time = time.time()

    for cycle in range(1, args.cycles + 1):
        # 1. Seleciona um conceito para estudar
        if args.focus:
            # Foco: sempre em torno do conceito alvo
            related = [n for n in concept_names if args.focus.lower() in n.lower()]
            if not related:
                related = concept_names
            concept_name = random.choice(related)
        elif args.domain:
            # Dominio especifico
            domain_concepts = [n for n, c in kg.concepts.items()
                              if c.domain == args.domain]
            concept_name = random.choice(domain_concepts) if domain_concepts else random.choice(concept_names)
        else:
            concept_name = random.choice(concept_names)

        concept = kg.get_concept(concept_name)
        if not concept:
            continue

        # 2. Alimenta ConceptAgents
        for agent in concept_registry.get_all():
            if concept.domain in agent.domains:
                agent.observe(concept.domain)

        # 3. Alimenta ConceptEconomy
        concept_economy.observe_concept(
            concept.name, {concept.domain or "general"},
            confidence=concept.confidence,
        )

        # 4. Gera hipóteses causais sobre este conceito
        hypotheses = hyp_causal.generate_hypotheses(
            concept_name, max_hypotheses=3
        )

        # 5. Periodicamente: economia, leis, trace
        if cycle % args.snapshot == 0:
            economy_actions = concept_economy.run()

            # Stats
            hyp_stats = hyp_causal.get_stats()
            snap = analyzer.snapshot(cycle)

            elapsed = time.time() - start_time
            print(f"  [{cycle}/{args.cycles}] {elapsed:.0f}s | "
                  f"conceitos={len(kg.concepts)} "
                  f"relacoes={len(kg.relations)} "
                  f"hipoteses={hyp_stats['total_hypotheses']} "
                  f"novas={hyp_stats['novel_hypotheses']}")

        # 6. Simula processamento com sleep minimo
        if cycle % 1000 == 0:
            time.sleep(0.01)  # pequena pausa para simular processamento

    # ─── Relatorio Final ───
    elapsed = time.time() - start_time
    analyzer.snapshot(args.cycles)

    hyp_stats = hyp_causal.get_stats()
    print(f"\n{'='*60}")
    print(f"RELATORIO DE ESTUDO")
    print(f"{'='*60}")
    print(f"Duracao: {elapsed:.0f}s, {args.cycles} ciclos")
    print(f"\nKnowledge Graph:")
    print(f"  Conceitos: {len(kg.concepts)}")
    print(f"  Relacoes causais: {len(kg.relations)}")
    print(f"\nHipoteses Causais:")
    for k, v in hyp_stats.items():
        print(f"  {k}: {v}")

    # Top 5 hipoteses com maior confianca
    all_hyp = sorted(hyp_causal.hypotheses, key=lambda h: -h.confidence)[:10]
    if all_hyp:
        print(f"\nTop hipoteses:")
        for h in all_hyp:
            print(f"  {h.cause} → {h.effect} ({h.relation_type}, conf={h.confidence:.3f})")

    report_path = r"D:\novamind_data\traceability\study_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "cycles": args.cycles,
            "elapsed": round(elapsed, 1),
            "knowledge_graph": {
                "concepts": len(kg.concepts),
                "relations": len(kg.relations),
                "domains": len(domains),
            },
            "hypotheses": hyp_stats,
            "top_hypotheses": [h.to_dict() for h in all_hyp],
        }, f, indent=2)
    print(f"\nRelatorio: {report_path}")


if __name__ == "__main__":
    main()
