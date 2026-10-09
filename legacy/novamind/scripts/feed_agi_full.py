"""
feed_agi_full.py — Alimenta TODOS os módulos da AGI com dados reais do PubMed.

1. KnowledgeGraph (conceitos + relacoes causais)
2. ConceptAgentRegistry (cria agents para cada dominio descoberto)
3. WorldModel (alimenta com predicoes baseadas nos abstracts)
4. CGE (via MultilingualEncoder — se disponivel)
5. CognitiveExecutive (ciclos de estudo)

Uso:
    python scripts/feed_agi_full.py
    python scripts/feed_agi_full.py --ingest-only  # so o pipeline basico
    python scripts/feed_agi_full.py --with-cge     # inclui CGE (requer GPU)
"""
import sys, os, json, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def feed_knowledge_graph(kg, abstracts_dir, causal_file):
    """Alimenta o KnowledgeGraph com abstracts + relacoes."""
    print("\n--- KnowledgeGraph ---")

    # Carrega relacoes causais
    causal_relations = []
    if os.path.exists(causal_file):
        with open(causal_file, "r", encoding="utf-8") as f:
            causal_relations = json.load(f)
        print(f"  Relacoes causais: {len(causal_relations)}")

    for rel in causal_relations[:10000]:
        kg.add_causal_relation(
            source=rel.get("source", "unknown"),
            target=rel.get("target", "unknown"),
            relation_type=rel.get("type", "associado"),
            confidence=0.5,
            domain=rel.get("domain", "general"),
            source_text=rel.get("abstract", ""),
        )

    # Carrega abstracts por dominio
    total_abs = 0
    for fname in os.listdir(abstracts_dir):
        if not fname.endswith(".json") or fname in ("all_abstracts.json", "causal_relations.json"):
            continue
        domain = fname.replace(".json", "")
        filepath = os.path.join(abstracts_dir, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            abstracts = json.load(f)
        for ab in abstracts:
            title = ab.get("title", "")
            abstract = ab.get("abstract", "")
            kg.add_concept(
                name=title[:80] if title else f"pubmed_{ab.get('pmid','')}",
                aliases=[],
                domain=domain,
                description=abstract[:200],
                confidence=0.6,
                source=f"PubMed:{ab.get('pmid','')}",
            )
            total_abs += 1
    print(f"  Abstracts carregados: {total_abs}")
    print(f"  Total conceitos: {kg.total_ingested}")
    print(f"  Total relacoes:  {kg.total_relations}")


def feed_concept_agents(concept_registry, hypothesis_engine, kg):
    """Cria ConceptAgents para cada dominio descoberto nos dados."""
    print("\n--- ConceptAgents ---")

    # Agrupa conceitos por dominio
    domains = {}
    for name, concept in kg.concepts.items():
        d = concept.domain or "general"
        if d not in domains:
            domains[d] = []
        domains[d].append(name)

    print(f"  Dominios encontrados: {len(domains)}")

    from core.conceptual_discovery.concept_agent import ConceptAgent

    for domain, concepts in domains.items():
        agent = ConceptAgent(
            name=f"pubmed_{domain}",
            invariant_type="knowledge",
            description=f"domain: {domain} ({len(concepts)} concepts)",
        )
        # Adiciona predicoes baseadas nos conceitos do dominio
        top_concepts = concepts[:5]
        for c in top_concepts:
            agent.add_prediction(f"has_concept_{c[:20]}", True)
            agent.add_modal_signature(domain, [c[:20]], invariants=[f"domain_{domain}"])

        agent.observe(domain)
        agent.confidence = min(1.0, 0.3 + len(concepts) * 0.001)
        concept_registry.register(agent)
        hypothesis_engine.register_agent(agent)

    print(f"  Agents criados: {concept_registry.count}")


def feed_world_model(world_model, kg):
    """Alimenta o WorldModel com relacoes do conhecimento."""
    print("\n--- WorldModel ---")

    # Para cada relacao causal, alimenta como predicao
    for rel in kg.relations[:500]:
        obs = {
            "source": rel.source,
            "target": rel.target,
            "type": rel.relation_type,
            "domain": rel.domain,
        }
        pred = world_model.predict(obs)
        world_model.update(obs, {"confirmed": True}, pred)

    print(f"  Predicoes alimentadas: {min(len(kg.relations), 500)}")
    print(f"  Surpresa media: {world_model.get_surprise():.3f}")


def feed_cge(kg, encoder=None, cge=None):
    """Alimenta o CGE com conceitos via HDV."""
    print("\n--- CGE ---")
    if encoder is None or cge is None:
        print("  CGE offline (sem encoder) — pulando")
        return

    count = 0
    for name, concept in list(kg.concepts.items())[:500]:
        try:
            hdv = encoder.encode_bipolar(f"{name}: {concept.description[:100]}")
            cge.create_node(
                node_id=name[:50],
                semantic_vec=hdv,
            )
            count += 1
        except Exception as e:
            pass

    print(f"  Nos criados no CGE: {count}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-only", action="store_true")
    parser.add_argument("--with-cge", action="store_true")
    parser.add_argument("--max-relations", type=int, default=10000)
    args = parser.parse_args()

    print("=" * 60)
    print("FEED AGI — Alimentando TODOS os modulos")
    print("=" * 60)

    abstracts_dir = r"D:\novamind_data\pubmed_real"
    causal_file = r"D:\novamind_data\pubmed_real\causal_relations.json"
    kg_output = r"D:\novamind_data\knowledge_graph"

    if not os.path.exists(abstracts_dir):
        print("ERRO: Rode scripts/fetch_real_pubmed.py primeiro")
        return

    # ─── 1. KnowledgeGraph ───
    from core.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    feed_knowledge_graph(kg, abstracts_dir, causal_file)

    # ─── 2. ConceptAgents ───
    from core.hypothesis_engine import HypothesisEngine
    from core.conceptual_discovery.concept_agent import ConceptAgentRegistry

    concept_registry = ConceptAgentRegistry()
    hypothesis_engine = HypothesisEngine(concept_registry=concept_registry)
    feed_concept_agents(concept_registry, hypothesis_engine, kg)

    # ─── 3. WorldModel ───
    from core.world_model import WorldModel
    world_model = WorldModel()
    feed_world_model(world_model, kg)

    # ─── 4. CGE (opcional) ───
    encoder = None
    cge = None
    if args.with_cge:
        try:
            from core.multilingual_encoder import MultilingualEncoder
            from core.cognitive_graph_engine import CognitiveGraphEngine
            encoder = MultilingualEncoder(device='cuda')
            cge = CognitiveGraphEngine(device='cuda')
        except Exception as e:
            print(f"  CGE offline: {e}")
    feed_cge(kg, encoder, cge)

    # ─── 5. Salva estado ───
    os.makedirs(kg_output, exist_ok=True)

    # Conceitos
    concepts_path = os.path.join(kg_output, "concepts.json")
    concepts_data = {
        n: {"domain": c.domain, "description": c.description[:100],
            "confidence": c.confidence, "source": c.source}
        for n, c in kg.concepts.items()
    }
    with open(concepts_path, "w", encoding="utf-8") as f:
        json.dump(concepts_data, f, indent=2, ensure_ascii=False)
    concept_count = sum(len(v) for v in concepts_data.values()) if isinstance(next(iter(concepts_data.values())), dict) and False else len(concepts_data)
    print(f"\n  Conceitos salvos:     {concepts_path} ({concept_count})")

    # Relacoes
    relations_path = os.path.join(kg_output, "relations.json")
    relations_data = [r.to_dict() for r in kg.relations[:args.max_relations]]
    with open(relations_path, "w", encoding="utf-8") as f:
        json.dump(relations_data, f, indent=2, ensure_ascii=False)
    print(f"  Relacoes salvas:      {relations_path} ({len(relations_data)})")

    # Stats
    stats = kg.get_stats()
    stats.update({
        "concept_agents": concept_registry.count,
        "world_surprise": round(world_model.get_surprise(), 3),
        "world_accuracy": round(world_model.get_accuracy(), 3),
    })
    stats_path = os.path.join(kg_output, "feed_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats:               {stats_path}")

    print(f"\n{'='*60}")
    print(f"RESUMO DA ALIMENTACAO")
    print(f"{'='*60}")
    print(f"  KnowledgeGraph: {concept_count} conceitos, {kg.total_relations} relacoes")
    print(f"  ConceptAgents:  {concept_registry.count}")
    print(f"  WorldModel:     surprise={world_model.get_surprise():.3f}")
    print(f"  CGE online:     {cge is not None}")

    # ─── 6. Se nao for ingest-only, roda estudo curto ───
    if not args.ingest_only and concept_count > 0:
        print("\n--- Estudo rapido pos-alimentacao ---")
        from core.hypothesis_engine_causal import HypothesisEngineCausal
        hc = HypothesisEngineCausal(knowledge_graph=kg)

        targets = ["Alzheimer", "cancer", "depression", "diabetes", "stroke",
                    "autism", "schizophrenia", "Parkinson", "inflammation",
                    "amyloid", "dopamine", "synaptic", "insulin", "CRISPR", "T cell"]
        for t in targets:
            hyps = hc.generate_hypotheses(t, max_hypotheses=3)
            if hyps:
                print(f"  {t:15s}: {len(hyps):2d} hipoteses")
                for h in hyps[:1]:
                    print(f"           {h.cause[:45]:45s} -> {h.effect[:45]}")
            else:
                print(f"  {t:15s}: 0 hipoteses (dados insuficientes)")


if __name__ == "__main__":
    main()
