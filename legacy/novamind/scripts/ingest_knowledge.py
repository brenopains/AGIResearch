"""
IngestKnowledge — Pipeline texto → HDV → CGE → KnowledgeGraph.

Le os arquivos de D:\novamind_data\knowledge_raw\ e:
  1. Para cada dominio, le fatos de texto
  2. Usa MultilingualEncoder para gerar HDV (10000D bipolar)
  3. Extrai relações causais (X causa Y, X inibe Y, etc.)
  4. Alimenta o CGE com nós e arestas
  5. Alimenta o KnowledgeGraph
  6. Salva estado processado em D:\novamind_data\knowledge_graph\

Uso:
    python scripts/ingest_knowledge.py
    python scripts/ingest_knowledge.py --domain medicine  # so um dominio
    python scripts/ingest_knowledge.py --delete-raw  # deleta brutos apos processar
"""
import sys, os, json, re, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np


# Padroes de relacao causal em texto
CAUSAL_PATTERNS = [
    (r"(.+?)\s+causes\s+(.+)", "causa"),
    (r"(.+?)\s+leads to\s+(.+)", "causa"),
    (r"(.+?)\s+results in\s+(.+)", "causa"),
    (r"(.+?)\s+activates\s+(.+)", "ativa"),
    (r"(.+?)\s+inhibits\s+(.+)", "inibe"),
    (r"(.+?)\s+blocks\s+(.+)", "inibe"),
    (r"(.+?)\s+suppresses\s+(.+)", "inibe"),
    (r"(.+?)\s+treats\s+(.+)", "trata"),
    (r"(.+?)\s+prevents\s+(.+)", "previne"),
    (r"(.+?)\s+regulates\s+(.+)", "regula"),
    (r"(.+?)\s+modulates\s+(.+)", "modula"),
    (r"(.+?)\s+interacts with\s+(.+)", "interage"),
    # Relacoes implícitas
    (r"(.+?)\s+is characterized by\s+(.+)", "caracteriza"),
    (r"(.+?)\s+is caused by\s+(.+)", "causado_por"),
    (r"(.+?)\s+is associated with\s+(.+)", "associado"),
    (r"(.+?)\s+is linked to\s+(.+)", "associado"),
    (r"(.+?)\s+is involved in\s+(.+)", "envolvido"),
    (r"(.+?)\s+contributes to\s+(.+)", "contribui"),
    (r"(.+?)\s+is mediated by\s+(.+)", "mediado"),
    (r"(.+?)\s+is regulated by\s+(.+)", "regulado"),
    (r"(.+?)\s+works by\s+(.+)", "mecanismo"),
    (r"(.+?)\s+is defined by\s+(.+)", "definido"),
]

# Palavras de parada para extrair conceitos
STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
              "have", "has", "had", "do", "does", "did", "will", "would",
              "can", "could", "may", "might", "shall", "should", "to",
              "of", "in", "for", "on", "with", "at", "by", "from",
              "this", "that", "these", "those", "it", "its", "and", "or",
              "not", "no", "but", "as", "also", "very", "just", "so",
              "about", "than", "into"}

# Extrai o conceito principal de cada fato
MAIN_CONCEPT_RULES = {
    "medicine": [
        (r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s", 1),
        (r"^(Type\s+\d+\s+(?:diabetes|error)?)", 1),
    ],
    "default": [
        (r"^(The\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", 2),
        (r"^([A-Z][a-z]+)", 1),
    ],
}

# Palavras que NAO sao conceitos validos
BAD_CONCEPTS = {"the", "this", "that", "these", "those", "type", "an",
                "long", "short", "high", "low", "new", "old", "some",
                "each", "every", "both", "most", "few", "many"}


def extract_concepts(text: str, domain: str) -> list:
    """Extrai conceitos-chave de um texto factual.

    Usa regras especificas do dominio para identificar o conceito principal.
    Retorna conceitos CURTOS (max 5 palavras).
    """
    concepts = []

    # Tenta regras do dominio primeiro
    rules = MAIN_CONCEPT_RULES.get(domain, MAIN_CONCEPT_RULES["default"])
    for pattern, group_idx in rules:
        match = re.match(pattern, text)
        if match:
            main = match.group(group_idx)
            if main and main.lower() not in STOP_WORDS and main.lower() not in BAD_CONCEPTS:
                # Limita a 5 palavras
                words = main.strip().split()
                concepts.append(" ".join(words[:5]))
                break

    # Extrai conceitos adicionais: termos entre aspas
    for match in re.finditer(r'"([^"]+)"', text):
        term = match.group(1).strip()
        if term and len(term) > 2:
            words = term.split()
            concepts.append(" ".join(words[:3]))

    # Extrai termos compostos (max 5 palavras)
    for match in re.finditer(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})', text):
        term = match.group(0).strip()
        if term and len(term) > 3 and term.lower() not in STOP_WORDS \
                and term.lower() not in BAD_CONCEPTS:
            if term not in concepts:
                # So adiciona se nao for frase longa (max 6 palavras)
                words = term.split()
                if len(words) <= 6:
                    concepts.append(term)

    return concepts[:10]  # max 10 conceitos por fato


def parse_causal_relations(text: str, domain: str) -> list:
    """Extrai relações causais de um texto factual.

    Divide o texto em sentencas e extrai relacoes de cada uma.
    Limpa alvos: remove sufixos como 'in X', 'of Y', 'from Z'.
    """
    relations = []
    seen_pairs = set()

    # Divide em sentencas e processa cada uma
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue

        # Remove prefixo ABSTRACT: se presente
        sentence = re.sub(r'^ABSTRACT:\s*', '', sentence)

        for pattern, rel_type in CAUSAL_PATTERNS:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for source_match, target_match in matches:
                source_clean = source_match.strip().rstrip(".,;:")
                target_clean = target_match.strip().rstrip(".,;:")

                # Remove sufixos de localizacao: " in Alzheimer", " in the brain", etc.
                for suffix in [" in Alzheimer", " in the brain", " in the hippocampus",
                               " in patients", " in the prefrontal cortex", " in the striatum",
                               " in animal models", " in vivo", " in vitro", " in mice",
                               " in Drosophila", " of Alzheimer", " of disease"]:
                    if target_clean.endswith(suffix):
                        target_clean = target_clean[:-len(suffix)]

                # Limita a 5 palavras para ambos source e target
                s_words = source_clean.split()[:5]
                t_words = target_clean.split()[:5]
                source_clean = " ".join(s_words)
                target_clean = " ".join(t_words)

                # Remove entidades compostas com "and" no final
                if source_clean.endswith(" and"):
                    source_clean = source_clean[:-4]
                if target_clean.endswith(" and"):
                    target_clean = target_clean[:-4]

                pair = (source_clean, target_clean)
                if (source_clean and target_clean
                    and len(source_clean) > 3 and len(target_clean) > 3
                    and pair not in seen_pairs):
                    seen_pairs.add(pair)
                    relations.append({
                        "source": source_clean,
                        "target": target_clean,
                        "type": rel_type,
                        "evidence": sentence,
                        "domain": domain,
                    })

    return relations[:10]  # max 10 relacoes por texto


def ingest_domain(domain: str, kg, encoder=None, cge=None,
                  raw_dir: str = r"D:\novamind_data\knowledge_raw",
                  delete_raw: bool = False):
    """Ingere todos os fatos de um dominio."""
    filepath = os.path.join(raw_dir, f"{domain}.txt")
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} nao encontrado")
        return 0

    print(f"\n--- Ingerindo: {domain} ---")
    with open(filepath, "r", encoding="utf-8") as f:
        facts = [line.strip() for line in f if line.strip()]

    concept_count = 0
    relation_count = 0

    for i, fact in enumerate(facts):
        # 1. Extrai conceitos
        concepts = extract_concepts(fact, domain)
        main_concept = concepts[0] if concepts else fact.split()[0]

        # 2. Adiciona ao KnowledgeGraph
        kg.add_concept(
            name=main_concept,
            aliases=concepts[1:],
            domain=domain,
            description=fact,
            confidence=0.6,
            source=filepath,
        )

        # 3. Extrai relações causais
        relations = parse_causal_relations(fact, domain)
        for rel in relations:
            kg.add_causal_relation(
                source=rel["source"],
                target=rel["target"],
                relation_type=rel["type"],
                confidence=0.5,
                evidence=rel["evidence"],
                domain=domain,
                source_text=fact,
            )
            relation_count += 1

        concept_count += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(facts)}] {concept_count} conceitos, {relation_count} relacoes")

    print(f"  OK: {concept_count} conceitos, {relation_count} relacoes causais")

    # Opcional: deleta raw apos processar
    if delete_raw:
        os.remove(filepath)
        print(f"  [DELETE] {filepath}")

    return concept_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest knowledge into CGE")
    parser.add_argument("--domain", default=None,
                        help="Dominio especifico (opcional: processa todos)")
    parser.add_argument("--delete-raw", action="store_true",
                        help="Deletar arquivos brutos apos processar")
    parser.add_argument("--output", default=r"D:\novamind_data\knowledge_graph",
                        help="Diretorio de saida")
    parser.add_argument("--no-cge", action="store_true",
                        help="Nao conectar ao CGE (so knowledge graph)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Setup encoder e CGE
    encoder = None
    cge = None
    if not args.no_cge:
        try:
            from core.multilingual_encoder import MultilingualEncoder
            from core.cognitive_graph_engine import CognitiveGraphEngine
            encoder = MultilingualEncoder(device='cuda')
            cge = CognitiveGraphEngine(device='cuda')
            print("Encoder+CGE: OK")
        except Exception as e:
            print(f"  Encoder/CGE offline: {e}")
            print("  (continuando sem encoder — modo texto-only)")

    # Knowledge Graph
    from core.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph(encoder=encoder, cge=cge)

    # Carrega manifesto
    manifest_path = os.path.join(
        r"D:\novamind_data\knowledge_raw", "_manifest.json"
    )
    if not os.path.exists(manifest_path):
        print("ERRO: Rode scripts/download_knowledge.py primeiro")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    print(f"Dominios disponiveis: {manifest['domains']}")
    print(f"Total de fatos: {manifest['total_facts']}")

    # Processa dominios
    domains_to_process = [args.domain] if args.domain else manifest["domains"]
    total_concepts = 0
    total_relations = 0

    for domain in domains_to_process:
        if domain in manifest["domains"]:
            c = ingest_domain(
                domain, kg,
                encoder=encoder,
                cge=cge,
                delete_raw=args.delete_raw,
            )
            total_concepts += c

    total_relations = kg.total_relations
    print(f"\n=== RESUMO ===")
    print(f"Conceitos: {kg.total_ingested}")
    print(f"Relacoes causais: {kg.total_relations}")
    print(f"Dominios processados: {len(domains_to_process)}")

    # Salva estado do grafo
    stats = kg.get_stats()
    stats_path = os.path.join(args.output, "knowledge_graph_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats salvos: {stats_path}")

    # Salva conceitos e relacoes em JSON
    concepts_path = os.path.join(args.output, "concepts.json")
    concepts_data = {
        name: {
            "domain": c.domain,
            "aliases": c.aliases,
            "description": c.description[:100] if c.description else "",
            "confidence": c.confidence,
            "source": c.source,
        }
        for name, c in kg.concepts.items()
    }
    with open(concepts_path, "w", encoding="utf-8") as f:
        json.dump(concepts_data, f, indent=2, ensure_ascii=False)
    print(f"Conceitos salvos: {concepts_path} ({len(concepts_data)} entries)")

    relations_path = os.path.join(args.output, "relations.json")
    relations_data = [r.to_dict() for r in kg.relations[:10000]]
    with open(relations_path, "w", encoding="utf-8") as f:
        json.dump(relations_data, f, indent=2, ensure_ascii=False)
    print(f"Relacoes salvas: {relations_path} ({len(relations_data)} entries)")


if __name__ == "__main__":
    main()
