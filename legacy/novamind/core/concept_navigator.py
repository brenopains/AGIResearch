"""
concept_navigator.py — MOTOR DE NAVEGACAO CONCEITUAL (usando o cerebro REAL da AGI)

A AGI navega entre SEUS PROPRIOS conceitos usando:
  - HDV brain (560 MB, 10.000 dimensões) — memória de longo prazo
  - Knowledge graph (13.822 conceitos, 1.391 relações) — grafo semântico
  - Knowledge raw (48 textos) — conhecimento de domínio
  - SBML models (12 reais) — ground truth mecanístico

A AGI:
  1. Consulta o HDV brain por similaridade semântica
  2. Navega entre conceitos por associação HDV + relações KG
  3. Combina conceitos de domínios diferentes -> novos blends
  4. Testa blends contra SBMLs reais
  5. Retém o que funciona -> expande o próprio conhecimento
"""
import sys, os, json, time, random, copy
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class ConceptNode:
    """Um conceito na memoria da AGI."""
    id: str
    name: str
    domain: str
    source: str = "unknown"       # hdv, kg, raw, sbml, blend
    description: str = ""
    variables: List[str] = field(default_factory=list)
    mechanism_type: str = "unknown"
    hdv_similarity: float = 0.0
    blended_from: List[str] = field(default_factory=list)
    fitness: float = 0.0
    times_tested: int = 0
    times_survived: int = 0
    metadata: dict = field(default_factory=dict)

    def survival_rate(self) -> float:
        return self.times_survived / max(1, self.times_tested)

    def to_dict(self):
        return {
            "id": self.id, "name": self.name[:60], "domain": self.domain,
            "source": self.source, "fitness": round(self.fitness, 3),
            "survival": round(self.survival_rate(), 3),
            "blended": len(self.blended_from) > 0,
            "hdv_sim": round(self.hdv_similarity, 3),
        }


class ConceptNavigator:
    """Navega o espaco conceitual REAL da AGI.

    Usa o cerebro HDV (560 MB) como memoria associativa principal,
    o grafo de conhecimento como indices estrutural,
    e SBMLs como ground truth para testar hipoteses.
    """

    def __init__(self):
        self.concepts: Dict[str, ConceptNode] = {}
        self.hdv_memory = None
        self.kg_concepts: Dict = {}
        self.kg_relations: List = []
        self.gt = None
        self.blend_history: List[dict] = []
        self.navigation_log: List[dict] = []

    # ═══════════════════════════════════════
    # CARGA DO CONHECIMENTO
    # ═══════════════════════════════════════

    def load_hdv_brain(self):
        """Carrega o cerebro HDV (560 MB) — a memoria real da AGI."""
        try:
            from core.hdc_memory import HDCMemory
            self.hdv_memory = HDCMemory()
            stats = self.hdv_memory.get_stats()
            print(f"  HDV brain: {stats['total_items']} itens, {stats['total_ram_kb']:.0f} KB RAM")
            print(f"  Fontes: {stats['sources']}")
            return True
        except Exception as e:
            print(f"  Erro carregando HDV: {e}")
            return False

    def load_knowledge_graph(self, kg_path: str):
        """Carrega o grafo de conhecimento (13.822 conceitos + 1.391 relacoes)."""
        concepts_file = os.path.join(kg_path, "concepts.json")
        relations_file = os.path.join(kg_path, "relations.json")
        loaded = 0
        if os.path.exists(concepts_file):
            with open(concepts_file, "r", encoding="utf-8") as f:
                self.kg_concepts = json.load(f)
            for name, data in self.kg_concepts.items():
                domain = data.get("domain", "unknown") if isinstance(data, dict) else "unknown"
                node = ConceptNode(
                    id=f"kg_{loaded}",
                    name=str(name)[:100],
                    domain=domain,
                    source="kg",
                    description=str(data.get("description", ""))[:200] if isinstance(data, dict) else "",
                )
                self.concepts[node.id] = node
                loaded += 1
            print(f"  KG: {loaded} conceitos carregados")

        if os.path.exists(relations_file):
            with open(relations_file, "r", encoding="utf-8") as f:
                self.kg_relations = json.load(f)
            print(f"  KG: {len(self.kg_relations)} relacoes carregadas")

    def load_knowledge_raw(self, raw_path: str):
        """Carrega os textos de conhecimento (48 dominios)."""
        if not os.path.exists(raw_path):
            return
        loaded = 0
        for fname in os.listdir(raw_path):
            if fname.endswith(".txt"):
                domain = fname.replace(".txt", "")
                fpath = os.path.join(raw_path, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()
                # Cria um conceito para o dominio
                node = ConceptNode(
                    id=f"raw_{domain}",
                    name=f"Knowledge: {domain}",
                    domain=domain,
                    source="raw",
                    description=text[:200],
                    metadata={"file": fname, "size": len(text)},
                )
                self.concepts[node.id] = node
                loaded += 1
        print(f"  Raw: {loaded} dominios carregados")

    def load_ground_truth(self, gt):
        """Conecta o ground truth (modelos SBML reais do BioModels)."""
        self.gt = gt
        if not hasattr(gt, 'AVAILABLE_MODELS'):
            return
        for key, info in gt.AVAILABLE_MODELS.items():
            vars_list = []
            params_list = []
            try:
                mi = gt.get_model_info(key)
                if mi:
                    vars_list = mi.species_names
                    params_list = mi.parameter_names
            except:
                pass
            node = ConceptNode(
                id=f"sbml_{key}",
                name=info["name"],
                domain="molecular_biology",
                source="sbml",
                description=info["description"],
                variables=vars_list,
                mechanism_type="sbml_model",
                metadata={"model_key": key, "n_params": len(params_list)},
            )
            self.concepts[node.id] = node
        print(f"  SBML: {len(gt.AVAILABLE_MODELS)} modelos carregados")

    def load_all(self, kg_path: str = None, raw_path: str = None, gt=None):
        """Carrega TODAS as fontes de conhecimento da AGI."""
        print("Carregando conhecimento da AGI...")
        self.load_hdv_brain()
        if kg_path:
            self.load_knowledge_graph(kg_path)
        if raw_path:
            self.load_knowledge_raw(raw_path)
        if gt:
            self.load_ground_truth(gt)
        print(f"\nTotal: {len(self.concepts)} conceitos carregados")

    # ═══════════════════════════════════════
    # NAVEGACAO POR HDV
    # ═══════════════════════════════════════

    def query_brain(self, text: str, top_k: int = 10) -> List[Tuple[float, dict, str]]:
        """Consulta o cerebro HDV da AGI por similaridade semantica."""
        if self.hdv_memory is None:
            return []
        return self.hdv_memory.query(text, top_k=top_k)

    def retrieve_context(self, text: str, top_k: int = 5) -> str:
        """Obtem contexto relevante do cerebro da AGI para um texto."""
        if self.hdv_memory is None:
            return ""
        return self.hdv_memory.get_context(text, top_k=top_k)

    def navigate_by_association(self, seed_concept: str, n_hops: int = 3,
                                 top_k: int = 5) -> List[ConceptNode]:
        """Navega por associacao HDV: de um conceito, encontra similares.

        A AGI 'pensa' associativamente: dado um conceito,
        busca no proprio cerebro conceitos similares,
        depois busca similares dos similares (n hops).
        """
        visited = set()
        path = []
        current_text = seed_concept

        for _ in range(n_hops):
            results = self.query_brain(current_text, top_k=top_k)
            for sim, meta, source in results:
                label = meta.get("text", meta.get("id", ""))
                if label not in visited:
                    visited.add(label)
                    node = ConceptNode(
                        id=f"hdv_{len(visited)}",
                        name=label[:100],
                        domain=meta.get("domain", "unknown"),
                        source="hdv",
                        hdv_similarity=sim,
                        metadata={"source_brain": source},
                    )
                    self.concepts[node.id] = node
                    path.append(node)
                    if len(path) >= n_hops * top_k:
                        break
            if path:
                # Proximo hop: usa o ultimo conceito encontrado
                current_text = path[-1].name[:50]
        return path

    # ═══════════════════════════════════════
    # EXPLORACAO DO GRAFO
    # ═══════════════════════════════════════

    def get_concepts_by_domain(self, domain: str) -> List[ConceptNode]:
        """Retorna todos os conceitos de um dominio."""
        return [n for n in self.concepts.values() if n.domain == domain]

    def search_concepts(self, query: str, top_k: int = 20) -> List[ConceptNode]:
        """Busca conceitos por nome."""
        q = query.lower()
        results = []
        for node in self.concepts.values():
            if q in node.name.lower() or q in node.domain.lower():
                results.append(node)
        results.sort(key=lambda n: n.fitness, reverse=True)
        return results[:top_k]

    def identify_gaps(self) -> List[dict]:
        """Identifica dominios sem modelo SBML (lacunas de conhecimento mecanistico)."""
        domains_with_sbml = set()
        sbml_vars = set()
        for node in self.concepts.values():
            if node.source == "sbml":
                domains_with_sbml.add(node.domain)
                for v in node.variables:
                    sbml_vars.add(v.lower())

        domains_count = defaultdict(int)
        for node in self.concepts.values():
            domains_count[node.domain] += 1

        gaps = []
        for domain, count in sorted(domains_count.items(), key=lambda x: -x[1]):
            if count > 30 and domain not in domains_with_sbml and domain != "unknown":
                # Amostra de conceitos deste dominio
                sample = [n.name[:60] for n in self.concepts.values()
                          if n.domain == domain][:5]
                gaps.append({
                    "domain": domain,
                    "n_concepts": count,
                    "sample": sample,
                })
        return gaps

    # ═══════════════════════════════════════
    # BLENDING ENGINE
    # ═══════════════════════════════════════

    def find_blend_candidates(self, target: str = "schizophrenia",
                               n_pairs: int = 5) -> List[Tuple[ConceptNode, ConceptNode]]:
        """Encontra pares de conceitos para blend.

        Usa o HDV brain para encontrar conceitos relacionados ao target
        em dominios DIFERENTES, criando combinacoes inter-domínio.
        """
        # Consulta o cerebro sobre o target
        brain_results = self.query_brain(target, top_k=20)

        # Agrupa por dominio
        by_domain = defaultdict(list)
        for sim, meta, source in brain_results:
            label = meta.get("text", meta.get("id", ""))
            domain = meta.get("domain", "unknown")
            by_domain[domain].append((label, sim, source))

        # Cria candidatos de dominios diferentes
        domain_list = list(by_domain.keys())
        candidates = []
        for i in range(len(domain_list)):
            for j in range(i + 1, len(domain_list)):
                d1, d2 = domain_list[i], domain_list[j]
                if by_domain[d1] and by_domain[d2]:
                    c1 = by_domain[d1][0]
                    c2 = by_domain[d2][0]
                    n1 = ConceptNode(id=f"candidate_a_{len(candidates)}",
                                     name=c1[0], domain=d1, source="hdv",
                                     hdv_similarity=c1[1])
                    n2 = ConceptNode(id=f"candidate_b_{len(candidates)}",
                                     name=c2[0], domain=d2, source="hdv",
                                     hdv_similarity=c2[1])
                    candidates.append((n1, n2))

        random.shuffle(candidates)
        return candidates[:n_pairs]

    def blend(self, concept_a: ConceptNode, concept_b: ConceptNode) -> ConceptNode:
        """Combina dois conceitos em um NOVO conceito.

        O blend cria algo que nao existia em nenhum dos dois isoladamente.
        A AGI decide o nome, dominio e estrutura do blend.
        """
        blend_id = f"blend_{int(time.time())}_{len(self.blend_history)}"
        name = f"{concept_a.name[:20]} + {concept_b.name[:20]}"
        domain = "cross_domain" if concept_a.domain != concept_b.domain else concept_a.domain

        blend = ConceptNode(
            id=blend_id,
            name=name,
            domain=domain,
            source="blend",
            description=f"Blend: {concept_a.source}/{concept_a.domain} + {concept_b.source}/{concept_b.domain}",
            blended_from=[concept_a.id, concept_b.id],
            variables=list(set(concept_a.variables + concept_b.variables)),
            metadata={
                "parent_a": concept_a.name[:50],
                "parent_b": concept_b.name[:50],
                "parent_a_domain": concept_a.domain,
                "parent_b_domain": concept_b.domain,
            },
        )
        self.concepts[blend_id] = blend
        self.blend_history.append({
            "blend_id": blend_id,
            "name": name,
            "parents": [concept_a.name[:50], concept_b.name[:50]],
            "domains": [concept_a.domain, concept_b.domain],
            "time": time.time(),
        })
        return blend

    # ═══════════════════════════════════════
    # TESTE CONTRA GROUND TRUTH
    # ═══════════════════════════════════════

    def test_blend(self, blend: ConceptNode) -> dict:
        """Testa um blend contra SBML real.

        Retorna: sobreviveu ou nao.
        """
        if self.gt is None:
            return {"survived": False, "reason": "no ground truth"}

        # Encontra SBML com maior overlap de conceitos
        best_key, best_overlap = None, 0
        for key in self.gt.AVAILABLE_MODELS:
            mi = self.gt.get_model_info(key)
            if mi is None:
                continue
            overlap = len(set(blend.variables) & set(mi.species_names))
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key
                best_ratio = overlap / max(1, len(mi.species_names))

        result = {
            "survived": best_overlap > 0,
            "matched_sbml": best_key,
            "overlap": best_overlap,
            "overlap_ratio": best_ratio if best_key else 0,
            "n_sbml_species": len(self.gt.get_model_info(best_key).species_names) if best_key and self.gt.get_model_info(best_key) else 0,
        }

        blend.times_tested += 1
        if result["survived"]:
            blend.times_survived += 1
            blend.fitness = min(1.0, result["overlap_ratio"])

        return result

    # ═══════════════════════════════════════
    # CICLO AUTONOMO
    # ═══════════════════════════════════════

    def autonomous_cycle(self, target: str = "schizophrenia") -> dict:
        """Ciclo autonomo completo.

        1. Consulta o cerebro HDV sobre o target
        2. Encontra candidatos a blend em dominios diferentes
        3. Executa blends
        4. Testa contra SBML
        5. Registra resultados
        """
        log = {
            "time": time.time(),
            "target": target,
            "n_concepts_before": len(self.concepts),
            "n_blends": 0,
            "n_survived": 0,
            "hdv_context": "",
        }

        # 1. Consulta o cerebro
        ctx = self.retrieve_context(target, top_k=5)
        log["hdv_context"] = ctx
        print(f"\n  HDV context for '{target}':")
        for line in ctx.split("\n")[:5]:
            print(f"    {line[:100]}")

        # 2. Encontra candidatos
        candidates = self.find_blend_candidates(target, n_pairs=3)
        print(f"  Candidatos a blend: {len(candidates)} pares")

        # 3. Blends
        for a, b in candidates:
            if log["n_blends"] >= 3:
                break
            blend = self.blend(a, b)
            log["n_blends"] += 1
            print(f"  Blend {log['n_blends']}: {a.name[:25]} + {b.name[:25]}")

            # 4. Testa
            result = self.test_blend(blend)
            if result["survived"]:
                log["n_survived"] += 1
                print(f"    -> Sobreviveu! (overlap={result['overlap_ratio']:.2f} com {result['matched_sbml']})")
            else:
                print(f"    -> Descartado")

        log["n_concepts_after"] = len(self.concepts)
        self.navigation_log.append(log)
        return log

    # ═══════════════════════════════════════
    # RELATORIO
    # ═══════════════════════════════════════

    def report(self) -> str:
        """Relatorio do estado atual da AGI."""
        lines = []
        lines.append("=" * 70)
        lines.append("  ESTADO DO CONHECIMENTO DA AGI")
        lines.append("=" * 70)
        lines.append(f"  Conceitos totais: {len(self.concepts)}")

        by_source = defaultdict(int)
        by_domain = defaultdict(int)
        for n in self.concepts.values():
            by_source[n.source] += 1
            by_domain[n.domain] += 1

        lines.append("  Por fonte:")
        for s, c in sorted(by_source.items(), key=lambda x: -x[1]):
            lines.append(f"    {s}: {c}")

        lines.append("\n  Top dominios:")
        for d, c in sorted(by_domain.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    {d}: {c}")

        lines.append(f"\n  Blends criados: {len(self.blend_history)}")
        if self.hdv_memory:
            stats = self.hdv_memory.get_stats()
            lines.append(f"  Cerebro HDV: {stats['total_items']} itens, {stats['total_ram_kb']:.0f} KB")

        gaps = self.identify_gaps()
        lines.append(f"\n  Lacunas (dominios sem modelo SBML):")
        for g in gaps[:5]:
            lines.append(f"    {g['domain']}: {g['n_concepts']} conceitos")

        lines.append("=" * 70)
        return "\n".join(lines)
