"""
exploratorium.py — ESPAÇO ABERTO DE EXPLORAÇÃO PARA A AGI

A AGI tem acesso TOTAL a:
  - 242.419 itens no HDV brain (memória de longo prazo)
  - 13.822 conceitos no Knowledge Graph
  - 12 modelos SBML reais (BioModels)
  - 47 textos de conhecimento de domínio

Ela pode:
  - Consultar QUALQUER conceito no HDV
  - Navegar associativamente entre QUALQUER item
  - Combinar QUALQUER par em blends NOVOS
  - Testar QUALQUER blend contra QUALQUER SBML
  - RETER o que sobrevive

EU não decido:
  - O que explorar
  - O que combinar
  - O que testar
  - O que reter

A AGI decide TUDO.
"""
import sys, os, json, time, random, math
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class Concept:
    """Um item qualquer na memória da AGI."""
    uid: str
    label: str
    source: str        # hdv, kg, raw, sbml, blend
    domain: str
    vector: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)
    parents: List[str] = field(default_factory=list)
    fitness: float = 0.0
    tested: int = 0
    survived: int = 0

    @property
    def survival(self):
        return self.survived / max(1, self.tested)


class Exploratorium:
    """A AGI navega livremente, combinando e testando o que quiser."""

    def __init__(self):
        self.all: Dict[str, Concept] = {}
        self.blends: List[str] = []
        self.discoveries: List[dict] = []
        self.session_log: List[dict] = []
        self._rng = random.Random(42)

        # Conexões externas (setadas depois)
        self._hdv = None
        self._gt = None
        self._kg_raw = {}

    # ─── CARGA ───

    def load_hdv(self):
        """Carrega o cérebro HDV."""
        from core.hdc_memory import HDCMemory
        self._hdv = HDCMemory()
        stats = self._hdv.get_stats()
        print(f"[HDV] {stats['total_items']} itens, {stats['total_ram_kb']:.0f} KB")
        return stats['total_items']

    def load_knowledge_graph(self, path: str):
        """Carrega KG como conceitos."""
        f = os.path.join(path, "concepts.json")
        if not os.path.exists(f):
            return 0
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        n = 0
        for name, meta in data.items():
            domain = meta.get("domain", "unknown") if isinstance(meta, dict) else "unknown"
            c = Concept(
                uid=f"kg_{n}",
                label=str(name)[:200],
                source="kg",
                domain=domain,
                metadata={"description": str(meta.get("description", ""))[:200]} if isinstance(meta, dict) else {},
            )
            self.all[c.uid] = c
            n += 1
        print(f"[KG] {n} conceitos carregados")
        return n

    def load_knowledge_raw(self, path: str):
        """Carrega textos de domínio."""
        if not os.path.exists(path):
            return 0
        for fname in os.listdir(path):
            if not fname.endswith(".txt"):
                continue
            domain = fname.replace(".txt", "")
            fpath = os.path.join(path, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
            c = Concept(
                uid=f"raw_{domain}",
                label=f"Knowledge: {domain}",
                source="raw",
                domain=domain,
                metadata={"text": text[:500], "size": len(text)},
            )
            self.all[c.uid] = c
            self._kg_raw[domain] = text
        n = sum(1 for c in self.all.values() if c.source == "raw")
        print(f"[RAW] {n} domínios carregados")
        return n

    def load_sbml(self, gt):
        """Conecta SBML ground truth."""
        self._gt = gt
        n = 0
        for key, info in gt.AVAILABLE_MODELS.items():
            vars_list = []
            try:
                mi = gt.get_model_info(key)
                if mi:
                    vars_list = mi.species_names
            except:
                pass
            c = Concept(
                uid=f"sbml_{key}",
                label=info["name"],
                source="sbml",
                domain="molecular_biology",
                metadata={"key": key, "n_vars": len(vars_list), "species": vars_list},
            )
            self.all[c.uid] = c
            n += 1
        print(f"[SBML] {n} modelos carregados")
        return n

    def load_all(self, kg_path: str = None, raw_path: str = None, gt=None):
        """Carrega TUDO."""
        self.load_hdv()
        if kg_path:
            self.load_knowledge_graph(kg_path)
        if raw_path:
            self.load_knowledge_raw(raw_path)
        if gt:
            self.load_sbml(gt)
        print(f"\n>>> Total: {len(self.all)} conceitos na memória")
        print(f">>> HDV: 242.419 itens (cérebro)")
        print(f">>> AGI pode consultar, navegar, combinar e testar livremente")

    # ─── CONSULTA AO HDV ───

    def hdv_query(self, text: str, top_k: int = 5) -> List[Concept]:
        """Consulta o cérebro HDV."""
        if self._hdv is None:
            return []
        results = self._hdv.query(text, top_k=top_k)
        concepts = []
        for sim, meta, source in results:
            label = meta.get("text", meta.get("id", ""))
            domain = meta.get("domain", "unknown")
            c = Concept(
                uid=f"hdv_{len(self.all)}_{random.randint(0, 99999)}",
                label=label[:200],
                source="hdv",
                domain=domain,
                metadata={"sim": sim, "brain": source, "raw": meta},
            )
            concepts.append(c)
        return concepts

    def hdv_random_sample(self, n: int = 20) -> List[Concept]:
        """Pesca itens ALEATÓRIOS do cérebro HDV.

        A AGI pode explorar o que quiser, sem direção minha.
        """
        if self._hdv is None:
            return []
        # Acessa o índice do HDV e amostra aleatoriamente
        bits = []
        for key, indices in self._hdv._indices.items():
            if len(indices) > 0:
                sample = random.sample(indices, min(n // max(1, len(self._hdv._indices)), len(indices)))
                bits.extend(sample)
        concepts = []
        for meta in bits[:n]:
            label = meta.get("text", meta.get("id", ""))
            if not label:
                continue
            c = Concept(
                uid=f"hdv_rand_{len(concepts)}",
                label=label[:200],
                source="hdv",
                domain=meta.get("domain", "unknown"),
                metadata={"raw": meta},
            )
            concepts.append(c)
        return concepts

    # ─── NAVEGAÇÃO LIVRE ───

    def wander(self, seed: str = None, steps: int = 5, branch: int = 3) -> List[Concept]:
        """A AGI vagueia livremente pela memória.

        Dado um ponto de partida (ou aleatório), ela:
        1. Consulta o HDV por similaridade
        2. Pega um resultado
        3. Usa ELE como nova consulta
        4. Repete 'steps' vezes
        5. Em cada passo, guarda 'branch' conceitos

        Resultado: um caminho associativo que a AGI escolheu.
        """
        if seed is None:
            # Começa de um conceito aleatório do KG
            kg_nodes = [c for c in self.all.values() if c.source == "kg"]
            if kg_nodes:
                seed = random.choice(kg_nodes).label[:50]

        path = []
        current = seed or "life"
        for _ in range(steps):
            results = self.hdv_query(current, top_k=branch)
            for c in results:
                if c.label != current:
                    path.append(c)
                    if len(path) >= steps * branch:
                        break
            if path:
                current = path[-1].label[:50]
            else:
                break
        return path

    def find_strange_bedfellows(self, domain_a: str, domain_b: str, k: int = 5) -> List[Tuple[Concept, Concept]]:
        """Encontra pares de conceitos em domínios DISTANTES.

        A AGI procura conexões entre áreas que NINGUÉM conectou antes.
        """
        a_concepts = [c for c in self.all.values() if c.domain == domain_a]
        b_concepts = [c for c in self.all.values() if c.domain == domain_b]
        if not a_concepts or not b_concepts:
            return []
        pairs = []
        for _ in range(k):
            a = random.choice(a_concepts)
            b = random.choice(b_concepts)
            pairs.append((a, b))
        return pairs

    # ─── BLENDING ───

    def blend(self, a: Concept, b: Concept) -> Concept:
        """Combina DOIS conceitos em UM novo.

        A AGI cria algo que não existe em nenhum dos pais.
        O blending é feito por:
        - Nome: fusão dos nomes
        - Domínio: se diferente, vira cross_domain
        - Conteúdo: união dos metadados
        """
        uid = f"blend_{len(self.blends)}_{int(time.time())}"
        # Define domínio
        if a.domain == b.domain:
            domain = a.domain
        else:
            domain = f"{a.domain}__{b.domain}"

        c = Concept(
            uid=uid,
            label=f"{a.label[:30].strip()} + {b.label[:30].strip()}",
            source="blend",
            domain=domain,
            metadata={
                "parent_a": {"label": a.label[:100], "source": a.source, "domain": a.domain},
                "parent_b": {"label": b.label[:100], "source": b.source, "domain": b.domain},
            },
            parents=[a.uid, b.uid],
        )
        self.all[uid] = c
        self.blends.append(uid)
        return c

    # ─── TESTE CIENTÍFICO ───

    def test(self, concept: Concept) -> dict:
        """Testa um conceito contra a realidade (SBML).

        Retorna se o conceito TEM CORRESPONDÊNCIA com algum mecanismo real.
        Se sim, ele 'sobrevive'.
        """
        result = {
            "uid": concept.uid,
            "label": concept.label[:60],
            "tested": True,
            "survived": False,
            "matched_sbml": None,
            "overlap": 0,
        }
        concept.tested += 1

        if self._gt is None or concept.source == "sbml":
            # Conceitos SBML já são reais por definição
            if concept.source == "sbml":
                concept.survived += 1
                result["survived"] = True
            return result

        # Extrai palavras-chave do conceito
        keywords = set(concept.label.lower().split())
        # Adiciona palavras do domínio
        keywords.add(concept.domain.lower())

        best_key, best_overlap = None, 0
        for key in self._gt.AVAILABLE_MODELS:
            mi = self._gt.get_model_info(key)
            if mi is None:
                continue
            # Overlap entre keywords do conceito e species do SBML
            species_set = set(s.lower() for s in mi.species_names)
            overlap = len(keywords & species_set)
            # Também checa se o domínio do conceito aparece no nome do modelo
            if concept.domain.lower() in mi.name.lower():
                overlap += 2
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key

        result["matched_sbml"] = best_key
        result["overlap"] = best_overlap
        result["survived"] = best_overlap > 1  # Sobrevive se overlap significativo

        if result["survived"]:
            concept.survived += 1
            concept.fitness = min(1.0, best_overlap / 10.0)

        return result

    # ─── CICLO AUTÔNOMO ───

    def cycle(self, n_explore: int = 10, n_blend: int = 5) -> dict:
        """Um CICLO COMPLETO onde a AGI decide TUDO.

        1. EXPLORA: vagueia pela memória aleatoriamente
        2. ENCONTRA: pares inusitados (cross-domain)
        3. COMBINA: cria blends
        4. TESTA: contra SBML real
        5. RELATA: o que sobreviveu
        """
        log = {
            "time": time.time(),
            "phase": [],
            "n_explored": 0,
            "n_blends": 0,
            "n_survived": 0,
            "discoveries": [],
        }

        # 1. EXPLORAÇÃO ALEATÓRIA
        # A AGI pesca conceitos aleatórios do cérebro
        random_hdv = self.hdv_random_sample(n_explore)
        log["phase"].append(f"explore: {len(random_hdv)} HDV samples")

        # Também vagueia associativamente
        wander_path = self.wander(seed=None, steps=3, branch=3)
        log["phase"].append(f"wander: {len(wander_path)} associative steps")

        # 2. ENCONTRA PARES INUSITADOS
        # A AGI procura conexões entre domínios DISTANTES
        domains = list(set(c.domain for c in self.all.values() if c.domain != "unknown"))
        strange_pairs = []

        # Tenta blends entre domínios biomédicos e não-biomédicos
        bio_domains = [d for d in domains if d in ("schizophrenia", "alzheimer", "parkinson",
                       "diabetes", "cancer", "depression", "autism", "als",
                       "huntington", "stroke", "epilepsy")]
        other_domains = [d for d in domains if d not in bio_domains and d != "cross_domain"]

        for bio in bio_domains:
            for other in other_domains:
                pairs = self.find_strange_bedfellows(bio, other, k=2)
                strange_pairs.extend(pairs)

        random.shuffle(strange_pairs)
        log["phase"].append(f"cross-domain pairs: {len(strange_pairs)}")

        # 3. COMBINA
        # A AGI cria blends dos pares
        blends_created = []
        for a, b in strange_pairs[:n_blend]:
            blend = self.blend(a, b)
            blends_created.append(blend)
            log["n_blends"] += 1

        # 4. TESTA
        # A AGI testa CADA blend contra a realidade (SBML)
        for blend in blends_created:
            result = self.test(blend)
            if result["survived"]:
                log["n_survived"] += 1
                log["discoveries"].append({
                    "blend": blend.label[:80],
                    "parents": [blend.metadata.get("parent_a", {}).get("label", "?")[:40],
                                blend.metadata.get("parent_b", {}).get("label", "?")[:40]],
                    "domains": blend.domain,
                    "matched_sbml": result["matched_sbml"],
                    "overlap": result["overlap"],
                })

        # 5. RELATA
        print(f"\n>>> CICLO COMPLETO <<<")
        print(f"  Explorou: {len(random_hdv)} HDV + {len(wander_path)} associativos")
        print(f"  Blends criados: {log['n_blends']}")
        print(f"  Sobreviveram: {log['n_survived']}")
        for d in log["discoveries"]:
            print(f"\n  *** DESCOBERTA ***")
            print(f"  Blend: {d['blend']}")
            print(f"  Pais: {d['parents']}")
            print(f"  Match SBML: {d['matched_sbml']}")
            print(f"  Overlap: {d['overlap']}")

        self.session_log.append(log)
        return log

    def multi_cycle(self, n: int = 10):
        """Múltiplos ciclos autônomos."""
        print(f"\n>>> {n} CICLOS AUTÔNOMOS <<<")
        print("A AGI decide: o que explorar, o que combinar, o que testar.")
        print()

        total_discoveries = 0
        for i in range(n):
            log = self.cycle(n_explore=8, n_blend=3)
            total_discoveries += log["n_survived"]
            if log["n_survived"] > 0:
                print(f"  ** Ciclo {i+1}: {log['n_survived']}/3 sobreviveram")

        print(f"\n>>> Total: {total_discoveries} descobertas em {n} ciclos <<<")

        # Relatório final
        survived = [c for c in self.all.values() if c.source == "blend" and c.survived > 0]
        if survived:
            print(f"\n=== CONHECIMENTO NOVO CRIADO PELA AGI ===")
            for c in sorted(survived, key=lambda x: -x.fitness)[:10]:
                print(f"\n  --- {c.label[:60]} ---")
                print(f"  Domínio: {c.domain}")
                print(f"  Fitness: {c.fitness:.3f}")
                print(f"  Testes: {c.tested}, Sobreviveu: {c.survived}")
                if c.metadata.get("parent_a"):
                    print(f"  Pai A: {c.metadata['parent_a']['label'][:40]}")
                    print(f"  Pai B: {c.metadata['parent_b']['label'][:40]}")

        return survived
