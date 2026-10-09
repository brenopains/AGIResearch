"""
LifeStream — Pipeline de Aprendizado Continuo Multimodal

Fluxo completo:

Fonte (Wikipedia, YouTube, procedural, texto)
  │
  ├── Source Reliability Filter
  │     └── confianca por fonte (0.98 Wikipedia, 0.15 comentario, etc)
  │
  ├── Opportunity Engine
  │     └── expected_gain = knowledge × skill × compression × curiosity × world_model
  │
  └── Episode Stream
        └── frame → evento → HDV → CGE → descarta frame bruto

Integra com SelfModel (curiosidade → gaps) e SimulationEngine (branches → oportunidade).
"""
import time
import math
import random
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


# ─── Source Reliability ─────────────────────────────────────

@dataclass
class SourceProfile:
    """Perfil de confiabilidade de uma fonte de conhecimento."""
    name: str
    base_reliability: float = 0.5  # 0..1
    domain: str = "general"
    evidence_count: int = 0
    contradiction_count: int = 0

    @property
    def reliability(self) -> float:
        """Confiabilidade ajustada por evidencias e contradicoes."""
        r = self.base_reliability
        if self.evidence_count + self.contradiction_count > 0:
            ratio = self.evidence_count / max(1, self.evidence_count + self.contradiction_count)
            r = r * 0.7 + ratio * 0.3
        return max(0.0, min(1.0, r))


SOURCE_RELIABILITY_DEFAULTS = {
    "wikipedia": SourceProfile("wikipedia", 0.95, "general"),
    "arxiv": SourceProfile("arxiv", 0.98, "science"),
    "mit_ocw": SourceProfile("mit_ocw", 0.97, "education"),
    "nature": SourceProfile("nature", 0.98, "science"),
    "youtube_educational": SourceProfile("youtube_educational", 0.85, "general"),
    "youtube_random": SourceProfile("youtube_random", 0.40, "general"),
    "comment": SourceProfile("comment", 0.15, "opinion"),
    "advertisement": SourceProfile("advertisement", 0.05, "marketing"),
    "procedural": SourceProfile("procedural", 0.90, "simulation"),
    "self_generated": SourceProfile("self_generated", 0.70, "internal"),
}


# ─── Content Item ───────────────────────────────────────────

@dataclass
class ContentItem:
    """Um item de conteudo a ser processado."""
    id: str
    source: str
    source_reliability: float = 0.5
    modality: str = "text"  # text, video, audio, interactive
    title: str = ""
    body: str = ""  # texto extraido
    concepts: List[str] = field(default_factory=list)
    domain: str = "general"
    estimated_duration: float = 60.0
    estimated_value: float = 0.5
    is_ad: bool = False
    is_clickbait: bool = False

    @property
    def effective_value(self) -> float:
        """Valor ajustado por confiabilidade da fonte."""
        return self.estimated_value * self.source_reliability


# ─── Opportunity Engine ─────────────────────────────────────

@dataclass
class Opportunity:
    """Uma oportunidade de aprendizado."""
    content_id: str
    title: str = ""
    knowledge_gain: float = 0.0  # quanto novo conhecimento
    skill_gain: float = 0.0     # quanto melhora habilidades
    compression_gain: float = 0.0  # quanto reorganiza conhecimento
    curiosity_gain: float = 0.0  # quanto sacia curiosidade
    world_model_gain: float = 0.0  # quanto melhora modelo do mundo
    estimated_duration: float = 60.0
    cost: float = 0.0

    @property
    def expected_gain(self) -> float:
        """Ganho esperado multidimensional."""
        return (self.knowledge_gain * 0.25 + self.skill_gain * 0.20 +
                self.compression_gain * 0.15 + self.curiosity_gain * 0.25 +
                self.world_model_gain * 0.15)


class OpportunityEngine:
    """Mede Expected Gain multidimensional para cada conteudo disponivel."""

    def __init__(self, self_model=None, simulation_engine=None):
        self.self_model = self_model
        self.simulation_engine = simulation_engine
        self.opportunities: List[Opportunity] = []

    def evaluate(self, content: ContentItem,
                 cognitive_state: Optional[Dict] = None) -> Opportunity:
        """Avalia o ganho esperado de um conteudo."""
        if cognitive_state is None:
            cognitive_state = {}

        # Knowledge gain: quantos conceitos novos vs ja conhecidos
        known = 0
        unknown = 0
        if self.self_model:
            for c in content.concepts:
                if c in self.self_model.identity.beliefs:
                    known += 1
                else:
                    unknown += 1
        total = known + unknown
        knowledge_gain = unknown / max(1, total) if total > 0 else 0.3

        # Curiosity gain: quantas lacunas de conhecimento este conteudo preenche
        curiosity_gain = 0.3
        if self.self_model:
            gaps = self.self_model.get_curiosity_targets(5)
            matching_gaps = sum(1 for c in content.concepts if c in gaps)
            curiosity_gain = min(1.0, matching_gaps * 0.3 + 0.2)

        # Compression gain: quantas crencas existentes seriam reorganizadas
        compression_gain = 0.2 * unknown / max(1, total) if total > 0 else 0.2

        # Skill gain: relevancia para dominios de interesse
        skill_gain = 0.3
        if self.self_model:
            weaknesses = self.self_model.identity.capabilities.get_weaknesses()
            if content.domain in weaknesses:
                skill_gain = 0.8
            strengths = self.self_model.identity.capabilities.get_strengths()
            if content.domain in strengths:
                skill_gain = 0.4  # menos ganho se ja e forte

        # World model gain: quantas simulacoes existentes seriam afetadas
        world_model_gain = 0.2
        if self.simulation_engine and self.simulation_engine.branches:
            n = len(self.simulation_engine.branches)
            world_model_gain = min(0.8, unknown * 0.15 + 0.1)

        # Custo: tempo * (1 - prioridade)
        cost = content.estimated_duration / 3600.0  # normalizar para horas

        opp = Opportunity(
            content_id=content.id,
            title=content.title,
            knowledge_gain=knowledge_gain,
            skill_gain=skill_gain,
            compression_gain=compression_gain,
            curiosity_gain=curiosity_gain,
            world_model_gain=world_model_gain,
            estimated_duration=content.estimated_duration,
            cost=cost,
        )
        return opp

    def rank(self, contents: List[ContentItem],
             n: int = 5) -> List[Opportunity]:
        """Ranqueia oportunidades por expected_gain / cost."""
        opportunities = []
        for c in contents:
            opp = self.evaluate(c)
            opportunities.append(opp)

        # Ordenar por expected_gain / max(0.01, cost)
        opportunities.sort(key=lambda o: o.expected_gain / max(0.01, o.cost), reverse=True)
        self.opportunities = opportunities[:n]
        return self.opportunities


# ─── Episode ────────────────────────────────────────────────

@dataclass
class Episode:
    """Uma experiencia completa — unidade de aprendizado."""
    id: str
    source: str = ""
    modality: str = "text"
    concepts: List[str] = field(default_factory=list)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)
    # (conceito_a, relacao, conceito_b)
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.3
    reliability: float = 0.5
    duration_seconds: float = 0.0
    raw_content_discarded: bool = True  # sempre True — nao guardamos raw
    created_at: float = field(default_factory=time.time)


class EpisodeStream:
    """Fluxo continuo de episodios — converte conteudo bruto em experiencias.

    O conteudo bruto (texto, frames, audio) e processado e descartado.
    So sobram conceitos, relacoes, hipoteses no CGE.
    """

    def __init__(self, cge=None, bge_encoder=None):
        self.cge = cge
        self.bge_encoder = bge_encoder
        self.episodes: List[Episode] = []
        self.processed_count: int = 0

    def process_text(self, content: ContentItem) -> Episode:
        """Processa texto → episodio."""
        # Extrair conceitos do texto (simples: dividir por espacos)
        words = content.body.lower().split()[:200]
        concepts = list(set(w for w in words if len(w) > 3))[:20]

        # Extrair relacoes simples (co-ocorrencia)
        relations = []
        for i in range(len(concepts) - 1):
            relations.append((concepts[i], "co_occur", concepts[i + 1]))

        episode = Episode(
            id=f"ep_{content.id}_{int(time.time())}",
            source=content.source,
            modality=content.modality,
            concepts=concepts,
            relations=relations,
            reliability=content.source_reliability,
            duration_seconds=content.estimated_duration,
        )
        self.episodes.append(episode)
        self.processed_count += 1
        return episode

    def feed_to_cge(self, episode: Episode):
        """Alimenta o CGE com o episodio — o conteudo bruto e descartado."""
        if self.cge is None:
            return

        # Criar conceitos no CGE
        for concept in episode.concepts[:10]:  # limitar a 10 por vez
            if concept and concept not in self.cge.nodes:
                if self.bge_encoder:
                    try:
                        vec = self.bge_encoder.encode_bipolar([concept]).squeeze(0)
                    except Exception:
                        import torch
                        import hashlib
                        h = hashlib.md5(concept.encode()).hexdigest()
                        seed = int(h[:8], 16)
                        rng = torch.Generator()
                        rng.manual_seed(seed)
                        vec = torch.sign(torch.randn(10000, generator=rng))
                else:
                    import torch
                    import hashlib
                    h = hashlib.md5(concept.encode()).hexdigest()
                    seed = int(h[:8], 16)
                    rng = torch.Generator()
                    rng.manual_seed(seed)
                    vec = torch.sign(torch.randn(10000, generator=rng))

                self.cge.create_node(concept, vec)
                # Definir emocao depois da criacao
                if concept in self.cge.nodes:
                    self.cge.nodes[concept]["emotional_valence"] = episode.emotional_valence
                    self.cge.nodes[concept]["emotional_arousal"] = episode.emotional_arousal

        # Criar arestas entre conceitos co-ocorrentes
        for a, rel, b in episode.relations[:20]:
            if a in self.cge.nodes and b in self.cge.nodes:
                self.cge.add_edge(a, b, self.cge.EDGE_CAUSES)

        # O episodio pode ser descartado depois — o conhecimento ficou no CGE
        return episode.concepts[:10]

    def get_state(self) -> Dict:
        return {
            "total_episodes": len(self.episodes),
            "processed": self.processed_count,
            "last_episode": self.episodes[-1].concepts[:5] if self.episodes else [],
        }


# ─── LifeStream Scheduler ───────────────────────────────────

class ContentSource:
    """Fonte de conteudo — gera ContentItems sob demanda."""
    pass


class WikipediaSource(ContentSource):
    """Fonte Wikipedia — busca artigos por tema."""
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.topics: List[str] = []

    def fetch(self, topic: str) -> Optional[ContentItem]:
        """Busca artigo da Wikipedia."""
        try:
            import urllib.request
            import json
            url = f"https://{self.lang}.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
            req = urllib.request.Request(url, headers={"User-Agent": "NovaMind/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if "extract" in data:
                    return ContentItem(
                        id=f"wiki_{topic}_{int(time.time())}",
                        source="wikipedia",
                        source_reliability=SOURCE_RELIABILITY_DEFAULTS["wikipedia"].reliability,
                        modality="text",
                        title=data.get("title", topic),
                        body=data.get("extract", ""),
                        concepts=[topic] + data.get("extract", "").split()[:5],
                        estimated_value=0.8,
                    )
        except Exception as e:
            pass
        return None


class ProceduralSource(ContentSource):
    """Fonte procedural — gera experiencias sobre fisica/materiais."""
    def __init__(self, concepts_pool: List[str] = None):
        self.concepts_pool = concepts_pool or [
            "gravity", "energy", "conductivity", "force", "mass",
            "velocity", "acceleration", "friction", "momentum", "inertia",
        ]

    def generate(self, topic: str) -> ContentItem:
        """Gera um item procedural sobre um tema."""
        concepts = [topic] + random.sample(self.concepts_pool, 3)
        body = f"Exploring {topic} and its relationship to {', '.join(concepts[1:])}."
        return ContentItem(
            id=f"proc_{topic}_{int(time.time())}",
            source="procedural",
            source_reliability=SOURCE_RELIABILITY_DEFAULTS["procedural"].reliability,
            modality="text",
            title=f"Exploracao procedural: {topic}",
            body=body,
            concepts=concepts,
            domain="physics",
            estimated_value=0.7,
        )


class LifeStream:
    """Orquestrador de aprendizado continuo.

    Ciclo:
      1. Opportunity Engine: avalia Expected Gain de cada conteudo disponivel
      2. Source Filter: ajusta por confiabilidade
      3. Episode Stream: processa conteudo → episodio → CGE
      4. SelfModel: atualiza crencas, capacidades, perfil
      5. Descarta raw content
    """

    def __init__(self, cge=None, bge_encoder=None, self_model=None,
                 simulation_engine=None):
        self.cge = cge
        self.bge_encoder = bge_encoder
        self.self_model = self_model
        self.sim_engine = simulation_engine

        # Subsistemas
        self.opportunity = OpportunityEngine(self_model, simulation_engine)
        self.episode_stream = EpisodeStream(cge, bge_encoder)
        self.sources: Dict[str, ContentSource] = {}
        self.source_reliability = dict(SOURCE_RELIABILITY_DEFAULTS)

        # Conteudo disponivel e processado
        self.content_queue: List[ContentItem] = []
        self.processed_count: int = 0
        self.total_opportunities: int = 0

    def add_source(self, name: str, source: ContentSource):
        """Registra uma fonte de conteudo."""
        self.sources[name] = source

    def add_content(self, content: ContentItem):
        """Adiciona conteudo a fila de processamento."""
        self.content_queue.append(content)

    def fetch_content(self, topic: str, source_name: str = "wikipedia") -> bool:
        """Busca conteudo de uma fonte registrada."""
        source = self.sources.get(source_name)
        if source:
            content = source.fetch(topic)
            if content:
                self.add_content(content)
                return True
        return False

    def generate_content(self, topic: str, source_name: str = "procedural") -> bool:
        """Gera conteudo procedural."""
        source = self.sources.get(source_name)
        if source and hasattr(source, 'generate'):
            content = source.generate(topic)
            self.add_content(content)
            return True
        return False

    def tick(self, max_items: int = 1) -> List[str]:
        """Um ciclo do LifeStream: avaliar, selecionar, processar.

        Returns:
            Lista de conceitos aprendidos neste ciclo.
        """
        if not self.content_queue:
            return []

        # 1. Opportunity Engine: avaliar Expected Gain
        opportunities = self.opportunity.rank(self.content_queue, n=max_items)

        # 2. Selecionar melhor oportunidade
        learned_concepts = []
        processed = 0
        for opp in opportunities[:max_items]:
            content = next((c for c in self.content_queue
                           if c.id == opp.content_id), None)
            if content is None:
                continue

            # 3. Processar → episodio
            episode = self.episode_stream.process_text(content)

            # 4. Alimentar CGE
            concepts = self.episode_stream.feed_to_cge(episode)

            # 5. Atualizar SelfModel
            if self.self_model:
                for c in episode.concepts[:5]:
                    self.self_model.learn_from_experience(
                        concept=c,
                        evidence=content.body[:100],
                        reliability=content.source_reliability,
                        domain=content.domain,
                        modality=content.modality,
                        success=True,
                    )

            # Remover da fila (conteudo bruto descartado)
            self.content_queue = [c for c in self.content_queue
                                  if c.id != content.id]
            self.processed_count += 1
            learned_concepts.extend(episode.concepts[:5])
            processed += 1

        self.total_opportunities += processed
        return learned_concepts

    def get_curriculum_plan(self, topics: List[str]) -> List[str]:
        """Gera um plano de estudos baseado em topicos."""
        plan = []
        for topic in topics:
            # Buscar e processar
            if self.fetch_content(topic):
                plan.append(topic)
        return plan

    def get_state(self) -> Dict:
        return {
            "queue_size": len(self.content_queue),
            "processed": self.processed_count,
            "opportunities": self.total_opportunities,
            "sources": list(self.sources.keys()),
            "episodes": self.episode_stream.get_state(),
            "recent_reliability": {
                k: round(v.reliability, 2)
                for k, v in self.source_reliability.items()
            },
        }
