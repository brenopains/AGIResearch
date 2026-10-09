"""
CognitiveArchive — Memoria em Camadas L0-L4 com Arqueologia Cognitiva

Camadas:
  L0 — Working: estado atual (volatil, ~30s)
  L1 — Active: usado frequentemente (dias)
  L2 — Dormant: raramente usado (meses)
  L3 — Archive: quase nunca usado (anos)
  L4 — Fossil: nunca mais usado, mas preservado

Cada item tem:
  - utility: utilidade observada
  - generality: quao geral
  - reuse_count: frequencia de uso
  - last_used: timestamp
  - activation_cost: custo para reativar
  - novelty: novidade intrinseca
  - rarity: raridade (itens raros podem ser valiosos)

Arqueologia Cognitiva:
  Periodicamente, a Forja revisita camadas frias (L3, L4)
  e tenta combinar itens para novas descobertas.
"""
import time, hashlib, math, random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class MemoryLayer(Enum):
    L0_WORKING = "L0"
    L1_ACTIVE = "L1"
    L2_DORMANT = "L2"
    L3_ARCHIVE = "L3"
    L4_FOSSIL = "L4"


@dataclass
class ArchiveItem:
    """Um item na memoria em camadas."""
    id: str = ""
    content_type: str = ""  # "algorithm", "strategy", "theory", "heuristic", "stance"
    name: str = ""
    data: Dict = field(default_factory=dict)
    layer: MemoryLayer = MemoryLayer.L1_ACTIVE

    # Metricas
    utility: float = 0.5
    generality: float = 0.3
    reuse_count: int = 0
    last_used: float = field(default_factory=time.time)
    activation_cost: float = 0.1
    novelty: float = 0.3
    rarity: float = 0.5

    created_at: float = field(default_factory=time.time)

    @property
    def archival_value(self) -> float:
        """Valor para arquivamento: utilidade * generalidade * (1 + raridade)."""
        return self.utility * self.generality * (1.0 + self.rarity)


class CognitiveArchive:
    """Memoria em camadas L0-L4 com arqueologia.

    Itens nunca sao apagados — apenas movidos entre camadas
    baseado em frequencia de uso e valor arquivistico.
    """

    def __init__(self, cge=None):
        self.cge = cge
        self.items: Dict[str, ArchiveItem] = {}
        # Indices por camada
        self._layer_index: Dict[MemoryLayer, List[str]] = {
            MemoryLayer.L0_WORKING: [],
            MemoryLayer.L1_ACTIVE: [],
            MemoryLayer.L2_DORMANT: [],
            MemoryLayer.L3_ARCHIVE: [],
            MemoryLayer.L4_FOSSIL: [],
        }

    def store(self, item: ArchiveItem) -> str:
        """Armazena um item (ou atualiza se ja existe)."""
        if item.id in self.items:
            existing = self.items[item.id]
            existing.reuse_count += 1
            existing.last_used = time.time()
            existing.utility = (existing.utility + item.utility) / 2
            self._promote(existing.id)
            return existing.id
        self.items[item.id] = item
        self._layer_index[item.layer].append(item.id)
        return item.id

    def access(self, item_id: str) -> Optional[ArchiveItem]:
        """Acessa um item (atualiza metadados)."""
        item = self.items.get(item_id)
        if item is None:
            return None
        item.reuse_count += 1
        item.last_used = time.time()
        self._promote(item_id)
        return item

    def _promote(self, item_id: str):
        """Promove um item para camada mais quente se estiver sendo usado."""
        item = self.items.get(item_id)
        if item is None:
            return
        now = time.time()
        # Se foi usado recentemente, garantir que esta em L1 no minimo
        if item.layer in (MemoryLayer.L3_ARCHIVE, MemoryLayer.L4_FOSSIL):
            if now - item.last_used < 3600 * 24 * 7:  # usado na ultima semana
                self._move_to(item_id, MemoryLayer.L2_DORMANT)
        if item.layer == MemoryLayer.L2_DORMANT:
            if now - item.last_used < 3600 * 24:  # usado no ultimo dia
                self._move_to(item_id, MemoryLayer.L1_ACTIVE)

    def _move_to(self, item_id: str, target_layer: MemoryLayer):
        """Move um item para uma camada especifica."""
        item = self.items.get(item_id)
        if item is None:
            return
        old_layer = item.layer
        if old_layer == target_layer:
            return
        item.layer = target_layer
        # Atualizar indices
        if old_layer in self._layer_index and item_id in self._layer_index[old_layer]:
            self._layer_index[old_layer].remove(item_id)
        self._layer_index[target_layer].append(item_id)

    def tick_maintenance(self):
        """Ciclo de manutencao: move itens entre camadas baseado em uso."""
        now = time.time()
        for item in list(self.items.values()):
            days_since_use = (now - item.last_used) / (3600 * 24)

            if days_since_use > 365:  # 1 ano
                self._move_to(item.id, MemoryLayer.L4_FOSSIL)
            elif days_since_use > 90:  # 3 meses
                self._move_to(item.id, MemoryLayer.L3_ARCHIVE)
            elif days_since_use > 14:  # 2 semanas
                self._move_to(item.id, MemoryLayer.L2_DORMANT)
            elif days_since_use < 1 and item.reuse_count > 5:  # ativo
                self._move_to(item.id, MemoryLayer.L1_ACTIVE)

    def archaeology(self, n_pairs: int = 3) -> List[Tuple[ArchiveItem, ArchiveItem]]:
        """Arqueologia cognitiva: combina itens de camadas frias.

        Pega pares de L3/L4 e retorna combinacoes promissoras.
        A Forja usa isso para novas descobertas.
        """
        cold_items = [self.items[iid] for iid in
                     self._layer_index[MemoryLayer.L3_ARCHIVE] +
                     self._layer_index[MemoryLayer.L4_FOSSIL]]
        if len(cold_items) < 2:
            return []

        pairs = []
        for _ in range(n_pairs):
            a, b = random.sample(cold_items, 2)
            # So combina se sao de tipos diferentes (ex: algoritmo + teoria)
            if a.content_type != b.content_type:
                pairs.append((a, b))
        return pairs

    def cross_reference(self, query: str, top_k: int = 5) -> List[ArchiveItem]:
        """Busca em TODAS as camadas (incluindo L4 fossil)."""
        q = query.lower()
        results = []
        for item in self.items.values():
            if q in item.name.lower() or q in str(item.data).lower()[:200]:
                results.append(item)
        results.sort(key=lambda i: (
            i.utility * (1.0 + i.rarity * 0.5) *
            (2.0 if i.layer == MemoryLayer.L0_WORKING else
             1.5 if i.layer == MemoryLayer.L1_ACTIVE else 1.0)
        ), reverse=True)
        return results[:top_k]

    def get_layer_stats(self) -> Dict:
        return {layer.value: len(ids) for layer, ids in self._layer_index.items()}

    def get_state(self) -> Dict:
        return {
            "total_items": len(self.items),
            "layers": self.get_layer_stats(),
            "cold_storage": len(self._layer_index[MemoryLayer.L3_ARCHIVE]) +
                           len(self._layer_index[MemoryLayer.L4_FOSSIL]),
            "oldest_item": min((i.last_used for i in self.items.values()), default=time.time()),
        }
