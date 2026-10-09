"""
CollectiveUnconscious — Cultura Compartilhada Multi-Agente
============================================================
Arquivo .hdv compartilhado entre múltiplas AGIs que armazena
apenas Arquétipos Universais, Leis, Axiomas e Cultura.

Cada AGI tem seu hipocampo local (HDV privado).
O Inconsciente Coletivo armazena apenas conceitos validados
por múltiplas AGIs.

Protocolos:
  - Promoção: conceito usado por >= 2 AGIs com FE < threshold → sobe
  - Demotion: conceito com FE consistentemente alta → desce
  - Consulta: busca por similaridade, retorna conceito + metadata
  - Propagação: quando uma AGI descobre algo novo, propaga para as outras

Referência: sent.txt (358-396) — Hipocampo Superior / Inconsciente Coletivo
"""

import os
import json
import time
import shutil
import threading
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class CulturalConcept:
    name: str
    vector: torch.Tensor         # vetor HDC 10000D
    domain: str                  # ex: "physics", "ethics", "mathematics"
    description: str             # descrição textual
    confidence: float            # 0-1, quantas AGIs validaram
    times_validated: int         # número de validações
    discovered_by: str           # AGI que descobriu
    created_at: float
    last_validated_at: float
    fe_history: List[float] = field(default_factory=list)  # FE ao longo do tempo
    archetype_type: str = "concept"  # concept, law, axiom, strategy, ethical_principle


class CollectiveUnconscious:
    """
    Memória cultural compartilhada entre múltiplas AGIs.

    Formato de arquivo: JSON estruturado com vetores HDC em base64.
    (Alternativa: .hdv binário + metadados JSON)

    Args:
        path: caminho para o arquivo collective_unconscious.hdv
        min_validations: mínimo de AGIs para promover conceito
        fe_promotion_threshold: FE máximo para promover
        fe_demotion_threshold: FE mínimo para remover
        device: device torch
    """

    VECTOR_DIM = 10000

    def __init__(
        self,
        path: str = "D:/temp/collective_unconscious.hdv",
        min_validations: int = 2,
        fe_promotion_threshold: float = 0.30,
        fe_demotion_threshold: float = 0.80,
        device: Optional[str] = None,
    ):
        self.path = path
        self.meta_path = os.path.splitext(path)[0] + "_meta.json"
        self.min_validations = min_validations
        self.fe_promotion_threshold = fe_promotion_threshold
        self.fe_demotion_threshold = fe_demotion_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._concepts: Dict[str, CulturalConcept] = {}
        self._lock = threading.RLock()
        self._agentes: Dict[str, str] = {}  # agent_id → ultimo heartbeat

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        self._load()

        # Dimensão atual do vetor (inferida do primeiro conceito carregado)
        self.dimension = self.VECTOR_DIM

        print(f"[COLUNCON] Inconsciente Coletivo: {self.path}")
        print(f"[COLUNCON] Conceitos: {len(self._concepts)}")
        print(f"[COLUNCON] Promoção: FE<{fe_promotion_threshold}, "
              f"{min_validations} validações")

    # ─── Persistência ────────────────────────────────────────────

    def _save(self):
        """Salva no formato .hdv binário + metadados JSON."""
        try:
            # Concatena todos os vetores em um tensor
            if self._concepts:
                names = list(self._concepts.keys())
                vectors = torch.stack([self._concepts[n].vector.cpu() for n in names])
                # Salva .hdv (tensor binário)
                torch.save(vectors, self.path)

                # Salva metadados JSON
                meta = {
                    "version": 2,
                    "dimension": self.dimension,
                    "concepts": [
                        {
                            "name": n,
                            "index": i,
                            "domain": self._concepts[n].domain,
                            "description": self._concepts[n].description,
                            "confidence": self._concepts[n].confidence,
                            "times_validated": self._concepts[n].times_validated,
                            "discovered_by": self._concepts[n].discovered_by,
                            "created_at": self._concepts[n].created_at,
                            "last_validated_at": self._concepts[n].last_validated_at,
                            "fe_history": self._concepts[n].fe_history[-20:],
                            "archetype_type": self._concepts[n].archetype_type,
                        }
                        for i, n in enumerate(names)
                    ],
                    "agentes": self._agentes,
                    "updated_at": time.time(),
                }
                with open(self.meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
            else:
                # Vazio: salva placeholder
                torch.save(torch.zeros(0, self.dimension), self.path)
                with open(self.meta_path, "w", encoding="utf-8") as f:
                    json.dump({"version": 2, "dimension": self.dimension,
                               "concepts": [], "agentes": {},
                               "updated_at": time.time()}, f, indent=2)
        except Exception as e:
            print(f"[COLUNCON] Erro salvando: {e}")

    def _load(self):
        """Carrega do formato .hdv binário + metadados JSON."""
        self._concepts = {}

        if not os.path.exists(self.meta_path):
            return

        try:
            # Carrega metadados
            with open(self.meta_path, "r") as f:
                meta = json.load(f)

            self.dimension = meta.get("dimension", self.VECTOR_DIM)
            self._agentes = meta.get("agentes", {})

            # Carrega vetores
            if os.path.exists(self.path) and meta.get("concepts"):
                vectors = torch.load(self.path, map_location="cpu", weights_only=True)
                if vectors.dim() == 2 and vectors.shape[0] == len(meta["concepts"]):
                    for i, cinfo in enumerate(meta["concepts"]):
                        vec = vectors[i] if i < vectors.shape[0] else torch.zeros(self.dimension)
                        self._concepts[cinfo["name"]] = CulturalConcept(
                            name=cinfo["name"],
                            vector=vec,
                            domain=cinfo.get("domain", "unknown"),
                            description=cinfo.get("description", ""),
                            confidence=cinfo.get("confidence", 0.5),
                            times_validated=cinfo.get("times_validated", 0),
                            discovered_by=cinfo.get("discovered_by", "unknown"),
                            created_at=cinfo.get("created_at", 0),
                            last_validated_at=cinfo.get("last_validated_at", 0),
                            fe_history=cinfo.get("fe_history", []),
                            archetype_type=cinfo.get("archetype_type", "concept"),
                        )
        except Exception as e:
            print(f"[COLUNCON] Erro carregando: {e}")

    # ─── API de Conceitos ────────────────────────────────────────

    def register_agent(self, agent_id: str, metadata: Optional[dict] = None):
        """Registra um agente no coletivo."""
        with self._lock:
            self._agentes[agent_id] = json.dumps({
                "last_seen": time.time(),
                "metadata": metadata or {},
            })
            self._save()

    def get_agents(self) -> List[str]:
        with self._lock:
            return list(self._agentes.keys())

    def add_concept(
        self,
        name: str,
        vector: torch.Tensor,
        domain: str = "general",
        description: str = "",
        discovered_by: str = "unknown",
        archetype_type: str = "concept",
    ) -> bool:
        """
        Adiciona um novo conceito ao inconsciente coletivo.

        Se o conceito já existe, atualiza com merge.
        Se é novo, adiciona com confiança inicial baixa.
        """
        with self._lock:
            now = time.time()

            if name in self._concepts:
                # Merge: média dos vetores
                existing = self._concepts[name]
                old_vec = existing.vector.to(self.device)
                new_vec = vector.to(self.device)
                merged = torch.sign(old_vec + new_vec)
                existing.vector = merged.cpu()
                existing.times_validated += 1
                existing.last_validated_at = now
                existing.confidence = min(1.0, existing.confidence + 0.1)

                # Mede FE do merge
                fe = self._measure_fe(merged, old_vec, new_vec)
                existing.fe_history.append(fe)
                print(f"[COLUNCON] Conceito atualizado: {name} "
                      f"(validações={existing.times_validated}, FE={fe:.2f})")
            else:
                # Novo conceito
                self._concepts[name] = CulturalConcept(
                    name=name,
                    vector=vector.cpu(),
                    domain=domain,
                    description=description,
                    confidence=0.3,  # começa baixo
                    times_validated=1,
                    discovered_by=discovered_by,
                    created_at=now,
                    last_validated_at=now,
                    fe_history=[],
                    archetype_type=archetype_type,
                )
                print(f"[COLUNCON] Novo conceito: {name} (domain={domain}, "
                      f"by={discovered_by})")

            self._save()
            return True

    def validate_concept(self, name: str, vector: torch.Tensor) -> bool:
        """
        Valida um conceito existente com novo vetor.
        Incrementa contagem de validação.
        Se atingir min_validations, promove (aumenta confiança).
        """
        with self._lock:
            if name not in self._concepts:
                return False

            concept = self._concepts[name]
            old_vec = concept.vector.to(self.device)
            new_vec = vector.to(self.device)
            fe = self._measure_fe(new_vec, old_vec, old_vec)

            if fe < self.fe_promotion_threshold:
                # Validação bem sucedida
                concept.times_validated += 1
                concept.last_validated_at = time.time()
                concept.fe_history.append(fe)
                # Merge do vetor
                merged = torch.sign(old_vec + new_vec)
                concept.vector = merged.cpu()

                if concept.times_validated >= self.min_validations:
                    concept.confidence = min(1.0, concept.confidence + 0.15)
                    print(f"[COLUNCON] Conceito PROMOVIDO: {name} "
                          f"(confidence={concept.confidence:.2f})")
                else:
                    print(f"[COLUNCON] Conceito validado: {name} "
                          f"({concept.times_validated}/{self.min_validations})")

                self._save()
                return True
            else:
                print(f"[COLUNCON] Validação falhou: {name} (FE={fe:.2f})")
                return False

    def demote_concept(self, name: str, reason: str = "") -> bool:
        """Remove conceito do inconsciente coletivo (FE consistentemente alta)."""
        with self._lock:
            if name not in self._concepts:
                return False

            concept = self._concepts[name]
            print(f"[COLUNCON] Conceito REMOVIDO: {name} "
                  f"(confidence={concept.confidence:.2f}, reason={reason})")
            del self._concepts[name]
            self._save()
            return True

    def query(
        self,
        query_vec: torch.Tensor,
        top_k: int = 10,
        domain: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Tuple[str, CulturalConcept, float]]:
        """
        Busca conceitos por similaridade de cosseno.

        Args:
            query_vec: vetor de consulta (10000D)
            top_k: quantos resultados
            domain: filtrar por domínio (opcional)
            min_confidence: confiança mínima

        Returns:
            lista de (name, concept, similarity)
        """
        with self._lock:
            if not self._concepts:
                return []

            q = query_vec.flatten().float().to(self.device)
            results = []

            for name, concept in self._concepts.items():
                if domain and concept.domain != domain:
                    continue
                if concept.confidence < min_confidence:
                    continue

                v = concept.vector.float().to(self.device)
                cos = float(F.cosine_similarity(q.unsqueeze(0), v.unsqueeze(0)))
                results.append((name, concept, cos))

            results.sort(key=lambda x: -x[2])
            return results[:top_k]

    def get_concept(self, name: str) -> Optional[CulturalConcept]:
        with self._lock:
            return self._concepts.get(name)

    def get_all_concepts(self) -> List[CulturalConcept]:
        with self._lock:
            return list(self._concepts.values())

    def count(self) -> int:
        with self._lock:
            return len(self._concepts)

    # ─── Protocolo de Propagação ────────────────────────────────

    def propagate_to_agent(self, concept_name: str, agent_id: str) -> Optional[torch.Tensor]:
        """
        Propaga um conceito do coletivo para um agente específico.
        Retorna o vetor HDC do conceito.
        """
        with self._lock:
            concept = self._concepts.get(concept_name)
            if not concept:
                return None
            print(f"[COLUNCON] Propagando {concept_name} → {agent_id}")
            return concept.vector.clone()

    def share_discovery(
        self,
        agent_id: str,
        concept_name: str,
        concept_vector: torch.Tensor,
        domain: str,
        description: str,
    ):
        """
        Uma AGI compartilha uma descoberta com o coletivo.
        Outras AGIs podem então puxar e validar.
        """
        print(f"[COLUNCON] {agent_id} compartilhou: {concept_name} ({domain})")
        self.add_concept(concept_name, concept_vector, domain,
                         description, agent_id)

    def get_pending_validations(self, agent_id: str) -> List[Tuple[str, CulturalConcept]]:
        """
        Retorna conceitos que precisam ser validados por este agente.
        (Conceitos descobertos por OUTROS agentes com baixa confiança)
        """
        pending = []
        with self._lock:
            for name, concept in self._concepts.items():
                if (concept.discovered_by != agent_id
                        and concept.times_validated < self.min_validations):
                    pending.append((name, concept))
        return pending

    # ─── Utilitários ─────────────────────────────────────────────

    def _measure_fe(self, merged: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """FE = 1 - média das similaridades entre merged e originais."""
        m = merged.flatten().float()
        a = v1.flatten().float()
        b = v2.flatten().float()

        cos1 = float(F.cosine_similarity(m.unsqueeze(0), a.unsqueeze(0)))
        cos2 = float(F.cosine_similarity(m.unsqueeze(0), b.unsqueeze(0)))
        return 1.0 - (cos1 + cos2) / 2.0

    def get_summary(self) -> dict:
        with self._lock:
            domains = {}
            for c in self._concepts.values():
                domains[c.domain] = domains.get(c.domain, 0) + 1

            return {
                "path": self.path,
                "total_concepts": len(self._concepts),
                "domains": domains,
                "agents": len(self._agentes),
                "min_validations": self.min_validations,
                "fe_promotion_threshold": self.fe_promotion_threshold,
                "dimension": self.dimension,
            }

    def get_archetypes(self, archetype_type: str = "axiom") -> List[CulturalConcept]:
        """Retorna conceitos de um tipo específico (axiom, law, strategy, etc)."""
        with self._lock:
            return [c for c in self._concepts.values()
                    if c.archetype_type == archetype_type]
