"""
Invariant Discovery — Encontra padroes que PERMANECEM IGUAIS entre observacoes.

Nao pergunta "qual primitiva resolve?".
Pergunta "o que permaneceu igual?".

Exemplo:
  - bola bate na parede e volta
  - som ecoa e volta
  - luz reflete e volta
  → Invariante: "trajetoria inverte direcao apos colisao"

Cada invariante e um candidato a conceito.
"""
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class Invariant:
    """Um padrao que permanece igual atraves de observacoes.
    
    Attributes:
        structure_type: "reflection", "branching", "expansion", "contraction", etc
        domains: em quais dominios o padrao foi observado
        signature: assinatura estrutural do padrao (dict de metricas)
        n_observations: quantas observacoes confirmam este invariante
        confidence: quao confiavel e este invariante
    """
    structure_type: str
    domains: Set[str] = field(default_factory=set)
    signature: Dict[str, float] = field(default_factory=dict)
    n_observations: int = 0
    confidence: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "type": self.structure_type,
            "domains": list(self.domains),
            "n_observations": self.n_observations,
            "confidence": round(self.confidence, 3),
        }


class InvariantDiscovery:
    """Encontra invariantes atraves de observacoes.
    
    Uso:
        idisc = InvariantDiscovery()
        idisc.observe({"action": "throw", "result": "return", "domain": "physics"})
        idisc.observe({"action": "echo", "result": "return", "domain": "audio"})
        
        invariants = idisc.get_invariants(min_confidence=0.3)
    """
    
    def __init__(self):
        self._invariants: Dict[str, Invariant] = {}
        self._observations: List[dict] = []
    
    def observe(self, observation: dict):
        """Observa um evento e atualiza invariantes.
        
        Aceita dicts com 'type', 'domain', 'content', 'program', etc.
        """
        # Normaliza: se vier com "content" string, extrai tipo do dominio
        if "content" in observation and "type" not in observation:
            observation["type"] = "text_observation"
        if "domain" not in observation:
            observation["domain"] = "general"
        
        self._observations.append(observation)
        self._update_invariants(observation)
    
    def observe_program_result(self, program_name: str, input_shape: tuple,
                                output_shape: tuple, success: bool,
                                domain: str = "program"):
        """Observa o resultado da execucao de um programa."""
        obs = {
            "type": "program_result",
            "program": program_name,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "success": success,
            "domain": domain,
        }
        self.observe(obs)
    
    def observe_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray,
                                program_name: str, domain: str = "arc"):
        """Observa uma transformacao de grid."""
        in_objs = int((input_grid > 0).sum())
        out_objs = int((output_grid > 0).sum())
        in_shape = input_grid.shape
        out_shape = output_grid.shape
        
        obs = {
            "type": "transformation",
            "program": program_name,
            "in_shape": in_shape,
            "out_shape": out_shape,
            "in_objs": in_objs,
            "out_objs": out_objs,
            "shape_changed": in_shape != out_shape,
            "object_count_changed": in_objs != out_objs,
            "domain": domain,
        }
        self.observe(obs)
    
    def _update_invariants(self, obs: dict):
        """Atualiza invariantes baseado em uma nova observacao."""
        domain = obs.get("domain", "unknown")
        
        # --- INVARIANTE: BRANCHING (ramificacao) ---
        content = str(obs.get("content", ""))
        obs_type = obs.get("type", "")
        
        # Detecta branching de observacoes textuais
        branching_keywords = ["branch", "split", "ramifica", "bifurca", "fork",
                            "divide", "ramify", "dendrite", "tributary", "delta",
                            "galho", "ramo", "bifurcation", "bifurcates"]
        if any(kw in content.lower() for kw in branching_keywords):
            self._update_invariant("branching", domain, {
                "pattern": "structural_division",
                "detected_in": "text",
            })
        
        # Detecta branching de eventos explicitos
        if "branch" in obs_type or "split" in obs_type or "ramif" in obs_type:
            level = obs.get("level", 0)
            self._update_invariant("branching", domain, {
                "pattern": "explicit_event",
                "depth": level,
            })
        
        # Detecta branching de programas (tile, copy)
        if obs.get("type") == "program_result":
            program = obs.get("program", "")
            if "tile" in program or "copy" in program or "repeat" in program:
                self._update_invariant("branching", domain, {
                    "pattern": "programmatic_replication",
                    "program": program,
                })
        
        # --- INVARIANTE: REFLECTION (trajetoria inverte) ---
        if obs.get("type") == "program_result":
            program = obs.get("program", "")
            input_shape = obs.get("input_shape", (0, 0))
            output_shape = obs.get("output_shape", (0, 0))
            
            # Rotacao/flip sao exemplos de reflection
            if "rotate" in program or "flip" in program:
                self._update_invariant("reflection", domain, {
                    "preserves_area": input_shape[0] * input_shape[1] == 
                                     output_shape[0] * output_shape[1]
                    if len(input_shape) == 2 and len(output_shape) == 2 else True,
                    "changes_orientation": True,
                })
                self._update_invariant("spatial_transformation", domain, {})
        
        # --- DETECCAO GEOMETRICA (sem depender de nome de programa) ---
        if "in_shape" in obs and "out_shape" in obs:
            in_sh = obs["in_shape"]
            out_sh = obs["out_shape"]
            if len(in_sh) == 2 and len(out_sh) == 2:
                # Reflection: shape muda mas area preserva
                if in_sh != out_sh:
                    in_area = in_sh[0] * in_sh[1]
                    out_area = out_sh[0] * out_sh[1]
                    if abs(in_area - out_area) <= max(in_sh):
                        # Provavelmente rotacao ou transpose (area preserva)
                        if (in_sh[0] == out_sh[1] and in_sh[1] == out_sh[0]):
                            self._update_invariant("reflection", domain, {
                                "detected_by": "geometry",
                                "type": "rotation_or_transpose",
                            })
                            self._update_invariant("spatial_transformation", domain, {})
        
        # --- INVARIANTE: OBJECT_COUNT_CHANGE ---
        if obs.get("object_count_changed"):
            if obs.get("out_objs", 0) > obs.get("in_objs", 0):
                self._update_invariant("expansion", domain, {
                    "object_ratio": obs["out_objs"] / max(1, obs["in_objs"]),
                })
            else:
                self._update_invariant("contraction", domain, {
                    "object_ratio": obs["in_objs"] / max(1, obs["out_objs"]),
                })
        
        # --- INVARIANTE: SHAPE_CHANGE ---
        if obs.get("shape_changed"):
            in_shape = obs.get("in_shape", (0, 0))
            out_shape = obs.get("out_shape", (0, 0))
            if len(in_shape) >= 2 and len(out_shape) >= 2:
                h_ratio = out_shape[0] / max(1, in_shape[0])
                w_ratio = out_shape[1] / max(1, in_shape[1])
                if h_ratio > 1.2 or w_ratio > 1.2:
                    self._update_invariant("expansion", domain, {
                        "shape_h_ratio": h_ratio,
                        "shape_w_ratio": w_ratio,
                    })
                elif h_ratio < 0.8 or w_ratio < 0.8:
                    self._update_invariant("contraction", domain, {
                        "shape_h_ratio": h_ratio,
                        "shape_w_ratio": w_ratio,
                    })
        
        # --- INVARIANTE: CAUSAL_CHAIN (A -> B -> C) ---
        if obs.get("type") in ("program_result", "transformation"):
            # Toda transformacao e uma cadeia causal: input -> programa -> output
            self._update_invariant("causal_chain", domain, {
                "has_precondition": True,
                "has_action": True,
                "has_result": True,
            })
        
        # --- INVARIANTE: SYMMETRY (preservacao de estrutura) ---
        if obs.get("type") == "transformation":
            if not obs.get("shape_changed") and not obs.get("object_count_changed"):
                self._update_invariant("symmetry", domain, {
                    "preserves_shape": True,
                    "preserves_count": True,
                })
    
    def _update_invariant(self, inv_type: str, domain: str, signature: dict):
        """Cria ou reforca um invariante."""
        if inv_type not in self._invariants:
            self._invariants[inv_type] = Invariant(
                structure_type=inv_type,
                signature=signature,
            )
        
        inv = self._invariants[inv_type]
        inv.domains.add(domain)
        inv.n_observations += 1
        inv.confidence = min(1.0, inv.n_observations * 0.02)
        
        # Atualiza assinatura com media (so para valores numericos)
        for k, v in signature.items():
            if isinstance(v, (int, float)):
                if k in inv.signature and isinstance(inv.signature[k], (int, float)):
                    inv.signature[k] = (inv.signature[k] * (inv.n_observations - 1) + v) / inv.n_observations
                else:
                    inv.signature[k] = v
            else:
                # Para valores nao numericos (strings, etc), substitui
                inv.signature[k] = v
    
    def get_invariants(self, min_confidence: float = 0.0,
                        min_domains: int = 1) -> List[Invariant]:
        """Retorna invariantes com filtros."""
        return [i for i in self._invariants.values()
                if i.confidence >= min_confidence and len(i.domains) >= min_domains]
    
    def get_by_type(self, inv_type: str) -> Optional[Invariant]:
        return self._invariants.get(inv_type)
    
    def get_cross_domain_invariants(self, min_domains: int = 2) -> List[Invariant]:
        """Invariantes que aparecem em MULTIPLOS dominios."""
        return self.get_invariants(min_domains=min_domains)
    
    def get_stats(self) -> dict:
        return {
            "total_invariants": len(self._invariants),
            "cross_domain": len(self.get_cross_domain_invariants()),
            "total_observations": len(self._observations),
            "invariants": {t: {"domains": list(i.domains), "confidence": round(i.confidence, 2)}
                          for t, i in self._invariants.items()},
        }
