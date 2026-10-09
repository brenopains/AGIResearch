"""
SelfDescription — Modelo Explicito de Si Mesma
=================================================
A arquitetura gera um relato estruturado de:
  - Modulos que possui (tipo, nome, estado)
  - Conexoes entre modulos (fluxo de dados)
  - Capacidades que cada modulo oferece
  - Gaps detectados por falhas recentes

Isto e o "metamodelo" — a representacao que o sistema
usa para decidir o que modificar em si mesma.
"""
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ModuleDescriptor:
    """Descricao de um modulo da arquitetura."""
    name: str = ""
    module_type: str = ""
    status: str = "active"  # active, idle, failed
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)
    last_error_rate: float = 0.0
    n_calls: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name, "type": self.module_type,
            "status": self.status, "capabilities": self.capabilities,
            "dependencies": self.dependencies, "connections": self.connections,
            "error_rate": round(self.last_error_rate, 3),
        }


@dataclass
class ArchitectureSnapshot:
    """Instantaneo completo da arquitetura em um momento."""
    timestamp: float = 0.0
    modules: Dict[str, ModuleDescriptor] = field(default_factory=dict)
    connections: List[Tuple[str, str, str]] = field(default_factory=list)
    gaps: List[dict] = field(default_factory=list)
    trial_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "n_modules": len(self.modules),
            "n_connections": len(self.connections),
            "n_gaps": len(self.gaps),
            "trial_scores": self.trial_scores,
            "modules": {k: v.to_dict() for k, v in self.modules.items()},
        }


class SelfModel:
    """
    Modelo de si mesma: a arquitetura se descreve.
    
    Mantem um snapshot atualizado do que existe,
    detecta gaps comparando capacidades com demandas,
    e alimenta o loop evolucionario.
    """

    def __init__(self, structural_plasticity=None, cge=None,
                 world_model=None, language_engine=None,
                 journal=None, calibration_engine=None):
        self.sp = structural_plasticity
        self.cge = cge
        self.world_model = world_model
        self.language = language_engine
        self.journal = journal
        self.calibration_engine = calibration_engine

        self._snapshots: List[ArchitectureSnapshot] = []
        self._gap_history: List[dict] = []

    def snapshot(self, trial_scores: Dict[str, float] = None) -> ArchitectureSnapshot:
        """Tira um instantaneo da arquitetura."""
        snap = ArchitectureSnapshot(timestamp=time.time())
        snap.trial_scores = trial_scores or {}

        # Modulos
        if self.sp is not None and hasattr(self.sp, 'arch'):
            for nid, node in self.sp.arch.nodes.items():
                caps = []
                if node.module_type == "cge":
                    caps = ["concept_storage", "relation_traversal", "spread_activation"]
                elif node.module_type == "world_model":
                    caps = ["prediction", "intervention", "counterfactual"]
                elif node.module_type == "language":
                    caps = ["text_generation", "internal_monologue"]
                elif node.module_type == "calibration":
                    caps = ["confidence_scaling", "ece_measurement"]
                elif node.module_type == "journal":
                    caps = ["prediction_recording", "outcome_resolution"]
                snap.modules[nid] = ModuleDescriptor(
                    name=node.name, module_type=node.module_type,
                    capabilities=caps,
                    last_error_rate=node.error_rate,
                    n_calls=node.activation_count,
                )

        # Conexoes
        if self.sp is not None and hasattr(self.sp, 'arch'):
            for edge in self.sp.arch.edges:
                snap.connections.append(
                    (edge.from_id, edge.to_id, f"bandwidth={edge.bandwidth:.1f}")
                )

        # Gaps detectados
        snap.gaps = list(self._gap_history[-5:])

        self._snapshots.append(snap)
        return snap

    def detect_gaps(self, trial_scores: Dict[str, float]) -> List[dict]:
        """
        Detecta gaps da arquitetura comparando scores com thresholds.
        
        Returns: lista de gaps com tipo, trial, e causa provavel.
        """
        gaps = []
        thresholds = {"T1": 0.35, "T2": 0.4, "T3": 0.4,
                      "T4": 0.4, "T5": 0.3, "T6": 0.3, "T7": 0.3}

        for trial, score in trial_scores.items():
            thresh = thresholds.get(trial, 0.3)
            if score < thresh:
                gap = self._classify_gap(trial, score, thresh)
                gaps.append(gap)
                self._gap_history.append(gap)

        return gaps

    def _classify_gap(self, trial: str, score: float,
                      threshold: float) -> dict:
        """Classifica a causa provavel de um gap."""
        gap = {
            "trial": trial, "score": score, "threshold": threshold,
            "gap_size": round(threshold - score, 3),
            "timestamp": time.time(),
        }

        if trial == "T3" and score < 0.1:
            gap["type"] = "calibration_missing"
            gap["cause"] = "no_prediction_journal_data_for_domain"
            gap["fix"] = "generate_platt_scaler_per_domain"
            if self.journal is not None:
                stats = self.journal.get_all_stats()
                gap["journal_n"] = stats.get("n_resolved", 0)
                gap["journal_domains"] = stats.get("n_domains", 0)
                if gap["journal_n"] < 50:
                    gap["fix_detail"] = "run_bootcamp_with_diverse_domains"
                else:
                    gap["fix_detail"] = "retrain_calibration_engine"

        elif trial == "T7" and score < 0.1:
            gap["type"] = "missing_primitive"
            gap["cause"] = "value_transform_not_in_search_space"
            gap["fix"] = "induce_value_map_from_examples"
            gap["fix_detail"] = "run_value_map_induction_on_failure_examples"

        elif trial == "T4" and score < 0.2:
            gap["type"] = "poor_function_fitting"
            gap["cause"] = "no_log_linear_in_causal_core"
            if self.world_model is not None and hasattr(self.world_model, 'causal_core'):
                cc = self.world_model.causal_core
                if cc is not None:
                    for var, params in cc.function_params.items():
                        if params.get("type") == "exogenous":
                            gap["cause"] = f"variable_{var}_could_not_be_fit"
                            break
            gap["fix"] = "add_function_family_selection"
            gap["fix_detail"] = "enable_log_linear_and_polynomial_in_SCM"

        elif trial == "T1" and score < 0.3:
            gap["type"] = "confounding_not_detected"
            gap["cause"] = "causal_core_not_used"
            gap["fix"] = "route_queries_through_causal_core"
            gap["fix_detail"] = "connect_world_model.predict_to_causal_core"

        elif trial == "T6" and score < 0.2:
            gap["type"] = "poor_experiment_design"
            gap["cause"] = "no_information_gain_based_selection"
            gap["fix"] = "add_eig_based_experiment_selection"
            gap["fix_detail"] = "compute_expected_information_gain_per_experiment"

        else:
            gap["type"] = "general_performance"
            gap["cause"] = "insufficient_parameter_tuning"
            gap["fix"] = "param_mutation"
            gap["fix_detail"] = "mutate_relevant_parameters"

        return gap

    def summary(self, trial_scores: Dict[str, float] = None) -> str:
        """Gera relato textual do estado da arquitetura."""
        lines = ["=== SELF-DESCRIPTION ==="]
        lines.append(f"Modules: {len(self._snapshots[-1].modules) if self._snapshots else 0}")
        if self._snapshots:
            for mid, m in self._snapshots[-1].modules.items():
                lines.append(f"  {m.name} ({m.module_type}): {', '.join(m.capabilities[:3])}")
        if trial_scores:
            lines.append(f"Scores: {trial_scores}")
            gaps = self.detect_gaps(trial_scores)
            if gaps:
                lines.append("Gaps detected:")
                for g in gaps:
                    lines.append(f"  {g['trial']}: {g['type']} -> {g.get('fix_detail', g['fix'])}")
        return "\n".join(lines)

    def get_state(self) -> dict:
        return {
            "n_snapshots": len(self._snapshots),
            "n_gaps_detected": len(self._gap_history),
            "current_gaps": self._gap_history[-5:] if self._gap_history else [],
        }
