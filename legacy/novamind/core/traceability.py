"""
TraceabilityAnalyzer — Mede auto-organizacao do sistema ao longo do tempo.

Metricas capturadas em cada snapshot:
  - Compression ratio (conceitos antes/depois da economia)
  - Numero de leis (WorldLawRegistry)
  - Conflitos detectados vs resolvidos
  - Transicoes de modo cognitivo
  - Metas de curiosidade geradas
  - Energia media
  - Surpresa media (WorldModel)
  - Consistencia do modelo do mundo
  - Cobertura cross-dominio

Gera relatorio em D:\novamind_data\traceability\
"""
import json, os, time
from typing import List, Dict
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MetricSnapshot:
    """Snapshot de metricas em um ponto no tempo."""
    cycle: int
    timestamp: float

    # Modelo do mundo
    compression_ratio: float = 1.0
    n_concepts: int = 0
    n_laws: int = 0
    n_discovered_laws: int = 0
    world_accuracy: float = 0.5
    world_surprise: float = 0.0
    world_consistency: float = 1.0

    # Conflitos
    n_conflicts: int = 0
    n_resolved: int = 0
    n_unresolved: int = 0

    # Modos cognitivos
    current_mode: str = "focused"
    mode_transition_count: int = 0
    mode_cycle_counts: Dict[str, int] = field(default_factory=dict)

    # Curiosidade
    n_intrinsic_goals: int = 0
    n_curiosity_goals: int = 0
    n_mastery_goals: int = 0
    n_novelty_goals: int = 0

    # Self
    energy: float = 1.0
    mood: str = "curious"
    dopamine: float = 0.5
    noradrenaline: float = 0.3

    # Economia de conceitos
    economy_merges: int = 0
    economy_splits: int = 0
    economy_prunes: int = 0

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "elapsed_s": round(self.timestamp, 1),
            "compression": round(self.compression_ratio, 3),
            "n_concepts": self.n_concepts,
            "n_laws": self.n_laws,
            "discovered_laws": self.n_discovered_laws,
            "accuracy": round(self.world_accuracy, 3),
            "surprise": round(self.world_surprise, 3),
            "consistency": round(self.world_consistency, 3),
            "conflicts": self.n_conflicts,
            "resolved": self.n_resolved,
            "mode": self.current_mode,
            "transitions": self.mode_transition_count,
            "goals": self.n_intrinsic_goals,
            "energy": round(self.energy, 2),
            "mood": self.mood,
        }


class TraceabilityAnalyzer:
    """Analisa traceability do sistema ao longo do tempo.

    Uso:
        analyzer = TraceabilityAnalyzer()
        analyzer.snapshot(cycle=100, cognitive_executive=exec_)
        analyzer.snapshot(cycle=200, cognitive_executive=exec_)
        report = analyzer.generate_report()
    """

    def __init__(self, output_dir: str = r"D:\novamind_data\traceability"):
        self.output_dir = output_dir
        self.snapshots: List[MetricSnapshot] = []
        self._events: List[dict] = []
        os.makedirs(output_dir, exist_ok=True)

    def snapshot(self, cycle: int, exec_=None) -> MetricSnapshot:
        """Captura um snapshot do estado do sistema."""
        snap = MetricSnapshot(cycle=cycle, timestamp=time.time())

        if exec_ is not None:
            # World model
            if exec_.world:
                snap.world_accuracy = exec_.world.get_accuracy()
                snap.world_surprise = exec_.world.get_surprise()
                snap.world_consistency = exec_.world.get_consistency()

            # Mode manager
            if exec_.mode_mgr:
                snap.current_mode = exec_.mode_mgr.current_mode.value
                snap.mode_transition_count = len(exec_.mode_mgr.mode_history)
                # Count cycles per mode
                mode_counts = defaultdict(int)
                for h in exec_.mode_mgr.mode_history:
                    mode_counts[h.get("to", "?")] += 1
                snap.mode_cycle_counts = dict(mode_counts)

            # Self
            if exec_.self_model:
                snap.energy = exec_.self_model.energy
                snap.mood = exec_.self_model.current_mood
                snap.dopamine = exec_.self_model.dopamine
                snap.noradrenaline = exec_.self_model.noradrenaline

            # Conflict resolver
            if exec_.conflict_resolver:
                stats = exec_.conflict_resolver.get_stats()
                snap.n_conflicts = stats["total_conflicts"]
                snap.n_resolved = stats["resolved"]
                snap.n_unresolved = stats["unresolved"]

            # Law registry
            if exec_.law_registry:
                stats = exec_.law_registry.get_stats()
                snap.n_laws = stats["total_laws"]
                snap.n_discovered_laws = stats["discovered_laws"]

            # Concept economy
            if exec_.concept_economy:
                stats = exec_.concept_economy.get_stats()
                snap.compression_ratio = stats["compression_ratio"]
                snap.economy_merges = stats["merges"]
                snap.economy_splits = stats["splits"]
                snap.economy_prunes = stats["prunes"]

            # Concept registry
            if exec_.concept_registry:
                snap.n_concepts = exec_.concept_registry.count

            # Intrinsic goals from recent history
            if exec_.history:
                recent = [h for h in exec_.history[-50:] if "intrinsic_goals" in h]
                for h in recent:
                    for g in h.get("intrinsic_goals", []):
                        snap.n_intrinsic_goals += 1
                        drive = g.get("drive", "")
                        if "curiosity" in drive:
                            snap.n_curiosity_goals += 1
                        elif "mastery" in drive:
                            snap.n_mastery_goals += 1
                        elif "novelty" in drive:
                            snap.n_novelty_goals += 1

        self.snapshots.append(snap)
        return snap

    def log_event(self, event_type: str, description: str,
                  data: dict = None):
        """Registra um evento relevante."""
        self._events.append({
            "type": event_type,
            "description": description,
            "data": data or {},
            "timestamp": time.time(),
            "cycle": len(self.snapshots),
        })

    def generate_report(self) -> dict:
        """Gera relatorio de auto-organizacao."""
        if not self.snapshots:
            return {"error": "no snapshots"}

        first = self.snapshots[0]
        last = self.snapshots[-1]

        # Deltas
        delta_compression = last.compression_ratio - first.compression_ratio
        delta_laws = last.n_laws - first.n_laws
        delta_concepts = last.n_concepts - first.n_concepts
        delta_accuracy = last.world_accuracy - first.world_accuracy

        # Modos mais frequentes
        all_modes = defaultdict(int)
        for s in self.snapshots:
            all_modes[s.current_mode] += 1

        # Eventos por tipo
        events_by_type = defaultdict(int)
        for e in self._events:
            events_by_type[e["type"]] += 1

        # Conclusao
        conclusions = []
        if delta_compression > 0:
            conclusions.append(
                f"Compressao aumentou {delta_compression:.1%}: "
                f"conceitos estao sendo fundidos com sucesso"
            )
        if delta_laws > 0:
            conclusions.append(
                f"{delta_laws} novas leis descobertas: "
                f"o modelo do mundo esta se expandindo"
            )
        if last.n_resolved > first.n_resolved:
            conclusions.append(
                f"{last.n_resolved - first.n_resolved} conflitos resolvidos: "
                f"o sistema esta aprendendo com contradicoes"
            )
        if last.world_accuracy > first.world_accuracy:
            conclusions.append(
                f"Acuracia do modelo do mundo subiu {delta_accuracy:.1%}: "
                f"predicoes estao melhorando"
            )

        report = {
            "total_cycles": last.cycle,
            "elapsed_seconds": round(last.timestamp - first.timestamp, 1),
            "snapshots": len(self.snapshots),
            "initial": first.to_dict(),
            "final": last.to_dict(),
            "deltas": {
                "compression_ratio": round(delta_compression, 3),
                "n_laws": delta_laws,
                "n_concepts": delta_concepts,
                "world_accuracy": round(delta_accuracy, 3),
            },
            "mode_distribution": dict(all_modes),
            "events_by_type": dict(events_by_type),
            "total_events": len(self._events),
            "conclusions": conclusions,
            "self_organized": delta_compression > 0 or delta_laws > 0 or delta_accuracy > 0,
        }

        # Salva
        report_path = os.path.join(self.output_dir, "traceability_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Salva snapshots
        snap_path = os.path.join(self.output_dir, "snapshots.json")
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in self.snapshots], f, indent=2)

        return report

    def print_summary(self, report: dict):
        """Imprime resumo legivel."""
        print(f"\n{'='*60}")
        print(f"RELATORIO DE AUTO-ORGANIZACAO")
        print(f"{'='*60}")
        print(f"Duracao: {report['elapsed_seconds']:.0f}s, "
              f"{report['total_cycles']} ciclos")
        print(f"\nEvolucao:")
        print(f"  Compressao: {report['initial']['compression']} → "
              f"{report['final']['compression']} "
              f"({report['deltas']['compression_ratio']:+.3f})")
        print(f"  Leis: {report['initial']['n_laws']} → "
              f"{report['final']['n_laws']} "
              f"({report['deltas']['n_laws']:+d})")
        print(f"  Acuracia mundo: {report['initial']['accuracy']} → "
              f"{report['final']['accuracy']} "
              f"({report['deltas']['world_accuracy']:+.3f})")
        print(f"  Conflitos resolvidos: {report['final'].get('resolved', 0)}")

        print(f"\nDistribuicao de modos:")
        for mode, count in sorted(
            report['mode_distribution'].items(),
            key=lambda x: -x[1]
        ):
            pct = count / report['snapshots'] * 100
            print(f"  {mode}: {count} ciclos ({pct:.0f}%)")

        print(f"\nConclusoes:")
        if report['conclusions']:
            for c in report['conclusions']:
                print(f"  ✅ {c}")
        else:
            print(f"  ⚠️  Nenhuma auto-organizacao significativa detectada")

        if report['self_organized']:
            print(f"\n✅ O SISTEMA SE AUTO-ORGANIZOU")
        else:
            print(f"\n⚠️  AUTO-ORGANIZACAO NAO DETECTADA")
