"""
IntrinsicMotivationEngine — Gera METAS INTERNAS a partir dos drives do self.

Tipos de drive:
  - Curiosidade: o que eu nao sei ainda? (lacunas no CGE)
  - Mestria: o que eu quero melhorar? (habilidades com baixa proficiencia)
  - Novidade: o que e novo/diferente? (padroes nunca vistos)

Cada drive gera GOALS que o sistema persegue autonomamente.
Sem isso, o sistema so faz o que mandam — e uma ferramenta.
Com isso, vira um agente com iniciativa propria.
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import random


@dataclass
class IntrinsicGoal:
    """Meta gerada internamente por um drive."""
    drive: str                    # "curiosity", "mastery", "novelty"
    description: str
    priority: float = 0.5        # 0-1
    domain: str = "general"
    target_concept: str = ""
    strategy: str = ""            # "explore", "practice", "experiment"
    duration: int = 3             # quantos steps persistir

    def to_dict(self) -> dict:
        return {
            "drive": self.drive,
            "description": self.description[:60],
            "priority": round(self.priority, 2),
            "strategy": self.strategy,
        }


class IntrinsicMotivationEngine:
    """Motor de motivacao intrinseca.

    Gera metas a partir do estado do self e do CGE.
    """

    def __init__(self, self_model=None, cge=None):
        self.self_model = self_model
        self.cge = cge
        self.active_goals: List[IntrinsicGoal] = []
        self._goal_counter = 0

    def generate_goals(self) -> List[IntrinsicGoal]:
        """Gera metas baseadas nos drives atuais.

        Cada drive avalia o estado do self e do mundo e gera
        uma ou mais metas para reduzir sua frustracao.
        """
        goals = []

        if self.self_model is None:
            return goals

        # Drive 1: CURIOSIDADE — buscar lacunas no conhecimento
        curiosity_drive = self.self_model.drives.get("curiosity")
        if curiosity_drive and curiosity_drive.is_pressing:
            cg = self._curiosity_goal()
            if cg:
                goals.append(cg)

        # Drive 2: MESTRIA — melhorar habilidades
        mastery_drive = self.self_model.drives.get("mastery")
        if mastery_drive and mastery_drive.is_pressing:
            mg = self._mastery_goal()
            if mg:
                goals.append(mg)

        # Drive 3: NOVIDADE — buscar experiencias novas
        novelty_drive = self.self_model.drives.get("novelty")
        if novelty_drive and novelty_drive.is_pressing:
            ng = self._novelty_goal()
            if ng:
                goals.append(ng)

        # Se nenhum drive gerou meta, fallback: curiosidade geral
        if not goals:
            goals.append(IntrinsicGoal(
                drive="curiosity",
                description="explore anything new",
                priority=0.3,
                strategy="explore",
            ))

        self.active_goals = goals
        return goals

    def _curiosity_goal(self) -> Optional[IntrinsicGoal]:
        """Gera meta de curiosidade: busca conceitos com baixa confianca no CGE."""
        if self.cge is None:
            return IntrinsicGoal(
                drive="curiosity",
                description="explore the environment",
                priority=0.6,
                strategy="explore",
            )

        # Busca conceitos com baixa confianca no CGE
        low_conf_nodes = []
        if hasattr(self.cge, 'nodes'):
            for nid, ndata in self.cge.nodes.items():
                conf = float(ndata.get("confidence", 0.5))
                if conf < 0.3:
                    low_conf_nodes.append((nid, conf))

        if low_conf_nodes:
            # Pega um conceito aleatorio com baixa confianca
            target = random.choice(low_conf_nodes)
            return IntrinsicGoal(
                drive="curiosity",
                description=f"investigate uncertain concept: {target[0]}",
                priority=0.7,
                domain="concept_learning",
                target_concept=target[0],
                strategy="explore",
            )

        # Fallback: busca qualquer conceito com acesso recente
        return IntrinsicGoal(
            drive="curiosity",
            description="search for new knowledge",
            priority=0.5,
            strategy="explore",
        )

    def _mastery_goal(self) -> Optional[IntrinsicGoal]:
        """Gera meta de mestria: melhorar habilidade com menor proficiencia."""
        if self.self_model is None:
            return None

        # Encontra a habilidade com menor proficiencia
        if self.self_model.capabilities:
            worst_skill = min(self.self_model.capabilities,
                             key=self.self_model.capabilities.get)
            worst_prof = self.self_model.capabilities[worst_skill]

            return IntrinsicGoal(
                drive="mastery",
                description=f"practice skill: {worst_skill} (current={worst_prof:.2f})",
                priority=0.6 if worst_prof < 0.3 else 0.4,
                domain="skill_development",
                target_concept=worst_skill,
                strategy="practice",
            )

        return None

    def _novelty_goal(self) -> Optional[IntrinsicGoal]:
        """Gera meta de novidade: algo que o sistema nunca tentou."""
        return IntrinsicGoal(
            drive="novelty",
            description="try a new approach to problem solving",
            priority=0.5,
            strategy="experiment",
        )

    def record_outcome(self, goal: IntrinsicGoal, success: bool):
        """Registra o resultado de perseguir uma meta."""
        if self.self_model is None:
            return

        # Satisfaz o drive correspondente
        drive = self.self_model.drives.get(goal.drive)
        if drive:
            if success:
                drive.satisfy()
                self.self_model.dopamine = min(1.0, self.self_model.dopamine + 0.05)
            else:
                drive.frustrate(0.05)

        # Atualiza memoria autobiografica
        self.self_model.memory.record(
            event_type="success" if success else "failure",
            description=f"intrinsic goal: {goal.description}",
            concepts=[goal.drive, goal.target_concept] if goal.target_concept else [goal.drive],
            valence=0.5 if success else -0.3,
        )

    def get_stats(self) -> dict:
        return {
            "active_goals": len(self.active_goals),
            "goal_types": list(set(g.drive for g in self.active_goals)),
        }
