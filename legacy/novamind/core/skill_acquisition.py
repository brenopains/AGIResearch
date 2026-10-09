"""
SkillAcquisition — Compilacao de Sequencias Frequentes em Habilidades

Diferenca entre CONHECER algo e DOMINAR um procedimento:
  - Conhecimento: "sei que 2+2=4" (CGE)
  - Habilidade: "sei somar mentalmente" (Skill)

Uma skill e uma sequencia percepcao → decisao → acao que foi
"compilada" em uma unidade reutilizavel e otimizada.

Exemplos de habilidades:
  - resolver equacao quadratica
  - tocar uma escala musical
  - dirigir em linha reta
  - compilar um programa
  - fazer uma busca no grafo

Caracteristicas de uma skill:
  - executa quase automaticamente (baixo custo cognitivo)
  - melhora com pratica (refinamento continuo)
  - pode ser combinada com outras skills
  - e compilada da experiencia, nao programada
"""
import time, hashlib, math, random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Skill:
    """Uma habilidade compilada."""
    id: str = ""
    name: str = ""
    description: str = ""
    sequence: List[str] = field(default_factory=list)  # acoes/operadores

    # Metricas de performance
    accuracy: float = 0.5          # precisao atual
    latency: float = 1.0           # custo relativo (1.0 = baseline)
    n_executions: int = 0
    last_executed: float = field(default_factory=time.time)

    # Composicao
    sub_skills: List[str] = field(default_factory=list)  # skills que a compoem
    parent_skills: List[str] = field(default_factory=list)  # skills que a usam

    # Evolucao
    confidence: float = 0.3
    created_at: float = field(default_factory=time.time)
    last_optimized: float = field(default_factory=time.time)

    @property
    def is_automatic(self) -> bool:
        """Uma skill se torna automatica quando executada N vezes."""
        return self.n_executions >= 10 and self.accuracy > 0.8

    def to_dict(self) -> Dict:
        return {"id": self.id[:8], "name": self.name,
                "accuracy": round(self.accuracy, 2), "latency": round(self.latency, 2),
                "n_executions": self.n_executions, "automatic": self.is_automatic}


class SkillAcquisition:
    """Sistema de aquisicao de habilidades.

    Compila sequencias bem-sucedidas em skills reutilizaveis.
    Skills sao refinadas com a pratica.
    Skills podem ser decompostas e recombinadas.
    """

    def __init__(self, cge=None):
        self.cge = cge
        self.skills: Dict[str, Skill] = {}
        self._history: List[Dict] = []
        self._execution_log: List[Dict] = []

    def compile(self, sequence: List[str], name: str = "",
                parent_ids: Optional[List[str]] = None) -> Optional[Skill]:
        """Compila uma sequencia frequente em uma skill."""
        if len(sequence) < 2:
            return None

        sid = f"skill_{hashlib.md5('_'.join(sequence).encode()).hexdigest()[:8]}"
        if sid in self.skills:
            return self.skills[sid]

        skill = Skill(
            id=sid,
            name=name or f"skill_{len(self.skills)+1}",
            description=f"Sequencia compilada: {' → '.join(sequence[:5])}",
            sequence=sequence,
            accuracy=0.5,
            latency=max(0.3, len(sequence) * 0.15),
            confidence=0.3,
            sub_skills=parent_ids or [],
        )
        self.skills[sid] = skill
        self._history.append({"action": "compile", "name": skill.name,
                              "n_steps": len(sequence), "time": time.time()})
        return skill

    def execute(self, skill_id: str, outcome: bool, latency: float = 1.0):
        """Executa uma skill e registra o resultado (refinamento)."""
        skill = self.skills.get(skill_id)
        if skill is None:
            return
        skill.n_executions += 1
        skill.last_executed = time.time()

        # Refinamento: atualiza accuracy com media movel
        old_acc = skill.accuracy
        skill.accuracy = old_acc * 0.9 + (1.0 if outcome else 0.0) * 0.1

        # Refinamento: latencia diminui com pratica
        skill.latency = skill.latency * 0.95 + latency * 0.05
        if skill.n_executions > 5:
            skill.latency *= 0.98  # fica mais rapido com pratica

        # Confianca aumenta com acertos
        if outcome:
            skill.confidence = min(1.0, skill.confidence + 0.05)
        else:
            skill.confidence = max(0.1, skill.confidence - 0.02)

        self._execution_log.append({
            "skill_id": skill_id, "outcome": outcome,
            "latency": latency, "time": time.time(),
        })
        if len(self._execution_log) > 200:
            self._execution_log = self._execution_log[-200:]

    def decompose(self, skill_id: str) -> List[Skill]:
        """Decompoe uma skill em sub-skills."""
        skill = self.skills.get(skill_id)
        if skill is None:
            return []
        sub_skills = []
        for sub_id in skill.sub_skills:
            sub = self.skills.get(sub_id)
            if sub:
                sub_skills.append(sub)
        return sub_skills

    def compose(self, skill_ids: List[str], name: str = "") -> Optional[Skill]:
        """Combina multiplas skills em uma habilidade mais complexa."""
        if len(skill_ids) < 2:
            return None
        # Concatena as sequencias
        combined_sequence = []
        for sid in skill_ids:
            s = self.skills.get(sid)
            if s:
                combined_sequence.extend(s.sequence)

        if not combined_sequence:
            return None

        skill = self.compile(combined_sequence, name=name,
                             parent_ids=skill_ids)
        return skill

    def mine_from_traces(self, trace_data: List[Dict]) -> List[Skill]:
        """Minera traces para compilar novas habilidades."""
        new_skills = []

        # Extrai n-grams de operadores/modos com alta recompensa
        seq_counts = defaultdict(list)
        for td in trace_data:
            steps = td.get("steps", []) if isinstance(td, dict) else []
            if not steps:
                continue
            modes = [s.get("mode", "") for s in steps if isinstance(s, dict)]
            rewards = [s.get("reward", 0) for s in steps if isinstance(s, dict)]

            # 4-grams e 5-grams
            for n in [4, 5]:
                for i in range(len(modes) - n + 1):
                    gram = tuple(modes[i:i+n])
                    avg_r = sum(rewards[i:i+n]) / max(1, n) if len(rewards) >= i+n else 0
                    seq_counts[gram].append(avg_r)

        for gram, rewards in seq_counts.items():
            if len(rewards) < 3:
                continue
            avg_r = sum(rewards) / len(rewards)
            if avg_r < 0.5:
                continue

            skill = self.compile(list(gram))
            if skill:
                new_skills.append(skill)

        # Tenta compor skills que aparecem juntas
        all_skill_names = [s.name for s in self.skills.values() if s.n_executions > 0]
        if len(all_skill_names) >= 2:
            for i in range(len(all_skill_names)):
                for j in range(i+1, len(all_skill_names)):
                    if random.random() < 0.5:  # exploracao
                        self.compose([all_skill_names[i], all_skill_names[j]])

        return new_skills

    def get_state(self) -> Dict:
        return {
            "n_skills": len(self.skills),
            "automatic": sum(1 for s in self.skills.values() if s.is_automatic),
            "total_executions": sum(s.n_executions for s in self.skills.values()),
            "top": [s.to_dict() for s in sorted(
                self.skills.values(), key=lambda s: -s.n_executions)[:3]],
        }
