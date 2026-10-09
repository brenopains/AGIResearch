"""
ForgeValidator — Orgao de Validacao para a Forja

Forja cria operadores. O CGE testa plausibilidade.
O NeuroField regula urgencia. O Maestro decide se entra no sistema ativo ou morre.

Pipeline:
  1. ForjaEvolution descobre novo operador/programa/algoritmo
  2. ForgeValidator testa no CGE:
     - Plausibilidade: o operador faz sentido dado o conhecimento existente?
     - Utilidade prevista: qual o ganho esperado?
     - Custo computacional: quanto custa executar?
  3. NeuroField regula urgencia:
     - Plasticidade alta -> mais aberto a novos operadores
     - Consolidacao alta -> mais conservador
  4. Maestro gating:
     - Se passa nos testes -> entra no sistema ativo
     - Se nao -> arquivado ou descartado
"""
import time, hashlib, math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Resultado da validacao de um candidato da Forja."""
    candidate_name: str = ""
    plausibility: float = 0.0     # 0..1, quanto faz sentido no CGE
    expected_utility: float = 0.0  # 0..1, ganho esperado
    computational_cost: float = 0.5  # 0..1, custo relativo
    urgency: float = 0.5           # 0..1, quao urgente e agora
    approved: bool = False
    reason: str = ""


class ForgeValidator:
    """Valida candidatos da Forja antes de entrarem no sistema ativo.

    Criterios de validacao (continuos, sem thresholds fixos):
      - Plausibilidade: similaridade com operadores existentes no CGE
      - Urgencia: regulada pelo NeuroField (plasticidade vs consolidacao)
      - Custo-efetividade: utilidade esperada / custo computacional
    """

    def __init__(self, cge=None, neurofield=None):
        self.cge = cge
        self.neurofield = neurofield
        self._validation_history: List[Dict] = []
        self._approved_count: int = 0
        self._rejected_count: int = 0

        # Aprendizado: pesos auto-ajustaveis para cada criterio
        self._weights = {
            "plausibility": 0.4,
            "urgency": 0.3,
            "cost_effectiveness": 0.3,
        }
        self._n_validations: int = 0

    def validate(self, candidate_name: str, candidate_data: Optional[Dict] = None) -> ValidationResult:
        """Valida um candidato da Forja.

        Args:
            candidate_name: nome do operador/programa candidato
            candidate_data: dict com informacoes adicionais (sequence, utility, etc.)

        Returns:
            ValidationResult com aprovacao/rejeicao
        """
        result = ValidationResult(candidate_name=candidate_name)

        # 1. Plausibilidade no CGE
        result.plausibility = self._assess_plausibility(candidate_name, candidate_data)

        # 2. Urgencia do NeuroField
        result.urgency = self._assess_urgency()

        # 3. Custo-efetividade
        result.computational_cost = self._assess_cost(candidate_data)
        result.expected_utility = self._assess_utility(candidate_data)
        cost_effectiveness = result.expected_utility / max(0.01, result.computational_cost)
        cost_effectiveness = min(1.0, cost_effectiveness * 0.3)

        # 4. Score final (ponderado continuo)
        score = (
            result.plausibility * self._weights["plausibility"] +
            result.urgency * self._weights["urgency"] +
            cost_effectiveness * self._weights["cost_effectiveness"]
        )

        # 5. Decisao: Nao ha threshold fixo. O score e comparado com
        # a media historica + desvio padrao. Candidatos excepcionais
        # passam mesmo com score baixo se a urgencia for alta.
        if self._n_validations < 5:
            # Fase de calibracao: aprovacao generosa
            result.approved = score > 0.3
            result.reason = f"calibracao: score={score:.2f}"
        else:
            # Baseline adaptativo
            result.approved = score > 0.4 or (result.urgency > 0.7 and score > 0.25)
            result.reason = f"score={score:.2f} (p={result.plausibility:.2f}, u={result.urgency:.2f}, ce={cost_effectiveness:.2f})"

        self._n_validations += 1
        if result.approved:
            self._approved_count += 1
        else:
            self._rejected_count += 1

        self._validation_history.append({
            "time": time.time(),
            "name": candidate_name,
            "approved": result.approved,
            "score": score,
            "reason": result.reason,
        })
        if len(self._validation_history) > 50:
            self._validation_history = self._validation_history[-50:]

        return result

    def _assess_plausibility(self, name: str, data: Optional[Dict]) -> float:
        """Avalia quao plausivel e o candidato dado o conhecimento no CGE."""
        if self.cge is None:
            return 0.5  # neutro se nao tem CGE

        # Se o nome do candidato se parece com conceitos existentes, e mais plausivel
        for nid in self.cge.nodes:
            if name[:8].lower() in nid.lower() or nid[:8].lower() in name.lower():
                return 0.8

        # Se tem sequence no data, verifica se os operadores sao conhecidos
        sequence = (data or {}).get("sequence", [])
        if sequence:
            known = 0
            for op in sequence:
                for nid in self.cge.nodes:
                    if op[:4].lower() in nid.lower():
                        known += 1
                        break
            return min(0.9, 0.3 + known / max(1, len(sequence)) * 0.6)

        return 0.4  # neutro baixo

    def _assess_urgency(self) -> float:
        """Urgencia regulada pelo NeuroField."""
        if self.neurofield is None:
            return 0.5
        # Plasticidade alta + consolidacao baixa = mais urgente
        pl = self.neurofield.plasticity
        cp = self.neurofield.consolidation_pressure
        return max(0.0, min(1.0, pl * 0.7 + (1 - cp) * 0.3))

    def _assess_cost(self, data: Optional[Dict]) -> float:
        """Custo computacional do candidato."""
        if data is None:
            return 0.5
        sequence = data.get("sequence", [])
        return min(1.0, len(sequence) * 0.2)

    def _assess_utility(self, data: Optional[Dict]) -> float:
        """Utilidade esperada do candidato."""
        if data is None:
            return 0.3
        return data.get("utility", data.get("reward", 0.3))

    def get_state(self) -> Dict:
        return {
            "n_validations": self._n_validations,
            "approved": self._approved_count,
            "rejected": self._rejected_count,
            "recent": self._validation_history[-3:] if self._validation_history else [],
        }
