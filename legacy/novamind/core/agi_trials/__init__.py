"""
AGI Trials — 7 test provas que somente uma AGI genuina pode passar
====================================================================
Cada trial e proceduralmente gerado (semente aleatoria, nunca repete),
impossivel de passar por memorizacao, e mede uma capacidade fundamental:

  1. Descoberta Causal  — discrimina correlacao de causalidade
  2. Aterramento        — composicao one-shot com simbolos novos
  3. Calibracao         — sabe quando NAO sabe
  4. Contrafactual      — simulacao de mundos alternativos
  5. Transferencia      — reconhecimento de isomorfismo
  6. Experimentacao     — agencia cientifica autonoma
  7. Invencao           — cria primitivas que nao existem

Cada trial retorna TrialResult. O score final e a media geometrica.
Um gate rigido exige que todos > chance. Se qualquer trial e zero,
o sistema nao e AGI — por definicao.
"""
from .scoring import TrialResult, TrialSuiteResult, compute_agi_score
from .procedural_worlds import CausalWorld, PhysicsWorld, CompositionWorld, TransferWorld

__all__ = [
    "TrialResult", "TrialSuiteResult", "compute_agi_score",
    "CausalWorld", "PhysicsWorld", "CompositionWorld", "TransferWorld",
]
