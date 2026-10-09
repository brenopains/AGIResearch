"""Executa todas as etapas do curriculo."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from core.curriculum import CurriculumTrainer

ct = CurriculumTrainer()
ct.run_all()
print("\n=== CURRICULO CONCLUIDO ===")
