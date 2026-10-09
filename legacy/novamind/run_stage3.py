"""Executa etapa 3 - Fala (LibriSpeech)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from core.curriculum import CurriculumTrainer

ct = CurriculumTrainer()
ct.resume_or_start()
ct.run_stage(3)
