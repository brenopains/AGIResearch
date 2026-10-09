"""Limitar BLAS antes de NumPy; evitar consultas WMI do runtime Windows."""
import os
import platform
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'deps'))
if sys.platform == 'win32':
    platform.machine = platform._get_machine_win32
    platform.system = lambda: 'Windows'
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
