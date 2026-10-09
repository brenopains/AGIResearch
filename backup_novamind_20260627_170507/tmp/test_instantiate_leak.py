import os
import sys

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_mem(step_name):
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024
        print(f"[MEM] {step_name}: {mem:.2f} MB")
    except ImportError:
        print(f"[MEM] {step_name}")

print_mem("Initial startup")

from core.perception import PerceptionEngine
print_mem("After importing PerceptionEngine")
pe = PerceptionEngine()
print_mem("After instantiating PerceptionEngine")

from core.memory import MemorySystem
print_mem("After importing MemorySystem")
ms = MemorySystem()
print_mem("After instantiating MemorySystem")

from core.language_cortex import NativeLanguageCortex
print_mem("After importing LanguageCortex")
lc = NativeLanguageCortex()
print_mem("After instantiating LanguageCortex")

from core.diva_controller import DIVAController
print_mem("After importing DIVAController")
dc = DIVAController()
print_mem("After instantiating DIVAController")

from core.motor_cortex import MotorCortex
print_mem("After importing MotorCortex")
mc = MotorCortex()
print_mem("After instantiating MotorCortex")
