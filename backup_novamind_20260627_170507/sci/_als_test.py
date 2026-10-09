import os
os.environ['LLM_MODEL'] = 'qwen3-235b-a22b-instruct-2507'   # FORCA o modelo forte ANTES dos imports
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
from unified.run import run
run("amyotrophic lateral sclerosis (ALS)", generation_cycles=5, top_for_tournament=6,
    tourn_rounds=2, compose_every=2, deep_dive_top=2)
