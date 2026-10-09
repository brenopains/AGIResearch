import sys, os, time, pathlib
os.environ.setdefault("LLM_MODEL","zai-org-glm-4.7")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import v3_bridge as v3
t=time.time()
res = v3.run(n_tasks=20, seed=7, gens=2, refine=1, fresh_lib=True, tag="llm_breadth")
print("ELAPSED %.0fs"%(time.time()-t))
