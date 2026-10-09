import importlib, sys, os
# Force reload
for mod in list(sys.modules.keys()):
    if 'core' in mod or 'word_ngram' in mod or 'novacrush' in mod:
        del sys.modules[mod]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check source file timestamp
spec = importlib.util.find_spec('core.language_cortex')
print(f"Loading from: {spec.origin}")
print(f"pyc: {spec.cached}")
with open(spec.origin) as f:
    content = f.read()
has_gen_marker = 'returning ngram result' in content
print(f"Source has [GEN] markers: {has_gen_marker}")

from core.language_cortex import NativeLanguageCortex
import inspect
source = inspect.getsource(NativeLanguageCortex.generate_fluent_thought)
has_print = 'import sys' in source and 'print(' in source
print(f"Loaded function has debug prints: {has_print}")
if has_print:
    print("Function source excerpt (first 200 chars):")
    print(source[:200])
