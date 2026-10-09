"""Teste cientifico: geracao de codigo composicional"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from core.grounding_engine import GroundingEngine
from core.hdc_memory import HDCMemory
from core.autoregressive_generator import AutoregressiveGenerator
from core.positional_binder import PositionalBinder, hdv_projector
from core.broca_decoder import BrocaDecoder
from core.temporal_buffer import TemporalBuffer
from core.code_extractor import sanitize_code, desanitize_code

engine = GroundingEngine(device='cuda')
binder = PositionalBinder(engine, dim=10000)
decoder = BrocaDecoder(engine)
buffer = TemporalBuffer(engine)
memory = HDCMemory(engine)
memory.load_all()

# 1. Verificar vocabulario para operacoes-chave
print("=" * 60)
print("VERIFICANDO CONCEITOS NO HDV")
print("=" * 60)
test_words = ['divisible', 'mod', 'and', '7', '11',
              'divisivel', 'modulo', 'remainder',
              'if', 'else', 'return', 'int',
              'prime', 'even', 'odd', 'multiple']

for word in test_words:
    q = hdv_projector(engine.encode_text(sanitize_code(word)).tensor.float().cpu())
    results = memory.query(q[0], top_k=3)
    if results:
        best_text, best_sim = results[0]
        print(f"  {word:15s} -> {best_text:.50s}  (sim={best_sim:.3f})")
    else:
        print(f"  {word:15s} -> (vazio)")

# 2. Buscar no HDV funcoes que usam MOD e AND juntos
print()
print("=" * 60)
print("BUSCANDO FUNCOES COM MOD + AND")
print("=" * 60)
# Codigo sanitizado de "n % 7 == 0 and n % 11 == 0"
mod_and_text = "n _MOD_ 7 _EQ_ 0 and n _MOD_ 11 _EQ_ 0"
q_mod_and = hdv_projector(engine.encode_text(mod_and_text).tensor.float().cpu())
results = memory.query(q_mod_and[0], top_k=10)
for txt, sim in results:
    visible = desanitize_code(txt) if '<' in txt or '_COLON_' in txt else txt
    print(f"  sim={sim:.3f}  {visible:.80s}")

# 3. Buscar funcao exata de divisivel por 7 e 11
print()
print("=" * 60)
print("BUSCANDO is_divisible_by_7_and_11")
print("=" * 60)
q_exact = hdv_projector(engine.encode_text(sanitize_code("def is_divisible_by_7_and_11(n):")).tensor.float().cpu())
results = memory.query(q_exact[0], top_k=5)
for txt, sim in results:
    print(f"  sim={sim:.3f}  {desanitize_code(txt):.100s}")

# 4. TESTE PRINCIPAL: geracao
print()
print("=" * 60)
print("TESTE PRINCIPAL: GERACAO COMPOSICIONAL")
print("=" * 60)
gen = AutoregressiveGenerator(engine, binder, decoder, memory, buffer, verbose=True)
result = gen.generate("def is_divisible_by_7_and_11(n):", max_chunks=8, chunk_size=4, min_sim=0.0)
print()
print(">>> RESULTADO FINAL:")
print(result)
print()

# 5. Validar se o codigo gerado e compilavel
print("=" * 60)
print("VALIDACAO: O CODIGO COMPILA?")
print("=" * 60)
try:
    compile(result, '<test>', 'exec')
    print("  ✅ COMPILA SEM ERROS!")
except SyntaxError as e:
    print(f"  ❌ SYNTAX ERROR: {e}")

# 6. Testar execucao se compilar
if 'def is_divisible_by_7_and_11' in result:
    try:
        namespace = {}
        exec(result, namespace)
        fn = namespace.get('is_divisible_by_7_and_11')
        if fn:
            test_cases = [(77, True), (7, False), (11, False), (154, True), (1, False), (0, False)]
            print()
            print("=" * 60)
            print("TESTANDO LOGICA: n % 7 == 0 and n % 11 == 0")
            print("=" * 60)
            all_pass = True
            for n, expected in test_cases:
                actual = fn(n)
                if n < 0: continue
                status = "✅" if actual == expected else "❌"
                if actual != expected: all_pass = False
                print(f"  f({n:3d}) = {str(actual):5s}  esperado={str(expected):5s}  {status}")
            print()
            if all_pass:
                print("  ✅ TODOS OS TESTES PASSARAM!")
            else:
                print("  ⚠ ALGUNS TESTES FALHARAM")
    except Exception as e:
        print(f"  ⚠ ERRO NA EXECUCAO: {e}")
