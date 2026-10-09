"""
Compila funcoes Python do projeto novamind e HumanEval em .hdv.
Cada funcao vira UM manifold no espaco encode_text.

Uso:
    python scripts/compile_python_hdv.py
"""
import sys, os, ast, re, inspect, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import torch
from core.coglang import Manifold
from core.grounding_engine import GroundingEngine
from core.coglang_compiler import StreamProcessor, HDV_MAGIC
from core.code_extractor import sanitize_code

# Diretorios de codigo-fonte para extrair funcoes
CODE_DIRS = [
    ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind\\core').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')),
]
OUTPUT_HDV = r'D:\novamind_data\hdv\python_brain.hdv'

def extract_functions(filepath):
    """Extrai funcoes de um arquivo .py, retorna lista de dicts."""
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return []
    
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            name = node.name
            # Extrai argumentos
            args = [a.arg for a in node.args.args]
            # Extrai docstring
            docstring = ast.get_docstring(node) or ""
            # Extrai decorators
            decos = [d.id if isinstance(d, ast.Name) else '' for d in node.decorator_list]
            
            # Linhas da funcao
            try:
                lines, start = inspect.getsourcelines(open(filepath).read().splitlines())
            except:
                lines = []
            
            # Monta texto descritivo para o HDV
            # Formato: "nome_func arg1 arg2 ... docstring_resumida"
            args_str = " ".join(a for a in args[:5])  # max 5 args
            doc_first_line = docstring.split('\n')[0][:100] if docstring else ""
            
            # Texto principal para busca semantica
            text_parts = [name]
            if args_str:
                text_parts.append(args_str)
            if doc_first_line:
                text_parts.append(doc_first_line)
            text = " ".join(text_parts)
            
            if len(text) < 10:
                continue
            
            functions.append({
                'id': f"novamind/{os.path.basename(filepath)}/{name}",
                'text': text,
                'source': 'novamind',
                'type': 'code',
                'func': name,
                'args': args_str,
                'doc': doc_first_line[:200],
            })
    return functions

def compile_python_brain():
    engine = GroundingEngine()
    
    # Coleta funcoes de todos os arquivos
    all_funcs = []
    for code_dir in CODE_DIRS:
        for py_file in sorted(Path(code_dir).glob("*.py")):
            funcs = extract_functions(str(py_file))
            all_funcs.extend(funcs)
            if funcs:
                print(f"  {py_file.name}: {len(funcs)} funcoes")
    
    print(f"\nTotal: {len(all_funcs)} funcoes extraidas")
    
    # Escreve .hdv
    output_path = Path(OUTPUT_HDV)
    os.makedirs(output_path.parent, exist_ok=True)
    
    with StreamProcessor(engine, str(output_path.parent)).open("python_brain") as sp:
        for func in all_funcs:
            try:
                mf = engine.encode_text(func['text'][:5000])
                sp.write_item(mf, func)
            except Exception as e:
                print(f"  Erro ao codificar {func['id']}: {e}")
                continue
    
    print(f"\nPython brain compilado: {OUTPUT_HDV}")
    print(f"  {len(all_funcs)} funcoes")

def quick_compile_sample():
    """Versao rapida: compila apenas algumas funcoes representativas."""
    engine = GroundingEngine()
    
    samples = [
        {
            'id': 'math/soma',
            'text': 'soma a b return a plus b sum two numbers',
            'source': 'sample',
            'type': 'code',
            'func': 'soma',
            'args': 'a b',
        },
        {
            'id': 'math/fibonacci',
            'text': 'fibonacci n return nth fibonacci number sequence',
            'source': 'sample',
            'type': 'code',
            'func': 'fibonacci',
            'args': 'n',
        },
        {
            'id': 'list/reverse',
            'text': 'reverse_list lst return reversed list elements',
            'source': 'sample',
            'type': 'code',
            'func': 'reverse_list',
            'args': 'lst',
        },
        {
            'id': 'sort/bubble',
            'text': 'bubble_sort arr sort array using bubble sort algorithm',
            'source': 'sample',
            'type': 'code',
            'func': 'bubble_sort',
            'args': 'arr',
        },
        {
            'id': 'math/is_even',
            'text': 'is_even n check if number is even return boolean modulus',
            'source': 'sample',
            'type': 'code',
            'func': 'is_even',
            'args': 'n',
        },
        {
            'id': 'string/palindrome',
            'text': 'is_palindrome s check if string is palindrome return boolean',
            'source': 'sample',
            'type': 'code',
            'func': 'is_palindrome',
            'args': 's',
        },
        {
            'id': 'math/factorial',
            'text': 'factorial n compute factorial of n recursive product',
            'source': 'sample',
            'type': 'code',
            'func': 'factorial',
            'args': 'n',
        },
        {
            'id': 'search/binary',
            'text': 'binary_search arr target find element in sorted array',
            'source': 'sample',
            'type': 'code',
            'func': 'binary_search',
            'args': 'arr target',
        },
        {
            'id': 'math/is_prime',
            'text': 'is_prime n check if number is prime return boolean',
            'source': 'sample',
            'type': 'code',
            'func': 'is_prime',
            'args': 'n',
        },
        {
            'id': 'math/max_of',
            'text': 'max_of a b return maximum of two numbers comparison',
            'source': 'sample',
            'type': 'code',
            'func': 'max_of',
            'args': 'a b',
        },
        # Novamind-specific functions
        {
            'id': 'hdc/encode_text',
            'text': 'encode_text text encode string to HDC manifold hash unigram bigrams trigrams',
            'source': 'sample',
            'type': 'code',
            'func': 'encode_text',
            'args': 'text',
        },
        {
            'id': 'hdc/bind',
            'text': 'bind a b bind two manifolds using circular convolution hdc',
            'source': 'sample',
            'type': 'code',
            'func': 'bind',
            'args': 'a b',
        },
        {
            'id': 'hdc/cosine_similarity',
            'text': 'cosine_similarity a b compute cosine similarity between two vectors compare',
            'source': 'sample',
            'type': 'code',
            'func': 'cosine_similarity',
            'args': 'a b',
        },
        {
            'id': 'hdc/bundle',
            'text': 'bundle a b sum two manifolds in hyperdimensional space',
            'source': 'sample',
            'type': 'code',
            'func': 'bundle',
            'args': 'a b',
        },
        {
            'id': 'sort/merge_sort',
            'text': 'merge_sort arr sort array using merge sort divide and conquer',
            'source': 'sample',
            'type': 'code',
            'func': 'merge_sort',
            'args': 'arr',
        },
        # Wikipedia-style texts (simulated)
        {
            'id': 'wiki/gravity',
            'text': 'gravity is a natural phenomenon by which all things with mass are brought toward one another gravitational force',
            'source': 'sample',
            'type': 'wiki',
            'func': 'gravity',
            'args': '',
        },
        {
            'id': 'wiki/dna',
            'text': 'DNA deoxyribonucleic acid is a molecule that carries genetic instructions used in growth development functioning and reproduction of all living organisms',
            'source': 'sample',
            'type': 'wiki',
            'func': 'dna',
            'args': '',
        },
        {
            'id': 'wiki/algorithm',
            'text': 'algorithm is a finite sequence of rigorous instructions typically used to solve a class of specific problems or to perform a computation',
            'source': 'sample',
            'type': 'wiki',
            'func': 'algorithm',
            'args': '',
        },
        {
            'id': 'wiki/photosynthesis',
            'text': 'photosynthesis is the process by which plants use sunlight water and carbon dioxide to produce oxygen and energy in the form of glucose sugar',
            'source': 'sample',
            'type': 'wiki',
            'func': 'photosynthesis',
            'args': '',
        },
        {
            'id': 'wiki/entropy',
            'text': 'entropy is a measure of disorder or randomness in a thermodynamic system second law of thermodynamics states entropy increases',
            'source': 'sample',
            'type': 'wiki',
            'func': 'entropy',
            'args': '',
        },
    ]
    
    existing_names = {s['func'] for s in samples}
    
    # Extrai funcoes reais do novamind e adiciona (sem duplicatas)
    for code_dir in CODE_DIRS:
        for py_file in sorted(Path(code_dir).glob("*.py")):
            funcs = extract_functions(str(py_file))
            for f in funcs:
                if f['func'] not in existing_names:
                    samples.append(f)
                    existing_names.add(f['func'])
            if funcs:
                print(f"  {py_file.name}: {len(funcs)} funcoes, {sum(1 for f in funcs if f['func'] in existing_names)} novas")
    
    # Adiciona implementacoes reais de codigo (serao sanitizadas)
    # NOTA: usa id para dedup, nao func — uma funcao pode ter descricao + codigo
    code_impls = [
        {'id': 'code/is_even_impl', 'text': 'def is_even(n): return n % 2 == 0', 'source': 'code', 'type': 'code', 'func': 'is_even', 'args': 'n'},
        {'id': 'code/soma_impl', 'text': 'def soma(a, b): return a + b', 'source': 'code', 'type': 'code', 'func': 'soma', 'args': 'a b'},
        {'id': 'code/fib_impl', 'text': 'def fibonacci(n): if n <= 1: return n else: return fibonacci(n-1) + fibonacci(n-2)', 'source': 'code', 'type': 'code', 'func': 'fibonacci', 'args': 'n'},
        {'id': 'code/factorial_impl', 'text': 'def factorial(n): if n == 0: return 1 else: return n * factorial(n-1)', 'source': 'code', 'type': 'code', 'func': 'factorial', 'args': 'n'},
        {'id': 'code/bubble_impl', 'text': 'def bubble_sort(arr): for i in range(len(arr)): for j in range(len(arr)-1): if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]', 'source': 'code', 'type': 'code', 'func': 'bubble_sort', 'args': 'arr'},
        {'id': 'code/palindrome_impl', 'text': 'def is_palindrome(s): return s == s[::-1]', 'source': 'code', 'type': 'code', 'func': 'is_palindrome', 'args': 's'},
        {'id': 'code/binary_search_impl', 'text': 'def binary_search(arr, target): low = 0; high = len(arr) - 1; while low <= high: mid = (low + high) // 2; if arr[mid] == target: return mid; elif arr[mid] < target: low = mid + 1; else: high = mid - 1; return -1', 'source': 'code', 'type': 'code', 'func': 'binary_search', 'args': 'arr target'},
        {'id': 'code/is_prime_impl', 'text': 'def is_prime(n): if n < 2: return False; for i in range(2, int(n**0.5) + 1): if n % i == 0: return False; return True', 'source': 'code', 'type': 'code', 'func': 'is_prime', 'args': 'n'},
    ]
    existing_ids = {s['id'] for s in samples}
    for impl in code_impls:
        if impl['id'] not in existing_ids:
            samples.append(impl)
            existing_ids.add(impl['id'])
    
    print(f"Total: {len(samples)} itens para compilar")
    
    output_path = Path(OUTPUT_HDV)
    os.makedirs(output_path.parent, exist_ok=True)
    
    with StreamProcessor(engine, str(output_path.parent)).open("python_brain") as sp:
        for func in samples:
            try:
                text = func['text'][:5000]
                # Sanitiza codigo (traduz simbolos Python para tokens HDC)
                if func.get('type') == 'code':
                    text = sanitize_code(text)
                mf = engine.encode_text(text)
                sp.write_item(mf, func)
            except Exception as e:
                print(f"  Erro ao codificar {func['id']}: {e}")
    
    print(f"\nPython brain compilado: {OUTPUT_HDV}")
    print(f"  {len(samples)} itens (sample + wiki)")


if __name__ == '__main__':
    print("Compilando Python Brain HDV...")
    # Usa amostra rapida por enquanto
    quick_compile_sample()
    print("Pronto!")
