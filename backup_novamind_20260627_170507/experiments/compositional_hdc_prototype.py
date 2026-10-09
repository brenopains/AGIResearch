"""
PROTOTIPO HDC COMPOSICIONAL — Fase 1
======================================
Valida se bind/unbind/parameter recovery funcionam num espaco
HDC controlado antes de migrar o sistema inteiro.

Arquitetura:
  - Vetores atomicos: ±1 bipolar, 10000 dim
  - Binding: Hadamard product (elementwise XOR em bipolar)
  - Bundling: majority sum + sign
  - Unbinding: Hadamard product com o role vector
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

torch.manual_seed(42)
np.random.seed(42)

DIM = 10000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def random_bipolar(dim: int = DIM, device: str = DEVICE) -> torch.Tensor:
    """Gera um vetor bipolar aleatorio (±1)."""
    return torch.randint(0, 2, (dim,), device=device).float() * 2 - 1


def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Binding: Hadamard product. Em bipolar, equivale a XOR."""
    return a * b


def unbind(concept: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
    """Unbinding: concept * role. Recupera filler ± ruido."""
    return concept * role


def bundle(vectors: List[torch.Tensor]) -> torch.Tensor:
    """Bundling: majority sum. Soma e aplica sign()."""
    s = torch.stack(vectors).sum(dim=0)
    return torch.sign(s)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def normalize(v: torch.Tensor) -> torch.Tensor:
    """Normaliza para ±1 (bipolar)."""
    return torch.sign(v)


# ===============================================================
# FASE 1: CRIAR VOCABULARIO ATOMICO
# ===============================================================
print("=" * 70)
print("FASE 1: CRIANDO VOCABULARIO ATOMICO")
print("=" * 70)

atoms = {}

# Numeros 0-20
for i in range(21):
    atoms[f'num_{i}'] = random_bipolar()
atoms['num_7'] = random_bipolar()  # sobrescreve com seed limpa
atoms['num_11'] = random_bipolar()
atoms['num_3'] = random_bipolar()
atoms['num_5'] = random_bipolar()
atoms['num_77'] = random_bipolar()
atoms['num_154'] = random_bipolar()

# Operacoes
atoms['op_mod'] = random_bipolar()
atoms['op_eq'] = random_bipolar()
atoms['op_and'] = random_bipolar()
atoms['op_return'] = random_bipolar()
atoms['op_if'] = random_bipolar()

# Tipos
atoms['type_number'] = random_bipolar()
atoms['type_boolean'] = random_bipolar()
atoms['type_function'] = random_bipolar()

# Roles (posicoes estruturais)
atoms['role_param1'] = random_bipolar()
atoms['role_param2'] = random_bipolar()
atoms['role_operation'] = random_bipolar()
atoms['role_return_type'] = random_bipolar()
atoms['role_name'] = random_bipolar()
atoms['role_body'] = random_bipolar()
atoms['role_template'] = random_bipolar()

# Palavras de funcao
atoms['word_def'] = random_bipolar()
atoms['word_is'] = random_bipolar()
atoms['word_divisible'] = random_bipolar()
atoms['word_by'] = random_bipolar()
atoms['word_and'] = random_bipolar()
atoms['word_return'] = random_bipolar()
atoms['word_n'] = random_bipolar()
atoms['word_mod'] = random_bipolar()
atoms['word_eq'] = random_bipolar()
atoms['word_zero'] = random_bipolar()

# Templates de funcao (atomicos por enquanto)
atoms['template_divisible_by_X_and_Y'] = random_bipolar()
atoms['template_return_n_mod_X_eq_0_and_n_mod_Y_eq_0'] = random_bipolar()
atoms['template_is_even'] = random_bipolar()
atoms['template_is_odd'] = random_bipolar()

print(f"  {len(atoms)} vetores atomicos criados ({DIM} dim, {DEVICE})")

# Verificar: vetores atomicos sao ortogonais?
print()
sims = []
keys = list(atoms.keys())[:10]
for i, k1 in enumerate(keys):
    for k2 in keys[i+1:]:
        s = cosine(atoms[k1], atoms[k2])
        sims.append(abs(s))
print(f"  Similaridade media entre atomicos: {np.mean(sims):.4f} (ideal: ~0.0)")
print(f"  Similaridade maxima: {np.max(sims):.4f}")
print(f"  Auto-similaridade: {cosine(atoms['num_7'], atoms['num_7']):.3f} (ideal: 1.0)")


# ===============================================================
# FASE 2: BINDING/UNBINDING BASICO
# ===============================================================
print()
print("=" * 70)
print("FASE 2: BINDING / UNBINDING BASICO")
print("=" * 70)

# Criar conceito composto: role_param1 * num_7
concept_7 = bind(atoms['role_param1'], atoms['num_7'])
concept_11 = bind(atoms['role_param1'], atoms['num_11'])

# Unbinding: recuperar num_7 de concept_7
recovered_7 = unbind(concept_7, atoms['role_param1'])  # = num_7 + ruido
sim_recovered = cosine(recovered_7, atoms['num_7'])
print(f"\n  Unbind(role_param1 * num_7, role_param1) -> sim(num_7) = {sim_recovered:.3f}")
print(f"  Unbind(role_param1 * num_7, role_param1) -> sim(num_11) = {cosine(recovered_7, atoms['num_11']):.3f}")
print(f"  (1.0 = recuperacao perfeita do filler)")

# Binding com ruido de bundle: (role1 * 7) + (role2 * 11)
concept_dual = bundle([bind(atoms['role_param1'], atoms['num_7']),
                       bind(atoms['role_param2'], atoms['num_11'])])
recovered_from_dual = unbind(concept_dual, atoms['role_param1'])
sim_dual = cosine(recovered_from_dual, atoms['num_7'])
sim_dual_noise = cosine(recovered_from_dual, atoms['num_11'])
print(f"\n  Bundle[(role1 * 7) + (role2 * 11)]")
print(f"  Unbind(role1) -> sim(num_7) = {sim_dual:.3f}")
print(f"  Unbind(role1) -> sim(num_11) = {sim_dual_noise:.3f}")

# ===============================================================
# FASE 3: CODIFICACAO DE FUNCAO COMPLETA
# ===============================================================
print()
print("=" * 70)
print("FASE 3: CODIFICACAO DE FUNCAO COMPOSICIONAL")
print("=" * 70)

# Codificar: def is_divisible_by_7_and_11(n): return n % 7 == 0 and n % 11 == 0
# Como estrutura role-filler:
function_7_11 = bundle([
    bind(atoms['role_name'], atoms['word_divisible']),
    bind(atoms['role_param1'], atoms['num_7']),
    bind(atoms['role_param2'], atoms['num_11']),
    bind(atoms['role_operation'], atoms['op_mod']),
    bind(atoms['role_return_type'], atoms['type_boolean']),
])

# Codificar: def is_divisible_by_3_and_5(n): ... (template alternativo)
function_3_5 = bundle([
    bind(atoms['role_name'], atoms['word_divisible']),
    bind(atoms['role_param1'], atoms['num_3']),
    bind(atoms['role_param2'], atoms['num_5']),
    bind(atoms['role_operation'], atoms['op_mod']),
    bind(atoms['role_return_type'], atoms['type_boolean']),
])

# Armazenar num pequeno HDV
hdv_experimental = {
    'is_divisible_by_7_and_11': function_7_11,
    'is_divisible_by_3_and_5': function_3_5,
    'is_even': bundle([
        bind(atoms['role_name'], atoms['word_is']),
        bind(atoms['role_param1'], atoms['num_2']),
        bind(atoms['role_operation'], atoms['op_mod']),
    ]),
}

print(f"  {len(hdv_experimental)} funcoes codificadas no HDV experimental")

# TESTE 1: Recuperar parametro 7 da funcao composta
print()
print("--- TESTE 1: Recuperar num_7 de is_divisible_by_7_and_11 ---")
recovered_7_fn = unbind(function_7_11, atoms['role_param1'])
rec_sim_7 = cosine(recovered_7_fn, atoms['num_7'])
rec_sim_3 = cosine(recovered_7_fn, atoms['num_3'])
rec_sim_11 = cosine(recovered_7_fn, atoms['num_11'])
print(f"  sim(num_7)  = {rec_sim_7:.3f}  (deveria ser o mais alto)")
print(f"  sim(num_3)  = {rec_sim_3:.3f}")
print(f"  sim(num_11) = {rec_sim_11:.3f}")
print(f"  -> {'OK: parametro 7 recuperavel' if rec_sim_7 > rec_sim_3 and rec_sim_7 > rec_sim_11 else 'FALHA: nao distingue parametro'}")

# TESTE 2: Recuperar parametro 11 da funcao composta
print()
print("--- TESTE 2: Recuperar num_11 de is_divisible_by_7_and_11 ---")
recovered_11_fn = unbind(function_7_11, atoms['role_param2'])
rec2_sim_7 = cosine(recovered_11_fn, atoms['num_7'])
rec2_sim_11 = cosine(recovered_11_fn, atoms['num_11'])
rec2_sim_3 = cosine(recovered_11_fn, atoms['num_3'])
print(f"  sim(num_11) = {rec2_sim_11:.3f}  (deveria ser o mais alto)")
print(f"  sim(num_7)  = {rec2_sim_7:.3f}")
print(f"  sim(num_3)  = {rec2_sim_3:.3f}")
print(f"  -> {'OK: parametro 11 recuperavel' if rec2_sim_11 > rec2_sim_7 and rec2_sim_11 > rec2_sim_3 else 'FALHA: nao distingue parametro'}")

# TESTE 3: Similaridade entre funcoes preserva semantica
print()
print("--- TESTE 3: Similaridade entre funcoes ---")
for name1, vec1 in hdv_experimental.items():
    for name2, vec2 in hdv_experimental.items():
        if name1 < name2:
            s = cosine(vec1, vec2)
            print(f"  sim({name1}, {name2}) = {s:.3f}")

# TESTE 4: Busca no HDV experimental por similaridade com template
print()
print("--- TESTE 4: Busca por template composicional ---")
# Query: "quero uma funcao que usa mod com 7 e 11"
query = bundle([
    bind(atoms['role_operation'], atoms['op_mod']),
    bind(atoms['role_param1'], atoms['num_7']),
    bind(atoms['role_param2'], atoms['num_11']),
])
results = []
for name, vec in hdv_experimental.items():
    s = cosine(query, vec)
    results.append((s, name))
results.sort(reverse=True)
print(f"  Query: role_operation * op_mod + role_param1 * num_7 + role_param2 * num_11")
for s, name in results:
    print(f"  sim = {s:.3f}  -> {name}")

# ===============================================================
# FASE 4: COMPOSICAO DE NOVA FUNCAO (A FORJA EM PROTOTIPO)
# ===============================================================
print()
print("=" * 70)
print("FASE 4: COMPOSICAO DE NOVA FUNCAO (FORJA PROTOTIPO)")
print("=" * 70)

# Simular o que a Forja faria:
# 1. Recebe intencao: "def is_divisible_by_7_and_11(n):"
# 2. Extrai: template=divisible_by_X_and_Y, X=7, Y=11
# 3. Recupera template do HDV
# 4. Substitui parametros via unbind/bind

# Passo 1: Extrair template da funcao existente
template_7_11 = function_7_11.clone()  # copia

# Passo 2: Remover parametros antigos e inserir novos
# Na pratica: unbind(param1) descarta o valor, bind com novo valor
# Mas precisamos remover o binding antigo primeiro...
# Em HDC: nao ha "remocao" pura. Mas podemos refazer.

# Abordagem: construir nova funcao do zero com novos parametros
def compose_function(name_atom: torch.Tensor,
                     param1: torch.Tensor,
                     param2: torch.Tensor,
                     operation: torch.Tensor,
                     return_type: torch.Tensor) -> torch.Tensor:
    """Compoe uma nova funcao a partir de componentes atomicos."""
    return bundle([
        bind(atoms['role_name'], name_atom),
        bind(atoms['role_param1'], param1),
        bind(atoms['role_param2'], param2),
        bind(atoms['role_operation'], operation),
        bind(atoms['role_return_type'], return_type),
    ])

# Compor: is_divisible_by_77_and_154
new_fn = compose_function(
    name_atom=atoms['word_divisible'],
    param1=atoms['num_77'],
    param2=atoms['num_154'],
    operation=atoms['op_mod'],
    return_type=atoms['type_boolean'],
)

print(f"  Nova funcao composta: is_divisible_by_77_and_154")
print()

# Verificar: os parametros sao recuperaveis?
new_rec_77 = unbind(new_fn, atoms['role_param1'])
new_rec_154 = unbind(new_fn, atoms['role_param2'])
new_sim_77 = cosine(new_rec_77, atoms['num_77'])
new_sim_154 = cosine(new_rec_154, atoms['num_154'])
new_sim_7 = cosine(new_rec_77, atoms['num_7'])
print(f"  Unbind(role_param1) -> sim(num_77) = {new_sim_77:.3f} (deveria ser alto)")
print(f"  Unbind(role_param1) -> sim(num_7)  = {new_sim_7:.3f} (deveria ser baixo)")
print(f"  Unbind(role_param2) -> sim(num_154) = {new_sim_154:.3f} (deveria ser alto)")

# ===============================================================
# VEREDITO
# ===============================================================
print()
print("=" * 70)
print("VEREDITO DO PROTOTIPO COMPOSICIONAL")
print("=" * 70)

# Checar se todos os testes passaram
checks = []
checks.append(('Auto-similaridade', abs(cosine(atoms['num_7'], atoms['num_7']) - 1.0) < 0.01))
checks.append(('Unbind simples', rec_sim_7 > 0.1))
checks.append(('Parametro 7 > 3', rec_sim_7 > rec_sim_3))
checks.append(('Parametro 11 > 7', rec2_sim_11 > rec2_sim_7))
checks.append(('Composicao 77', new_sim_77 > 0.1))
checks.append(('Composicao 154', new_sim_154 > 0.1))
checks.append(('Nova fn != antiga 7', new_sim_7 < new_sim_77))

print()
for name, passed in checks:
    print(f"  {'PASS' if passed else 'FAIL'}: {name}")
print()

if all(p for _, p in checks):
    print("  >>> PROTOTIPO COMPOSICIONAL FUNCIONA <<<")
    print()
    print("  A representacao HDC com bind/unbind suporta:")
    print("    - Atomicos recuperaveis")
    print("    - Parametros extraiveis de funcoes compostas")
    print("    - Composicao de novas funcoes com parametros arbitrarios")
    print()
    print("  PROXIMO PASSO: Migrar HDV e Generator para este espaco.")
else:
    print("  >>> PROTOTIPO REQUER AJUSTES <<<")
    print("  Nem todos os testes passaram. Analisar causas.")

print()
print(f"  Capacidade do prototipo: {len(atoms)} atomicos, {len(hdv_experimental)} funcoes")
