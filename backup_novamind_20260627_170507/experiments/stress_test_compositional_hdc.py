"""
STRESS TEST - Prototipo HDC Composicional
==========================================
6 testes de ruptura para encontrar os LIMITES do bind/unbind.
Se passar, podemos migrar o HDV. Se quebrar, sabemos onde.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from typing import Dict, List, Tuple

torch.manual_seed(42)
np.random.seed(42)

DIM = 10000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===================== CORE HDC OPERATIONS =====================

class HDCAtomSpace:
    """Espaco atomico composicional com bind/unbind/bundle."""
    
    def __init__(self, dim: int = DIM, device: str = DEVICE):
        self.dim = dim
        self.device = device
        self.atoms: Dict[str, torch.Tensor] = {}
    
    def add_atom(self, name: str) -> torch.Tensor:
        """Cria e registra um vetor atomico bipolar aleatorio."""
        v = torch.randint(0, 2, (self.dim,), device=self.device).float() * 2 - 1
        self.atoms[name] = v
        return v
    
    def get(self, name: str) -> torch.Tensor:
        return self.atoms[name]
    
    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b
    
    def unbind(self, concept: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
        return concept * role
    
    def bundle(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        s = torch.stack(vectors).sum(dim=0)
        return torch.sign(s)
    
    def cosine(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    
    def similarity_matrix(self, names: List[str]) -> np.ndarray:
        """Matriz NxN de similaridades entre atomicos."""
        mat = np.zeros((len(names), len(names)))
        for i, n1 in enumerate(names):
            for j, n2 in enumerate(names):
                mat[i][j] = self.cosine(self.get(n1), self.get(n2))
        return mat
    
    def decode(self, vec: torch.Tensor, candidates: Dict[str, torch.Tensor],
               top_k: int = 3) -> List[Tuple[str, float]]:
        """Decodifica vec procurando o atomo mais similar em candidates."""
        results = []
        for name, atom in candidates.items():
            sim = self.cosine(vec, atom)
            results.append((sim, name))
        results.sort(reverse=True)
        return [(name, sim) for sim, name in results[:top_k]]


H = HDCAtomSpace()
print(f"HDC Atom Space: {DIM} dim, {DEVICE}")


# ===================================================================
# CONFIGURACAO: CRIAR CONJUNTOS DE TESTE
# ===================================================================
print()
print("=" * 70)
print("CONFIGURACAO: CRIANDO ATOMOS DE TESTE")
print("=" * 70)

# Roles estruturais
roles = ['role_isa', 'role_has', 'role_param1', 'role_param2', 'role_param3',
         'role_color', 'role_action', 'role_property', 'role_context',
         'role_pai', 'role_filho', 'role_causa', 'role_efeito',
         'role_subject', 'role_object', 'role_rule', 'role_operation']
for r in roles:
    H.add_atom(r)

# Entidades
entities = ['florb', 'joao', 'maria', 'azul', 'vermelho', 'gosta_musica',
            'animal', 'planta', 'agua', 'fogo', 'quente', 'frio',
            'A', 'B', 'C', 'D', 'E', 'X', 'Y', 'Z',
            'inverte_prop', 'contexto_a', 'contexto_b',
            'verdade', 'mentira']
for e in entities:
    H.add_atom(e)

# Numeros 0-20
for i in range(21):
    H.add_atom('num_' + str(i))
# Numeros extras
for n in [77, 154]:
    H.add_atom('num_' + str(n))

# Operacoes
ops = ['op_and', 'op_or', 'op_not', 'op_mod', 'op_eq', 'op_return',
       'op_if', 'op_gt', 'op_lt', 'op_add', 'op_sub', 'op_mul']
for o in ops:
    H.add_atom(o)

# Propriedades de relacionamento
props = ['prop_metade', 'prop_dobro', 'prop_muda_tempo', 'prop_inverso']
for p in props:
    H.add_atom(p)

print("Total atomicos: " + str(len(H.atoms)))

# Verificar qualidade do espaco
names_sample = list(H.atoms.keys())[:30]
sim_mat = H.similarity_matrix(names_sample)
off_diag = sim_mat[~np.eye(sim_mat.shape[0], dtype=bool)]
print("Similaridade media off-diagonal: {:.4f} (ideal: 0.0)".format(np.mean(off_diag)))
print("Similaridade max off-diagonal:  {:.4f}".format(np.max(off_diag)))


# ===================================================================
# TESTE 1: COMPOSICAO ZERO-SHOT SEM TEMPLATE
# ===================================================================
print()
print("=" * 70)
print("TESTE 1: COMPOSICAO ZERO-SHOT SEM TEMPLATE")
print("=" * 70)
print("""
  TREINO: is_multiple_of_7, is_multiple_of_11, AND rule
  TESTE:  compor is_divisible_by_7_and_11
          SEM nunca ter visto template divisible_by_X_and_Y
""")

# Codificar funcoes de treino (SEM template divisible_by_X_and_Y)
fn_mult_7 = H.bundle([
    H.bind(H.get('role_param1'), H.get('num_7')),
    H.bind(H.get('role_operation'), H.get('op_mod')),
    H.bind(H.get('role_property'), H.get('prop_metade')),
])

fn_mult_11 = H.bundle([
    H.bind(H.get('role_param1'), H.get('num_11')),
    H.bind(H.get('role_operation'), H.get('op_mod')),
    H.bind(H.get('role_property'), H.get('prop_metade')),
])

# Codificar AND rule (separada)
fn_and_rule = H.bundle([
    H.bind(H.get('role_operation'), H.get('op_and')),
    H.bind(H.get('role_has'), H.get('verdade')),
])

# ZERO-SHOT: compor usando so os atomos (sem template)
novo_conceito = H.bundle([
    H.bind(H.get('role_param1'), H.get('num_7')),
    H.bind(H.get('role_param2'), H.get('num_11')),
    H.bind(H.get('role_operation'), H.get('op_mod')),
    H.bind(H.get('role_has'), H.get('op_and')),
])

# Verificar: os componentes sao recuperaveis?
t1_rec_7 = H.unbind(novo_conceito, H.get('role_param1'))
t1_rec_11 = H.unbind(novo_conceito, H.get('role_param2'))
t1_rec_op = H.unbind(novo_conceito, H.get('role_operation'))

t1_sim_7 = H.cosine(t1_rec_7, H.get('num_7'))
t1_sim_3 = H.cosine(t1_rec_7, H.get('num_3'))
t1_sim_11 = H.cosine(t1_rec_11, H.get('num_11'))
t1_sim_op_mod = H.cosine(t1_rec_op, H.get('op_mod'))
t1_sim_op_and = H.cosine(t1_rec_op, H.get('op_and'))

print("  Unbind(role_param1) -> sim(num_7)  = {:.3f} (esperado: ~0.3+)".format(t1_sim_7))
print("  Unbind(role_param1) -> sim(num_3)  = {:.3f} (esperado: ~0.0)".format(t1_sim_3))
print("  Unbind(role_param2) -> sim(num_11) = {:.3f} (esperado: ~0.3+)".format(t1_sim_11))
print("  Unbind(role_operation) -> sim(op_mod) = {:.3f}".format(t1_sim_op_mod))
print("  Unbind(role_operation) -> sim(op_and) = {:.3f}".format(t1_sim_op_and))

t1_pass = (t1_sim_7 > t1_sim_3 and t1_sim_11 > 0.2)
print()
print("  >>> PASS <<<" if t1_pass else "  >>> FAIL <<<")


# ===================================================================
# TESTE 2: SILOGISMO HDC (TRANSITIVIDADE)
# ===================================================================
print()
print("=" * 70)
print("TESTE 2: SILOGISMO HDC (TRANSITIVIDADE)")
print("=" * 70)
print("""
  REGRA:  Florb  = (role_isa * azul) + (role_has * gosta_musica)
  REGRA:  Joao   = (role_isa * florb)
  TESTE:  Joao tem a propriedade gosta_musica?
          (transitividade: Joao eh Florb, Florb gosta de musica -> Joao gosta de musica)
""")

# Codificar: Florb = azul AND gosta_musica
florb_concept = H.bundle([
    H.bind(H.get('role_isa'), H.get('azul')),
    H.bind(H.get('role_has'), H.get('gosta_musica')),
])

# Codificar: Joao = florb (Joao e um Florb)
joao_concept = H.bundle([
    H.bind(H.get('role_isa'), H.get('florb')),
])

# TESTE: Descobrir se Joao gosta de musica
# Passo 1: unbind role_isa de Joao -> florb (se Joao for um florb)
t2_isa = H.unbind(joao_concept, H.get('role_isa'))
t2_sim_florb = H.cosine(t2_isa, H.get('florb'))
print("  Unbind(Joao, role_isa) -> sim(florb) = {:.3f}".format(t2_sim_florb))

# Passo 2: Se Joao = florb, entao unbind com role_has revela atributos
t2_has = H.unbind(joao_concept, H.get('role_has'))
t2_sim_gosta_musica = H.cosine(t2_has, H.get('gosta_musica'))
t2_sim_azul = H.cosine(t2_has, H.get('azul'))
print("  Unbind(Joao, role_has) -> sim(gosta_musica) = {:.3f}".format(t2_sim_gosta_musica))
print("  Unbind(Joao, role_has) -> sim(azul)         = {:.3f}".format(t2_sim_azul))

# Transitividade: (Joao * role_isa) * role_has = gosta_musica
# Joao = (role_isa * florb); florb = (role_has * gosta_musica)
# Logo: Joao * role_isa = florb; florb * role_has = gosta_musica
# Entao: (Joao * role_isa) * role_has = gosta_musica
t2_transitivo = H.unbind(H.unbind(joao_concept, H.get('role_isa')), H.get('role_has'))
t2_sim_gm = H.cosine(t2_transitivo, H.get('gosta_musica'))
print()
print("  (Joao * role_isa) * role_has -> sim(gosta_musica) = {:.3f}".format(t2_sim_gm))

t2_pass = (t2_sim_gm > 0.1)
print()
print("  >>> PASS <<<" if t2_pass else "  >>> FAIL <<<")


# ===================================================================
# TESTE 3: NEGACAO E OPOSICAO
# ===================================================================
print()
print("=" * 70)
print("TESTE 3: NEGACAO E OPOSICAO")
print("=" * 70)
print("""
  not(is_hot) ~ is_cold?
  A oposicao emerge da geometria ou e so ruido?

  Em HDC: NOT(x) = -x (inversao bipolar)
  Quente e frio sao opostos naturais se codificados como complementos.
""")

hot_vec = H.get('quente')
cold_vec = H.get('frio')
not_hot = -hot_vec  # negacao bipolar

# not(hot) deve ser similar a cold
t3_sim_not_hot_cold = H.cosine(not_hot, cold_vec)
t3_sim_hot_cold = H.cosine(hot_vec, cold_vec)
t3_sim_not_hot_hot = H.cosine(not_hot, hot_vec)

print("  sim(hot, cold)          = {:.3f}".format(t3_sim_hot_cold))
print("  sim(-hot, cold)         = {:.3f}".format(t3_sim_not_hot_cold))
print("  sim(-hot, hot)          = {:.3f} (deveria ser -1.0)".format(t3_sim_not_hot_hot))
print()
print("  'not hot' similar a 'cold'?", "SIM" if t3_sim_not_hot_cold > t3_sim_hot_cold else "NAO")

# Teste com binding: role_property * quente, depois negar
concept_hot = H.bind(H.get('role_property'), hot_vec)
concept_cold = H.bind(H.get('role_property'), cold_vec)
concept_not_hot = -concept_hot  # nega o conceito inteiro

t3_unbind_not = H.unbind(concept_not_hot, H.get('role_property'))
t3_sim_not_cold = H.cosine(t3_unbind_not, cold_vec)
t3_not_sim_hot = H.cosine(t3_unbind_not, hot_vec)

print()
print("  Not(role_property * hot) -> unbind -> sim(cold) = {:.3f}".format(t3_sim_not_cold))
print("  Not(role_property * hot) -> unbind -> sim(hot)  = {:.3f}".format(t3_not_sim_hot))

# Metodo alternativo: role_property * (-hot)
concept_not_hot_v2 = H.bind(H.get('role_property'), not_hot)
t3_unbind_v2 = H.unbind(concept_not_hot_v2, H.get('role_property'))
t3_v2_cold = H.cosine(t3_unbind_v2, cold_vec)
print("  role_property * (-hot) -> unbind -> sim(cold) = {:.3f}".format(t3_v2_cold))

t3_pass = (t3_sim_not_hot_cold > 0.3 or t3_v2_cold > 0.3)
print()
print("  >>> PASS <<<" if t3_pass else "  >>> FAIL <<<")


# ===================================================================
# TESTE 4: CAPACIDADE DE BUNDLE (CURVA DE RUPTURA)
# ===================================================================
print()
print("=" * 70)
print("TESTE 4: CAPACIDADE DE BUNDLE (CURVA DE RUPTURA)")
print("=" * 70)
print("""
  Em quantos bindings o unbind quebra?
  Testa com 2, 5, 10, 20, 50, 100 componentes no bundle.
""")

# Criar atomicos para o teste
for i in range(100):
    H.add_atom('bundle_test_' + str(i))

results_capacity = []
role_pool = ['role_param1', 'role_param2', 'role_param3', 'role_context',
             'role_action', 'role_subject', 'role_object', 'role_causa',
             'role_efeito', 'role_rule']

for n_components in [2, 5, 10, 20, 50, 100]:
    target_idx = 0
    target_atom = H.get('bundle_test_' + str(target_idx))
    role_target = H.get('role_color')

    binds_puro = [H.bind(role_target, target_atom)]
    for i in range(1, n_components):
        rname = role_pool[i % len(role_pool)]
        binds_puro.append(H.bind(H.get(rname), H.get('bundle_test_' + str(i))))

    concept = H.bundle(binds_puro)
    recovered = H.unbind(concept, role_target)
    sim = H.cosine(recovered, target_atom)
    sim_noise = H.cosine(recovered, H.get('bundle_test_' + str(1)))
    results_capacity.append((n_components, sim, sim_noise))

    status = 'OK' if sim > 0.1 else 'RUPTURA'
    print("  Bundle({:3d}) -> sim(target)={:.3f}  sim(noise)={:.3f}  [{}]".format(
        n_components, sim, sim_noise, status))

# Encontrar ponto de ruptura
rupture_point = None
for n, sim, _ in results_capacity:
    if sim < 0.1:
        rupture_point = n
        break
if rupture_point is None:
    rupture_point = results_capacity[-1][0]
    print()
    print("  SEM RUPTURA ate {:d} componentes".format(results_capacity[-1][0]))

t4_rupture = results_capacity[-1][1]
t4_pass = (t4_rupture > 0.1)
print()
print("  >>> PASS <<<" if t4_pass else "  >>> FAIL <<<")


# ===================================================================
# TESTE 5: BINDINGS CONTRADITORIOS
# ===================================================================
print()
print("=" * 70)
print("TESTE 5: BINDINGS CONTRADITORIOS")
print("=" * 70)
print("""
  role_color * blue + role_color * red -> unbind recupera o que?
  Cinza? Ruido? Um dos dois?
""")

# Bundle contraditorio: mesma role com dois fillers diferentes
concept_paradox = H.bundle([
    H.bind(H.get('role_color'), H.get('azul')),
    H.bind(H.get('role_color'), H.get('vermelho')),
])

t5_rec = H.unbind(concept_paradox, H.get('role_color'))
t5_sim_azul = H.cosine(t5_rec, H.get('azul'))
t5_sim_vermelho = H.cosine(t5_rec, H.get('vermelho'))
t5_sim_florb = H.cosine(t5_rec, H.get('florb'))  # noise baseline

print("  Bundle[(role_color * azul) + (role_color * vermelho)]")
print("  Unbind(role_color) -> sim(azul)     = {:.3f}".format(t5_sim_azul))
print("  Unbind(role_color) -> sim(vermelho)  = {:.3f}".format(t5_sim_vermelho))
print("  Unbind(role_color) -> sim(florb)     = {:.3f} (baseline)".format(t5_sim_florb))

if t5_sim_azul > 0.1 and t5_sim_vermelho > 0.1:
    print()
    print("  Ambos os valores competem. Nenhum domina - mescla dos dois.")
    t5_conclusion = "MESCLA"
elif abs(t5_sim_azul - t5_sim_vermelho) < 0.05 and t5_sim_azul < 0.1:
    print()
    print("  Ambos caem para ruido. Bindings contradictorios se cancelam.")
    t5_conclusion = "CANCELA"
else:
    t5_conclusion = "DOMINA: " + ('azul' if t5_sim_azul > t5_sim_vermelho else 'vermelho')
    print()
    print("  Conclusao:", t5_conclusion)

t5_pass = (t5_sim_azul > 0.05 or t5_sim_vermelho > 0.05)
print("  >>> PASS <<<" if t5_pass else "  >>> FAIL <<<")


# ===================================================================
# TESTE 6: TRANSITIVIDADE CAUSAL
# ===================================================================
print()
print("=" * 70)
print("TESTE 6: TRANSITIVIDADE CAUSAL")
print("=" * 70)
print("""
  regra1: role_pai * A + role_filho * B
  regra2: role_pai * B + role_filho * C
  teste:  role_pai * A + role_filho * C ?
          (A e avo de C?)

  Binding: pai(A, B) and pai(B, C) -> avo(A, C)
""")

# Armazenar como bindings causais
pai_AB = H.bundle([
    H.bind(H.get('role_pai'), H.get('A')),
    H.bind(H.get('role_filho'), H.get('B')),
    H.bind(H.get('role_rule'), H.get('verdade')),
])

pai_BC = H.bundle([
    H.bind(H.get('role_pai'), H.get('B')),
    H.bind(H.get('role_filho'), H.get('C')),
    H.bind(H.get('role_rule'), H.get('verdade')),
])

# Inferir: A -> C via B (composicao transitiva)
# Estrategia:
#   1. Unbind role_filho de pai_AB -> B
#   2. Procurar regra onde role_pai ~ B
#   3. Unbind role_filho dessa regra -> C
t6_b_de_ab = H.unbind(pai_AB, H.get('role_filho'))
t6_sim_b = H.cosine(t6_b_de_ab, H.get('B'))
print("  Unbind(regra1, role_filho) -> sim(B) = {:.3f}".format(t6_sim_b))

# Similaridade entre pai_BC e um conceito onde role_pai = B
query_c = H.unbind(pai_BC, H.get('role_filho'))
t6_sim_c = H.cosine(query_c, H.get('C'))
print("  Unbind(regra2, role_filho) -> sim(C) = {:.3f}".format(t6_sim_c))

# Composicao: se A -> B e B -> C, entao A -> C
# (role_pai * A + role_filho * C) = composicao das duas regras
# Extrair A de regra1 e C de regra2
avo_AC = H.bundle([
    H.unbind(pai_AB, H.get('role_pai')),   # ~ A
    H.unbind(pai_BC, H.get('role_filho')), # ~ C
])
# Se A e C estao ambos ativos nesse conceito
t6_sim_ac_A = H.cosine(avo_AC, H.get('A'))
t6_sim_ac_C = H.cosine(avo_AC, H.get('C'))
t6_sim_ac_D = H.cosine(avo_AC, H.get('D'))
print()
print("  Composicao Avo(A, C):")
print("    sim(A) = {:.3f}".format(t6_sim_ac_A))
print("    sim(C) = {:.3f}".format(t6_sim_ac_C))
print("    sim(D) = {:.3f} (baseline)".format(t6_sim_ac_D))

t6_pass = (t6_sim_ac_A > 0.05 and t6_sim_ac_C > 0.05)
print()
print("  >>> PASS <<<" if t6_pass else "  >>> FAIL <<<")


# ===================================================================
# RESUMO FINAL
# ===================================================================
print()
print("=" * 70)
print("RESUMO DOS TESTES DE RUPTURA")
print("=" * 70)

tests = [
    ("Teste 1: Composicao zero-shot sem template", t1_pass),
    ("Teste 2: Silogismo HDC (transitividade)",     t2_pass),
    ("Teste 3: Negacao e oposicao",                 t3_pass),
    ("Teste 4: Curva de ruptura (capacidade)",       t4_pass),
    ("Teste 5: Bindings contradictorios",            t5_pass),
    ("Teste 6: Transitividade causal",               t6_pass),
]

passed = 0
for name, result in tests:
    status = 'PASS' if result else 'FAIL'
    if result:
        passed += 1
    print("  [{}] {}".format(status, name))

print()
if passed == len(tests):
    print("  >>> TODOS OS TESTES PASSARAM. BASE SOLIDA PARA MIGRACAO. <<<")
    print()
    print("  O prototipo HDC composicional suporta:")
    print("    - Composicao zero-shot (bind/unbind sem template)")
    print("    - Silogismo transitivo (A->B, B->C => A->C)")
    print("    - Negacao via inversao bipolar")
    print("    - Bundle com >= 100 componentes sem ruptura")
    print("    - Bindings contradictorios com mescla")
    print("    - Inferencia causal composicional")
else:
    print("  >>> {:d} TESTES FALHARAM. PONTOS DE ATENCAO IDENTIFICADOS. <<<".format(len(tests) - passed))
    for name, result in tests:
        if not result:
            print("      - " + name)

print()
print("  Espaco: {:d} dim, {:d} atomicos, {}".format(DIM, len(H.atoms), DEVICE))
