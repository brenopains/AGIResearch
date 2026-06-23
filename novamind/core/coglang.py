import torch
import torchhd
import math
from typing import Optional, List, Tuple

class Manifold:
    """
    Representação de um vetor hiperdimensional (HDC) esparso/bipolar de 10.000 dimensões.
    Encapsula as propriedades de manifolds topológicos da CogLang.
    """
    DIMENSIONALITY = 10000

    def __init__(self, tensor=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        if tensor is None:
            # Cria um hypervector bipolar aleatório (-1 ou +1) por padrão
            self.tensor = torchhd.random(1, self.DIMENSIONALITY, "MAP", device=self.device).squeeze(0)
        else:
            self.tensor = tensor.to(self.device)
            # Garante que as dimensões estejam corretas
            if self.tensor.dim() > 1:
                self.tensor = self.tensor.view(-1)
            if self.tensor.shape[0] != self.DIMENSIONALITY:
                # Ajusta tamanho por interpolação/preenchimento se necessário
                if self.tensor.shape[0] < self.DIMENSIONALITY:
                    pad = torch.zeros(self.DIMENSIONALITY - self.tensor.shape[0], device=self.device)
                    self.tensor = torch.cat([self.tensor, pad])
                else:
                    self.tensor = self.tensor[:self.DIMENSIONALITY]

    def __mul__(self, other):
        """
        Operação de BINDING (Associação / Produto).
        Na lógica bipolar, o binding é o produto elemento a elemento (XOR lógico).
        Associa duas informações em um novo conceito em complexidade O(1).
        """
        if not isinstance(other, Manifold):
            raise TypeError("A operação de binding (*) exige outro Manifold.")
        return Manifold(self.tensor * other.tensor, device=self.device)

    def __add__(self, other):
        """
        Operação de BUNDLING (Superposição / Adição).
        Combina informações preservando características de ambos (união de conceitos).
        """
        if not isinstance(other, Manifold):
            raise TypeError("A operação de bundling (+) exige outro Manifold.")
        # Soma elemento a elemento
        bundled = self.tensor + other.tensor
        # Opcional: Normalizar para manter a escala
        norm = torch.norm(bundled)
        if norm > 0:
            bundled = bundled / norm * math.sqrt(self.DIMENSIONALITY)
        return Manifold(bundled, device=self.device)

    def similarity(self, other):
        """
        Calcula a similaridade cosseno entre dois manifolds.
        Retorna um valor de -1.0 a 1.0 (onde 1.0 significa identidade/compreensão).
        """
        if not isinstance(other, Manifold):
            raise TypeError("Similaridade exige outro Manifold.")
        # Calcula cosseno usando torchhd
        return float(torchhd.cosine_similarity(self.tensor, other.tensor))

    def inverse(self):
        """
        Inverso multiplicativo para UNBINDING.
        Em vetores bipolares, o inverso de A é o próprio A, pois A * A = 1.
        """
        return Manifold(self.tensor, device=self.device)

    def clone(self):
        return Manifold(self.tensor.clone(), device=self.device)

    def __repr__(self):
        return f"Manifold(dim={self.DIMENSIONALITY}, device={self.device})"


class Pressure:
    """
    Representação da pressão interna (necessidade/urgência) da CogLang.
    Varia de 0.0 (sem urgência) a 1.0 (bloqueio ou ação obrigatória).
    """
    def __init__(self, value=0.0):
        self.value = float(max(0.0, min(1.0, value)))

    def update(self, delta):
        self.value = float(max(0.0, min(1.0, self.value + delta)))
        return self

    def __gt__(self, other):
        if isinstance(other, Pressure):
            return self.value > other.value
        return self.value > float(other)

    def __lt__(self, other):
        if isinstance(other, Pressure):
            return self.value < other.value
        return self.value < float(other)

    def __repr__(self):
        return f"Pressure(val={self.value:.4f})"


class Archetype:
    """
    Um operador cognitivo (função matemática pura ou Spelke gene) que transforma manifolds.
    """
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def apply(self, manifold, *args, **kwargs):
        if not isinstance(manifold, Manifold):
            raise TypeError("Archetypes operam apenas sobre Manifolds.")
        result_tensor = self.func(manifold.tensor, *args, **kwargs)
        return Manifold(result_tensor, device=manifold.device)

    def __call__(self, manifold, *args, **kwargs):
        return self.apply(manifold, *args, **kwargs)

    def __repr__(self):
        return f"Archetype(name={self.name})"


class Stream:
    """
    Fluxo temporal contínuo de Manifolds (gerador/fila iterativa).
    Substitui listas discretas para evitar explosão combinatória.
    """
    def __init__(self, generator_func):
        self.generator = generator_func

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.generator)
        if not isinstance(item, Manifold):
            raise TypeError("Streams de CogLang devem retornar apenas Manifolds.")
        return item


# ─── Role Vectors para Sintaxe (Role-Filler Binding) ─────────
# Cada papel sintático é um vetor HDC fixo.
# Uma frase é armazenada como:
#   ROLE_SUBJECT * word_subj + ROLE_VERB * word_verb + ROLE_OBJECT * word_obj
# Para decodificar: unbind com ROLE_SUBJECT → encontra word_subj

def _make_role_vector(name: str, device=None) -> Manifold:
    """Cria um vetor de papel sintático determinístico a partir de um seed.
    
    Usa hashlib.sha256 em vez de hash() do Python (que é randomizado por
    PYTHONHASHSEED entre processos).
    """
    import hashlib
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=device)
    seed = int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2**31)
    g.manual_seed(seed)
    tensor = torchhd.random(1, Manifold.DIMENSIONALITY, "MAP", generator=g, device=device).squeeze(0)
    return Manifold(tensor, device=device)

ROLE_SUBJECT = _make_role_vector("ROLE_SUBJECT")
ROLE_VERB = _make_role_vector("ROLE_VERB")
ROLE_OBJECT = _make_role_vector("ROLE_OBJECT")
ROLE_CONDITION = _make_role_vector("ROLE_CONDITION")
ROLE_ADJECTIVE = _make_role_vector("ROLE_ADJECTIVE")
ROLE_ADVERB = _make_role_vector("ROLE_ADVERB")
ROLE_PREPOSITION = _make_role_vector("ROLE_PREPOSITION")
ROLE_CONJUNCTION = _make_role_vector("ROLE_CONJUNCTION")
ROLE_NEGATION = _make_role_vector("ROLE_NEGATION")

# Papéis de Programação (Fase 2: AGI Programadora)
ROLE_FUNCTION = _make_role_vector("ROLE_FUNCTION")
ROLE_ARGUMENT = _make_role_vector("ROLE_ARGUMENT")
ROLE_ACTION = _make_role_vector("ROLE_ACTION")

ALL_ROLES = [
    ("suj", ROLE_SUBJECT),
    ("ver", ROLE_VERB),
    ("obj", ROLE_OBJECT),
    ("adj", ROLE_ADJECTIVE),
    ("adv", ROLE_ADVERB),
    ("pre", ROLE_PREPOSITION),
    ("con", ROLE_CONJUNCTION),
    ("neg", ROLE_NEGATION),
    ("cond", ROLE_CONDITION),
    ("func", ROLE_FUNCTION),
    ("arg", ROLE_ARGUMENT),
    ("act", ROLE_ACTION),
]

CODE_ROLES = [
    ("func", ROLE_FUNCTION),
    ("arg", ROLE_ARGUMENT),
    ("act", ROLE_ACTION),
]

def bind_roles(*pairs) -> Manifold:
    """
    Constrói uma frase estruturada no espaço HDC via Role-Filler Binding.
    
    Uso:
        frase = bind_roles(("suj", mf_homem), ("ver", mf_corre), ("obj", mf_cavalo))
        # frase = sign(ROLE_SUBJECT * homem + ROLE_VERB * corre + ROLE_OBJECT * cavalo)
    
    O resultado é binarizado (sign) para manter a bipolaridade do espaço HDC,
    garantindo que o unbinding funcione com similaridade de cosseno normal.
    """
    role_map = dict(ALL_ROLES)
    accumulated = None
    for role_name, word_mf in pairs:
        if role_name not in role_map:
            raise ValueError(f"Role desconhecido: {role_name}. Use: {list(role_map.keys())}")
        role_mf = role_map[role_name]
        bound = role_mf * word_mf
        accumulated = bound if accumulated is None else accumulated + bound
    if accumulated is None:
        raise ValueError("bind_roles precisa de ao menos um par")
    # Binariza para manter bipolaridade (essencial para unbinding)
    binary = torch.sign(accumulated.tensor)
    # Garante que nenhum valor seja 0 (sign retorna 0 para inputs 0)
    binary[binary == 0] = 1
    return Manifold(binary, device=accumulated.device)

def unbind_role(phrase_manifold: Manifold, role_name: str) -> Manifold:
    """
    Extrai a palavra associada a um papel sintático via unbinding.
    
    Uso:
        sujeito = unbind_role(frase, "suj")
        # sujeito ≈ word_homem (o manifold da palavra original)
    
    Args:
        role_name: string em ("suj", "ver", "obj", "adj", ...
    
    Returns:
        Manifold aproximado da palavra original (aproximação por bundling)
    """
    role_map = dict(ALL_ROLES)
    if role_name not in role_map:
        raise ValueError(f"Role desconhecido: {role_name}")
    role_mf = role_map[role_name]
    # Unbinding: phrase * role = (ROLE * word) * ROLE + noise = word + noise
    return phrase_manifold * role_mf

def decode_roles(phrase_manifold: Manifold, vocab_matrix: torch.Tensor, words: list,
                 top_k: int = 3, min_sim: float = 0.1,
                 roles: Optional[List[Tuple[str, object]]] = None) -> list:
    """
    Decodifica uma frase estruturada em papéis sintáticos.
    Para cada papel, encontra a palavra mais similar no vocabulário.
    
    Usa matmul / D (cosseno bipolar) em vez de torchhd.cosine_similarity
    para evitar inconsistências de normalização.
    
    Args:
        roles: lista de (role_name, role_manifold) para usar.
               Default: ALL_ROLES (papéis sintáticos).
               Use CODE_ROLES para decodificar código.
    
    Returns:
        Lista de (role_name, palavra, similaridade)
    """
    results = []
    target_roles = roles if roles is not None else ALL_ROLES
    for role_name, role_mf in target_roles:
        unbound = phrase_manifold * role_mf
        sims = torch.matmul(vocab_matrix, unbound.tensor) / Manifold.DIMENSIONALITY
        best_idx = sims.argmax().item()
        best_sim = sims[best_idx].item()
        if best_sim >= min_sim:
            results.append((role_name, words[best_idx], best_sim))
    return sorted(results, key=lambda x: -x[2])


def calculate_free_energy(pred, target):
    """
    Calcula a energia livre de Helmholtz (frustração cognitiva) baseada no erro de similaridade.
    Varia de 0.0 (perfeita harmonia/compreensão) a 1.0 (máxima contradição).
    """
    if not isinstance(pred, Manifold) or not isinstance(target, Manifold):
        raise TypeError("Ambos os parâmetros devem ser Manifolds.")
    sim = pred.similarity(target)
    # Similaridade varia de -1.0 a 1.0; mapeia 1.0 -> 0.0 de energia livre, e valores menores -> maiores energias livres
    return float(max(0.0, min(1.0, (1.0 - sim) / 2.0 if sim < 0 else 1.0 - sim)))


class ArchetypeForge:
    """
    Forja de Arquétipos (Archetype Forge):
    Instancia operadores neurais e matemáticos sob altos limiares de frustração (Energia Livre).
    """
    def __init__(self):
        self.registry = {}

    def register(self, archetype):
        self.registry[archetype.name] = archetype
        return archetype

    def trigger_forge(self, pred_manifold, target_manifold, name="DynamicArchetype"):
        """
        Gera um novo arquétipo realizando busca combinatória (síntese de programa)
        no espaço de operadores hiperdimensionais para anular a Energia Livre.
        """
        print(f"[FORGE] Frustration threshold exceeded. Initiating HDC program synthesis for: {name}")
        
        # Calcula o resíduo do erro
        error_residue = target_manifold * pred_manifold.inverse()
        
        # Define as primitivas do espaço de busca HDC
        primitives = [
            ("identity", lambda x: x),
            ("inverse", lambda x: x.inverse()),
            ("negate", lambda x: Manifold(-x.tensor, device=x.device)),
            ("shift_r1", lambda x: Manifold(torch.roll(x.tensor, shifts=1), device=x.device)),
            ("shift_l1", lambda x: Manifold(torch.roll(x.tensor, shifts=-1), device=x.device)),
            ("shift_r10", lambda x: Manifold(torch.roll(x.tensor, shifts=10), device=x.device)),
            ("shift_l10", lambda x: Manifold(torch.roll(x.tensor, shifts=-10), device=x.device)),
            ("bind_residue", lambda x: x * error_residue),
        ]
        
        best_sim = -2.0
        best_formula = ""
        best_func = None
        
        # Nível 1: Operadores Simples
        for p_name, p_func in primitives:
            try:
                out = p_func(pred_manifold)
                sim = out.similarity(target_manifold)
                if sim > best_sim:
                    best_sim = sim
                    best_formula = f"{p_name}(x)"
                    best_func = p_func
            except Exception:
                continue
                
        # Nível 2: Composições
        for name1, func1 in primitives:
            for name2, func2 in primitives:
                if name1 == "identity" or name2 == "identity":
                    continue
                try:
                    out = func2(func1(pred_manifold))
                    sim = out.similarity(target_manifold)
                    if sim > best_sim:
                        best_sim = sim
                        best_formula = f"{name2}({name1}(x))"
                        
                        def make_composed(f1, f2):
                            return lambda x: f2(f1(x))
                        best_func = make_composed(func1, func2)
                except Exception:
                    continue
                    
        print(f"[FORGE] Program synthesis completed. Best candidate: {best_formula} | Score (Similarity): {best_sim:.4f}")
        
        # Cria a função wrapper que recebe o tensor brutos do Archetype e roda o melhor programa
        def wrapped_func(x_tensor, *args, **kwargs):
            m = Manifold(x_tensor, device=pred_manifold.device)
            out_m = best_func(m)
            return out_m.tensor
            
        new_archetype = Archetype(name, wrapped_func)
        self.register(new_archetype)
        return new_archetype

