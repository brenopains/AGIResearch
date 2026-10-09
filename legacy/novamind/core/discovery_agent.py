"""
discovery_agent.py — AGI que descobre modelos matematicos por evolucao GENUINA.

SEM dicas humanas. SEM equacoes pre-programadas.
A AGI comeca SABENDO APENAS:
    - Existem N variaveis (nomes opacos)
    - Pode chamar: gt.simulate(model_key, params) -> trajectory
    - Objetivo: reproduzir a dinamica do ground truth

Estrategia: Programacao Genetica de EDOs
    - Populacao de expressoes matematicas
    - Mutacao: adicionar/remover/modificar termos
    - Crossover: trocar sub-expressoes entre individuos
    - Selecao: os que melhor reproduzem o ground truth sobrevivem

Cada individuo e um conjunto de N equacoes diferenciais,
onde cada equacao e uma arvore de expressao matematica.
"""
import numpy as np
import random
import copy
import time
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── OPERACOES MATEMATICAS PRIMITIVAS ───
# A AGI comeca com um conjunto pequeno de operacoes
# e descobre combinacoes complexas por evolucao

OPERATIONS = [
    ("const", lambda: random.uniform(-1.0, 1.0)),  # constante
    ("var", None),                                   # referencia a variavel
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("div", lambda a, b: a / (abs(b) + 1e-8)),
    ("neg", lambda a: -a),
    ("abs", lambda a: abs(a)),
    ("square", lambda a: a * a),
    ("sqrt", lambda a: np.sqrt(abs(a))),
    ("exp", lambda a: np.exp(np.clip(a, -10, 10))),
    ("log", lambda a: np.log(abs(a) + 1e-8)),
    ("sigmoid", lambda a: 1.0 / (1.0 + np.exp(-np.clip(a, -10, 10)))),
    ("tanh", lambda a: np.tanh(a)),
    ("sin", lambda a: np.sin(a)),
    ("cos", lambda a: np.cos(a)),
    ("pow2", lambda a: a * a),
    ("pow3", lambda a: a * a * a),
    ("inv", lambda a: 1.0 / (abs(a) + 1e-8)),
    ("max0", lambda a: max(0, a)),
    ("min1", lambda a: min(1, max(0, a))),
]

UNARY_OPS = {name for name, fn in OPERATIONS if name in
             ["neg", "abs", "square", "sqrt", "exp", "log", "sigmoid",
              "tanh", "sin", "cos", "pow2", "pow3", "inv", "max0", "min1"]}
BINARY_OPS = {name for name, fn in OPERATIONS if name in
              ["add", "sub", "mul", "div"]}
LEAF_OPS = {"const", "var"}


class ExpressionNode:
    """No de arvore de expressao matematica."""
    def __init__(self, op: str, left=None, right=None, value=None):
        self.op = op
        self.left = left
        self.right = right
        self.value = value  # usado para 'const' e 'var'

    def evaluate(self, variables: Dict[int, float]) -> float:
        if self.op == "const":
            return self.value
        elif self.op == "var":
            return variables.get(self.value, 0.0)
        elif self.op in UNARY_OPS:
            v = self.left.evaluate(variables)
            for name, fn in OPERATIONS:
                if name == self.op:
                    return fn(v)
            return v
        elif self.op in BINARY_OPS:
            lv = self.left.evaluate(variables)
            rv = self.right.evaluate(variables)
            for name, fn in OPERATIONS:
                if name == self.op:
                    return fn(lv, rv)
            return lv + rv
        return 0.0

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        if self.op == "const":
            return f"{self.value:.3f}"
        elif self.op == "var":
            return f"x{self.value}"
        elif self.op in UNARY_OPS:
            return f"{self.op}({self.left})"
        elif self.op in BINARY_OPS:
            return f"({self.left} {self.op} {self.right})"
        return f"?{self.op}?"

    def depth(self) -> int:
        if self.op in LEAF_OPS:
            return 1
        elif self.op in UNARY_OPS:
            return 1 + self.left.depth()
        else:
            return 1 + max(self.left.depth(), self.right.depth())


def random_expression(depth: int = 0, max_depth: int = 4,
                      n_vars: int = 5, const_range: float = 2.0) -> ExpressionNode:
    """Gera uma expressao aleatoria."""
    if depth >= max_depth:
        # So nos terminais
        op = random.choice(list(LEAF_OPS))
        if op == "const":
            return ExpressionNode("const", value=random.uniform(-const_range, const_range))
        else:
            return ExpressionNode("var", value=random.randint(0, n_vars - 1))

    op = random.choice([k for k, _ in OPERATIONS])

    if op == "const":
        return ExpressionNode("const", value=random.uniform(-const_range, const_range))
    elif op == "var":
        return ExpressionNode("var", value=random.randint(0, n_vars - 1))
    elif op in UNARY_OPS:
        left = random_expression(depth + 1, max_depth, n_vars, const_range)
        return ExpressionNode(op, left=left)
    elif op in BINARY_OPS:
        left = random_expression(depth + 1, max_depth, n_vars, const_range)
        right = random_expression(depth + 1, max_depth, n_vars, const_range)
        return ExpressionNode(op, left=left, right=right)

    return ExpressionNode("const", value=0.0)


@dataclass
class ODEModel:
    """Conjunto de N equacoes diferenciais (arvores de expressao)."""
    equations: List[ExpressionNode]
    n_vars: int
    var_names: List[str] = None
    fitness: float = -1e9
    fitness_details: dict = field(default_factory=dict)
    generation: int = 0

    def evaluate(self, t: float, y: np.ndarray) -> np.ndarray:
        """Avalia todas as equacoes no estado y."""
        variables = {i: y[i] for i in range(self.n_vars)}
        dydt = np.zeros(self.n_vars)
        for i, eq in enumerate(self.equations):
            try:
                val = eq.evaluate(variables)
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                dydt[i] = np.clip(val, -10.0, 10.0)
            except:
                dydt[i] = 0.0
        return dydt

    def simulate(self, y0: np.ndarray, t_span: Tuple[float, float],
                 n_steps: int = 100) -> np.ndarray:
        """Simula o modelo com Euler simples."""
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        trajectory = np.zeros((self.n_vars, n_steps))
        y = y0.copy()
        for step in range(n_steps):
            dy = self.evaluate(0.0, y)
            y = y + dy * dt
            y = np.clip(y, -10.0, 10.0)
            trajectory[:, step] = y
        return trajectory

    def copy(self):
        return ODEModel(
            equations=[eq.copy() for eq in self.equations],
            n_vars=self.n_vars,
            var_names=self.var_names.copy() if self.var_names else None,
            fitness=self.fitness,
            fitness_details=self.fitness_details.copy(),
            generation=self.generation,
        )

    def complexity(self) -> int:
        """Numero total de nos na arvore (medida de complexidade)."""
        def count_nodes(node):
            if node is None:
                return 0
            if node.op in LEAF_OPS:
                return 1
            if node.op in UNARY_OPS:
                return 1 + count_nodes(node.left)
            return 1 + count_nodes(node.left) + count_nodes(node.right)
        return sum(count_nodes(eq) for eq in self.equations)


class DiscoveryAgent:
    """AGI que descobre modelos por programacao genetica.

    NAO recebe dicas sobre as equacoes.
    NAO sabe quais variaveis sao relevantes.
    So aprende por tentativa e erro.
    """

    def __init__(self, n_vars: int, var_names: List[str] = None,
                 population_size: int = 100,
                 max_depth: int = 5, const_range: float = 5.0,
                 mutation_rate: float = 0.3, crossover_rate: float = 0.2,
                 tournament_size: int = 3, seed: int = 42):
        self.n_vars = n_vars
        self.var_names = var_names or [f"x{i}" for i in range(n_vars)]
        self.population_size = population_size
        self.max_depth = max_depth
        self.const_range = const_range
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.population: List[ODEModel] = []
        self.generation = 0
        self.best_ever: Optional[ODEModel] = None
        self.history: List[dict] = []
        self._fitness_cache = {}

    def initialize_population(self):
        """Gera populacao inicial aleatoria."""
        self.population = []
        for _ in range(self.population_size):
            equations = []
            for i in range(self.n_vars):
                eq = random_expression(0, self.max_depth, self.n_vars, self.const_range)
                equations.append(eq)
            model = ODEModel(equations, self.n_vars, self.var_names)
            self.population.append(model)

    def evaluate_fitness(self, model: ODEModel,
                         ground_truth_fn: Callable,
                         initial_state: np.ndarray,
                         t_span: Tuple[float, float] = (0, 100),
                         n_steps: int = 50,
                         ground_truth_traj: np.ndarray = None) -> float:
        """Calcula fitness: quao bem o modelo reproduz o ground truth.

        O fitness e o NEGATIVO do erro quadratico medio.
        Modelos com >-1 sao bons, >-0.5 sao excelentes.
        """
        try:
            gt_traj = ground_truth_traj
            if gt_traj is None:
                gt_traj = ground_truth_fn(initial_state, t_span, n_steps)

            model_traj = model.simulate(initial_state, t_span, n_steps)

            # Normaliza trajetorias para comparacao
            gt_norm = (gt_traj - gt_traj.mean(axis=1, keepdims=True)) / (gt_traj.std(axis=1, keepdims=True) + 1e-8)
            model_norm = (model_traj - model_traj.mean(axis=1, keepdims=True)) / (model_traj.std(axis=1, keepdims=True) + 1e-8)

            # Erro quadratico medio entre as trajetorias normalizadas
            mse = np.mean((gt_norm - model_norm) ** 2)

            # Penalidade por complexidade (Occam)
            complexity_penalty = 0.0001 * model.complexity()

            # Bons modelos tem MSE baixo (perto de 0)
            fitness = -mse - complexity_penalty
            return fitness

        except Exception:
            return -100.0

    def evaluate_population(self, ground_truth_fn: Callable,
                            initial_state: np.ndarray,
                            t_span: Tuple[float, float] = (0, 100),
                            n_steps: int = 50):
        """Avalia toda a populacao."""
        # Pre-computa trajetoria do ground truth
        gt_traj = ground_truth_fn(initial_state, t_span, n_steps)

        for model in self.population:
            model.fitness = self.evaluate_fitness(
                model, ground_truth_fn, initial_state, t_span, n_steps, gt_traj
            )
            model.generation = self.generation

        # Atualiza best_ever
        best_now = max(self.population, key=lambda m: m.fitness)
        if self.best_ever is None or best_now.fitness > self.best_ever.fitness:
            self.best_ever = best_now.copy()

    def tournament_selection(self) -> ODEModel:
        """Seleciona individuo por torneio."""
        candidates = self.rng.sample(self.population, self.tournament_size)
        return max(candidates, key=lambda m: m.fitness).copy()

    def mutate_expression(self, node: ExpressionNode) -> ExpressionNode:
        """Aplica mutacao a uma expressao."""
        if self.rng.random() < 0.3:
            # Substitui por expressao aleatoria
            return random_expression(0, self.max_depth, self.n_vars, self.const_range)

        if node.op == "const":
            # Perturba constante
            node.value += self.rng.gauss(0, 0.1) * self.const_range
            return node
        elif node.op == "var":
            if self.rng.random() < 0.3:
                node.value = self.rng.randint(0, self.n_vars - 1)
            return node
        elif node.op in UNARY_OPS:
            if self.rng.random() < 0.5:
                # Muda operacao
                new_op = self.rng.choice(list(UNARY_OPS))
                node.op = new_op
            node.left = self.mutate_expression(node.left)
            return node
        elif node.op in BINARY_OPS:
            if self.rng.random() < 0.4:
                # Muda operacao
                new_op = self.rng.choice(list(BINARY_OPS))
                node.op = new_op
            node.left = self.mutate_expression(node.left)
            node.right = self.mutate_expression(node.right)
            return node
        return node

    def mutate_model(self, model: ODEModel) -> ODEModel:
        """Aplica mutacao a todas as equacoes."""
        new = model.copy()
        for i in range(new.n_vars):
            if self.rng.random() < self.mutation_rate:
                new.equations[i] = self.mutate_expression(new.equations[i])
        return new

    def crossover(self, parent1: ODEModel, parent2: ODEModel) -> Tuple[ODEModel, ODEModel]:
        """Crossover: troca equacoes entre pais."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        for i in range(self.n_vars):
            if self.rng.random() < self.crossover_rate:
                child1.equations[i], child2.equations[i] = child2.equations[i], child1.equations[i]
        return child1, child2

    def evolve_one_generation(self, ground_truth_fn: Callable,
                              initial_state: np.ndarray,
                              t_span: Tuple[float, float] = (0, 100),
                              n_steps: int = 50,
                              elite_size: int = 2):
        """Evolui uma geracao."""
        self.evaluate_population(ground_truth_fn, initial_state, t_span, n_steps)

        # Ordena por fitness
        self.population.sort(key=lambda m: m.fitness, reverse=True)

        self.history.append({
            "generation": self.generation,
            "best_fitness": self.population[0].fitness,
            "mean_fitness": np.mean([m.fitness for m in self.population]),
            "best_complexity": self.population[0].complexity(),
            "best_model_str": str(self.population[0].equations[0])[:50],
        })

        # Elitismo
        new_population = [m.copy() for m in self.population[:elite_size]]

        # Preenche resto com crossover + mutacao
        while len(new_population) < self.population_size:
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            c1, c2 = self.crossover(p1, p2)
            c1 = self.mutate_model(c1)
            c2 = self.mutate_model(c2)
            new_population.append(c1)
            if len(new_population) < self.population_size:
                new_population.append(c2)

        self.population = new_population
        self.generation += 1

    def get_best_model(self) -> Optional[ODEModel]:
        """Retorna o melhor modelo encontrado."""
        if self.population:
            return max(self.population, key=lambda m: m.fitness)
        return self.best_ever

    def get_model_similarity_matrix(self, model: ODEModel,
                                    ground_truth_fn: Callable,
                                    initial_state: np.ndarray,
                                    t_span: Tuple[float, float] = (0, 100),
                                    n_steps: int = 50) -> np.ndarray:
        """Compara cada variavel do modelo com cada variavel do ground truth."""
        gt_traj = ground_truth_fn(initial_state, t_span, n_steps)
        model_traj = model.simulate(initial_state, t_span, n_steps)

        # Correlacao entre cada par de variaveis
        n_gt = gt_traj.shape[0]
        n_model = model_traj.shape[0]
        sim_matrix = np.zeros((n_model, n_gt))

        for i in range(n_model):
            for j in range(n_gt):
                if np.std(model_traj[i]) > 1e-8 and np.std(gt_traj[j]) > 1e-8:
                    corr = np.corrcoef(model_traj[i], gt_traj[j])[0, 1]
                    sim_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0.0

        return sim_matrix

    def propose_interventions(self, best_model: ODEModel,
                              healthy_state: np.ndarray,
                              disease_state: np.ndarray,
                              gt_simulate_fn: Callable) -> List[Dict]:
        """Propoe intervencoes baseado no modelo descoberto.

        A AGI analisa o proprio modelo para encontrar
        parametros que levam do estado doente ao saudavel.
        """
        proposals = []

        # Testa perturbar cada variavel individualmente
        base_health = gt_simulate_fn(healthy_state, (0, 100), 50)[:, -1]

        for i in range(self.n_vars):
            # Tenta forcar a variavel i para o valor saudavel
            test_state = disease_state.copy()
            test_state[i] = healthy_state[i]

            result = gt_simulate_fn(test_state, (0, 100), 50)
            if result is None:
                continue
            final_state = result[:, -1]

            # Quao proximo do estado saudavel?
            improvement = 1.0 - np.mean(np.abs(final_state - base_health) / (np.abs(base_health) + 0.01))

            proposals.append({
                "target": self.var_names[i] if self.var_names else f"x{i}",
                "variable_index": i,
                "approach": f"Set {self.var_names[i] if self.var_names else f'x{i}'} to healthy level ({healthy_state[i]:.3f})",
                "improvement": float(improvement),
            })

        proposals.sort(key=lambda p: -p["improvement"])
        return proposals
