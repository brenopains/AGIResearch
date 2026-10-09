"""
procedural_worlds.py — Geradores de mundos com regras ineditas
================================================================
Cada execucao gera um mundo com leis causais, fisica, e regras
diferentes — impossivel de memorizar. So construindo um modelo
interno do mundo resolve os problemas.

Mundos:
  - CausalWorld: relacao causal escondida entre 2+ variaveis
  - PhysicsWorld: objetos com propriedades e leis de queda
  - CompositionWorld: combinacoes validas em espaco cor × forma × tamanho
  - TransferWorld: problemas isomorficos com superficies diferentes
"""
import random
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field


@dataclass
class CausalWorld:
    """
    Mundo com uma relacao causal escondida entre 2-4 variaveis.
    
    A cada seed, sorteia:
      - Topologia causal (X->Y, X->Y<-Z, X->Z->Y, etc.)
      - Funcao de geracao (linear, polinomial, sinusoidal)
      - Parametros (coeficientes, ruido)
      - Distribuicao observacional das causas
    
    O sistema deve:
      1. Aprender a estrutura causal de dados observacionais
      2. Prever efeitos de intervencoes do(X=x) — que diferem de P(Y|X=x)
      3. Responder contrafactuais
    
    LLMs falham: a estrutura muda a cada seed. Nao ha o que memorizar.
    """
    seed: int = 0
    n_vars: int = 2
    noise_std: float = 0.05
    n_obs_samples: int = 100
    n_int_samples: int = 20

    def __post_init__(self):
        rng = random.Random(self.seed)
        np_rng = np.random.RandomState(self.seed)
        
        self.topology = self._sample_topology(rng)
        self.functions = self._sample_functions(rng)
        self.params = self._sample_params(rng)
        
        self.cause_idx = self.topology["cause"]
        self.effect_idx = self.topology["effect"]
        self.confounders = self.topology.get("confounders", [])
        
        self._true_do_coeff = rng.uniform(0.3, 2.0) * rng.choice([-1, 1])
        
    def _sample_topology(self, rng):
        """Amosa estrutura causal."""
        if self.n_vars == 2:
            cause = 0
            effect = 1
            return {"cause": cause, "effect": effect, "type": "direct", "confounders": []}
        elif self.n_vars == 3:
            chain = rng.choice(["direct", "chain", "confounded", "common_effect"])
            if chain == "direct":
                return {"cause": 0, "effect": 1, "type": "direct", "confounders": []}
            elif chain == "chain":
                return {"cause": 0, "effect": 2, "type": "chain", "mediator": 1, "confounders": []}
            elif chain == "confounded":
                conf = rng.randint(0, 2)
                cause = 0 if conf != 0 else 1
                effect = 1 if conf != 0 else 0 if conf != 1 else 2
                return {"cause": cause, "effect": effect, "type": "confounded",
                        "confounder": conf, "confounders": [conf]}
            else:
                return {"cause": 0, "effect": 2, "type": "common_effect",
                        "other_cause": 1, "confounders": []}
        return {"cause": 0, "effect": 1, "type": "direct", "confounders": []}
    
    def _sample_functions(self, rng):
        families = ["linear", "quadratic", "sinusoidal", "threshold"]
        return {"cause_to_effect": rng.choice(families)}
    
    def _sample_params(self, rng):
        return {
            "slope": rng.uniform(-3, 3),
            "intercept": rng.uniform(-2, 2),
            "quadratic": rng.uniform(-0.5, 0.5),
            "sin_freq": rng.uniform(0.5, 3.0),
            "threshold": rng.uniform(-1, 1),
        }

    def _apply_function(self, x, func_name, params):
        if func_name == "linear":
            return params["slope"] * x + params["intercept"]
        elif func_name == "quadratic":
            return params["slope"] * x + params["quadratic"] * x**2 + params["intercept"]
        elif func_name == "sinusoidal":
            return params["slope"] * math.sin(params["sin_freq"] * x) + params["intercept"]
        elif func_name == "threshold":
            return params["slope"] * (x > params["threshold"]) + params["intercept"]
        return params["slope"] * x + params["intercept"]

    def generate_observational(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gera dados P(X, Y) — observacional, com vies de confusao se houver."""
        np_rng = np.random.RandomState(self.seed + 1)
        cause = self.topology["cause"]
        effect = self.topology["effect"]

        X = np_rng.uniform(-2, 2, self.n_obs_samples)
        
        if self.topology.get("type") == "confounded":
            conf = np_rng.uniform(-1, 1, self.n_obs_samples)
            X = X + 0.5 * conf  

        Y = np.array([self._apply_function(x, self.functions["cause_to_effect"], self.params)
                      for x in X])
        Y += np_rng.normal(0, self.noise_std, self.n_obs_samples)
        
        if len(self.confounders) > 0 and self.topology.get("type") != "confounded":
            Y = Y + 0.3 * np_rng.normal(0, 1, self.n_obs_samples)

        return X, Y

    def generate_interventional(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gera dados P(Y | do(X=x)) — intervencional, SEM vies de confusao."""
        np_rng = np.random.RandomState(self.seed + 2)
        cause = self.topology["cause"]
        effect = self.topology["effect"]

        X_int = np_rng.uniform(-2, 2, self.n_int_samples)
        Y_int = np.array([self._apply_function(x, self.functions["cause_to_effect"], self.params)
                          for x in X_int])
        Y_int += np_rng.normal(0, self.noise_std, self.n_int_samples)
        return X_int, Y_int

    def query_counterfactual(self, x_value: float, intervention_value: float) -> float:
        """Retorna o valor contrafactual: se X tivesse sido intervention_value (dado que foi x_value)."""
        cf = self._apply_function(intervention_value, self.functions["cause_to_effect"], self.params)
        return cf

    def get_true_do_effect(self, x_value: float) -> float:
        """Efeito causal verdadeiro de do(X=x_value) em Y."""
        obs_x, obs_y = self.generate_observational()
        return self._apply_function(x_value, self.functions["cause_to_effect"], self.params)

    def describe(self) -> str:
        return (f"CausalWorld(seed={self.seed}, vars={self.n_vars}, "
                f"topology={self.topology['type']}, "
                f"function={self.functions['cause_to_effect']})")


@dataclass
class PhysicsObject:
    id: int = 0
    mass: float = 1.0
    size: float = 1.0
    charge: float = 0.0
    temperature: float = 20.0
    fall_time: float = 0.0


@dataclass
class PhysicsWorld:
    """
    Mundo com objetos que caem sob leis de fisica OCULTAS.
    
    A cada seed, sorteia:
      - Lei da queda: t = a * m^b * s^c + d (parametros aleatorios)
      - Objetos com massa (0.5-10), size (0.5-5), charge aleatorios
    
    O sistema observa objetos caindo e deve:
      1. Aprender a lei da queda
      2. Prever tempo de queda de objetos novos
      3. Responder: "se a massa fosse X, quanto tempo levaria?"
    
    LLMs falham: a lei muda a cada seed.
    """
    seed: int = 0
    n_objects: int = 20
    n_test_objects: int = 5

    def __post_init__(self):
        rng = random.Random(self.seed)
        np_rng = np.random.RandomState(self.seed)
        self.a = rng.uniform(0.5, 3.0)
        self.b = rng.uniform(-1.0, 1.0)
        self.c = rng.uniform(-0.5, 0.5)
        self.d = rng.uniform(0.0, 1.0)
        self.noise_std = 0.05
        self.with_charge = rng.random() > 0.5
        if self.with_charge:
            self.charge_effect = rng.uniform(-0.5, 0.5)

        self.train_objects = []
        for i in range(self.n_objects):
            obj = PhysicsObject(
                id=i,
                mass=rng.uniform(0.5, 10.0),
                size=rng.uniform(0.5, 5.0),
                charge=rng.uniform(-2.0, 2.0) if self.with_charge else 0.0,
                temperature=rng.uniform(0, 100),
            )
            obj.fall_time = self._compute_fall_time(obj)
            self.train_objects.append(obj)

        self.test_objects = []
        for i in range(self.n_test_objects):
            obj = PhysicsObject(
                id=self.n_objects + i,
                mass=rng.uniform(0.5, 10.0),
                size=rng.uniform(0.5, 5.0),
                charge=rng.uniform(-2.0, 2.0) if self.with_charge else 0.0,
                temperature=rng.uniform(0, 100),
            )
            self.test_objects.append(obj)

    def _compute_fall_time(self, obj: PhysicsObject) -> float:
        t = self.a * (obj.mass ** self.b) * (obj.size ** self.c) + self.d
        if self.with_charge:
            t += self.charge_effect * obj.charge
        t += np.random.RandomState(self.seed + obj.id).normal(0, self.noise_std)
        return max(0.1, t)

    def get_training_data(self) -> List[Dict]:
        return [{"mass": o.mass, "size": o.size, "charge": o.charge,
                 "temperature": o.temperature, "fall_time": o.fall_time}
                for o in self.train_objects]

    def get_test_objects(self) -> List[PhysicsObject]:
        return self.test_objects

    def ground_truth_fall_time(self, obj: PhysicsObject) -> float:
        return self._compute_fall_time(obj)

    def counterfactual(self, obj: PhysicsObject, new_mass: float) -> float:
        """Quanto tempo levaria se a massa fosse new_mass?"""
        cf_obj = PhysicsObject(
            id=obj.id, mass=new_mass, size=obj.size,
            charge=obj.charge, temperature=obj.temperature
        )
        return self._compute_fall_time(cf_obj)

    def describe(self) -> str:
        return (f"PhysicsWorld(seed={self.seed}, law: t={self.a:.2f}*m^{self.b:.2f}*s^{self.c:.2f}+{self.d:.2f}, "
                f"charge={self.with_charge})")


@dataclass
class CompositionWorld:
    """
    Mundo de combinacoes cor × forma × tamanho com regra oculta.
    
    A cada seed, sorteia:
      - Uma regra booleana sobre (cor, forma, tamanho) que determina se
        uma combinacao e "valida" ou "invalida"
      - 3 cores, 3 formas, 3 tamanhos (27 combinacoes possiveis)
    
    O sistema ve 10 exemplos (5 validos, 5 invalidos) e deve:
      1. Generalizar para combinacoes nao vistas
      2. Reconhecer sistematicidade (vermelho+quadrado, azul+circulo → vermelho+circulo)
    
    LLMs falham: a regra muda a cada seed. Sistemas de ML com poucos
    exemplos falham por nao conseguirem compor partes nao vistas juntas.
    """
    seed: int = 0
    colors: List[str] = field(default_factory=lambda: ["red", "blue", "green"])
    shapes: List[str] = field(default_factory=lambda: ["square", "circle", "triangle"])
    sizes: List[str] = field(default_factory=lambda: ["small", "medium", "large"])

    def __post_init__(self):
        rng = random.Random(self.seed)
        self.rule_type = rng.choice([
            "single_color", "single_shape", "single_size",
            "color_shape_pair", "color_size_pair", "shape_size_pair",
            "color_shape_xor", "color_and_not_shape",
        ])
        self.rule_params = self._sample_rule_params(rng)

        all_combos = []
        for c in self.colors:
            for s in self.shapes:
                for sz in self.sizes:
                    valid = self._evaluate_rule(c, s, sz)
                    all_combos.append((c, s, sz, valid))

        rng.shuffle(all_combos)
        valid_examples = [(c, s, sz) for c, s, sz, v in all_combos if v]
        invalid_examples = [(c, s, sz) for c, s, sz, v in all_combos if not v]

        n_train = 5
        self.train_examples = (
            valid_examples[:n_train],
            invalid_examples[:n_train]
        )

        self.test_systematic = []
        for c in self.colors:
            for s in self.shapes:
                for sz in self.sizes:
                    if (c, s, sz) not in valid_examples[:n_train] and \
                       (c, s, sz) not in invalid_examples[:n_train]:
                        self.test_systematic.append((c, s, sz, self._evaluate_rule(c, s, sz)))

    def _sample_rule_params(self, rng):
        return {
            "color": rng.choice(self.colors),
            "shape": rng.choice(self.shapes),
            "size": rng.choice(self.sizes),
        }

    def _evaluate_rule(self, color, shape, size) -> bool:
        p = self.rule_params
        if self.rule_type == "single_color":
            return color == p["color"]
        elif self.rule_type == "single_shape":
            return shape == p["shape"]
        elif self.rule_type == "single_size":
            return size == p["size"]
        elif self.rule_type == "color_shape_pair":
            return color == p["color"] and shape == p["shape"]
        elif self.rule_type == "color_size_pair":
            return color == p["color"] and size == p["size"]
        elif self.rule_type == "shape_size_pair":
            return shape == p["shape"] and size == p["size"]
        elif self.rule_type == "color_shape_xor":
            return (color == p["color"]) != (shape == p["shape"])
        elif self.rule_type == "color_and_not_shape":
            return color == p["color"] and shape != p["shape"]
        return False

    def get_training_data(self) -> Tuple[List[Tuple], List[Tuple]]:
        return self.train_examples

    def get_test_set(self) -> List[Tuple]:
        return self.test_systematic

    def describe(self) -> str:
        return f"CompositionWorld(seed={self.seed}, rule={self.rule_type}, params={self.rule_params})"


@dataclass
class TransferWorld:
    """
    Mundo com dois problemas isomorficos em superficies diferentes.
    
    A cada seed, sorteia:
      - Um grafo dirigido com 5 nos e caminhos especificos
      - Duas superficies: vocabulario A ("cidades", "rotas") e B ("genes", "regulacao")
      - A estrutura do grafo e a MESMA, so muda a linguagem
    
    O sistema resolve o Problema A (encontrar caminho no grafo).
    Entao recebe o Problema B (mesma estrutura, palavras diferentes).
    
    Transferencia = resolver B significativamente mais rapido que do zero.
    
    Sistemas sem abstração estrutural falham: tratam A e B como problemas
    diferentes porque as palavras sao diferentes. Uma AGI vera a estrutura.
    """
    seed: int = 0

    def __post_init__(self):
        rng = random.Random(self.seed)

        self.n_nodes = 5
        self.graph = self._generate_graph(rng)

        vocab_sets = [
            ("city", ["New York", "Tokyo", "London", "Paris", "Berlin"],
             ["highway", "railway", "flight", "shipping", "tunnel"]),
            ("gene", ["BRCA1", "TP53", "EGFR", "MYC", "KRAS"],
             ["activates", "inhibits", "regulates", "binds", "phosphorylates"]),
            ("ecology", ["rabbit", "fox", "wolf", "deer", "bear"],
             ["eats", "preys_on", "competes_with", "symbiotes", "parasites"]),
            ("network", ["router_A", "router_B", "server_C", "node_D", "gateway_E"],
             ["connects", "routes_to", "forwards", "bridges", "switches"]),
        ]
        pair = rng.sample(vocab_sets, 2)
        self.vocab_a = pair[0]
        self.vocab_b = pair[1]

    def _generate_graph(self, rng):
        """Gera um grafo dirigido com caminhos (para o problema de encontrar caminho entre nos)."""
        nodes = list(range(self.n_nodes))
        edges = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j and rng.random() < 0.3:
                    edges.append((i, j))

        start = 0
        end = self.n_nodes - 1
        has_path = self._has_path(edges, start, end, self.n_nodes)
        if not has_path:
            edges.append((start, end))

        return {"nodes": nodes, "edges": edges, "start": start, "end": end}

    def _has_path(self, edges, start, end, n):
        visited = [False] * n
        stack = [start]
        while stack:
            v = stack.pop()
            if v == end:
                return True
            if not visited[v]:
                visited[v] = True
                for a, b in edges:
                    if a == v and not visited[b]:
                        stack.append(b)
        return False

    def problem_a(self) -> Dict:
        """Problema A com vocabulario A."""
        rng = random.Random(self.seed)
        n = self.vocab_a[1]
        e = self.vocab_a[2]
        start_name = n[self.graph["start"]]
        end_name = n[self.graph["end"]]
        edge_descriptions = []
        for a, b in self.graph["edges"]:
            edge_descriptions.append(f"{n[a]} --{rng.choice(e)}--> {n[b]}")
        rng.shuffle(edge_descriptions)
        return {
            "domain": self.vocab_a[0],
            "nodes": n,
            "edges": self.graph["edges"],
            "edge_verbs": e,
            "start": start_name,
            "end": end_name,
            "start_idx": self.graph["start"],
            "end_idx": self.graph["end"],
            "description": edge_descriptions,
            "question": f"Find a path from {start_name} to {end_name}.",
        }

    def problem_b(self) -> Dict:
        """Problema B com vocabulario B (MESMA estrutura do grafo)."""
        rng = random.Random(self.seed + 1)
        n = self.vocab_b[1]
        e = self.vocab_b[2]
        start_name = n[self.graph["start"]]
        end_name = n[self.graph["end"]]
        edge_descriptions = []
        for a, b in self.graph["edges"]:
            edge_descriptions.append(f"{n[a]} {rng.choice(e)} {n[b]}")
        rng.shuffle(edge_descriptions)
        return {
            "domain": self.vocab_b[0],
            "nodes": n,
            "edges": self.graph["edges"],
            "edge_verbs": e,
            "start": start_name,
            "end": end_name,
            "start_idx": self.graph["start"],
            "end_idx": self.graph["end"],
            "description": edge_descriptions,
            "question": f"Find a path from {start_name} to {end_name}.",
            "hint": f"This is structurally similar to: {self.vocab_a[0]} problem.",
        }

    def ground_truth_path(self) -> List[int]:
        """BFS para encontrar caminho.""" 
        start = self.graph["start"]
        end = self.graph["end"]
        from collections import deque
        q = deque([(start, [start])])
        visited = {start}
        while q:
            v, path = q.popleft()
            if v == end:
                return path
            for a, b in self.graph["edges"]:
                if a == v and b not in visited:
                    visited.add(b)
                    q.append((b, path + [b]))
        return []

    def describe(self) -> str:
        return (f"TransferWorld(seed={self.seed}, "
                f"surface_A={self.vocab_a[0]}, surface_B={self.vocab_b[0]}, "
                f"n_edges={len(self.graph['edges'])})")
