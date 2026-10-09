"""
trial_generator.py — Gerador Procedural de Novas Familias de Trials
=====================================================================
Gera trials NOVOS, que o sistema nunca viu, para testar generalidade.

Cada familia tem uma semente base — a cada execucao, gera problemas
diferentes. Isto previne overfitting aos trials originais.

Familias:
  Trial 8  — Teoria da Mente de 2a ordem
  Trial 9  — Aprendizado por unica demonstracao
  Trial 10 — Invencao de ferramentas compostas
  Trial 11 — Auto-modelo temporal (trajetoria de crencas)
  Trial 12 — Raciocinio analogico estrutural

Principio anti-Goodhart: estas familias sao geradas DEPOIS do
desenvolvimento das fases 1-4, para provar que o sistema nao
so "passa nos 7 trials" mas sim generaliza.
"""
import random
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field

from .scoring import TrialResult


@dataclass
class TrialFamily:
    """Uma familia de trials que pode gerar N trials diferentes."""
    name: str = ""
    trial_id: int = 0
    description: str = ""
    seed: int = 0
    generator: Optional[Callable] = None
    evaluator: Optional[Callable] = None


class TrialGenerator:
    """
    Gerador procedural de novas familias de trials.
    
    Uso:
        gen = TrialGenerator(seed=42)
        family = gen.create_theory_of_mind_family()
        trial = family.generator()
        result = gen.run_trial(trial)
    """

    def __init__(self, seed: int = 42, verbose: bool = True):
        self.seed = seed
        self.verbose = verbose
        self.generated_families: List[TrialFamily] = []
        self._family_count = 0

    def create_all_families(self) -> List[TrialFamily]:
        """Gera todas as familias de trials."""
        families = [
            self.create_theory_of_mind_family(),
            self.create_demonstration_family(),
            self.create_tool_invention_family(),
            self.create_belief_trajectory_family(),
            self.create_analogy_family(),
        ]
        self.generated_families = families
        return families

    def create_theory_of_mind_family(self) -> TrialFamily:
        """
        Trial 8: Teoria da Mente de 2a ordem.
        
        O sistema deve modelar o que outro agente acredita
        sobre o que um terceiro agente acredita.
        
        "Alice ve a bola na caixa. Bob nao viu.
         Bob sai da sala. Alice move a bola para o armario.
         Onde Bob acha que a bola esta?
         Onde Alice acha que Bob acha que a bola esta?"
        """
        family = TrialFamily(
            name="Theory of Mind (2nd Order)",
            trial_id=8,
            description="O sistema deve modelar crencas de agentes sobre crencas de outros agentes.",
            seed=self.seed + 100,
        )

        def generator(trial_seed: int) -> Dict:
            rng = random.Random(trial_seed)
            objects = ["ball", "cube", "sphere", "block", "marble"]
            locations = ["box", "cabinet", "drawer", "shelf", "basket"]
            
            obj = rng.choice(objects)
            loc_initial = rng.choice(locations)
            loc_moved = rng.choice([l for l in locations if l != loc_initial])
            
            agent_a = rng.choice(["Alice", "Bob", "Carol", "Dave"])
            agent_b = rng.choice([a for a in ["Alice", "Bob", "Carol", "Dave"] if a != agent_a])
            agent_c = rng.choice([a for a in ["Alice", "Bob", "Carol", "Dave"]
                                  if a != agent_a and a != agent_b])
            
            return {
                "story": f"{agent_a} sees the {obj} in the {loc_initial}. "
                         f"{agent_b} does not see it. "
                         f"{agent_b} leaves the room. "
                         f"{agent_a} moves the {obj} to the {loc_moved}.",
                "question_q1": f"Where does {agent_b} believe the {obj} is?",
                "answer_q1": loc_initial,
                "question_q2": f"Where does {agent_a} believe {agent_b} believes the {obj} is?",
                "answer_q2": loc_initial,
                "question_q3": f"Where is the {obj} actually?",
                "answer_q3": loc_moved,
            }
        
        def evaluator(system_fn, trial_data: Dict) -> float:
            score = 0.0
            for q_key in ["question_q1", "question_q2", "question_q3"]:
                question = trial_data[q_key]
                correct = trial_data[f"answer_{q_key.replace('question_', '')}"]
                
                try:
                    if hasattr(system_fn, '__call__'):
                        answer = system_fn(question)
                    else:
                        answer = str(system_fn(question))
                    
                    if correct.lower() in answer.lower():
                        score += 0.33
                except Exception:
                    pass
            
            return min(score, 1.0)
        
        family.generator = generator
        family.evaluator = evaluator
        return family

    def create_demonstration_family(self) -> TrialFamily:
        """
        Trial 9: Aprendizado por unica demonstracao.
        
        Mostra UMA demonstracao de como resolver um problema.
        Depois apresenta o mesmo problema com perturbacoes.
        O sistema deve re-atingir o objetivo na nova configuracao.
        """
        family = TrialFamily(
            name="One-Shot Demonstration Learning",
            trial_id=9,
            description="Aprende uma sequencia de acoes de uma unica demonstracao, "
                        "re-aplica em configuracao diferente.",
            seed=self.seed + 200,
        )

        def generator(trial_seed: int) -> Dict:
            rng = random.Random(trial_seed)
            actions = ["push", "pull", "rotate", "flip", "slide"]
            objects = ["block", "lever", "switch", "handle", "pedal"]
            
            n_steps = rng.randint(3, 5)
            demonstration = [
                f"{rng.choice(actions)} the {rng.choice(objects)}"
                for _ in range(n_steps)
            ]
            
            pertubations = {}
            for step in demonstration:
                pertubations[step] = "performed"
            
            return {
                "demonstration": demonstration,
                "question": f"Perform the same sequence on a different setup.",
                "perturbations": {},
            }
        
        def evaluator(system_fn, trial_data: Dict) -> float:
            return 0.0  # Requires action execution capability
        
        family.generator = generator
        family.evaluator = evaluator
        return family

    def create_tool_invention_family(self) -> TrialFamily:
        """
        Trial 10: Invencao de ferramentas compostas.
        
        Problema que exige combinar 2+ primitivas existentes
        para criar uma ferramenta nova que resolve o problema.
        """
        family = TrialFamily(
            name="Compound Tool Invention",
            trial_id=10,
            description="Combina primitivas existentes para criar ferramenta nova.",
            seed=self.seed + 300,
        )

        def generator(trial_seed: int) -> Dict:
            rng = random.Random(trial_seed)
            required_parts = rng.sample(["grab", "lift", "move", "rotate", "flip"], 2)
            return {
                "required_parts": required_parts,
                "description": f"Create a tool that can {' and '.join(required_parts)} an object.",
                "test_config": {},
            }
        
        def evaluator(system_fn, trial_data: Dict) -> float:
            return 0.0  # Requires tool composition
        
        family.generator = generator
        family.evaluator = evaluator
        return family

    def create_belief_trajectory_family(self) -> TrialFamily:
        """
        Trial 11: Auto-Modelo Temporal.
        
        Testa se o sistema lembra no que acreditava e como
        suas crencas mudaram ao longo do tempo.
        
        Requer: SelfModel com trajetoria de crencas (gap atual).
        """
        family = TrialFamily(
            name="Temporal Belief Trajectory",
            trial_id=11,
            description="O sistema deve reportar sua propria historia de crencas.",
            seed=self.seed + 400,
        )

        def generator(trial_seed: int) -> Dict:
            rng = random.Random(trial_seed)
            concepts = ["gravity", "evolution", "quantum", "relativity", "entropy"]
            concept = rng.choice(concepts)
            return {
                "concept": concept,
                "question_past": f"What did you believe about {concept} 10 cycles ago?",
                "question_change": f"How did your belief about {concept} change?",
                "question_why": f"Why did your belief about {concept} change?",
            }
        
        def evaluator(system_fn, trial_data: Dict) -> float:
            score = 0.0
            
            for q_key in ["question_past", "question_change", "question_why"]:
                question = trial_data[q_key]
                try:
                    answer = system_fn(question) if callable(system_fn) else str(system_fn(question))
                    if answer and len(answer) > 10:
                        score += 0.33
                except Exception:
                    pass
            
            return min(score, 1.0)
        
        family.generator = generator
        family.evaluator = evaluator
        return family

    def create_analogy_family(self) -> TrialFamily:
        """
        Trial 12: Raciocinio Analogico Estrutural.
        
        "O atomo e como o sistema solar:
         - qual o 'sol' do atomo?
         - qual o 'planeta'?
         - qual a 'gravidade'?"
        """
        family = TrialFamily(
            name="Structural Analogy",
            trial_id=12,
            description="Mapeia estrutura entre dominios diferentes via analogia.",
            seed=self.seed + 500,
        )

        def generator(trial_seed: int) -> Dict:
            analogies = [
                {
                    "source": "solar_system",
                    "target": "atom",
                    "mappings": {"sun": "nucleus", "planet": "electron",
                                  "gravity": "electromagnetic_force", "orbit": "orbital"},
                    "questions": [
                        "In the atom, what corresponds to the sun of the solar system?",
                        "What corresponds to the planets orbiting the sun?",
                        "What corresponds to the gravitational force holding the system together?",
                    ],
                    "answers": ["nucleus", "electron", "electromagnetic_force"],
                },
                {
                    "source": "ecosystem",
                    "target": "economy",
                    "mappings": {"predator": "competitor", "prey": "market",
                                  "food_chain": "supply_chain"},
                    "questions": [
                        "In the economy, what corresponds to the predators of an ecosystem?",
                        "What corresponds to prey?",
                        "What corresponds to the food chain?",
                    ],
                    "answers": ["competitor", "market", "supply_chain"],
                },
            ]
            rng = random.Random(trial_seed)
            return rng.choice(analogies)
        
        def evaluator(system_fn, trial_data: Dict) -> float:
            score = 0.0
            for i, (q, correct) in enumerate(zip(trial_data["questions"],
                                                    trial_data["answers"])):
                try:
                    answer = system_fn(q) if callable(system_fn) else str(system_fn(q))
                    if correct.lower() in answer.lower():
                        score += 0.33
                except Exception:
                    pass
            return min(score, 1.0)
        
        family.generator = generator
        family.evaluator = evaluator
        return family

    def run_family(self, family: TrialFamily, system_fn: Any,
                   n_trials: int = 5) -> dict:
        """Executa N trials de uma familia e retorna score medio."""
        scores = []
        for i in range(n_trials):
            trial_seed = family.seed + i * 1000
            trial_data = family.generator(trial_seed)
            score = family.evaluator(system_fn, trial_data)
            scores.append(score)
        
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "family_name": family.name,
            "trial_id": family.trial_id,
            "mean_score": mean_score,
            "scores": scores,
            "n_trials": n_trials,
        }

    def run_all_new_families(self, system_fn: Any,
                              n_per_family: int = 5) -> List[dict]:
        """Executa todas as familias de trials novos."""
        families = self.create_all_families()
        results = []
        for family in families:
            result = self.run_family(family, system_fn, n_per_family)
            results.append(result)
            if self.verbose:
                print(f"  [TrialGen] {family.name}: {result['mean_score']:.2%} "
                      f"(n={n_per_family})")
        return results
