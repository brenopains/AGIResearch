import random

class PoetManager:
    """
    POET-Lite (Paired Open-Ended Trailblazer).
    Maintains a population of Agent-Environment pairs.
    Mutates environments and filters them by Goldilocks criteria (not too easy, not too hard).
    """
    def __init__(self, min_reward=10.0, max_reward=90.0, reproduction_rate=1):
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reproduction_rate = reproduction_rate
        self.pairs = []

    def initialize(self, pairs):
        self.pairs = pairs

    def evaluate_all(self):
        # Cross evaluation (N agents x M envs)
        scores = {}
        for a, _ in self.pairs:
            for _, e in self.pairs:
                scores[(a.id, e.id)] = a.evaluate(e)
        return scores

    def step(self):
        """
        Executes one POET macro-step:
        1. Mutate environments.
        2. Evaluate all agents on mutated environments.
        3. Filter mutated environments using Goldilocks.
        4. Create new pairs.
        """
        new_pairs = []
        
        for agent, env in self.pairs:
            for _ in range(self.reproduction_rate):
                mutated_env = env.mutate()
                
                # Check Goldilocks criteria with the current agent
                reward = agent.evaluate(mutated_env)
                
                if self.min_reward <= reward <= self.max_reward:
                    mutated_env.passed_goldilocks = True
                    new_pairs.append((agent, mutated_env))
                    
        # Update current population (in a real scenario we'd do cap size, transfer, etc.)
        # Here we just combine them and return for testing purposes
        self.pairs.extend(new_pairs)
        
        # Keep a max capacity to prevent explosion
        if len(self.pairs) > 10:
            self.pairs = self.pairs[-10:]
            
        return self.pairs
