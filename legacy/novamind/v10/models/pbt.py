import random
import numpy as np

class PBTController:
    """
    Population-Based Training Controller.
    Exploits (copies weights/state from top performers to bottom performers) and 
    Explores (mutates hyperparameters of copied models).
    """
    def __init__(self, population_size=4, mutation_factors=(0.8, 1.2)):
        self.population_size = population_size
        self.mutation_factors = mutation_factors

    def exploit_and_explore(self, agents: list, performances: dict, bottom_pct=0.25, top_pct=0.25):
        """
        Takes a list of agent objects and their fitness scores.
        agents must implement: .agent_id, .hyperparams, .get_weights(), .set_weights()
        """
        # Sort agents by performance
        sorted_ids = sorted(performances.keys(), key=lambda x: performances[x], reverse=True)
        
        num_bottom = max(1, int(self.population_size * bottom_pct))
        num_top = max(1, int(self.population_size * top_pct))
        
        bottom_ids = sorted_ids[-num_bottom:]
        top_ids = sorted_ids[:num_top]
        
        agent_dict = {a.agent_id: a for a in agents}
        
        for bottom_id in bottom_ids:
            # Exploit: Pick a random top agent to clone
            target_top_id = random.choice(top_ids)
            
            bottom_agent = agent_dict[bottom_id]
            top_agent = agent_dict[target_top_id]
            
            # Clone weights
            bottom_agent.set_weights(top_agent.get_weights())
            
            # Explore: Mutate hyperparams
            for hp_name, hp_val in top_agent.hyperparams.items():
                factor = random.choice(self.mutation_factors)
                bottom_agent.hyperparams[hp_name] = hp_val * factor
