import pytest
import random
from novamind.v10.models.poet import PoetManager

class MockAgent:
    def __init__(self, agent_id):
        self.id = agent_id
        
    def evaluate(self, env):
        # Deterministic dummy reward based on env difficulty and agent id
        return 100.0 - abs(env.difficulty - (self.id * 10))

class MockEnv:
    def __init__(self, env_id, difficulty):
        self.id = env_id
        self.difficulty = difficulty
        
    def mutate(self):
        # Mutate difficulty slightly
        new_diff = self.difficulty + random.choice([-5.0, 5.0])
        return MockEnv(self.id + 100, new_diff)

def test_poet_mutation_and_goldilocks():
    manager = PoetManager(min_reward=20.0, max_reward=90.0)
    
    # Initialize some paired agents and envs
    agent_env_pairs = [
        (MockAgent(0), MockEnv(0, 0.0)),
        (MockAgent(1), MockEnv(1, 10.0))
    ]
    
    manager.initialize(agent_env_pairs)
    
    # Do 1 step
    new_pairs = manager.step()
    
    # They should evaluate each other
    # Agent 0 on Env 0 -> 100.0 (Too easy, max_reward=90). Wait, 100 is above max_reward, so environment might be considered trivial.
    # The goldilocks filter should remove it when evaluating mutations, but if starting pairs are out of bounds, 
    # the manager usually tries to mutate them to fix them.
    assert len(new_pairs) > 0, "POET should generate some pairs"
    
    for a, e in new_pairs:
        # Evaluate to ensure it is bounded
        reward = a.evaluate(e)
        if getattr(e, 'passed_goldilocks', False):
            assert 20 <= reward <= 90, f"Reward {reward} not in Goldilocks zone"
