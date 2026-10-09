import pytest
from novamind.v10.models.pbt import PBTController

class MockAgent:
    def __init__(self, agent_id, lr=0.1):
        self.agent_id = agent_id
        self.hyperparams = {'lr': lr}
        self.weights = {"param1": agent_id * 10} # Mock weights tied to ID
        
    def get_weights(self):
        return self.weights.copy()
        
    def set_weights(self, weights):
        self.weights = weights.copy()

def test_pbt_exploit_explore():
    controller = PBTController(population_size=4)
    # Init 4 agents
    agents = [MockAgent(i) for i in range(4)]
    
    # Let's say agents 0, 1 act badly, 2, 3 act well
    performances = {
        0: 10.0,
        1: 20.0,
        2: 90.0,
        3: 100.0 # Best
    }
    
    # Run PBT exploit and explore
    controller.exploit_and_explore(agents, performances, bottom_pct=0.25, top_pct=0.25)
    
    # 0.25 of 4 is 1. Bottom 1 is index 0. Top 1 is index 3.
    # Agent 0 should have its weights replaced by Agent 3's weights.
    # Agent 0 should have its hyperparams mutated.
    
    assert agents[0].weights["param1"] == 30, "Agent 0 (worst) should have copied weights of Agent 3 (best)"
    assert agents[0].hyperparams['lr'] != 0.1, "Agent 0 hyperparams should be mutated"
    
    # Top agents should remain unchanged
    assert agents[3].weights["param1"] == 30
    assert agents[3].hyperparams['lr'] == 0.1
