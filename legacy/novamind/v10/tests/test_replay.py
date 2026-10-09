import pytest
import numpy as np
from novamind.v10.models.replay_buffer import ReplayBuffer

def test_replay_buffer_storage():
    buffer = ReplayBuffer(capacity=100)
    
    # Store dummy episodes
    # Let's say episode format is a dict with 'obs', 'action', 'reward'
    ep1 = {'obs': np.zeros((50, 64)), 'action': np.zeros((50, 10))}
    ep2 = {'obs': np.ones((50, 64)), 'action': np.ones((50, 10))}
    
    buffer.add(ep1)
    buffer.add(ep2)
    
    assert len(buffer) == 2
    
def test_replay_buffer_sampling():
    buffer = ReplayBuffer(capacity=10)
    for i in range(15):
        buffer.add({'obs': np.full((10, 64), i)})
        
    # Should only hold 10
    assert len(buffer) == 10
    
    # Check shape of batch
    batch = buffer.sample(batch_size=4)
    # Batch is a dict of lists or stacked arrays
    assert batch['obs'].shape == (4, 10, 64)
