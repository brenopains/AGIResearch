import random
import numpy as np

class ReplayBuffer:
    """
    Experience replay buffer for storing trajectories.
    Uses Reservoir sampling or simple FIFO for capacity management.
    """
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self._next_idx = 0

    def add(self, episode):
        """
        Add an entire episode (dict of numpy arrays of length T).
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(episode)
        else:
            self.buffer[self._next_idx] = episode
            self._next_idx = (self._next_idx + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of episodes uniformly.
        Returns a dict of stacked numpy arrays [BatchSize, Time, ...]
        """
        indices = random.sample(range(len(self.buffer)), batch_size)
        
        # Batch transpose dicts
        batch = {}
        for key in self.buffer[0].keys():
            batch[key] = np.stack([self.buffer[i][key] for i in indices], axis=0)
            
        return batch

    def __len__(self):
        return len(self.buffer)
