import os
import json
import torch
import random

class ARCDataset:
    """Loads ARC JSON files and pads them to fixed max dimensions for transformer consumption."""
    def __init__(self, data_dir, max_h=30, max_w=30, pad_val=10, dev='cpu'):
        self.data_dir = data_dir
        self.max_h = max_h
        self.max_w = max_w
        self.pad_val = pad_val
        self.dev = dev
        self.tasks = []
        
        if not os.path.exists(data_dir):
            print(f"Warning: ARC data dir {data_dir} not found.")
            return
            
        for f in os.listdir(data_dir):
            if f.endswith('.json'):
                with open(os.path.join(data_dir, f), 'r') as file:
                    self.tasks.append((f, json.load(file)))
        
        print(f"Loaded {len(self.tasks)} ARC tasks from {data_dir}.")
                    
    def pad_grid(self, grid):
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        
        padded = torch.full((self.max_h, self.max_w), self.pad_val, dtype=torch.long, device=self.dev)
        
        # Trim if larger than max
        h_end = min(h, self.max_h)
        w_end = min(w, self.max_w)
        
        if h_end > 0 and w_end > 0:
            grid_tensor = torch.tensor(grid, dtype=torch.long, device=self.dev)
            padded[:h_end, :w_end] = grid_tensor[:h_end, :w_end]
            
        return padded.reshape(-1)
        
    def sample_batch(self, batch_size=16, k_support=3, q_queries=1):
        xs_batch, ys_batch = [], []
        xq_batch, yq_batch = [], []
        
        for _ in range(batch_size):
            name, task = random.choice(self.tasks)
            train_pairs = task['train']
            test_pairs = task['test']
            
            # Support
            xs, ys = [], []
            for _ in range(k_support):
                pair = random.choice(train_pairs)
                xs.append(self.pad_grid(pair['input']))
                ys.append(self.pad_grid(pair['output']))
                
            xs_batch.append(torch.stack(xs))
            ys_batch.append(torch.stack(ys))
            
            # Query
            xq, yq = [], []
            for _ in range(q_queries):
                pair = random.choice(test_pairs)
                xq.append(self.pad_grid(pair['input']))
                yq.append(self.pad_grid(pair['output']))
                
            xq_batch.append(torch.stack(xq))
            yq_batch.append(torch.stack(yq))
            
        return (torch.stack(xs_batch), torch.stack(ys_batch)), (torch.stack(xq_batch), torch.stack(yq_batch))
