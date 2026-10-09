import torch
import torch.nn as nn

class EWC:
    """
    Elastic Weight Consolidation regularizer.
    Penalizes weight modifications based on Fisher Information Matrix diagonal.
    """
    def __init__(self, model: nn.Module, lambda_ewc: float = 0.5):
        self.model = model
        self.lambda_ewc = lambda_ewc
        # Supports multi-task memory, store by task index
        self._means = {}
        self._fishers = {}
        
    def register_task(self, params_dict: dict, fisher_dict: dict):
        """
        Register a set of parameters and their diagonal fisher matrix values.
        """
        task_id = len(self._means)
        self._means[task_id] = params_dict
        self._fishers[task_id] = fisher_dict

    def penalty(self) -> torch.Tensor:
        """
        Computes the EWC penalty to be added to the loss.
        """
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        if not self._means:
            return loss # No previously registered tasks
            
        # Iterate over all stored tasks and apply penalty
        for task_id in self._means:
            for name, param in self.model.named_parameters():
                if name in self._means[task_id] and name in self._fishers[task_id]:
                    # L2 penalty weighted by Fisher info
                    fisher = self._fishers[task_id][name]
                    mean = self._means[task_id][name]
                    loss += torch.sum(fisher * (param - mean) ** 2)
                    
        return self.lambda_ewc * (loss / 2.0)
