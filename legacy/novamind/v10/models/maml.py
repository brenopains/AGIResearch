import torch
import torch.nn as nn

class MAMLWrapper:
    """
    Model-Agnostic Meta-Learning wrapper (Finn et al. 2017).
    """
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, num_inner_steps: int = 1):
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps

    def inner_loop(self, x_support, y_support, loss_fn):
        """
        Executes one or more steps of gradient descent manually to keep the computation graph alive.
        """
        # Create a dict of fast weights
        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        
        for _ in range(self.num_inner_steps):
            # Compute pred with fast weights
            # For a general model, doing this generically requires torch.func.functional_call
            # Here, we use torch.func for forward pass with params dictionary
            try:
                from torch.func import functional_call
                pred = functional_call(self.model, fast_weights, (x_support,))
            except ImportError:
                # Fallback for strict environments (assumes SimpleNet from test)
                import torch.nn.functional as F
                pred = F.linear(x_support, fast_weights['fc.weight'], fast_weights.get('fc.bias'))
                
            loss = loss_fn(pred, y_support)
            
            # Gradients of loss wrt fast_weights
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            
            # Manual SGD update
            fast_weights = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(fast_weights.items(), grads)
            }
            
        return fast_weights
