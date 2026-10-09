import torch
import torch.nn as nn
import pytest
from novamind.v10.models.self_play import SelfDistillationLoop

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.fc(x)

def test_self_distillation():
    student = SimpleNet()
    # Initialize teacher differently to verify EMA syncs it
    teacher = SimpleNet()
    
    loop = SelfDistillationLoop(student, teacher, momentum=0.5)
    
    x = torch.randn(2, 10)
    
    # Run step 1
    # Check that gradients flow to student and teacher is updated
    loss = loop.compute_loss(x, x)
    assert loss.requires_grad, "Loss must be connected to student graph"
    
    # Store old teacher weights
    old_teacher_weight = teacher.fc.weight.clone()
    
    # Update teacher
    loop.update_teacher()
    
    # Check EMA updated the teacher weights
    assert not torch.allclose(old_teacher_weight, teacher.fc.weight)
