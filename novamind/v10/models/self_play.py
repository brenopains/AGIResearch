import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfDistillationLoop:
    """
    Manages the self-distillation / self-play process.
    The Student learns to predict the Teacher's outputs, and the Teacher is slowly updated via EMA.
    """
    def __init__(self, student: nn.Module, teacher: nn.Module, momentum: float = 0.996):
        self.student = student
        self.teacher = teacher
        self.momentum = momentum
        
        # Teacher does not require gradients
        for param in self.teacher.parameters():
            param.requires_grad = False

    def compute_loss(self, student_inputs, teacher_inputs):
        """
        Computes Mean Squared Error distance between student and teacher representations.
        Useful for JEPA-style or Data2Vec style distillation.
        """
        student_repr = self.student(student_inputs)
        
        with torch.no_grad():
            teacher_repr = self.teacher(teacher_inputs)
            
        # L2 Loss or smooth L1 loss
        return F.mse_loss(student_repr, teacher_repr)

    def update_teacher(self):
        """
        Polyak Averaging / Exponential Moving Average (EMA) update for teacher.
        """
        with torch.no_grad():
            for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
                param_t.data.mul_(self.momentum).add_((1.0 - self.momentum) * param_s.detach().data)
